/*
 * FLUX Main Implementation
 *
 * Main entry point for the FLUX.2 klein 4B inference engine.
 * Ties together all components: tokenizer, text encoder, VAE, transformer, sampling.
 */

#include "flux.h"
#include "flux_kernels.h"
#include "flux_safetensors.h"
#include "flux_qwen3.h"
#include "embcache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

#ifdef USE_METAL
#include "flux_metal.h"
#endif

/* ========================================================================
 * Forward Declarations for Internal Types
 * ======================================================================== */

typedef struct flux_tokenizer flux_tokenizer;
typedef struct flux_vae flux_vae_t;
typedef struct flux_transformer flux_transformer_t;

/* Internal function declarations */
extern flux_tokenizer *flux_tokenizer_load(const char *path);
extern void flux_tokenizer_free(flux_tokenizer *tok);
extern int *flux_tokenize(flux_tokenizer *tok, const char *text,
                          int *num_tokens, int max_len);

extern flux_vae_t *flux_vae_load_safetensors(safetensors_file_t *sf);
extern void flux_vae_free(flux_vae_t *vae);
extern float *flux_vae_encode(flux_vae_t *vae, const float *img,
                              int batch, int H, int W, int *out_h, int *out_w);
extern flux_image *flux_vae_decode(flux_vae_t *vae, const float *latent,
                                   int batch, int latent_h, int latent_w);
extern float *flux_image_to_tensor(const flux_image *img);

extern flux_transformer_t *flux_transformer_load_safetensors(safetensors_file_t *sf);
extern flux_transformer_t *flux_transformer_load_safetensors_mmap(safetensors_file_t *sf);
extern void flux_transformer_free(flux_transformer_t *tf);
extern float *flux_transformer_forward(flux_transformer_t *tf,
                                        const float *img_latent, int img_h, int img_w,
                                        const float *txt_emb, int txt_seq,
                                        float timestep);

extern float *flux_sample_euler(void *transformer, void *text_encoder,
                                float *z, int batch, int channels, int h, int w,
                                const float *text_emb, int text_seq,
                                const float *schedule, int num_steps,
                                void (*progress_callback)(int step, int total));
extern float *flux_sample_euler_with_refs(void *transformer, void *text_encoder,
                                          float *z, int batch, int channels, int h, int w,
                                          const float *ref_latent, int ref_h, int ref_w,
                                          int t_offset,
                                          const float *text_emb, int text_seq,
                                          const float *schedule, int num_steps,
                                          void (*progress_callback)(int step, int total));

/* Multi-reference support */
typedef struct {
    const float *latent;
    int h, w;
    int t_offset;
} flux_ref_t;

extern float *flux_sample_euler_with_multi_refs(void *transformer, void *text_encoder,
                                                float *z, int batch, int channels, int h, int w,
                                                const flux_ref_t *refs, int num_refs,
                                                const float *text_emb, int text_seq,
                                                const float *schedule, int num_steps,
                                                void (*progress_callback)(int step, int total));

extern float *flux_linear_schedule(int num_steps);
extern float *flux_official_schedule(int num_steps, int image_seq_len);
extern float *flux_init_noise(int batch, int channels, int h, int w, int64_t seed);

/* ========================================================================
 * Text Encoder (Qwen3)
 * ======================================================================== */

/* Qwen3 text encoder is implemented in flux_qwen3.c */

/* ========================================================================
 * Main Context Structure
 * ======================================================================== */

struct flux_ctx {
    /* Components */
    flux_tokenizer *tokenizer;
    qwen3_encoder_t *qwen3_encoder;
    flux_vae_t *vae;
    flux_transformer_t *transformer;

    /* Configuration */
    int max_width;
    int max_height;
    int default_steps;

    /* Model info */
    char model_name[64];
    char model_version[32];
    char model_dir[512];  /* For reloading text encoder if released */

    /* Memory mode */
    int use_mmap;  /* Use mmap for text encoder (lower memory, slower) */

    /* Generation behavior */
    int strict_mode;
    int warned_missing_embeddings;

    /* Prompt embedding cache */
    emb_cache_t *emb_cache;
    int embed_cache_enabled;
    char tokenizer_config_id[80];
    char model_revision_id[80];
};

/* Global error message */
static char g_error_msg[256] = {0};

const char *flux_get_error(void) {
    return g_error_msg;
}

void flux_set_step_image_callback(flux_ctx *ctx, flux_step_image_cb_t callback) {
    flux_step_image_callback = callback;
    flux_step_image_vae = callback ? ctx->vae : NULL;
}

static void set_error(const char *msg) {
    strncpy(g_error_msg, msg, sizeof(g_error_msg) - 1);
    g_error_msg[sizeof(g_error_msg) - 1] = '\0';
}

/* ========================================================================
 * Model Loading from HuggingFace-style directory with safetensors files
 * ======================================================================== */

static int file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0;
}


static uint64_t fnv1a_bytes(const unsigned char *data, size_t n) {
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t hash_file_contents(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    unsigned char buf[4096];
    uint64_t h = 14695981039346656037ULL;
    for (;;) {
        size_t n = fread(buf, 1, sizeof(buf), f);
        if (n > 0) {
            for (size_t i = 0; i < n; i++) {
                h ^= buf[i];
                h *= 1099511628211ULL;
            }
        }
        if (n < sizeof(buf)) break;
    }
    fclose(f);
    return h;
}

static void init_model_ids(flux_ctx *ctx, const char *model_dir) {
    char path[1024];

    snprintf(path, sizeof(path), "%s/tokenizer/tokenizer_config.json", model_dir);
    uint64_t tok_hash = hash_file_contents(path);
    if (tok_hash == 0) {
        snprintf(path, sizeof(path), "%s/tokenizer/tokenizer.json", model_dir);
        tok_hash = hash_file_contents(path);
    }
    if (tok_hash == 0) tok_hash = fnv1a_bytes((const unsigned char *)"unknown", 7);
    snprintf(ctx->tokenizer_config_id, sizeof(ctx->tokenizer_config_id), "tokcfg:%016llx",
             (unsigned long long)tok_hash);

    const char *env_rev = getenv("FLUX_MODEL_REVISION");
    if (env_rev && env_rev[0]) {
        strncpy(ctx->model_revision_id, env_rev, sizeof(ctx->model_revision_id) - 1);
        return;
    }

    snprintf(path, sizeof(path), "%s/REVISION", model_dir);
    FILE *rf = fopen(path, "r");
    if (!rf) {
        snprintf(path, sizeof(path), "%s/.revision", model_dir);
        rf = fopen(path, "r");
    }
    if (rf) {
        if (fgets(ctx->model_revision_id, sizeof(ctx->model_revision_id), rf)) {
            size_t len = strlen(ctx->model_revision_id);
            while (len > 0 &&
                  (ctx->model_revision_id[len - 1] == '\n' ||
                   ctx->model_revision_id[len - 1] == '\r')) {
                ctx->model_revision_id[--len] = '\0';
            }
        }
        fclose(rf);
    }

    if (ctx->model_revision_id[0] == '\0') {
        strncpy(ctx->model_revision_id, "unknown", sizeof(ctx->model_revision_id) - 1);
    }
}

flux_ctx *flux_load_dir(const char *model_dir) {
    char path[1024];

    flux_ctx *ctx = calloc(1, sizeof(flux_ctx));
    if (!ctx) {
        set_error("Out of memory");
        return NULL;
    }

    /* Set defaults - max 1792x1792 (requires ~18GB VAE work buffers) */
    ctx->max_width = FLUX_VAE_MAX_DIM;
    ctx->max_height = FLUX_VAE_MAX_DIM;
    ctx->default_steps = 4;
    strncpy(ctx->model_name, "FLUX.2-klein-4B", sizeof(ctx->model_name) - 1);
    strncpy(ctx->model_version, "1.0", sizeof(ctx->model_version) - 1);
    strncpy(ctx->model_dir, model_dir, sizeof(ctx->model_dir) - 1);
    ctx->emb_cache = emb_lru_create(8);
    ctx->embed_cache_enabled = 1;
    init_model_ids(ctx, model_dir);

    /* Load VAE only at startup (~300MB).
     * Transformer and text encoder are loaded on-demand during generation
     * to support systems with limited RAM (e.g., 16GB). */
    snprintf(path, sizeof(path), "%s/vae/diffusion_pytorch_model.safetensors", model_dir);
    if (file_exists(path)) {
        safetensors_file_t *sf = safetensors_open(path);
        if (sf) {
            ctx->vae = flux_vae_load_safetensors(sf);
            safetensors_close(sf);
        }
    }

    /* Verify VAE is loaded */
    if (!ctx->vae) {
        set_error("Failed to load VAE - cannot generate images");
        flux_free(ctx);
        return NULL;
    }

    /* Verify transformer file exists (will be loaded on-demand) */
    snprintf(path, sizeof(path), "%s/transformer/diffusion_pytorch_model.safetensors", model_dir);
    if (!file_exists(path)) {
        set_error("Transformer model file not found");
        flux_free(ctx);
        return NULL;
    }
    /* Text encoder and transformer are loaded on-demand to reduce peak memory. */

    /* Initialize RNG */
    flux_rng_seed((uint64_t)time(NULL));

    return ctx;
}

void flux_free(flux_ctx *ctx) {
    if (!ctx) return;

    flux_tokenizer_free(ctx->tokenizer);
    qwen3_encoder_free(ctx->qwen3_encoder);
    flux_vae_free(ctx->vae);
    flux_transformer_free(ctx->transformer);
    emb_lru_destroy(ctx->emb_cache);

    free(ctx);
}

void flux_set_mmap(flux_ctx *ctx, int enable) {
    if (ctx) ctx->use_mmap = enable;
}

void flux_set_strict(flux_ctx *ctx, int enable) {
    if (ctx) ctx->strict_mode = enable ? 1 : 0;
}

void flux_set_embed_cache(flux_ctx *ctx, int enable) {
    if (!ctx) return;
    ctx->embed_cache_enabled = enable ? 1 : 0;
    if (ctx->emb_cache) emb_lru_set_enabled(ctx->emb_cache, ctx->embed_cache_enabled);
}

void flux_release_text_encoder(flux_ctx *ctx) {
    if (!ctx || !ctx->qwen3_encoder) return;

    qwen3_encoder_free(ctx->qwen3_encoder);
    ctx->qwen3_encoder = NULL;

#ifdef USE_METAL
    /* Reset all GPU state to ensure clean slate for transformer.
     * This clears weight caches, activation pools, and pending commands. */
    flux_metal_reset();
#endif
}

/* Load transformer on-demand if not already loaded */
static int flux_load_transformer_if_needed(flux_ctx *ctx) {
    if (ctx->transformer) return 1;  /* Already loaded */

    char path[1024];
    snprintf(path, sizeof(path), "%s/transformer/diffusion_pytorch_model.safetensors",
             ctx->model_dir);

    if (flux_phase_callback) flux_phase_callback("Loading FLUX.2 transformer", 0);
    safetensors_file_t *sf = safetensors_open(path);
    if (sf) {
        if (ctx->use_mmap) {
            /* Mmap mode: load only small weights, keep sf open for on-demand loading.
             * The transformer takes ownership of sf and will close it on free. */
            ctx->transformer = flux_transformer_load_safetensors_mmap(sf);
        } else {
            ctx->transformer = flux_transformer_load_safetensors(sf);
            safetensors_close(sf);
        }
    }
    if (flux_phase_callback) flux_phase_callback("Loading FLUX.2 transformer", 1);

    if (!ctx->transformer) {
        set_error("Failed to load transformer");
        return 0;
    }
    return 1;
}

/* Get transformer for debugging */
void *flux_get_transformer(flux_ctx *ctx) {
    return ctx ? ctx->transformer : NULL;
}

/* ========================================================================
 * Text Encoding
 * ======================================================================== */

float *flux_encode_text(flux_ctx *ctx, const char *prompt, int *out_seq_len) {
    if (!ctx || !prompt) {
        *out_seq_len = 0;
        return NULL;
    }

    if (ctx->embed_cache_enabled && ctx->emb_cache) {
        float *cached = NULL;
        int cached_seq = 0, cached_dim = 0;
        if (emb_lru_lookup(ctx->emb_cache, prompt,
                             ctx->tokenizer_config_id, ctx->model_revision_id,
                             &cached, &cached_seq, &cached_dim)) {
            if (cached && cached_seq > 0 && cached_dim == QWEN3_TEXT_DIM) {
                *out_seq_len = cached_seq;
                return cached;
            }
            free(cached);
        }
    }

    /* Load encoder if not already loaded */
    if (!ctx->qwen3_encoder && ctx->model_dir[0]) {
        if (flux_phase_callback) flux_phase_callback("Loading Qwen3 encoder", 0);
        ctx->qwen3_encoder = qwen3_encoder_load(ctx->model_dir, ctx->use_mmap);
        if (flux_phase_callback) flux_phase_callback("Loading Qwen3 encoder", 1);
        if (!ctx->qwen3_encoder && ctx->strict_mode) {
            set_error("Strict mode: failed to load Qwen3 text encoder");
        }
    }

    if (!ctx->qwen3_encoder) {
        if (ctx->strict_mode) {
            set_error("Strict mode: text embeddings unavailable (encoder missing)");
            *out_seq_len = 0;
            return NULL;
        }
        if (!ctx->warned_missing_embeddings) {
            fprintf(stderr, "Warning: Qwen3 text encoder unavailable; using zero embeddings. Use --strict to fail instead.\n");
            ctx->warned_missing_embeddings = 1;
        }
        *out_seq_len = QWEN3_MAX_SEQ_LEN;
        return (float *)calloc(QWEN3_MAX_SEQ_LEN * QWEN3_TEXT_DIM, sizeof(float));
    }

    /* Encode text using Qwen3 */
    if (flux_phase_callback) flux_phase_callback("encoding text", 0);
    float *embeddings = qwen3_encode_text(ctx->qwen3_encoder, prompt);
    if (flux_phase_callback) flux_phase_callback("encoding text", 1);

    if (!embeddings) {
        if (ctx->strict_mode) {
            set_error("Strict mode: failed to encode prompt embeddings");
            *out_seq_len = 0;
            return NULL;
        }
        if (!ctx->warned_missing_embeddings) {
            fprintf(stderr, "Warning: Prompt encoding failed; using zero embeddings. Use --strict to fail instead.\n");
            ctx->warned_missing_embeddings = 1;
        }
        *out_seq_len = QWEN3_MAX_SEQ_LEN;
        return (float *)calloc(QWEN3_MAX_SEQ_LEN * QWEN3_TEXT_DIM, sizeof(float));
    }

    *out_seq_len = QWEN3_MAX_SEQ_LEN;  /* Always 512 */
    if (ctx->embed_cache_enabled && ctx->emb_cache) {
        emb_lru_store(ctx->emb_cache, prompt,
                        ctx->tokenizer_config_id, ctx->model_revision_id,
                        embeddings, QWEN3_MAX_SEQ_LEN, QWEN3_TEXT_DIM);
    }
    return embeddings;
}

static flux_params flux_normalize_params(const flux_params *params) {
    flux_params p = params ? *params : (flux_params)FLUX_PARAMS_DEFAULT;

    if (p.width <= 0) p.width = FLUX_DEFAULT_WIDTH;
    if (p.height <= 0) p.height = FLUX_DEFAULT_HEIGHT;
    if (p.num_steps <= 0) p.num_steps = 4;

    p.width = (p.width / 16) * 16;
    p.height = (p.height / 16) * 16;
    if (p.width < 64) p.width = 64;
    if (p.height < 64) p.height = 64;

    return p;
}

static int flux_validate_params(flux_ctx *ctx, const flux_params *params) {
    if (params->width > FLUX_VAE_MAX_DIM || params->height > FLUX_VAE_MAX_DIM) {
        set_error("Image dimensions exceed maximum (1792x1792)");
        return 0;
    }
    (void)ctx;
    return 1;
}

/* ========================================================================
 * Image Generation
 * ======================================================================== */

flux_image *flux_generate(flux_ctx *ctx, const char *prompt,
                          const flux_params *params) {
    if (!ctx || !prompt) {
        set_error("Invalid context or prompt");
        return NULL;
    }

    flux_params p = flux_normalize_params(params);
    if (!flux_validate_params(ctx, &p)) {
        return NULL;
    }

    /* Encode text */
    int text_seq;
    float *text_emb = flux_encode_text(ctx, prompt, &text_seq);
    if (!text_emb) {
        set_error("Failed to encode prompt");
        return NULL;
    }

    /* Release text encoder to free ~8GB before loading transformer */
    flux_release_text_encoder(ctx);

    /* Load transformer now (after text encoder is freed to reduce peak memory) */
    if (!flux_load_transformer_if_needed(ctx)) {
        free(text_emb);
        return NULL;
    }

    /* Compute latent dimensions */
    int latent_h = p.height / 16;
    int latent_w = p.width / 16;
    int image_seq_len = latent_h * latent_w;

    /* Initialize noise */
    int64_t seed = (p.seed < 0) ? (int64_t)time(NULL) : p.seed;
    float *z = flux_init_noise(1, FLUX_LATENT_CHANNELS, latent_h, latent_w, seed);

    /* Get official FLUX.2 schedule (matches Python) */
    float *schedule = flux_official_schedule(p.num_steps, image_seq_len);

    /* Sample */
    float *latent = flux_sample_euler(
        ctx->transformer, ctx->qwen3_encoder,
        z, 1, FLUX_LATENT_CHANNELS, latent_h, latent_w,
        text_emb, text_seq,
        schedule, p.num_steps,
        NULL
    );

    free(z);
    free(schedule);
    free(text_emb);

    if (!latent) {
        set_error("Sampling failed");
        return NULL;
    }

    /* Decode latent to image */
    flux_image *img = NULL;
    if (ctx->vae) {
        if (flux_phase_callback) flux_phase_callback("decoding image", 0);
        img = flux_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
        if (flux_phase_callback) flux_phase_callback("decoding image", 1);
    }

    free(latent);

    return img;
}

/* ========================================================================
 * Generation with Pre-computed Embeddings
 * ======================================================================== */

flux_image *flux_generate_with_embeddings(flux_ctx *ctx,
                                           const float *text_emb, int text_seq,
                                           const flux_params *params) {
    if (!ctx || !text_emb) {
        set_error("Invalid context or embeddings");
        return NULL;
    }

    /* Load transformer if not already loaded */
    if (!flux_load_transformer_if_needed(ctx)) {
        return NULL;
    }

    flux_params p = flux_normalize_params(params);
    if (!flux_validate_params(ctx, &p)) {
        return NULL;
    }

    /* Compute latent dimensions */
    int latent_h = p.height / 16;
    int latent_w = p.width / 16;
    int image_seq_len = latent_h * latent_w;

    /* Initialize noise */
    int64_t seed = (p.seed < 0) ? (int64_t)time(NULL) : p.seed;
    float *z = flux_init_noise(1, FLUX_LATENT_CHANNELS, latent_h, latent_w, seed);

    /* Get official FLUX.2 schedule (matches Python) */
    float *schedule = flux_official_schedule(p.num_steps, image_seq_len);

    /* Sample */
    float *latent = flux_sample_euler(
        ctx->transformer, ctx->qwen3_encoder,
        z, 1, FLUX_LATENT_CHANNELS, latent_h, latent_w,
        text_emb, text_seq,
        schedule, p.num_steps,
        NULL  /* progress_callback */
    );

    free(z);
    free(schedule);

    if (!latent) {
        set_error("Sampling failed");
        return NULL;
    }

    /* Decode latent to image */
    flux_image *img = NULL;
    if (ctx->vae) {
        if (flux_phase_callback) flux_phase_callback("decoding image", 0);
        img = flux_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
        if (flux_phase_callback) flux_phase_callback("decoding image", 1);
    } else {
        set_error("No VAE loaded");
        free(latent);
        return NULL;
    }

    free(latent);
    return img;
}

/* Generate with external embeddings and external noise */
flux_image *flux_generate_with_embeddings_and_noise(flux_ctx *ctx,
                                                     const float *text_emb, int text_seq,
                                                     const float *noise, int noise_size,
                                                     const flux_params *params) {
    if (!ctx || !text_emb || !noise) {
        set_error("Invalid context, embeddings, or noise");
        return NULL;
    }

    /* Load transformer if not already loaded */
    if (!flux_load_transformer_if_needed(ctx)) {
        return NULL;
    }

    flux_params p = flux_normalize_params(params);
    if (!flux_validate_params(ctx, &p)) {
        return NULL;
    }

    /* Compute latent dimensions */
    int latent_h = p.height / 16;
    int latent_w = p.width / 16;
    int image_seq_len = latent_h * latent_w;
    int expected_noise_size = FLUX_LATENT_CHANNELS * latent_h * latent_w;

    if (noise_size != expected_noise_size) {
        char err[256];
        snprintf(err, sizeof(err), "Noise size mismatch: got %d, expected %d",
                 noise_size, expected_noise_size);
        set_error(err);
        return NULL;
    }

    /* Copy external noise */
    float *z = (float *)malloc(expected_noise_size * sizeof(float));
    memcpy(z, noise, expected_noise_size * sizeof(float));

    /* Get official FLUX.2 schedule (matches Python) */
    float *schedule = flux_official_schedule(p.num_steps, image_seq_len);

    /* Sample */
    float *latent = flux_sample_euler(
        ctx->transformer, ctx->qwen3_encoder,
        z, 1, FLUX_LATENT_CHANNELS, latent_h, latent_w,
        text_emb, text_seq,
        schedule, p.num_steps,
        NULL  /* progress_callback */
    );

    free(z);
    free(schedule);

    if (!latent) {
        set_error("Sampling failed");
        return NULL;
    }

    /* Decode latent to image */
    flux_image *img = NULL;
    if (ctx->vae) {
        if (flux_phase_callback) flux_phase_callback("decoding image", 0);
        img = flux_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
        if (flux_phase_callback) flux_phase_callback("decoding image", 1);
    } else {
        set_error("No VAE loaded");
        free(latent);
        return NULL;
    }

    free(latent);
    return img;
}

/* ========================================================================
 * Image-to-Image Generation
 * ======================================================================== */

flux_image *flux_img2img(flux_ctx *ctx, const char *prompt,
                         const flux_image *input, const flux_params *params) {
    if (!ctx || !prompt || !input) {
        set_error("Invalid parameters");
        return NULL;
    }

    flux_params p;
    if (params) {
        p = *params;
    } else {
        p = (flux_params)FLUX_PARAMS_DEFAULT;
    }

    /* Use input image dimensions if not specified */
    if (p.width <= 0) p.width = input->width;
    if (p.height <= 0) p.height = input->height;

    /* Clamp to VAE max dimensions, preserving aspect ratio */
    if (p.width > FLUX_VAE_MAX_DIM || p.height > FLUX_VAE_MAX_DIM) {
        float scale = (float)FLUX_VAE_MAX_DIM /
                      (p.width > p.height ? p.width : p.height);
        p.width = (int)(p.width * scale);
        p.height = (int)(p.height * scale);
    }

    /* Ensure divisible by 16 */
    p.width = (p.width / 16) * 16;
    p.height = (p.height / 16) * 16;

    /* Resize input if needed */
    flux_image *resized = NULL;
    const flux_image *img_to_use = input;
    if (input->width != p.width || input->height != p.height) {
        resized = flux_image_resize(input, p.width, p.height);
        if (!resized) {
            set_error("Failed to resize input image");
            return NULL;
        }
        img_to_use = resized;
    }

    /* Encode text */
    int text_seq;
    float *text_emb = flux_encode_text(ctx, prompt, &text_seq);
    if (!text_emb) {
        if (resized) flux_image_free(resized);
        set_error("Failed to encode prompt");
        return NULL;
    }


    /* Release text encoder to free ~8GB before loading transformer */
    flux_release_text_encoder(ctx);

    /* Load transformer now (after text encoder is freed to reduce peak memory) */
    if (!flux_load_transformer_if_needed(ctx)) {
        free(text_emb);
        return NULL;
    }

    /* Encode image to latent */
    float *img_tensor = flux_image_to_tensor(img_to_use);
    if (resized) flux_image_free(resized);

    int latent_h, latent_w;
    float *img_latent = NULL;

    if (ctx->vae) {
        img_latent = flux_vae_encode(ctx->vae, img_tensor, 1,
                                     p.height, p.width, &latent_h, &latent_w);
    } else {
        /* Placeholder if no VAE */
        latent_h = p.height / 16;
        latent_w = p.width / 16;
        img_latent = (float *)calloc(FLUX_LATENT_CHANNELS * latent_h * latent_w, sizeof(float));
    }

    free(img_tensor);

    if (!img_latent) {
        free(text_emb);
        set_error("Failed to encode image");
        return NULL;
    }

    /*
     * FLUX.2 img2img uses in-context conditioning:
     * - Reference image is encoded to latent with T offset in RoPE (T=10)
     * - Target image starts from pure noise (T=0)
     * - Both are concatenated as tokens in the transformer
     * - Model attends to reference via joint attention
     * - Only target tokens are output
     *
     * This is fundamentally different from traditional img2img that adds
     * noise directly to the encoded image.
     */
    int num_steps = p.num_steps;
    int image_seq_len = latent_h * latent_w;  /* For schedule calculation */

    /* Use official FLUX.2 schedule */
    float *schedule = flux_official_schedule(num_steps, image_seq_len);

    /* Initialize target latent with pure noise */
    int64_t seed = (p.seed < 0) ? (int64_t)time(NULL) : p.seed;
    float *z = flux_init_noise(1, FLUX_LATENT_CHANNELS, latent_h, latent_w, seed);

    /* Reference image latent is img_latent, with T offset = 10 */
    int t_offset = 10;

    /* Sample using in-context conditioning */
    float *latent = flux_sample_euler_with_refs(
        ctx->transformer, ctx->qwen3_encoder,
        z, 1, FLUX_LATENT_CHANNELS, latent_h, latent_w,
        img_latent, latent_h, latent_w,  /* Reference latent */
        t_offset,
        text_emb, text_seq,
        schedule, num_steps,
        NULL  /* progress_callback */
    );

    free(z);
    free(img_latent);
    free(schedule);
    free(text_emb);

    if (!latent) {
        set_error("Sampling failed");
        return NULL;
    }

    /* Decode */
    flux_image *result = NULL;
    if (ctx->vae) {
        if (flux_phase_callback) flux_phase_callback("decoding image", 0);
        result = flux_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
        if (flux_phase_callback) flux_phase_callback("decoding image", 1);
    }

    free(latent);
    return result;
}

/* ========================================================================
 * Multi-Reference Generation
 * ======================================================================== */

flux_image *flux_multiref(flux_ctx *ctx, const char *prompt,
                          const flux_image **refs, int num_refs,
                          const flux_params *params) {
    if (!ctx || !prompt) {
        set_error("Invalid parameters");
        return NULL;
    }

    /* No references - text-to-image */
    if (!refs || num_refs == 0) {
        return flux_generate(ctx, prompt, params);
    }

    /* Single reference - use optimized path */
    if (num_refs == 1) {
        return flux_img2img(ctx, prompt, refs[0], params);
    }

    flux_params p;
    if (params) {
        p = *params;
    } else {
        p = (flux_params)FLUX_PARAMS_DEFAULT;
    }

    /* Use first reference dimensions if not specified */
    if (p.width <= 0) p.width = refs[0]->width;
    if (p.height <= 0) p.height = refs[0]->height;

    /* Clamp to VAE max dimensions */
    if (p.width > FLUX_VAE_MAX_DIM || p.height > FLUX_VAE_MAX_DIM) {
        float scale = (float)FLUX_VAE_MAX_DIM /
                      (p.width > p.height ? p.width : p.height);
        p.width = (int)(p.width * scale);
        p.height = (int)(p.height * scale);
    }

    p.width = (p.width / 16) * 16;
    p.height = (p.height / 16) * 16;

    /* Encode text */
    int text_seq;
    float *text_emb = flux_encode_text(ctx, prompt, &text_seq);
    if (!text_emb) {
        set_error("Failed to encode prompt");
        return NULL;
    }

    flux_release_text_encoder(ctx);

    if (!flux_load_transformer_if_needed(ctx)) {
        free(text_emb);
        return NULL;
    }

    /* Encode all reference images at their native sizes */
    flux_ref_t *ref_latents = (flux_ref_t *)malloc(num_refs * sizeof(flux_ref_t));
    float **ref_data = (float **)malloc(num_refs * sizeof(float *));
    flux_image **resized_imgs = (flux_image **)calloc(num_refs, sizeof(flux_image *));

    for (int i = 0; i < num_refs; i++) {
        const flux_image *ref = refs[i];
        const flux_image *img_to_use = ref;

        /* Compute native size rounded to multiple of 16 */
        int ref_w = (ref->width / 16) * 16;
        int ref_h = (ref->height / 16) * 16;

        /* Clamp to VAE max */
        if (ref_w > FLUX_VAE_MAX_DIM) ref_w = FLUX_VAE_MAX_DIM;
        if (ref_h > FLUX_VAE_MAX_DIM) ref_h = FLUX_VAE_MAX_DIM;

        /* Resize only if dimensions changed after rounding/clamping */
        if (ref->width != ref_w || ref->height != ref_h) {
            resized_imgs[i] = flux_image_resize(ref, ref_w, ref_h);
            if (!resized_imgs[i]) {
                for (int j = 0; j < i; j++) {
                    free(ref_data[j]);
                    if (resized_imgs[j]) flux_image_free(resized_imgs[j]);
                }
                free(ref_latents);
                free(ref_data);
                free(resized_imgs);
                free(text_emb);
                set_error("Failed to resize reference image");
                return NULL;
            }
            img_to_use = resized_imgs[i];
        }

        /* Encode to latent at reference's own size */
        float *tensor = flux_image_to_tensor(img_to_use);
        int lat_h, lat_w;
        ref_data[i] = flux_vae_encode(ctx->vae, tensor, 1,
                                       img_to_use->height, img_to_use->width,
                                       &lat_h, &lat_w);
        free(tensor);

        if (!ref_data[i]) {
            for (int j = 0; j < i; j++) {
                free(ref_data[j]);
                if (resized_imgs[j]) flux_image_free(resized_imgs[j]);
            }
            if (resized_imgs[i]) flux_image_free(resized_imgs[i]);
            free(ref_latents);
            free(ref_data);
            free(resized_imgs);
            free(text_emb);
            set_error("Failed to encode reference image");
            return NULL;
        }

        ref_latents[i].latent = ref_data[i];
        ref_latents[i].h = lat_h;
        ref_latents[i].w = lat_w;
        ref_latents[i].t_offset = 10 * (i + 1);  /* 10, 20, 30, ... */
    }

    /* Free resized images (latents are now encoded) */
    for (int i = 0; i < num_refs; i++) {
        if (resized_imgs[i]) flux_image_free(resized_imgs[i]);
    }
    free(resized_imgs);

    int latent_h = p.height / 16;
    int latent_w = p.width / 16;
    int image_seq_len = latent_h * latent_w;

    float *schedule = flux_official_schedule(p.num_steps, image_seq_len);
    int64_t seed = (p.seed < 0) ? (int64_t)time(NULL) : p.seed;
    float *z = flux_init_noise(1, FLUX_LATENT_CHANNELS, latent_h, latent_w, seed);

    /* Sample with multi-reference conditioning */
    float *latent = flux_sample_euler_with_multi_refs(
        ctx->transformer, ctx->qwen3_encoder,
        z, 1, FLUX_LATENT_CHANNELS, latent_h, latent_w,
        ref_latents, num_refs,
        text_emb, text_seq,
        schedule, p.num_steps,
        NULL
    );

    /* Cleanup */
    free(z);
    for (int i = 0; i < num_refs; i++) {
        free(ref_data[i]);
    }
    free(ref_data);
    free(ref_latents);
    free(schedule);
    free(text_emb);

    if (!latent) {
        set_error("Sampling failed");
        return NULL;
    }

    /* Decode */
    flux_image *result = NULL;
    if (ctx->vae) {
        if (flux_phase_callback) flux_phase_callback("decoding image", 0);
        result = flux_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
        if (flux_phase_callback) flux_phase_callback("decoding image", 1);
    }

    free(latent);
    return result;
}

/* ========================================================================
 * Utility Functions
 * ======================================================================== */

void flux_set_seed(int64_t seed) {
    flux_rng_seed((uint64_t)seed);
}

const char *flux_model_info(flux_ctx *ctx) {
    static char info[256];
    if (!ctx) {
        return "No model loaded";
    }
    snprintf(info, sizeof(info), "%s v%s (max %dx%d, %d steps)",
             ctx->model_name, ctx->model_version,
             ctx->max_width, ctx->max_height, ctx->default_steps);
    return info;
}

/* ========================================================================
 * Low-level API
 * ======================================================================== */

float *flux_encode_image(flux_ctx *ctx, const flux_image *img,
                         int *out_h, int *out_w) {
    if (!ctx || !img || !ctx->vae) {
        *out_h = *out_w = 0;
        return NULL;
    }

    float *tensor = flux_image_to_tensor(img);
    if (!tensor) return NULL;

    float *latent = flux_vae_encode(ctx->vae, tensor, 1,
                                    img->height, img->width, out_h, out_w);
    free(tensor);
    return latent;
}

flux_image *flux_decode_latent(flux_ctx *ctx, const float *latent,
                               int latent_h, int latent_w) {
    if (!ctx || !latent || !ctx->vae) return NULL;
    if (flux_phase_callback) flux_phase_callback("decoding image", 0);
    flux_image *img = flux_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
    if (flux_phase_callback) flux_phase_callback("decoding image", 1);
    return img;
}

float *flux_denoise_step(flux_ctx *ctx, const float *z, float t,
                         const float *text_emb, int text_len,
                         int latent_h, int latent_w) {
    if (!ctx || !z || !text_emb) return NULL;

    /* Load transformer if not already loaded */
    if (!flux_load_transformer_if_needed(ctx)) {
        return NULL;
    }

    return flux_transformer_forward(ctx->transformer,
                                    z, latent_h, latent_w,
                                    text_emb, text_len, t);
}


/* ========================================================================
 * Finalized status-based API wrappers
 * ======================================================================== */

static flux_status_t flux_status_from_ptr(const void *p) {
    return p ? FLUX_STATUS_OK : FLUX_STATUS_RUNTIME_ERROR;
}

flux_status_t flux_ctx_load(flux_ctx **out_ctx, const char *model_dir) {
    if (!out_ctx || !model_dir) return FLUX_STATUS_INVALID_ARGUMENT;
    *out_ctx = flux_load_dir(model_dir);
    return flux_status_from_ptr(*out_ctx);
}

flux_status_t flux_ctx_free(flux_ctx *ctx) {
    if (!ctx) return FLUX_STATUS_INVALID_ARGUMENT;
    flux_free(ctx);
    return FLUX_STATUS_OK;
}

flux_status_t flux_ctx_release_text_encoder(flux_ctx *ctx) {
    if (!ctx) return FLUX_STATUS_INVALID_ARGUMENT;
    flux_release_text_encoder(ctx);
    return FLUX_STATUS_OK;
}

flux_status_t flux_ctx_set_mmap(flux_ctx *ctx, int enable) {
    if (!ctx) return FLUX_STATUS_INVALID_ARGUMENT;
    flux_set_mmap(ctx, enable);
    return FLUX_STATUS_OK;
}

flux_status_t flux_ctx_set_strict(flux_ctx *ctx, int enable) {
    if (!ctx) return FLUX_STATUS_INVALID_ARGUMENT;
    flux_set_strict(ctx, enable);
    return FLUX_STATUS_OK;
}

flux_status_t flux_ctx_set_embed_cache(flux_ctx *ctx, int enable) {
    if (!ctx) return FLUX_STATUS_INVALID_ARGUMENT;
    flux_set_embed_cache(ctx, enable);
    return FLUX_STATUS_OK;
}

flux_status_t flux_generate_status(flux_ctx *ctx, const char *prompt,
                                   const flux_params *params, flux_image **out_image) {
    if (!ctx || !prompt || !out_image) return FLUX_STATUS_INVALID_ARGUMENT;
    *out_image = flux_generate(ctx, prompt, params);
    return flux_status_from_ptr(*out_image);
}

flux_status_t flux_img2img_status(flux_ctx *ctx, const char *prompt,
                                  const flux_image *input, const flux_params *params,
                                  flux_image **out_image) {
    if (!ctx || !prompt || !input || !out_image) return FLUX_STATUS_INVALID_ARGUMENT;
    *out_image = flux_img2img(ctx, prompt, input, params);
    return flux_status_from_ptr(*out_image);
}

flux_status_t flux_multiref_status(flux_ctx *ctx, const char *prompt,
                                   const flux_image **refs, int num_refs,
                                   const flux_params *params, flux_image **out_image) {
    if (!ctx || !prompt || !refs || num_refs < 0 || !out_image) return FLUX_STATUS_INVALID_ARGUMENT;
    *out_image = flux_multiref(ctx, prompt, refs, num_refs, params);
    return flux_status_from_ptr(*out_image);
}

flux_status_t flux_generate_with_embeddings_status(flux_ctx *ctx,
                                                   const float *text_emb, int text_seq,
                                                   const flux_params *params,
                                                   flux_image **out_image) {
    if (!ctx || !text_emb || text_seq <= 0 || !out_image) return FLUX_STATUS_INVALID_ARGUMENT;
    *out_image = flux_generate_with_embeddings(ctx, text_emb, text_seq, params);
    return flux_status_from_ptr(*out_image);
}

flux_status_t flux_generate_with_embeddings_and_noise_status(flux_ctx *ctx,
                                                             const float *text_emb, int text_seq,
                                                             const float *noise, int noise_size,
                                                             const flux_params *params,
                                                             flux_image **out_image) {
    if (!ctx || !text_emb || text_seq <= 0 || !noise || noise_size <= 0 || !out_image) {
        return FLUX_STATUS_INVALID_ARGUMENT;
    }
    *out_image = flux_generate_with_embeddings_and_noise(ctx, text_emb, text_seq,
                                                         noise, noise_size, params);
    return flux_status_from_ptr(*out_image);
}
