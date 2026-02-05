#include "embcache.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    uint64_t key_hash;
    char *prompt;
    char *tokenizer_config_id;
    char *model_revision_id;
    float *embedding;
    int seq_len;
    int dim;
    unsigned long long last_used;
    int occupied;
} emb_cache_entry_t;

struct emb_cache {
    emb_cache_entry_t *entries;
    int capacity;
    unsigned long long clock;
    int enabled;
};

static uint64_t fnv1a_update(uint64_t h, const char *s) {
    while (*s) { h ^= (unsigned char)*s++; h *= 1099511628211ULL; }
    return h;
}

static uint64_t emb_key_hash(const char *prompt, const char *tok, const char *rev) {
    uint64_t h = 14695981039346656037ULL;
    h = fnv1a_update(h, prompt ? prompt : "");
    h = fnv1a_update(h, "|");
    h = fnv1a_update(h, tok ? tok : "");
    h = fnv1a_update(h, "|");
    h = fnv1a_update(h, rev ? rev : "");
    return h;
}

static void entry_clear(emb_cache_entry_t *e) {
    free(e->prompt); free(e->tokenizer_config_id); free(e->model_revision_id); free(e->embedding);
    memset(e, 0, sizeof(*e));
}

emb_cache_t *emb_lru_create(int capacity) {
    if (capacity <= 0) capacity = 8;
    emb_cache_t *c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->entries = calloc((size_t)capacity, sizeof(emb_cache_entry_t));
    if (!c->entries) { free(c); return NULL; }
    c->capacity = capacity;
    c->enabled = 1;
    return c;
}

void emb_lru_destroy(emb_cache_t *cache) {
    if (!cache) return;
    for (int i = 0; i < cache->capacity; i++) entry_clear(&cache->entries[i]);
    free(cache->entries);
    free(cache);
}

void emb_lru_set_enabled(emb_cache_t *cache, int enable) { if (cache) cache->enabled = enable ? 1 : 0; }
int emb_lru_is_enabled(const emb_cache_t *cache) { return cache && cache->enabled; }

static int key_match(const emb_cache_entry_t *e, uint64_t h, const char *p, const char *t, const char *r) {
    return e->occupied && e->key_hash == h && strcmp(e->prompt,p)==0 && strcmp(e->tokenizer_config_id,t)==0 && strcmp(e->model_revision_id,r)==0;
}

int emb_lru_lookup(emb_cache_t *cache, const char *prompt, const char *tok, const char *rev,
                   float **out_embedding, int *out_seq_len, int *out_dim) {
    if (!cache || !cache->enabled || !prompt || !tok || !rev || !out_embedding || !out_seq_len || !out_dim) return 0;
    uint64_t h = emb_key_hash(prompt, tok, rev);
    for (int i = 0; i < cache->capacity; i++) {
        emb_cache_entry_t *e = &cache->entries[i];
        if (!key_match(e, h, prompt, tok, rev)) continue;
        size_t n = (size_t)e->seq_len * (size_t)e->dim;
        float *copy = malloc(n * sizeof(float));
        if (!copy) return 0;
        memcpy(copy, e->embedding, n * sizeof(float));
        cache->clock++; e->last_used = cache->clock;
        *out_embedding = copy; *out_seq_len = e->seq_len; *out_dim = e->dim;
        return 1;
    }
    return 0;
}

void emb_lru_store(emb_cache_t *cache, const char *prompt, const char *tok, const char *rev,
                   const float *embedding, int seq_len, int dim) {
    if (!cache || !cache->enabled || !prompt || !tok || !rev || !embedding || seq_len<=0 || dim<=0) return;
    uint64_t h = emb_key_hash(prompt, tok, rev);
    int target = -1; unsigned long long oldest = ~0ULL;
    for (int i = 0; i < cache->capacity; i++) {
        emb_cache_entry_t *e = &cache->entries[i];
        if (key_match(e, h, prompt, tok, rev)) { target = i; break; }
        if (!e->occupied) { target = i; oldest = 0; break; }
        if (e->last_used < oldest) { oldest = e->last_used; target = i; }
    }
    if (target < 0) return;
    size_t n = (size_t)seq_len * (size_t)dim;
    float *copy = malloc(n * sizeof(float));
    if (!copy) return;
    memcpy(copy, embedding, n * sizeof(float));

    emb_cache_entry_t *e = &cache->entries[target];
    entry_clear(e);
    e->prompt = strdup(prompt);
    e->tokenizer_config_id = strdup(tok);
    e->model_revision_id = strdup(rev);
    if (!e->prompt || !e->tokenizer_config_id || !e->model_revision_id) { free(copy); entry_clear(e); return; }
    e->embedding = copy;
    e->seq_len = seq_len;
    e->dim = dim;
    e->key_hash = h;
    cache->clock++; e->last_used = cache->clock;
    e->occupied = 1;
}

/* Legacy wrappers */
static emb_cache_t *g_legacy_cache = NULL;
void emb_cache_init(void) { if (!g_legacy_cache) g_legacy_cache = emb_lru_create(4); }
void emb_cache_free(void) { emb_lru_destroy(g_legacy_cache); g_legacy_cache = NULL; }
float *emb_cache_lookup(const char *prompt) {
    if (!g_legacy_cache) emb_cache_init();
    float *emb = NULL; int seq = 0, dim = 0;
    if (emb_lru_lookup(g_legacy_cache, prompt, "legacy-tok", "legacy-rev", &emb, &seq, &dim)) return emb;
    return NULL;
}
void emb_cache_store(const char *prompt, const float *embedding, int num_elements) {
    if (!g_legacy_cache) emb_cache_init();
    emb_lru_store(g_legacy_cache, prompt, "legacy-tok", "legacy-rev", embedding, 1, num_elements);
}
