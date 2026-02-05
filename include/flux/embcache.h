#ifndef EMBCACHE_H
#define EMBCACHE_H

#include <stdint.h>

typedef struct emb_cache emb_cache_t;

/* New keyed LRU API */
emb_cache_t *emb_lru_create(int capacity);
void emb_lru_destroy(emb_cache_t *cache);
void emb_lru_set_enabled(emb_cache_t *cache, int enable);
int emb_lru_is_enabled(const emb_cache_t *cache);
int emb_lru_lookup(emb_cache_t *cache,
                   const char *prompt,
                   const char *tokenizer_config_id,
                   const char *model_revision_id,
                   float **out_embedding,
                   int *out_seq_len,
                   int *out_dim);
void emb_lru_store(emb_cache_t *cache,
                   const char *prompt,
                   const char *tokenizer_config_id,
                   const char *model_revision_id,
                   const float *embedding,
                   int seq_len,
                   int dim);

/* Legacy global API used by interactive CLI */
void emb_cache_init(void);
void emb_cache_free(void);
float *emb_cache_lookup(const char *prompt);
void emb_cache_store(const char *prompt, const float *embedding, int num_elements);

#endif
