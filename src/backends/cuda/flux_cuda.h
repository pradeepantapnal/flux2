#ifndef FLUX_CUDA_H
#define FLUX_CUDA_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint16_t flux_fp16;

#define CUDA_HOME_STR "/usr/local/cuda-13.0"

#define FLUX_CUDA_CHECK(expr) \
    do { \
        int _err = (expr); \
        if (_err != 0) { \
            fprintf(stderr, "CUDA error code %d at %s:%d\n", _err, __FILE__, __LINE__); \
            return -1; \
        } \
    } while (0)

int flux_cuda_init(int verbose);
void flux_cuda_shutdown(void);

int flux_cuda_malloc(void **ptr, size_t bytes);
int flux_cuda_free(void *ptr);
int flux_cuda_memcpy_h2d(void *dst, const void *src, size_t bytes);
int flux_cuda_memcpy_d2h(void *dst, const void *src, size_t bytes);

int flux_cuda_gemm_fp16(const flux_fp16 *d_A,
                        const flux_fp16 *d_B,
                        flux_fp16 *d_C,
                        int m,
                        int n,
                        int k);

int flux_cuda_test_gemm(void);

#ifdef __cplusplus
}
#endif

#endif
