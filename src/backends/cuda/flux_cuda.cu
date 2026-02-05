#include "flux_cuda.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include <math.h>
#include <stdio.h>

#define CUDA_RT_CHECK(expr) \
    do { \
        cudaError_t _err = (expr); \
        if (_err != cudaSuccess) { \
            fprintf(stderr, "CUDA runtime error %s at %s:%d\n", cudaGetErrorString(_err), __FILE__, __LINE__); \
            return -1; \
        } \
    } while (0)

static cublasHandle_t g_cublas = NULL;
static cublasLtHandle_t g_cublaslt = NULL;
static int g_cuda_ready = 0;

static int flux_cublas_check(cublasStatus_t status, const char *label) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error (%s): %d\n", label, (int)status);
        return -1;
    }
    return 0;
}

int flux_cuda_init(int verbose) {
    if (g_cuda_ready) return 0;

    CUDA_RT_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CUDA_RT_CHECK(cudaGetDeviceProperties(&prop, 0));

    int runtime_version = 0;
    CUDA_RT_CHECK(cudaRuntimeGetVersion(&runtime_version));

    if (verbose) {
        fprintf(stderr,
                "diag: cuda.device=%s cc=%d.%d runtime=%d CUDA_HOME=%s\n",
                prop.name,
                prop.major,
                prop.minor,
                runtime_version,
                CUDA_HOME_STR);
    }

    if (flux_cublas_check(cublasCreate(&g_cublas), "cublasCreate") != 0) return -1;
    if (flux_cublas_check(cublasLtCreate(&g_cublaslt), "cublasLtCreate") != 0) {
        cublasDestroy(g_cublas);
        g_cublas = NULL;
        return -1;
    }

    g_cuda_ready = 1;
    return 0;
}

void flux_cuda_shutdown(void) {
    if (g_cublaslt) cublasLtDestroy(g_cublaslt);
    if (g_cublas) cublasDestroy(g_cublas);
    g_cublaslt = NULL;
    g_cublas = NULL;
    g_cuda_ready = 0;
}

int flux_cuda_malloc(void **ptr, size_t bytes) {
    CUDA_RT_CHECK(cudaMalloc(ptr, bytes));
    return 0;
}

int flux_cuda_free(void *ptr) {
    CUDA_RT_CHECK(cudaFree(ptr));
    return 0;
}

int flux_cuda_memcpy_h2d(void *dst, const void *src, size_t bytes) {
    CUDA_RT_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
    return 0;
}

int flux_cuda_memcpy_d2h(void *dst, const void *src, size_t bytes) {
    CUDA_RT_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
    return 0;
}

int flux_cuda_gemm_fp16(const flux_fp16 *d_A,
                        const flux_fp16 *d_B,
                        flux_fp16 *d_C,
                        int m,
                        int n,
                        int k) {
    if (!g_cuda_ready && flux_cuda_init(0) != 0) return -1;

    cublasLtMatmulDesc_t op_desc = NULL;
    cublasLtMatrixLayout_t a_desc = NULL;
    cublasLtMatrixLayout_t b_desc = NULL;
    cublasLtMatrixLayout_t c_desc = NULL;

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasOperation_t trans = CUBLAS_OP_N;

    if (flux_cublas_check(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "matmulDescCreate") != 0) goto fail;
    if (flux_cublas_check(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)), "set transa") != 0) goto fail;
    if (flux_cublas_check(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)), "set transb") != 0) goto fail;

    if (flux_cublas_check(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16F, m, k, m), "layoutA") != 0) goto fail;
    if (flux_cublas_check(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16F, k, n, k), "layoutB") != 0) goto fail;
    if (flux_cublas_check(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16F, m, n, m), "layoutC") != 0) goto fail;

    if (flux_cublas_check(cublasLtMatmul(g_cublaslt,
                                         op_desc,
                                         &alpha,
                                         (const __half *)d_A,
                                         a_desc,
                                         (const __half *)d_B,
                                         b_desc,
                                         &beta,
                                         (__half *)d_C,
                                         c_desc,
                                         (__half *)d_C,
                                         c_desc,
                                         NULL,
                                         NULL,
                                         0,
                                         0),
                         "cublasLtMatmul") != 0) {
        goto fail;
    }

    CUDA_RT_CHECK(cudaDeviceSynchronize());

    cublasLtMatrixLayoutDestroy(c_desc);
    cublasLtMatrixLayoutDestroy(b_desc);
    cublasLtMatrixLayoutDestroy(a_desc);
    cublasLtMatmulDescDestroy(op_desc);
    return 0;

fail:
    if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
    if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
    if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
    if (op_desc) cublasLtMatmulDescDestroy(op_desc);
    return -1;
}

int flux_cuda_test_gemm(void) {
    const int m = 2, n = 2, k = 2;
    const __half h_A[4] = {__float2half(1.0f), __float2half(3.0f), __float2half(2.0f), __float2half(4.0f)};
    const __half h_B[4] = {__float2half(5.0f), __float2half(7.0f), __float2half(6.0f), __float2half(8.0f)};
    const float expected[4] = {19.0f, 43.0f, 22.0f, 50.0f};

    __half *d_A = NULL;
    __half *d_B = NULL;
    __half *d_C = NULL;
    __half h_C[4] = {0};

    if (flux_cuda_init(1) != 0) return -1;

    if (flux_cuda_malloc((void **)&d_A, sizeof(h_A)) != 0) goto fail;
    if (flux_cuda_malloc((void **)&d_B, sizeof(h_B)) != 0) goto fail;
    if (flux_cuda_malloc((void **)&d_C, sizeof(h_C)) != 0) goto fail;

    if (flux_cuda_memcpy_h2d(d_A, h_A, sizeof(h_A)) != 0) goto fail;
    if (flux_cuda_memcpy_h2d(d_B, h_B, sizeof(h_B)) != 0) goto fail;

    if (flux_cuda_gemm_fp16((const flux_fp16 *)d_A, (const flux_fp16 *)d_B, (flux_fp16 *)d_C, m, n, k) != 0) goto fail;
    if (flux_cuda_memcpy_d2h(h_C, d_C, sizeof(h_C)) != 0) goto fail;

    for (int i = 0; i < 4; i++) {
        const float got = __half2float(h_C[i]);
        if (fabsf(got - expected[i]) > 0.2f) {
            fprintf(stderr, "CUDA GEMM test FAIL at %d: got %.3f expected %.3f\n", i, got, expected[i]);
            goto fail;
        }
    }

    fprintf(stderr, "CUDA GEMM test PASS\n");
    flux_cuda_free(d_C);
    flux_cuda_free(d_B);
    flux_cuda_free(d_A);
    return 0;

fail:
    if (d_C) flux_cuda_free(d_C);
    if (d_B) flux_cuda_free(d_B);
    if (d_A) flux_cuda_free(d_A);
    fprintf(stderr, "CUDA GEMM test FAIL\n");
    return -1;
}
