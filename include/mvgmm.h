#ifndef MVNORM_H
#define MVNORM_H

#include <cublas_v2.h>
#include <cusolverDn.h>

void rmvgmm(cublasHandle_t cublasHandle, cusolverDnHandle_t cusolverHandle,
                    int N, int D, int K, const float *d_weights, const float *d_means,
                    const float *d_covariances, float *d_randomValues);

void dmvgmm(cublasHandle_t cublasHandle, int N, int D, int K, const float *d_data,
                    const float *h_weights, const float *d_means, const float *d_covariances,
                    float *d_logpdf, bool log = true);

#endif
