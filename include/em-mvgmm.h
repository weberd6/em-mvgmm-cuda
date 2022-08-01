#ifndef EM_MVGMM_H
#define EM_MVGMM_H

#include <cublas_v2.h>
#include <cusolverDn.h>

void expectationMaximizationMVGMM(cublasHandle_t cublasHandle, cusolverDnHandle_t cusolverHandle,
                                  int D, int N, int K, const float *d_data, float *h_weights,
                                  float *d_means, float *d_covariances);

#endif
