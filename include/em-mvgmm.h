#ifndef EM_MVN_H
#define EM_MVN_H

#include <cublas_v2.h>
#include <cusolverDn.h>

void expectationMaximizationMVGMM(cublasHandle_t cublasHandle, cusolverDnHandle_t cusolverHandle,
                                  int D, int N, int K, const float *d_data);

#endif
