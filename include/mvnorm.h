#ifndef MVNORM_H
#define MVNORM_H

#include <cublas_v2.h>
#include <cusolverDn.h>

void rmvnorm(cublasHandle_t cublasHandle, cusolverDnHandle_t cusolverHandle,
             int N, int D, const float *d_mean, const float *d_covariance, float *d_randomMvNorm);

#endif