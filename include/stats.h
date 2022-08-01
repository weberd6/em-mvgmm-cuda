#ifndef STATS_H
#define STATS_H

#include <cublas_v2.h>

void meanAndCovariance(cublasHandle_t cublasHandle, int D, int N, const float *d_data,
                       float *d_mean, float *d_covariance);

#endif