#ifndef EM_MVN_H
#define EM_MVN_H

void expectationMaximizationMultivariateNormal(int D, int N, int K, const float *d_data,
                                               float *d_weights, float *d_means, float *d_covariances);

#endif
