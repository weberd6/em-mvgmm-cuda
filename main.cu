#include <mvgmm.h>
#include <thrust/device_vector.h>

int main(int argc, char *argv[]) {

    cusolverDnHandle_t cusolverHandle;
    cublasHandle_t cublasHandle;

    int K = 4;
    int N = 32768;
    int D = 3;

    // Set values for K mean vectors and K covariance matrices

    float h_weights[K] = {0.2, 0.4, 0.3, 0.1};
    float h_means[K][D] = {{0, 0, 0}, {-10, 7, 3}, {5, 1, 12}, {1, -5, -2}};
    float h_covariances[K][D*D] = {{1, 0, 0, 0, 1, 0, 0, 0, 1},
                                   {2, 0.9, 0.1, 0.9, 1, 1.5, 0.1, 1.5, 0},
                                   {1, -0.9, 1.2, -0.9, 4, 0.4, 1.2, 0.4, 2},
                                   {1.5, 0.7, -1.1, 0.7, 2.2, 0, -1.1, 0, 0.7}};

    // Create handles

    cublasCreate(&cublasHandle);
    cusolverDnCreate(&cusolverHandle);

    // Copy to device

    thrust::device_vector<float> d_weights(K);
    thrust::device_vector<float> d_means(K * D);
    thrust::device_vector<float> d_covariances(K * D * D);
    thrust::device_vector<float> d_randomValues(D * N);

    for (int k = 0; k < K; k++) {
        cudaMemcpy(thrust::raw_pointer_cast(d_weights.data() + k), &h_weights[k],
                   sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(thrust::raw_pointer_cast(d_means.data() + k * D), &h_means[k * D],
                   sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(thrust::raw_pointer_cast(d_covariances.data() + k * D * D), &h_covariances[k * D * D],
                   sizeof(float), cudaMemcpyHostToDevice);
    }

    rmvgmm(cublasHandle, cusolverHandle, N, D, K, thrust::raw_pointer_cast(d_weights.data()),
           thrust::raw_pointer_cast(d_means.data()), thrust::raw_pointer_cast(d_covariances.data()),
           thrust::raw_pointer_cast(d_randomValues.data()));

    // Cleanup

    cublasDestroy(cublasHandle);
    cusolverDnDestroy(cusolverHandle);

    return 0;
}
