#include <mvnorm.h>
#include <curand.h>
#include <thrust/device_vector.h>

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

// cusolver API error checking
#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusolver error");                                            \
        }                                                                                          \
    } while (0)

__global__ void rmvnorm_addMeans(int D, int N, float *d_data, const float *d_means)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N) {
        for (int d = 0; d < D; d++) {
            d_data[n * D + d] += d_means[d];
        }
    }
}

void rmvnorm(cublasHandle_t cublasHandle, cusolverDnHandle_t cusolverHandle,
             int N, int D, const float *d_mean, const float *d_covariance, float *d_randomMvNorm) {

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 3438);

    // Do cholesky decomposition on covariance matrices L*transpose(L)

    int Lwork;
 
    thrust::device_vector<float> d_covarianceLower(D * D);
    cudaMemcpy(thrust::raw_pointer_cast(d_covarianceLower.data()),
               d_covariance, D * D * sizeof(float), cudaMemcpyDeviceToDevice);
 
    CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(cusolverHandle, CUBLAS_FILL_MODE_LOWER, D,
                                thrust::raw_pointer_cast(d_covarianceLower.data()), D, &Lwork));

    thrust::device_vector<int> d_devInfo(1);
    thrust::device_vector<float> d_workspace(Lwork);
    CUSOLVER_CHECK(cusolverDnSpotrf(cusolverHandle, CUBLAS_FILL_MODE_LOWER, D,
                     thrust::raw_pointer_cast(d_covarianceLower.data()), D,
                     thrust::raw_pointer_cast(d_workspace.data()), Lwork,
                     thrust::raw_pointer_cast(d_devInfo.data())));

    // Generate standard normal variables, (D * N) samples

    thrust::device_vector<float> d_randomNorm(D * N);
    curandGenerateNormal(gen, thrust::raw_pointer_cast(d_randomNorm.data()), D * N, 0.0, 1.0);

    // Generate multivariate normal from standard normal with mean and covariance lower

    float alpha = 1.0;
    CUBLAS_CHECK(cublasStrmm(cublasHandle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                             CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, D, N, &alpha,
                             thrust::raw_pointer_cast(d_covarianceLower.data()), D,
                             thrust::raw_pointer_cast(d_randomNorm.data()), D,
                             d_randomMvNorm, D));

    int threadsPerBlock = 256;
    int blocks = std::ceil((float)N / threadsPerBlock);

    rmvnorm_addMeans<<<blocks, threadsPerBlock>>>(D, N, d_randomMvNorm, d_mean);

    curandDestroyGenerator(gen);
}