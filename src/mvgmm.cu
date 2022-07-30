#include <mvgmm.h>
#include <math_constants.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <curand.h>

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) {                                       \
    printf("Error at %s:%d\n",__FILE__,__LINE__);                                                  \
    return EXIT_FAILURE;}} while(0)

// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

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

__global__ void assignCluster(int K, int N, const float *d_weights, const float *d_uniformRand,
                              int *d_clusters) {

    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N) {
        float p = 0.0f;
        for (int k = 0; k < K; k++) {
            p += d_weights[k];
            if ((p - d_uniformRand[n]) >= 0) {
                d_clusters[n] = k;
                break;
            }
        }
    }
}

__global__ void setupBatchMultiply(int N, int D, const int *d_clusters, float *d_normalRand,
                                    float **d_meansArray, float **d_covariancesArray,
                                    float **d_ptrSamples, float **d_ptrMeans, float **d_ptrCovariances,
                                    float *d_resultMatrix)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N) {

        d_ptrSamples[n] = &d_normalRand[n * D];

        memcpy(&d_resultMatrix[n * D], d_meansArray[d_clusters[n]], D * sizeof(float));
        d_ptrMeans[n] = &d_resultMatrix[n * D];

        d_ptrCovariances[n] = d_covariancesArray[d_clusters[n]];
    }
}

void rmvgmm(cublasHandle_t cublasHandle, cusolverDnHandle_t cusolverHandle,
                    int N, int D, int K, const float *d_weights, const float *d_means,
                    const float *d_covariances, float *d_randomValues) {

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    // Generate N random values between 0.0 and 1.0

    thrust::device_vector<float> d_uniformRand(N);
    curandGenerateUniform(gen, thrust::raw_pointer_cast(d_uniformRand.data()), N);

    // Assign N clusters based on random uniform values and weights

    int threadsPerBlock = 256;
    int blocks = std::ceil(N / threadsPerBlock);
    thrust::device_vector<int> d_clusters(N);
    assignCluster<<<blocks, threadsPerBlock>>>(K, N, d_weights,
                                               thrust::raw_pointer_cast(d_uniformRand.data()),
                                               thrust::raw_pointer_cast(d_clusters.data()));

    // Copy means and covariances to arrays of pointers

    std::vector<float*> meansArray(K);
    std::vector<float*> covariancesArray(K);
    for (int k = 0; k < K; k++) {

        CUDA_CHECK(cudaMalloc(&meansArray[k], D * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(meansArray[k], &d_means[k * D], D * sizeof(float), cudaMemcpyDeviceToDevice));

        CUDA_CHECK(cudaMalloc(&covariancesArray[k], D * D * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(covariancesArray[k], &d_covariances[k * D * D],
                              D * D * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    thrust::device_vector<float*> d_meansArray = meansArray;
    thrust::device_vector<float*> d_covariancesArray = covariancesArray;

    // Do cholesky decomposition on covariance matrices L*transpose(L)

    thrust::device_vector<int> d_infoArray(K);
    CUSOLVER_CHECK(cusolverDnSpotrfBatched(cusolverHandle, CUBLAS_FILL_MODE_LOWER, D,
                                           thrust::raw_pointer_cast(d_covariancesArray.data()), D,
                                           thrust::raw_pointer_cast(d_infoArray.data()), K));

    // Generate standard normal variables, (D * N) samples

    thrust::device_vector<float> d_normalRand(D * N);
    curandGenerateNormal(gen, thrust::raw_pointer_cast(d_normalRand.data()), D * N, 0.0, 1.0);

    // For each sample, create pointer to assigned cluster mean and covariance lower

    thrust::device_vector<float*> d_ptrSamples(N);
    thrust::device_vector<float*> d_ptrCovariances(N);
    thrust::device_vector<float*> d_ptrMeans(N);
    setupBatchMultiply<<<blocks, threadsPerBlock>>>(N, D,
                                            thrust::raw_pointer_cast(d_clusters.data()),
                                            thrust::raw_pointer_cast(d_normalRand.data()),
                                            thrust::raw_pointer_cast(d_meansArray.data()),
                                            thrust::raw_pointer_cast(d_covariancesArray.data()),
                                            thrust::raw_pointer_cast(d_ptrSamples.data()),
                                            thrust::raw_pointer_cast(d_ptrMeans.data()),
                                            thrust::raw_pointer_cast(d_ptrCovariances.data()),
                                            d_randomValues);

    // Generate random multivarate normal by multiplying each data sample by covariance lower and adding mean

    float alpha = 1.0;
    float beta = 1.0;
    CUBLAS_CHECK(cublasSgemmBatched(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, D, 1, D, &alpha,
                                    thrust::raw_pointer_cast(d_ptrCovariances.data()), D,
                                    thrust::raw_pointer_cast(d_ptrSamples.data()), D,
                                    &beta, thrust::raw_pointer_cast(d_ptrMeans.data()), D, N));

    for (int k = 0; k < K; k++) {
        CUDA_CHECK(cudaFree(d_meansArray[k]));
        CUDA_CHECK(cudaFree(d_covariancesArray[k]));
    }

    curandDestroyGenerator(gen);
}

__global__ void centerData(int D, int N, const float *d_data, const float *d_means, float *d_dataZeroCentered)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N) {

        for (int d = 0; d < D; d++) {

            d_dataZeroCentered[n * D + d] = d_data[n * D + d] - d_means[d];
        }
    }
}

// d_data is (DxN) matrix column-major order
// d_means is (NxD) matrix column-major order
void mahalanobisDistanceSquared(cublasHandle_t cublasHandle, int D, int N, const float *d_data,
        const float *d_means, const float *d_covarianceInverse, float *d_mahalanobisSquared)
{
    thrust::device_vector<float> d_dataZeroCentered(D * N);
    thrust::device_vector<float> d_dataZeroCenteredScaled(D * N);
    float alpha, beta;

    // (x - mean) (DxN)

    int threadsPerBlock = 256;
    int blocks = std::ceil(N / threadsPerBlock);
    centerData<<<blocks, threadsPerBlock>>>(D, N, d_data, d_means,
                                            thrust::raw_pointer_cast(d_dataZeroCentered.data()));

    // transpose(x - mean) * covarianceInverse for all N data points (DxN)

    alpha = 1.0;
    beta = 0.0;
    CUBLAS_CHECK(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, D, N, N,
                             &alpha, d_covarianceInverse, D,
                             thrust::raw_pointer_cast(d_dataZeroCentered.data()), D,
                             &beta, thrust::raw_pointer_cast(d_dataZeroCenteredScaled.data()), D));

    // transpose(x - mean) * covarianceInverse * (x - mean) for all N data points (Nx1)

    alpha = 1.0;
    beta = 0.0;
    CUBLAS_CHECK(cublasSgemmStridedBatched(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, D, &alpha,
                                           thrust::raw_pointer_cast(d_dataZeroCenteredScaled.data()), 1, D,
                                           thrust::raw_pointer_cast(d_dataZeroCentered.data()), 1, D,
                                           &beta, d_mahalanobisSquared, D, 1, N));
}

void logDeterminants(int D, int K, float *const d_covariancesLUArray[], float *h_logDetCovariances)
{
    for (int k = 0; k < K; k++) {

        float eigs[D];

        CUBLAS_CHECK(cublasGetVector(D, sizeof(float), d_covariancesLUArray[k], D+1, eigs, 1));

        float sum_log_eigs = 0;
        for (int d = 0; d < D; d++) {
            sum_log_eigs += std::log(eigs[d]);
        }

        h_logDetCovariances[k] = sum_log_eigs;
    }
}

// d_mahalanobisSquared is Nx1 column-major order
__global__ void multivariateNormalLogPDF(int N, int D, float logDetCovariance,
                                         float weight, float *d_mahalanobisSquared_logpdf)
{

    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N) {

        float logpdf = log(weight) - 0.5 * (logDetCovariance +
                                            D * log(2*CUDART_PI_F) +
                                            d_mahalanobisSquared_logpdf[n]);

        d_mahalanobisSquared_logpdf[n] = logpdf;
    }
}

// d_matrix is NxK column-major order
__global__ void exponential(int N, int K, float *d_matrix)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N) {
        for (int k = 0; k < K; k++) {
            d_matrix[k * N + n] = expf(d_matrix[k * N + n]);
        }
    }
}

void dmvgmm(cublasHandle_t cublasHandle, int N, int D, int K, const float *d_data,
                    const float *h_weights, const float *d_means, const float *d_covariances,
                    float *d_logpdf, bool log) {

    float *d_covariancesLUArray[K];
    float *d_covariancesInverses[K];
    thrust::device_vector<int> d_pivotArray(D * K);
    thrust::device_vector<int> d_infoArray(K);

    float *h_logDetCovariances = (float*)malloc(K * sizeof(float));

    for (int k = 0; k < K; k++) {

        cudaMalloc(&d_covariancesLUArray[k], D * D * sizeof(float));
        cudaMalloc(&d_covariancesInverses[k], D * D * sizeof(float));

        cudaMemcpy(&d_covariancesLUArray[k], &d_covariances[k * D * D],
                   D * D * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // Get LU factorization of covariance matrices

    CUBLAS_CHECK(cublasSgetrfBatched(cublasHandle, D, d_covariancesLUArray, D,
                                     thrust::raw_pointer_cast(d_pivotArray.data()),
                                     thrust::raw_pointer_cast(d_infoArray.data()), K));

    // Get inverses of covariance matrices

    CUBLAS_CHECK(cublasSgetriBatched(cublasHandle, D, d_covariancesLUArray, D,
                                     thrust::raw_pointer_cast(d_pivotArray.data()), d_covariancesInverses, D,
                                     thrust::raw_pointer_cast(d_infoArray.data()), K));

    // Get determinants of covariance matrices

    logDeterminants(D, K, d_covariancesLUArray, h_logDetCovariances);

    for (int k = 0; k < K; k++) {

        // Calculate mahalanobis distance squared from mean and covariance

        mahalanobisDistanceSquared(cublasHandle, D, N, d_data, &d_means[k * D],
                                   d_covariancesInverses[k], &d_logpdf[k * N]);

        // Calculate log probabilities from squared mahlanobis distances

        int threadsPerBlock = 256;
        int blocks = std::ceil(N / threadsPerBlock);
        multivariateNormalLogPDF<<<blocks, threadsPerBlock>>>(N, D, h_logDetCovariances[k],
                                                                     h_weights[k], &d_logpdf[k * N]);
    }

    if (!log) {
        int threadsPerBlock = 256;
        int blocks = std::ceil(N / threadsPerBlock);
        exponential<<<blocks, threadsPerBlock>>>(N, K, d_logpdf);
    }

    for (int k = 0; k < K; k++) {
        cudaFree(d_covariancesLUArray[k]);
    }
}
