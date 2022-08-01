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

__global__ void rmvgmm_assignCluster(int K, int N, const float *d_weights, const float *d_uniformRand,
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

__global__ void rmvgmm_setupBatchMultiply(int N, int D, const int *d_clusters, float *d_normalRand,
                                    float **d_meansArray, float **d_covariancesArray,
                                    float **d_ptrSamples, float **d_ptrMeans, float **d_ptrCovariances,
                                    float *d_randVals)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N) {

        d_ptrSamples[n] = &d_normalRand[n * D];

        memcpy(&d_randVals[n * D], d_meansArray[d_clusters[n]], D * sizeof(float));
        d_ptrMeans[n] = &d_randVals[n * D];

        d_ptrCovariances[n] = d_covariancesArray[d_clusters[n]];
    }
}

void rmvgmm(cublasHandle_t cublasHandle, cusolverDnHandle_t cusolverHandle,
                    int N, int D, int K, const float *d_weights, const float *d_means,
                    const float *d_covariances, float *d_randomValues) {

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 148);

    // Generate N random values between 0.0 and 1.0

    thrust::device_vector<float> d_uniformRand(N);
    curandGenerateUniform(gen, thrust::raw_pointer_cast(d_uniformRand.data()), N);

    // Assign N clusters based on random uniform values and weights

    int threadsPerBlock = 256;
    int blocks = std::ceil((float)N / threadsPerBlock);
    thrust::device_vector<int> d_clusters(N);

    rmvgmm_assignCluster<<<blocks, threadsPerBlock>>>(K, N, d_weights,
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

    float *h_covarianceLower = (float *)malloc(D * D * sizeof(float));
    for (int k = 0; k < K; k++) {
        cudaMemcpy(h_covarianceLower, covariancesArray[k], D * D * sizeof(float), cudaMemcpyDeviceToHost);
        h_covarianceLower[3] = 0.0f;
        h_covarianceLower[6] = 0.0f;
        h_covarianceLower[7] = 0.0f;
        cudaMemcpy(covariancesArray[k], h_covarianceLower, D * D * sizeof(float), cudaMemcpyHostToDevice);
    }
    free(h_covarianceLower);

    // Generate standard normal variables, (D * N) samples

    thrust::device_vector<float> d_normalRand(D * N);
    curandGenerateNormal(gen, thrust::raw_pointer_cast(d_normalRand.data()), D * N, 0.0, 1.0);

    // For each sample, create pointer to assigned cluster mean and covariance lower

    thrust::device_vector<float*> d_ptrSamples(N);
    thrust::device_vector<float*> d_ptrCovariances(N);
    thrust::device_vector<float*> d_ptrMeans(N);

    rmvgmm_setupBatchMultiply<<<blocks, threadsPerBlock>>>(N, D,
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

void logDeterminants(cusolverDnHandle_t cusolverHandle, int D, int K, const float *d_covariances, float *h_logDetCovariances)
{
    gesvdjInfo_t gesvdj_params = NULL;
    CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&gesvdj_params));

    int lwork;
    thrust::device_vector<float> d_covariancesCopy(K * D * D);
    thrust::device_vector<float> d_eigenvalues(K * D);
    float *U = NULL;
    float *V = NULL;

    cudaMemcpy(thrust::raw_pointer_cast(d_covariancesCopy.data()), d_covariances,
               K * D * D * sizeof(float), cudaMemcpyDeviceToDevice);

    CUSOLVER_CHECK(cusolverDnSgesvdjBatched_bufferSize(cusolverHandle, CUSOLVER_EIG_MODE_NOVECTOR, D, D,
       thrust::raw_pointer_cast(d_covariancesCopy.data()), D,
       thrust::raw_pointer_cast(d_eigenvalues.data()), U, D, V, D, &lwork, gesvdj_params, K));

    thrust::device_vector<float> d_work(lwork);
    thrust::device_vector<int> d_info(K);
    CUSOLVER_CHECK(cusolverDnSgesvdjBatched(cusolverHandle, CUSOLVER_EIG_MODE_NOVECTOR, D, D,
                   thrust::raw_pointer_cast(d_covariancesCopy.data()), D,
                   thrust::raw_pointer_cast(d_eigenvalues.data()), U, D, V, D,
                   thrust::raw_pointer_cast(d_work.data()), lwork,
                   thrust::raw_pointer_cast(d_info.data()), gesvdj_params, K));

    for (int k = 0; k < K; k++) {

        float h_eigenvalues[D];

        cudaMemcpy(h_eigenvalues, thrust::raw_pointer_cast(&d_eigenvalues[k * D]),
                   D * sizeof(float), cudaMemcpyDeviceToHost);

        float sum_log_eigs = 0.0f;
        for (int d = 0; d < D; d++) {
            sum_log_eigs += std::log(h_eigenvalues[d]);
        }

        h_logDetCovariances[k] = sum_log_eigs;
    }
}

__global__ void mahalanobis_centerData(int D, int N, const float *d_data, const float *d_means, float *d_dataZeroCentered)
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
    int blocks = std::ceil((float)N / threadsPerBlock);
 
    mahalanobis_centerData<<<blocks, threadsPerBlock>>>(D, N, d_data, d_means,
                                    thrust::raw_pointer_cast(d_dataZeroCentered.data()));

    // transpose(x - mean) * covarianceInverse for all N data points (DxN)

    alpha = 1.0f;
    beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, D, N, D,
                             &alpha, d_covarianceInverse, D,
                             thrust::raw_pointer_cast(d_dataZeroCentered.data()), D,
                             &beta, thrust::raw_pointer_cast(d_dataZeroCenteredScaled.data()), D));

    // transpose(x - mean) * covarianceInverse * (x - mean) for all N data points (Nx1)

    alpha = 1.0f;
    beta = 0.0f;
    CUBLAS_CHECK(cublasSgemmStridedBatched(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, D, &alpha,
                                           thrust::raw_pointer_cast(d_dataZeroCenteredScaled.data()), 1, D,
                                           thrust::raw_pointer_cast(d_dataZeroCentered.data()), D, D,
                                           &beta, d_mahalanobisSquared, 1, 1, N));
}

// d_mahalanobisSquared is Nx1 column-major order
__global__ void dmvgmm_multivariateNormalLogPDF(int N, int D, float logDetCovariance,
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
__global__ void dmvgmm_exponential(int N, int K, float *d_matrix)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N) {
        for (int k = 0; k < K; k++) {
            d_matrix[k * N + n] = expf(d_matrix[k * N + n]);
        }
    }
}

void dmvgmm(cublasHandle_t cublasHandle, cusolverDnHandle_t cusolverHandle, int N, int D, int K,
            const float *d_data, const float *h_weights, const float *d_means,
            const float *d_covariances, float *d_logpdf, bool log) {

    thrust::host_vector<float*> covariancesArray(K);
    thrust::host_vector<float*> covariancesInversesArray(K);
    thrust::device_vector<int> d_pivotArray(D * K);
    thrust::device_vector<int> d_infoArray(K);
    thrust::host_vector<float> h_logDetCovariances(K);

    for (int k = 0; k < K; k++) {

        cudaMalloc(thrust::raw_pointer_cast(&covariancesArray[k]), D * D * sizeof(float));
        cudaMalloc(thrust::raw_pointer_cast(&covariancesInversesArray[k]), D * D * sizeof(float));

        cudaMemcpy(covariancesArray[k], &d_covariances[k * D * D],
                   D * D * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // Get inverses of covariance matrices

    thrust::device_vector<float*> d_covariancesArray = covariancesArray;
    thrust::device_vector<float*> d_covariancesInversesArray = covariancesInversesArray;
    CUBLAS_CHECK(cublasSmatinvBatched(cublasHandle, D,
                                      thrust::raw_pointer_cast(d_covariancesArray.data()), D,
                                      thrust::raw_pointer_cast(d_covariancesInversesArray.data()), D,
                                      thrust::raw_pointer_cast(d_infoArray.data()), K));

    // Get determinants of covariance matrices

    logDeterminants(cusolverHandle, D, K, d_covariances, thrust::raw_pointer_cast(h_logDetCovariances.data()));

    for (int k = 0; k < K; k++) {

        // Calculate mahalanobis distance squared from mean and covariance

        mahalanobisDistanceSquared(cublasHandle, D, N, d_data, &d_means[k * D],
                                   covariancesInversesArray[k], &d_logpdf[k * N]);

        // Calculate log probabilities from squared mahlanobis distances

        int threadsPerBlock = 256;
        int blocks = std::ceil((float)N / threadsPerBlock);

        dmvgmm_multivariateNormalLogPDF<<<blocks, threadsPerBlock>>>(N, D, h_logDetCovariances[k],
                                                                     h_weights[k], &d_logpdf[k * N]);
    }

    if (!log) {
 
        int threadsPerBlock = 256;
        int blocks = std::ceil((float)N / threadsPerBlock);

        dmvgmm_exponential<<<blocks, threadsPerBlock>>>(N, K, d_logpdf);
    }
}
