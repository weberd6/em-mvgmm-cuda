#include <em-mvgmm.h>
#include <mvgmm.h>
#include <mvnorm.h>
#include <stats.h>
#include <cublas_v2.h>
#include <curand.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

// d_matrix is NxK column-major order
__global__ void eStep_normalizeExponentialColumWise(int N, int K, float *d_matrix)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    float max_value = -INFINITY;
    float sum_exp = 0.0;

    // K x blockDim.x
    extern __shared__ float matrix_shared[];

    if (n < N) {

        for (int k = 0; k < K; k++) {
            matrix_shared[k * blockDim.x + threadIdx.x] = d_matrix[k * N + n];
        }

        for (int k = 0; k < K; k++) {
            max_value = max(max_value, matrix_shared[k * blockDim.x + threadIdx.x]);
        }

        for (int k = 0; k < K; k++) {
            matrix_shared[k * blockDim.x + threadIdx.x] =
                expf(matrix_shared[k * blockDim.x + threadIdx.x] - max_value);
        }

        for (int k = 0; k < K; k++) {
            sum_exp += matrix_shared[k * blockDim.x + threadIdx.x];
        }

        for (int k = 0; k < K; k++) {
            matrix_shared[k * blockDim.x + threadIdx.x] /= sum_exp;
        }

        for (int k = 0; k < K; k++) {
            d_matrix[k * N + n] = matrix_shared[k * blockDim.x + threadIdx.x];
        }
    }
}

// d_data is DxN column-major order
// d_responsibilities is NxK column-major order
void eStep(cublasHandle_t cublasHandle, cusolverDnHandle_t cusolverHandle, int D, int N, int K,
           const float *d_data, const float *h_weights, const float *d_means,
           const float *d_covariances, float *d_responsibilities)
{
    // Calculate log PDF for each data point to get log of new responsibility weights

    dmvgmm(cublasHandle, cusolverHandle, N, D, K, d_data, h_weights, d_means, d_covariances, d_responsibilities, true);

    // Get real responsibilites by exponentiating and renormalizing

    int threadsPerBlock = 256;
    int blocks = std::ceil((float)N / threadsPerBlock);
    int sharedMem = K * threadsPerBlock * sizeof(float);

    eStep_normalizeExponentialColumWise<<<blocks, threadsPerBlock, sharedMem>>>(N, K, d_responsibilities);
}

// d_data is DxN column-major order
// d_weights is Nx1 vector
// d_weighted_data is a DxN column-major
__global__ void mstep_calculateWeightedData(int D, int N, const float *d_data, const float *d_weights,
                                      float *d_weighted_data)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N) {

        float weight = d_weights[n];

        for (int d = 0; d < D; d++) {
            d_weighted_data[n * D + d] = weight * d_data[n * D + d];
        }
    }
}

// d_data is DxN column-major order
// d_responsibilities is NxK column-major order
void mStep(cublasHandle_t cublasHandle, int D, int N, int K, const float *d_data, float *d_responsibilities,
           float *h_weights, float *d_means, float *d_covariances)
{
    float *h_responsibilitiesSum = (float*)malloc(K * sizeof(float));
    float alpha;
    float beta;

    for (int k = 0; k < K; k++) {

        // Recalculate weights

        thrust::device_ptr<float> r(&d_responsibilities[k * N]);
        h_responsibilitiesSum[k] = thrust::reduce(r, r + N);
        h_weights[k] = h_responsibilitiesSum[k] / N;

        // Weight the data by responsibilities

        thrust::device_vector<float> d_weightedData(D * N);

        int threadsPerBlock = 256;
        int blocks = std::ceil((float)N / threadsPerBlock);

        mstep_calculateWeightedData<<<blocks, threadsPerBlock>>>(D, N, d_data, &d_responsibilities[k * N],
                                                           thrust::raw_pointer_cast(d_weightedData.data()));

        // Recalculate means

        // means(Dx1) = 1.0/responsibilitesSum * data(DxN) * responsibilites[k](Nx1)

        alpha = 1.0f / h_responsibilitiesSum[k];
        beta = 0.0f;
        CUBLAS_CHECK(cublasSgemv(cublasHandle, CUBLAS_OP_N, D, N,
                                 &alpha, d_data, D, &d_responsibilities[k * N], 1,
                                 &beta, &d_means[k * D], 1));

        // Recalculate covariances

        // C(DxD) = means(Dx1) * transpose(means)(1xD)

        alpha = 1.0f;
        beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, D, D, 1,
                                 &alpha, &d_means[k * D], D, &d_means[k * D], D,
                                 &beta, &d_covariances[k * D * D], D));

        // covariance(DxD) = 1.0/responsibilitesSum * (weighted_data(DxN) * transpose(data)(NxD)) - C(DxD) 

        alpha = 1.0f / h_responsibilitiesSum[k];
        beta = -1.0f;
        CUBLAS_CHECK(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, D, D, N,
                                 &alpha, thrust::raw_pointer_cast(d_weightedData.data()), D,
                                 d_data, D, &beta, &d_covariances[k * D * D], D));
    }

    free(h_responsibilitiesSum);
}



// Both d_weights and d_logPdf are NxK matrices column-major
__global__ void em_calculateWeightedLogPdf(int N, int K, const float *d_weights, float *d_logPdf)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N) {
        for (int k = 0; k < K; k++) {
            d_logPdf[k * N + n] *= d_weights[k * N + n];
        }
    }
}

void expectationMaximizationMVGMM(cublasHandle_t cublasHandle, cusolverDnHandle_t cusolverHandle,
                                  int D, int N, int K, const float *d_data,
                                  float *h_weights, float *d_means, float *d_covariances) {

    // Initialize weights

    for (int k = 0; k < K; k++) {
        h_weights[k] = 1.0f/K;
    }

    // Find mean and covariance of data

    thrust::device_vector<float> d_mean(D);
    thrust::device_vector<float> d_covariance(D * D);
    meanAndCovariance(cublasHandle, D, N, d_data,
                      thrust::raw_pointer_cast(d_mean.data()),
                      thrust::raw_pointer_cast(d_covariance.data()));

    // Generate random cluster means

    rmvnorm(cublasHandle, cusolverHandle, K, D, thrust::raw_pointer_cast(d_mean.data()),
            thrust::raw_pointer_cast(d_covariance.data()), d_means);

    // Initialize all covariances to be the same. covariance of the data divided by number of clusters

    thrust::constant_iterator<float> nclusters(K);

    thrust::transform(d_covariance.begin(), d_covariance.end(), nclusters,
                      d_covariance.begin(), thrust::divides<float>());

    for (int k = 0; k < K; k++) {
        cudaMemcpy(&d_covariances[k * D * D], thrust::raw_pointer_cast(d_covariance.data()),
                   D * D * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    bool converged = false;
    float epsilon = 1.0e-5;
    float Q = -INFINITY;
    thrust::device_vector<float> d_logPdf(N * K);
    thrust::device_vector<float> d_responsibilities(K * N);
    int threadsPerBlock = 256;
    int blocks = std::ceil((float)N / threadsPerBlock);

    while (!converged) {

        eStep(cublasHandle, cusolverHandle, D, N, K, d_data, h_weights, d_means,
              d_covariances, thrust::raw_pointer_cast(d_responsibilities.data()));

        mStep(cublasHandle, D, N, K, d_data,
              thrust::raw_pointer_cast(d_responsibilities.data()),
              h_weights, d_means, d_covariances);

        dmvgmm(cublasHandle, cusolverHandle, N, D, K, d_data,
               h_weights, d_means, d_covariances,
               thrust::raw_pointer_cast(d_logPdf.data()), true);

        em_calculateWeightedLogPdf<<<blocks, threadsPerBlock>>>(N, K,
                    thrust::raw_pointer_cast(d_responsibilities.data()),
                    thrust::raw_pointer_cast(d_logPdf.data()));

        float Qn = thrust::reduce(d_logPdf.begin(), d_logPdf.end());

        if (std::abs(Qn-Q)/std::abs(Qn) < epsilon) {
            converged = true;
        }

        Q = Qn;
    }
}
