#include <stats.h>
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

void meanAndCovariance(cublasHandle_t cublasHandle, int D, int N, const float *d_data,
                       float *d_mean, float *d_covariance)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    thrust::device_vector<float> d_cov1(D * D);
    thrust::device_vector<float> d_cov2(D * D);
    thrust::device_vector<float> d_ones(N, 1.0);

    // X*transpose(X) / N
    alpha = 1.0f / N;
    CUBLAS_CHECK(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, D, D, N,
                             &alpha, d_data, D, d_data, D,
                             &beta, thrust::raw_pointer_cast(d_cov1.data()), D));

    // Mean vector of each row 
    alpha = 1.0f / N;
    CUBLAS_CHECK(cublasSgemv(cublasHandle, CUBLAS_OP_N, D, N,
                             &alpha, d_data, D, thrust::raw_pointer_cast(d_ones.data()), 1,
                             &beta, d_mean, 1));

    // means * transpose(means)
    alpha = 1.0f;
    CUBLAS_CHECK(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, D, D, 1,
                             &alpha, d_mean, 1, d_mean, 1,
                             &beta, thrust::raw_pointer_cast(d_cov2.data()), D));

    //  (X*transpose(X) / N) -  means * transpose(means)
    alpha = 1.0f;
    beta = -1.0f;
    CUBLAS_CHECK(cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, D, D,
                             &alpha, thrust::raw_pointer_cast(d_cov1.data()), D,
                             &beta, thrust::raw_pointer_cast(d_cov2.data()), D, d_covariance, D));
}