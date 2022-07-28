#include <cublas_v2.h>

#include <string>
#include <iostream>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <library_types.h>
#include <cuda_fp16.h>

#include "args.h"
#include "cuda_err.h"

/*
 * Special thanks to the following codes and posts for ideas 
 * on how to run this benchmark:
 * https://github.com/hma02/cublasHgemm-P100
 * https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
 */

/*
 * Kernels
 */

//Copy kernel simplified for experiments used here
//from Elemental/src/hydrogen/blas/gpu/Copy.cu line 28
template<typename T, typename U>
__global__ void copy_2d(
    int m, int n,
    T const* __restrict__ src, int src_row_stride, int src_col_stride,
    U* __restrict__ dest, int dest_row_stride, int dest_col_stride)
{
	const int TILE_SIZE = 32;
	const int BLK_COLS = 8;
    __shared__ T** tile_shared[TILE_SIZE][TILE_SIZE+1];
    auto tile = reinterpret_cast<T(*)[TILE_SIZE+1]>(tile_shared);

    int const start_row = blockIdx.x * TILE_SIZE + threadIdx.x;
    int const start_col = blockIdx.y * TILE_SIZE + threadIdx.y;

    src += start_row*src_row_stride + start_col*src_col_stride;
    dest += start_row*dest_row_stride + start_col*dest_col_stride;
    if (start_row < m && start_col < n)
    {
        if (start_col + TILE_SIZE < n)
        {
            // Load the data
            #pragma unroll
            for (int ii = 0; ii < TILE_SIZE; ii += BLK_COLS)
                tile[threadIdx.y+ii][threadIdx.x] = src[ii*src_col_stride];

            // Store the data
            #pragma unroll
            for (int ii = 0; ii < TILE_SIZE; ii += BLK_COLS)
                dest[ii*dest_col_stride] = tile[threadIdx.y+ii][threadIdx.x];
        }
        else
        {
            // Load the data
            for (int ii = 0; ii < TILE_SIZE && start_col + ii < n; ii += BLK_COLS)
            {
                tile[threadIdx.y+ii][threadIdx.x] = src[ii*src_col_stride];
            }

            // Store the data
            for (int ii = 0; ii < TILE_SIZE && start_col + ii < n; ii += BLK_COLS)
            {
                dest[ii*dest_col_stride] = tile[threadIdx.y+ii][threadIdx.x];
            }
        }
    }
}

template<typename T>
__global__ void initA(int n, T* A)
{
	int index = threadIdx.x;
	int stride = blockDim.x;
	for (int i = index; i < n; i+= stride)
		A[i] = T(2.0);
}

template<typename T>
__global__ void initB(int n, T* B)
{
	int index = threadIdx.x;
	int stride = blockDim.x;
	for (int i = index; i < n; i+= stride)
		B[i] = T(3.0);
}

template<typename T>
__global__ void initC(int n, T* C)
{
	int index = threadIdx.x;
	int stride = blockDim.x;
	for (int i = index; i < n; i+= stride)
		C[i] = T(0.0);
}

template<typename T>
__global__ void initConsts(T* a, T* b)
{
	*a = T(1.0);
	*b = T(0.0);
}

/*
 * Experiment definitions
 */

float copy_experiment(Args);
float gemmex_experiment(Args);

/*
 * Main
 */

int main(int argc, char* argv[])
{
	Args a;
	if (!get_args(argc, argv, &a))
		return 1;

#if defined(DEBUG)
	std::cout << "m: " << a.m << " n: " << a.n << " k: " << a.k;
	if (a.is_copy)
		std::cout << " copy: True" << std::endl;
	else
		std::cout << " copy: False" << std::endl;
#endif

	float t = 0.0f;
	if (a.is_copy)
	{
		for (int i = 0; i < a.n_runs; i++)
			t += copy_experiment(a);

		std::cout << "Copy + SGEMM Ave Elapsed Time: " 
			<< t/a.n_runs << "s" << std::endl;
	}
	else
	{
		for (int i = 0; i < a.n_runs; i++)
			t += gemmex_experiment(a);
		std::cout << "GemmEX Ave Elapsed Time: " 
			<< t/a.n_runs << "s" << std::endl;
	}

	return 0;
}

float copy_experiment(Args args)
{
	__half *A_orig;
	float *A;
	__half *B_orig;
	float *B;
	float *C;
	float *alpha;
	float *beta;

	//Initializations
	int a_size = args.m * args.k;
	int b_size = args.k * args.n;
	int c_size = args.m * args.n;
	checkCuda(cudaMallocManaged(&A_orig, a_size*sizeof(__half)));
	checkCuda(cudaMallocManaged(&A, a_size*sizeof(float)));
	checkCuda(cudaMallocManaged(&B_orig, b_size*sizeof(__half)));
	checkCuda(cudaMallocManaged(&B, b_size*sizeof(float)));
	checkCuda(cudaMallocManaged(&C, c_size*sizeof(float)));
	checkCuda(cudaMallocManaged(&alpha, sizeof(float)));
	checkCuda(cudaMallocManaged(&beta, sizeof(float)));

	initA<<<1, 256>>>(a_size, A_orig);
	initB<<<1, 256>>>(b_size, B_orig);
	initC<<<1, 256>>>(c_size, C);
	initConsts<<<1, 1>>>(alpha, beta);

	//Experiment
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasStatus_t stat;
	cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	copy_2d<<<1, 256>>>(args.m, args.k, 
		A_orig, 1, 256, //src
		A, 1, 256); //dest
	copy_2d<<<1, 256>>>(args.m, args.k, 
		B_orig, 1, 256, //src
		B, 1, 256); //dest
	stat = cublasSgemm(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		args.m, args.n, args.k,
		alpha,
		A, args.m,
		B, args.k,
		beta,
		C, args.n);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		std::cerr << "cublas gemm Failed...." << std::endl;
		std::cerr << checkCublas(stat) << std::endl;
		exit(1);
	}
	checkCuda(cudaGetLastError());
	float elapsed;
	cudaEventElapsedTime(&elapsed, start, stop);
	elapsed /= 1000.0f;

	//Freedom
	cudaFree(A_orig);
	cudaFree(A);
	cudaFree(B);
	cudaFree(B_orig);
	cudaFree(C);
	cudaFree(alpha);
	cudaFree(beta);

	return elapsed;
}

float gemmex_experiment(Args args)
{
	__half *A;
	__half *B;
	float *C;
	float *alpha;
	float *beta;

	//Initializations
	int a_size = args.m * args.k;
	int b_size = args.k * args.n;
	int c_size = args.m * args.n;
	checkCuda(cudaMallocManaged(&A, a_size*sizeof(__half)));
	checkCuda(cudaMallocManaged(&B, b_size*sizeof(__half)));
	checkCuda(cudaMallocManaged(&C, c_size*sizeof(float)));
	checkCuda(cudaMallocManaged(&alpha, sizeof(float)));
	checkCuda(cudaMallocManaged(&beta, sizeof(float)));

	initA<<<1, 256>>>(a_size, A);
	initB<<<1, 256>>>(b_size, B);
	initC<<<1, 256>>>(c_size, C);
	initConsts<<<1, 1>>>(alpha, beta);

	//Experiment
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasStatus_t stat;
	cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	stat = cublasGemmEx(handle,
		CUBLAS_OP_N, CUBLAS_OP_N,
		args.m, args.n, args.k,
		alpha,
		A, CUDA_R_16F, args.m,
		B, CUDA_R_16F, args.k,
		beta,
		C, CUDA_R_32F, args.n,
		CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		std::cerr << "cublasGemmEx Failed...." << std::endl;
		std::cerr << "cublas Err no: " << stat << std::endl;
		std::cerr << checkCublas(stat) << std::endl;
		exit(1);
	}
	checkCuda(cudaGetLastError());
	float elapsed;
	cudaEventElapsedTime(&elapsed, start, stop);
	elapsed /= 1000.0f;

	//Freedom
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	cudaFree(alpha);
	cudaFree(beta);

	return elapsed;
}