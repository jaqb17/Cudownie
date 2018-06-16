
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;


	for (int i = index; i < n; i+=stride)
		y[i] = x[i] + y[i];
}

void add_wrapper(int N, float* x, float* y)
{
	
	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;

	// Allocate Unified Memory – accessible from CPU or GPU


	add<<<numBlocks, blockSize>>>(N, x, y);

	cudaDeviceSynchronize();
	// Free memory
	cudaFree(x);
	cudaFree(y);

}