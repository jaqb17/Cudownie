
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/mat.hpp>


__global__
void FloydSteinberg(unsigned char* input, unsigned char* output, int* error, unsigned int rows, unsigned int cols)
{
	int x = threadIdx.x;
	int y = blockIdx.x * blockDim.x;
	const int p = 128;
	const int black = 0;
	const int white = 255;

	int e = 0;

	if (input[y + x] + error[y + x] < p)
	{
		output[y + x] = black;
		e = input[y + x] + error[y + x];
	}
	else
	{
		output[y + x] = white;
		e = input[y + x] + error[y + x] - white;
	}

	if (x < blockDim.x - 1)
	{
		error[y + x + 1] += e * 7 / 16;

		if (blockIdx.x < rows - 1)
		{
			error[(blockIdx.x + 1)*blockDim.x + x + 1] += e * 1 / 16;
		}
	}

	if (blockIdx.x < cols - 1)
	{
		error[(blockIdx.x + 1)*blockDim.x + x] += e * 5 / 16;

		if (x > 0)
		{
			error[(blockIdx.x + 1)*blockDim.x + x - 1] += e * 3 / 16;
		}
	}
}

__global__
void FloydSteinbergST(unsigned char* input, unsigned char* output, int* error, unsigned int rows, unsigned int cols)
{
	const int p = 128;
	const int black = 0;
	const int white = 255;
	int width = cols;
	int heigth = rows;
	int e = 0;


	for (int y=0;y<rows;y++)
	{
		for (int x=0;x<cols;x++)
		{
			int e = 0;
			if(input[y*cols+x]+error[y*width+x]<p)
			{
				output[y*cols + x] = black;
				e = input[y*cols + x] + error[y*width + x];
			}
			else
			{
				output[y*cols + x] = white;
				e = input[y*cols + x] + error[y*width + x] - white;
			}

			if (x < width - 1)
			{
				error[y*width + x + 1] += e * 7 / 16;  //prawa

				if (y < heigth - 1)
				{
					error[(y + 1)*width + x + 1] += e * 1 / 16; //prawa dol
				}
			}

			if (y < heigth - 1)
			{
				error[(y + 1)*width + x] += e * 5 / 16;

				if (x > 0)
				{
					error[(y + 1)*width + x - 1] += e * 3 / 16;
				}
			}
		}
	}

}

void FloydSteinbergWrapper(const cv::Mat& in, cv::Mat& out)
{

	unsigned char *input_prt, *output_ptr;
	int *error_ptr;

	cudaMalloc<unsigned char>(&input_prt, in.rows*in.cols);
	cudaMalloc<unsigned char>(&output_ptr, out.rows*out.cols);
	cudaMalloc<int>(&error_ptr, in.cols*in.rows);

	cudaMemcpy(input_prt, in.ptr(), in.rows*in.cols, cudaMemcpyHostToDevice);


	//FloydSteinberg << <in.rows, in.cols >> > (input_prt, output_ptr, error_ptr, in.rows, in.cols);
	FloydSteinbergST << <1, 1 >> > (input_prt, output_ptr, error_ptr, in.rows, in.cols);

	cudaDeviceSynchronize();

	cudaMemcpy(out.ptr(), output_ptr, out.cols*out.rows, cudaMemcpyDeviceToHost);

	// Free memory
	cudaFree(input_prt);
	cudaFree(output_ptr);
	cudaFree(error_ptr);

}