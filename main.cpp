#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


using namespace cv;


extern void add_wrapper(int n, float *x, float *y);

int main(void)
{
	cv::Mat in;

	in = imread("C:\\Users\\Kuba\\Desktop\\cuda\\Cudownie\\Pepe.jpg", CV_LOAD_IMAGE_COLOR);
	int N = 1 << 20;
	float *x, *y;
	cudaMallocManaged(&x, N * sizeof(float));
	cudaMallocManaged(&y, N * sizeof(float));
	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}


	add_wrapper(N, x, y);
	




	namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Display window", in);                   // Show our image inside it.

	waitKey(0);

	return 0;
}