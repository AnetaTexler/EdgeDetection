#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"

#include <stdio.h>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

void imshowInWindow(const string& windowTitle, const Mat& image)
{
	namedWindow(windowTitle, WINDOW_AUTOSIZE); // Create a window for display
	//resizeWindow(windowTitle, windowWidth, windowHeight);
	//moveWindow(windowTitle, moveX, moveY);
	imshow(windowTitle, image); // Show image inside window
	waitKey(1);
}

Mat applyConvolution(const Mat& sourceImage, const Mat& convolutionKernel, const float kernelCoefficient)
{
	Mat resultImage = sourceImage.clone(); // deep copy
	float roiSum = 0;

	for (int row = (convolutionKernel.rows / 2); row < sourceImage.rows - (convolutionKernel.rows / 2); row++)
	{
		for (int col = (convolutionKernel.cols / 2); col < sourceImage.cols - (convolutionKernel.cols / 2); col++)
		{
			// region of interest (ROI)
			for (int roiRow = 0; roiRow < convolutionKernel.rows; roiRow++)
			{
				for (int roiCol = 0; roiCol < convolutionKernel.cols; roiCol++)
				{
					roiSum += sourceImage.at<float>(row - (convolutionKernel.rows / 2) + roiRow, col - (convolutionKernel.cols / 2) + roiCol)
						* convolutionKernel.at<float>(roiRow, roiCol);
				}
			}

			resultImage.at<float>(row, col) = roiSum * kernelCoefficient;
			roiSum = 0;
		}
	}
	return resultImage;
}

Mat getGradientMagnitude(const Mat& xGradientImage, const Mat& yGradientImage)
{
	Mat resultImage = xGradientImage.clone();

	for (int row = 0; row < xGradientImage.rows; row++)
	{
		for (int col = 0; col < xGradientImage.cols; col++)
		{
			resultImage.at<float>(row, col) = sqrt(xGradientImage.at<float>(row, col) * xGradientImage.at<float>(row, col)
				+ yGradientImage.at<float>(row, col) * yGradientImage.at<float>(row, col));
		}
	}

	return resultImage;
}

Mat getGradientDirection(const Mat& xGradientImage, const Mat& yGradientImage)
{
	Mat resultImage = xGradientImage.clone();

	for (int row = 0; row < xGradientImage.rows; row++)
	{
		for (int col = 0; col < xGradientImage.cols; col++)
		{
			resultImage.at<float>(row, col) = atan2f(yGradientImage.at<float>(row, col), xGradientImage.at<float>(row, col));
			//resultImage.at<float>(row, col) = atanf(yGradientImage.at<float>(row, col) / xGradientImage.at<float>(row, col));
		}
	}

	return resultImage;
}

Mat applyDoubleThresholdAndWeakEdgesSuppression(const Mat& src)
{
	float avg = 0;
	for (int row = 0; row < src.rows; row++)
	{
		for (int col = 0; col < src.cols; col++)
		{
			//if (src.at<float>(row, col) > max)
			//	max = src.at<float>(row, col);
			avg += src.at<float>(row, col);
		}
	}
	avg = avg / (src.rows * src.cols);

	float highThreshold = 4 * avg;
	float lowThreshold = 2 * avg;

	Mat resultImage = src.clone();

	for (int row = 1; row < src.rows - 1; row++)
	{
		for (int col = 1; col < src.cols - 1; col++)
		{
			if (src.at<float>(row, col) <= lowThreshold) // pixel value is lower than lowThreshold = suppression
				resultImage.at<float>(row, col) = 0;

			// 8-connectivity (check if neighborhood of weak pixel contains strong pixel)
			if (src.at<float>(row, col) > lowThreshold && src.at<float>(row, col) < highThreshold) // Weak pixel = pixel value is higher than lowThreshold and lower than highThreshold
			{
				// if weak pixel has NO strong neighbor, suppress it
				if (src.at<float>(row, col - 1) < highThreshold && /*horizontal neighbor - left*/
					src.at<float>(row, col + 1) < highThreshold && /*horizontal neighbor - right*/
					src.at<float>(row - 1, col) < highThreshold && /*vertical neighbor - up*/
					src.at<float>(row + 1, col) < highThreshold && /*vertical neighbor - down*/
					src.at<float>(row - 1, col - 1) < highThreshold && /*diagonal neighbor*/
					src.at<float>(row - 1, col + 1) < highThreshold && /*diagonal neighbor*/
					src.at<float>(row + 1, col - 1) < highThreshold && /*diagonal neighbor*/
					src.at<float>(row + 1, col + 1) < highThreshold)   /*diagonal neighbor*/
				{
					resultImage.at<float>(row, col) = 0;
				}
			}
		}
	}

	return resultImage;
}

__global__ void applyConvolution_CUDA(float* resultImageData, const float* sourceImageData, const int imageRowsSize, const int imageColsSize,
									 const float* convKernelData, const int convKernelSize, const float convKernelCoeff)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int row = index / imageColsSize;
	int col = index % imageColsSize;

	if (row < convKernelSize / 2 || col < convKernelSize / 2 || row > imageRowsSize - convKernelSize / 2 || col > imageColsSize - convKernelSize / 2)
	{
		return;
	}

	float roiSum = 0;
	for (int roiRow = 0; roiRow < convKernelSize; roiRow++)
	{
		for (int roiCol = 0; roiCol < convKernelSize; roiCol++)
		{
			int imageRow = row - (convKernelSize / 2) + roiRow;
			int imageCol = col - (convKernelSize / 2) + roiCol;
			roiSum += sourceImageData[imageRow*imageColsSize + imageCol] * convKernelData[roiRow * convKernelSize + roiCol];
		}
	}
	resultImageData[row * imageColsSize + col] = roiSum * convKernelCoeff;
}


int main()
{
	//string sourcePath = "sample_images\\valve.jpg";
	string sourcePath = "sample_images\\lion4k.jpg";
	//string sourcePath = "sample_images\\peacockHigh.jpg";
	//string sourcePath = "sample_images\\kingfisherHigh.jpg";
	//string sourcePath = "sample_images\\kingfisher.jpg";
	//string sourcePath = "sample_images\\kingfishers.jpg";

	Mat sourceImage;
	sourceImage = imread(sourcePath, CV_LOAD_IMAGE_GRAYSCALE); // Read the file

	if (!sourceImage.data) // Check for invalid input
	{
		cout << "Could not open or find the image" << endl;
		return -1;
	}

	Mat sourceImageF;
	sourceImage.convertTo(sourceImageF, CV_32FC1, 1 / 255.0); // convert image from uchar to float from 0.0 to 1.0 (due to negative values in Sobel operator)

	// Gaussian blur convolution kernel 5x5
	float dataGauss[25] = { 1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36, 24, 6, 4, 16, 24, 16, 4, 1, 4, 6, 4, 1 }; // *1/256
	Mat gaussianKernel = Mat(5, 5, CV_32FC1, dataGauss);
	float convKernelCoeff = 1 / 256.0;

	// ********************* CALL KERNEL FOR GPU ********************************
	unsigned int arraySize = sourceImage.rows * sourceImage.cols;
	unsigned int numOfThreadsInBlock = 512;
	unsigned int numOfBlocks = (arraySize + numOfThreadsInBlock - 1) / numOfThreadsInBlock;
	float *hostSourceImageData, *hostConvKernelData, *hostResultImageData;
	float *devSourceImageData, *devConvKernelData, *devResultImageData;

	auto start = chrono::system_clock::now();
	
	// Allocate memory on GPU
	cudaMalloc((void**)&devSourceImageData, arraySize * sizeof(float));
	cudaMalloc((void**)&devConvKernelData, gaussianKernel.rows * gaussianKernel.cols * sizeof(float));
	cudaMalloc((void**)&devResultImageData, arraySize * sizeof(float));

	// Allocate memory on CPU (possible even by malloc or new)
	cudaHostAlloc((void**)&hostSourceImageData, arraySize * sizeof(float), cudaHostAllocDefault);
	cudaHostAlloc((void**)&hostConvKernelData, gaussianKernel.rows * gaussianKernel.cols * sizeof(float), cudaHostAllocDefault);
	cudaHostAlloc((void**)&hostResultImageData, arraySize * sizeof(float), cudaHostAllocDefault);

	// Initialize arrays on the host
	for (int i = 0; i < arraySize; i++)
	{
		hostSourceImageData[i] = ((float*)sourceImageF.data)[i];
		hostResultImageData[i] = ((float*)sourceImageF.data)[i];
	}
	for (int i = 0; i < gaussianKernel.rows * gaussianKernel.cols; i++)
		hostConvKernelData[i] = dataGauss[i];

	// Copy data CPU -> GPU
	cudaMemcpy(devSourceImageData, hostSourceImageData, arraySize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devConvKernelData, hostConvKernelData, gaussianKernel.rows * gaussianKernel.cols * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devResultImageData, hostResultImageData, arraySize * sizeof(float), cudaMemcpyHostToDevice);

	// Launch a kernel on the GPU
	applyConvolution_CUDA<<<numOfBlocks, numOfThreadsInBlock>>>(devResultImageData, devSourceImageData, sourceImage.rows, sourceImage.cols, 
															   devConvKernelData, gaussianKernel.rows, convKernelCoeff);
	cudaDeviceSynchronize(); // wait for kernel end

	// Copy data GPU -> CPU
	cudaMemcpy(hostResultImageData, devResultImageData, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

	auto end = chrono::system_clock::now();
	long long millis = chrono::duration_cast<chrono::milliseconds>(end - start).count();
	printf("1st convolution: %.2f s\n", millis / 1000.0);

	Mat blurredImage(sourceImage.rows, sourceImage.cols, CV_32FC1, hostResultImageData); // not a deep copy of hostResultImageData!!!

	// free memory blocks on CPU
	cudaFreeHost(hostSourceImageData);
	cudaFreeHost(hostConvKernelData);
	//cudaFreeHost(hostResultImageData);
	// free memory blocks on GPU
	cudaFree(devSourceImageData);
	cudaFree(devConvKernelData);
	cudaFree(devResultImageData);
	// ***************************************************************************
	
	// WRITE
	Mat toWriteImage;
	blurredImage.convertTo(toWriteImage, CV_8UC1, 255.0);
	imwrite(sourcePath.substr(0, sourcePath.find(".jpg")) + "_gaussian_blur.jpg", toWriteImage);

	// Convolution kernel to get the image gradient in x direction (Sobel operator)
	float dataGx[9] = { 1, 0, -1, 2, 0, -2, 1, 0, -1 };
	Mat xGradientKernel = Mat(3, 3, CV_32FC1, dataGx);
	convKernelCoeff = 1;
	
	// ********************* CALL KERNEL FOR GPU *********************************
	// Allocate memory on GPU
	cudaMalloc((void**)&devSourceImageData, arraySize * sizeof(float));
	cudaMalloc((void**)&devConvKernelData, xGradientKernel.rows * xGradientKernel.cols * sizeof(float));
	cudaMalloc((void**)&devResultImageData, arraySize * sizeof(float));

	// Allocate memory on CPU (possible even by malloc or new)
	cudaHostAlloc((void**)&hostSourceImageData, arraySize * sizeof(float), cudaHostAllocDefault);
	cudaHostAlloc((void**)&hostConvKernelData, xGradientKernel.rows * xGradientKernel.cols * sizeof(float), cudaHostAllocDefault);
	cudaHostAlloc((void**)&hostResultImageData, arraySize * sizeof(float), cudaHostAllocDefault);

	// Initialize arrays on the host
	for (int i = 0; i < arraySize; i++)
	{
		hostSourceImageData[i] = ((float*)blurredImage.data)[i];
		hostResultImageData[i] = ((float*)blurredImage.data)[i];
	}
	for (int i = 0; i < xGradientKernel.rows * xGradientKernel.cols; i++)
		hostConvKernelData[i] = dataGx[i];

	// Copy data CPU -> GPU
	cudaMemcpy(devSourceImageData, hostSourceImageData, arraySize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devConvKernelData, hostConvKernelData, xGradientKernel.rows * xGradientKernel.cols * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devResultImageData, hostResultImageData, arraySize * sizeof(float), cudaMemcpyHostToDevice);

	// Launch a kernel on the GPU
	applyConvolution_CUDA<<<numOfBlocks, numOfThreadsInBlock>>>(devResultImageData, devSourceImageData, sourceImage.rows, sourceImage.cols,
															   devConvKernelData, xGradientKernel.rows, convKernelCoeff);
	cudaDeviceSynchronize(); // wait for kernel end

	// Copy data GPU -> CPU
	cudaMemcpy(hostResultImageData, devResultImageData, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

	Mat xGradientImage(blurredImage.rows, blurredImage.cols, CV_32FC1, hostResultImageData);

	// free memory blocks on CPU
	cudaFreeHost(hostSourceImageData);
	cudaFreeHost(hostConvKernelData);
	//cudaFreeHost(hostResultImageData);
	// free memory blocks on GPU
	cudaFree(devSourceImageData);
	cudaFree(devConvKernelData);
	cudaFree(devResultImageData);
	// ***************************************************************************

	// WRITE
	xGradientImage.convertTo(toWriteImage, CV_8UC1, 255.0);
	imwrite(sourcePath.substr(0, sourcePath.find(".jpg")) + "_gradient_X.jpg", toWriteImage);
	
	// Convolution kernel to get the image gradient in y direction (Sobel operator)
	float dataGy[9] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };
	Mat yGradientKernel = Mat(3, 3, CV_32FC1, dataGy);
	convKernelCoeff = 1;

	// ********************* CALL KERNEL FOR GPU **********************************
	// Allocate memory on GPU
	cudaMalloc((void**)&devSourceImageData, arraySize * sizeof(float));
	cudaMalloc((void**)&devConvKernelData, yGradientKernel.rows * yGradientKernel.cols * sizeof(float));
	cudaMalloc((void**)&devResultImageData, arraySize * sizeof(float));

	// Allocate memory on CPU (possible even by malloc or new)
	cudaHostAlloc((void**)&hostSourceImageData, arraySize * sizeof(float), cudaHostAllocDefault);
	cudaHostAlloc((void**)&hostConvKernelData, yGradientKernel.rows * yGradientKernel.cols * sizeof(float), cudaHostAllocDefault);
	cudaHostAlloc((void**)&hostResultImageData, arraySize * sizeof(float), cudaHostAllocDefault);

	// Initialize arrays on the host
	for (int i = 0; i < arraySize; i++)
	{
		hostSourceImageData[i] = ((float*)blurredImage.data)[i];
		hostResultImageData[i] = ((float*)blurredImage.data)[i];
	}
	for (int i = 0; i < yGradientKernel.rows * yGradientKernel.cols; i++)
		hostConvKernelData[i] = dataGy[i];

	// Copy data CPU -> GPU
	cudaMemcpy(devSourceImageData, hostSourceImageData, arraySize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devConvKernelData, hostConvKernelData, yGradientKernel.rows * yGradientKernel.cols * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devResultImageData, hostResultImageData, arraySize * sizeof(float), cudaMemcpyHostToDevice);

	// Launch a kernel on the GPU
	applyConvolution_CUDA<<<numOfBlocks, numOfThreadsInBlock>>>(devResultImageData, devSourceImageData, sourceImage.rows, sourceImage.cols,
															   devConvKernelData, yGradientKernel.rows, convKernelCoeff);
	cudaDeviceSynchronize(); // wait for kernel end

	// Copy data GPU -> CPU
	cudaMemcpy(hostResultImageData, devResultImageData, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

	Mat yGradientImage(blurredImage.rows, blurredImage.cols, CV_32FC1, hostResultImageData);

	// free memory blocks on CPU
	cudaFreeHost(hostSourceImageData);
	cudaFreeHost(hostConvKernelData);
	//cudaFreeHost(hostResultImageData);
	// free memory blocks on GPU
	cudaFree(devSourceImageData);
	cudaFree(devConvKernelData);
	cudaFree(devResultImageData);
	// *****************************************************************************

	// WRITE
	yGradientImage.convertTo(toWriteImage, CV_8UC1, 255.0);
	imwrite(sourcePath.substr(0, sourcePath.find(".jpg")) + "_gradient_Y.jpg", toWriteImage);

	// Get the gradient magnitude: G = sqrt(Gx^2 + Gy^2)
	Mat gradientMagnitudeImage = getGradientMagnitude(xGradientImage, yGradientImage);
	//imshowInWindow("Gradient Magnitude", gradientMagnitudeImage);

	// WRITE
	gradientMagnitudeImage.convertTo(toWriteImage, CV_8UC1, 255.0);
	imwrite(sourcePath.substr(0, sourcePath.find(".jpg")) + "_gradient_magnitude.jpg", toWriteImage);

	Mat finalResultImage = applyDoubleThresholdAndWeakEdgesSuppression(gradientMagnitudeImage);
	//imshowInWindow("Final Result", finalResultImage);

	// WRITE
	finalResultImage.convertTo(toWriteImage, CV_8UC1, 255.0);
	imwrite(sourcePath.substr(0, sourcePath.find(".jpg")) + "_final_result.jpg", toWriteImage);

	cudaFreeHost(hostResultImageData);

	//waitKey(0);
	system("pause");

	return 0;
}






cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int Xmain()
{
	const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n", c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
