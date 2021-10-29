#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;

void load(const char* path, float** imageData, int* imgRows, int* imgCols, float** convKernelData, int* convKernelSize, float* convKernelCoeff)
{
	FILE* file;
	file = fopen(path, "r");

	if (file == NULL)
	{
		printf("Cannot open file.\n");
		return;
	}
	
	fscanf(file, "%d %d ", imgRows, imgCols);
	*imageData = (float*)malloc(*imgRows * *imgCols * sizeof(float)); 
	for (int i = 0; i < *imgRows * *imgCols; i++)
		fscanf(file, "%f ", &(*imageData)[i]);

	fscanf(file, "%d %f ", convKernelSize, convKernelCoeff);
	*convKernelData = (float*)malloc(*convKernelSize * *convKernelSize * sizeof(float));
	for (int i = 0; i < *convKernelSize * *convKernelSize; i++)
		fscanf(file, "%f ", &(*convKernelData)[i]);

	fclose(file);
}

__global__ void applyConvolution_GPU(float* resultImageData, const float* sourceImageData, const int imageRowsSize, const int imageColsSize,
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
	float* imageData = NULL;
	float* convKernelData = NULL;
	int imgRows, imgCols, convKernelSize;
	float convKernelCoeff;

	load("srcImgData1.txt", &imageData, &imgRows, &imgCols, &convKernelData, &convKernelSize, &convKernelCoeff);
	
	unsigned int arraySize = imgRows * imgCols;
	unsigned int numOfThreadsInBlock = 512;
	unsigned int numOfBlocks = (arraySize + numOfThreadsInBlock - 1) / numOfThreadsInBlock;
	float *hostSourceImageData, *hostConvKernelData, *hostResultImageData;
	float *devSourceImageData, *devConvKernelData, *devResultImageData;

	// Choose which GPU to run on, change this on a multi-GPU system.
    	cudaError_t cudaStatus = cudaSetDevice(0);
    	if (cudaStatus != cudaSuccess) 
	{
        	fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        	return 1;
    	}
	
	// Allocate memory on GPU
	cudaMalloc((void**)&devSourceImageData, arraySize * sizeof(float));
	cudaMalloc((void**)&devConvKernelData, convKernelSize * convKernelSize * sizeof(float));
	cudaMalloc((void**)&devResultImageData, arraySize * sizeof(float));

	// Allocate memory on CPU (possible even by malloc or new)
	cudaHostAlloc((void**)&hostSourceImageData, arraySize * sizeof(float), cudaHostAllocDefault);
	cudaHostAlloc((void**)&hostConvKernelData, convKernelSize * convKernelSize * sizeof(float), cudaHostAllocDefault);
	cudaHostAlloc((void**)&hostResultImageData, arraySize * sizeof(float), cudaHostAllocDefault);

	// Initialize arrays on the host
	for (int i = 0; i < arraySize; i++)
	{
		hostSourceImageData[i] = imageData[i];
		hostResultImageData[i] = imageData[i];
	}
	for (int i = 0; i < convKernelSize * convKernelSize; i++)
		hostConvKernelData[i] = convKernelData[i];

	// Copy data CPU -> GPU
	cudaMemcpy(devSourceImageData, hostSourceImageData, arraySize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devConvKernelData, hostConvKernelData, convKernelSize * convKernelSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devResultImageData, hostResultImageData, arraySize * sizeof(float), cudaMemcpyHostToDevice);

	// Launch a kernel on the GPU
	applyConvolution_GPU<<<numOfBlocks, numOfThreadsInBlock>>>(devResultImageData, devSourceImageData, imgRows, imgCols, 
															   devConvKernelData, convKernelSize, 1 / 256.0);
	cudaDeviceSynchronize(); // wait for kernel end

	// Copy data GPU -> CPU
	cudaMemcpy(hostResultImageData, devResultImageData, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

	// free memory blocks on CPU
	cudaFreeHost(hostSourceImageData);
	cudaFreeHost(hostConvKernelData);
	cudaFreeHost(hostResultImageData);
	// free memory blocks on GPU
	cudaFree(devSourceImageData);
	cudaFree(devConvKernelData);
	cudaFree(devResultImageData);

	return 0;
}

