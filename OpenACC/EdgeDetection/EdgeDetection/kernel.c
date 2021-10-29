#define _CRT_SECURE_NO_WARNINGS
#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void load(const char* path, float** imageData, int* imgRows, int* imgCols, float** convolutionKernelData, int* kernelSize, float* kernelCoefficient)
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

	fscanf(file, "%d %f ", kernelSize, kernelCoefficient);
	*convolutionKernelData = (float*)malloc(*kernelSize * *kernelSize * sizeof(float));
	for (int i = 0; i < *kernelSize * *kernelSize; i++)
		fscanf(file, "%f ", &(*convolutionKernelData)[i]);

	fclose(file);
}

void save(const char* path, float* imageData, int imgRows, int imgCols, float* convolutionKernelData, int kernelSize, float kernelCoefficient)
{
	FILE* file;
	file = fopen(path, "w");

	if (file == NULL)
	{
		printf("Cannot open file.\n");
		return;
	}

	fprintf(file, "%d %d ", imgRows, imgCols);
	for (int i = 0; i < imgRows * imgCols; i++)
		fprintf(file, "%f ", imageData[i]);
	fprintf(file, "\n");

	fprintf(file, "%d %f ", kernelSize, kernelCoefficient);
	for (int i = 0; i < kernelSize * kernelSize; i++)
		fprintf(file, "%f ", convolutionKernelData[i]);
	fprintf(file, "\n");

	fclose(file);
}

float* applyConvolution_C(const float* sourceImageData, const int imageRowsSize, const int imageColsSize,
						  const float* convolutionKernelData, const int kernelSize, const float kernelCoefficient)
{
	float* resultImageData = (float*)malloc(imageRowsSize * imageColsSize * sizeof(float));

	for (int i = 0; i < imageRowsSize * imageColsSize; i++)
		resultImageData[i] = sourceImageData[i];
	
	#pragma acc data copyin(sourceImageData[0:imageRowsSize*imageColsSize]) copyin(convolutionKernelData[0:kernelSize*kernelSize]) copyout(resultImageData[0:imageRowsSize*imageColsSize])
	#pragma acc parallel num_gangs(1024) num_workers(1024)
	#pragma acc loop gang
	for (int row = kernelSize / 2; row < imageRowsSize - kernelSize / 2; row++) 
	{
		#pragma acc loop worker
		for (int col = kernelSize / 2; col < imageColsSize - kernelSize / 2; col++)
		{
			float roiSum = 0;
			for (int roiRow = 0; roiRow < kernelSize; roiRow++)
			{
				for (int roiCol = 0; roiCol < kernelSize; roiCol++)
				{
					int imageRow = row - (kernelSize / 2) + roiRow;
					int imageCol = col - (kernelSize / 2) + roiCol;
					roiSum += sourceImageData[imageRow*imageColsSize + imageCol] * convolutionKernelData[roiRow * kernelSize + roiCol];
				}
			}
			resultImageData[row * imageColsSize + col] = roiSum * kernelCoefficient;
		}
	}

	return resultImageData;
}

int main()
{
	float* imageData = NULL;
	float* convolutionKernelData = NULL;
	int imgRows, imgCols, kernelSize;
	float kernelCoefficient;

	load("srcImgData.txt", &imageData, &imgRows, &imgCols, &convolutionKernelData, &kernelSize, &kernelCoefficient);
	float* resultImageData = applyConvolution_C(imageData, imgRows, imgCols, convolutionKernelData, kernelSize, kernelCoefficient);
	save("outImgData.txt", resultImageData, imgRows, imgCols, convolutionKernelData, kernelSize, kernelCoefficient);

	return 0;
}