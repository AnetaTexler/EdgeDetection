#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
//#include "opencv2\imgproc\imgproc.hpp" // for Sobel function
#include <iostream>
#include <iomanip>
#include <chrono>
#include "kernel.h"

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

Mat applyConvolution(const Mat& sourceImage, const Mat& convolutionKernel, const float kernelCoefficient) // useful for blur image and for getting the gradient in x and y direction
{
	Mat resultImage = sourceImage.clone(); // deep copy
	float roiSum = 0;

	for (int row = (convolutionKernel.rows / 2); row < sourceImage.rows - (convolutionKernel.rows / 2); row++)
	{
		// (convolutionKernel.rows / 2) due to different size of convolution kernels
		for (int col = (convolutionKernel.cols / 2); col < sourceImage.cols - (convolutionKernel.cols / 2); col++)
		{
			// region of interest (ROI)
			for (int roiRow = 0; roiRow < convolutionKernel.rows; roiRow++)
			{
				for (int roiCol = 0; roiCol < convolutionKernel.cols; roiCol++)
				{
					int srcImgRow = row - (convolutionKernel.rows / 2) + roiRow;
					int srcImgCol = col - (convolutionKernel.cols / 2) + roiCol;

					roiSum += sourceImage.at<float>(srcImgRow, srcImgCol) * convolutionKernel.at<float>(roiRow, roiCol);
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
		}
	}

	return resultImage;
}

Mat applyDoubleThresholdAndWeakEdgesSuppression(const Mat& src)
{
	float max = 0;
	// get maximal value pixel
	for (int row = 0; row < src.rows; row++)
	{
		for (int col = 0; col < src.cols; col++)
		{
			if (src.at<float>(row, col) > max)
				max = src.at<float>(row, col);
		}
	}

	// thresholds are adjusted empirically
	float highThreshold = 2 * (max / 11);
	float lowThreshold = max / 6;

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


int main()
{
	string sourcePath = "sample_images\\valve.jpg";
	//string sourcePath = "sample_images\\lion4k.jpg";
	//string sourcePath = "sample_images\\peacockHigh.jpg";
	//string sourcePath = "sample_images\\kingfisherHigh.jpg";
	//string sourcePath = "sample_images\\kingfisher.jpg";
	//string sourcePath = "sample_images\\kingfishers.jpg";

	Mat sourceImage;
	sourceImage = imread(sourcePath, CV_LOAD_IMAGE_GRAYSCALE); // Read the file

	imwrite(sourcePath.substr(0, sourcePath.find(".jpg")) +"_grayscale.jpg", sourceImage);
	
	if (!sourceImage.data) // Check for invalid input
	{
		cout << "Could not open or find the image" << endl;
		return -1;
	}

	Mat sourceImageF;
	sourceImage.convertTo(sourceImageF, CV_32FC1, 1 / 255.0); // convert image from uchar to float from 0.0 to 1.0 (due to negative values in Sobel operator)

	//imshowInWindow("Original image", sourceImageF);
	
	// Gaussian blur convolution kernel 5x5
	float dataGauss[25] = { 1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36, 24, 6, 4, 16, 24, 16, 4, 1, 4, 6, 4, 1 }; // *1/256
	Mat gaussianKernel = Mat(5, 5, CV_32FC1, dataGauss);
	float kernelCoefficient = 1 / 256.0;
	
	//Mat blurredImage = applyConvolution(sourceImageF, gaussianKernel, kernelCoefficient);
	//imshowInWindow("Blurred", blurredImage);
	
	// ***************** CALL KERNEL FOR GPU *********************
	//float* blurredImageData = applyConvolution_C((float*)sourceImageF.data, sourceImage.rows, sourceImage.cols, dataGauss, gaussianKernel.rows, kernelCoefficient);
	//Mat blurredImage(sourceImage.rows, sourceImage.cols, CV_32FC1, blurredImageData);
	
	float* blurredImageData = NULL;
	save("srcImgData.txt", (float*)sourceImageF.data, sourceImage.rows, sourceImage.cols, dataGauss, gaussianKernel.rows, kernelCoefficient);

	auto start = chrono::system_clock::now();
	//system("kernel.exe");
	debug_main();
	auto end = chrono::system_clock::now();
	long long millis = chrono::duration_cast<chrono::milliseconds>(end - start).count();
	printf("1st convolution: %.2f s\n", millis / 1000.0);
	
	int rows, cols, size;
	float coeff;
	float* kernelData = NULL;
	load("outImgData.txt", &blurredImageData, &rows, &cols, &kernelData, &size, &coeff);
	Mat blurredImage(rows, cols, CV_32FC1, blurredImageData);
	// ***********************************************************

	// WRITE
	Mat toWriteImage;
	blurredImage.convertTo(toWriteImage, CV_8UC1, 255.0);
	imwrite(sourcePath.substr(0, sourcePath.find(".jpg")) + "_gaussian_blur.jpg", toWriteImage);
	
	// Convolution kernel to get the image gradient in x direction (Sobel operator)
	float dataGx[9] = { 1, 0, -1, 2, 0, -2, 1, 0, -1 };
	Mat xGradientKernel = Mat(3, 3, CV_32FC1, dataGx);

	//Mat xGradientImage = applyConvolution(blurredImage, xGradientKernel, 1);
	//imshowInWindow("Gradient X", xGradientImage);

	// ***************** CALL KERNEL FOR GPU *********************
	//float* xGradientImageData = applyConvolution_C((float*)blurredImage.data, blurredImage.rows, blurredImage.cols, dataGx, xGradientKernel.rows, 1);
	//Mat xGradientImage(blurredImage.rows, blurredImage.cols, CV_32FC1, xGradientImageData);

	float* xGradientImageData = NULL;
	save("srcImgData.txt", (float*)blurredImage.data, blurredImage.rows, blurredImage.cols, dataGx, xGradientKernel.rows, 1);
	
	start = chrono::system_clock::now();
	//system("kernel.exe");
	debug_main();
	end = chrono::system_clock::now();
	millis = chrono::duration_cast<chrono::milliseconds>(end - start).count();
	printf("2nd convolution: %.2f s\n", millis / 1000.0);
	
	load("outImgData.txt", &xGradientImageData, &rows, &cols, &kernelData, &size, &coeff);
	Mat xGradientImage(rows, cols, CV_32FC1, xGradientImageData);
	// ***********************************************************
	
	//Sobel(blurredImage, xGradientImage, CV_32FC1, 1, 0, 3);

	// WRITE
	xGradientImage.convertTo(toWriteImage, CV_8UC1, 255.0);
	imwrite(sourcePath.substr(0, sourcePath.find(".jpg")) + "_gradient_X.jpg", toWriteImage);

	// Convolution kernel to get the image gradient in y direction (Sobel operator)
	float dataGy[9] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };
	Mat yGradientKernel = Mat(3, 3, CV_32FC1, dataGy);

	//Mat yGradientImage = applyConvolution(blurredImage, yGradientKernel, 1);
	//imshowInWindow("Gradient Y", yGradientImage);
	
	// ***************** CALL KERNEL FOR GPU *********************
	//float* yGradientImageData = applyConvolution_C((float*)blurredImage.data, blurredImage.rows, blurredImage.cols, dataGy, yGradientKernel.rows, 1);
	//Mat yGradientImage(blurredImage.rows, blurredImage.cols, CV_32FC1, yGradientImageData);

	float* yGradientImageData = NULL;
	save("srcImgData.txt", (float*)blurredImage.data, blurredImage.rows, blurredImage.cols, dataGy, yGradientKernel.rows, 1);
	
	start = chrono::system_clock::now();
	//system("kernel.exe");
	debug_main();
	end = chrono::system_clock::now();
	millis = chrono::duration_cast<chrono::milliseconds>(end - start).count();
	printf("3th convolution: %.2f s\n", millis / 1000.0);

	load("outImgData.txt", &yGradientImageData, &rows, &cols, &kernelData, &size, &coeff);
	Mat yGradientImage(rows, cols, CV_32FC1, yGradientImageData);
	// ***********************************************************

	//Sobel(blurredImage, yGradientImage, CV_32FC1, 0, 1, 3);

	// WRITE
	yGradientImage.convertTo(toWriteImage, CV_8UC1, 255.0);
	imwrite(sourcePath.substr(0, sourcePath.find(".jpg")) + "_gradient_Y.jpg", toWriteImage);
	
	// Get the gradient magnitude: G = sqrt(Gx^2 + Gy^2)
	Mat gradientMagnitudeImage = getGradientMagnitude(xGradientImage, yGradientImage);
	//imshowInWindow("Gradient Magnitude", gradientMagnitudeImage);

	// Get the gradient direction: Theta = atan(Gy/Gx)
	//Mat gradientDirectionImage = getGradientDirection(xGradientImage, yGradientImage);
	//imshowInWindow("Gradient Direction", gradientDirectionImage);
	
	// WRITE
	gradientMagnitudeImage.convertTo(toWriteImage, CV_8UC1, 255.0);
	imwrite(sourcePath.substr(0, sourcePath.find(".jpg")) + "_gradient_magnitude.jpg", toWriteImage);

	Mat finalResultImage = applyDoubleThresholdAndWeakEdgesSuppression(gradientMagnitudeImage);
	//imshowInWindow("Final Result", finalResultImage);

	// WRITE
	finalResultImage.convertTo(toWriteImage, CV_8UC1, 255.0);
	imwrite(sourcePath.substr(0, sourcePath.find(".jpg")) + "_final_result.jpg", toWriteImage);

	//waitKey(0);
	system("pause");
	
	return 0;
}

