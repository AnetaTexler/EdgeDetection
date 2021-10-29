#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <iostream>
#include <iomanip>

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
	float max = 0;
	for (int row = 0; row < src.rows; row++)
	{
		for (int col = 0; col < src.cols; col++)
		{
			if (src.at<float>(row, col) > max)
				max = src.at<float>(row, col);
		}
	}

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
	Mat sourceImage;
	sourceImage = imread("sample_images\\valve.jpg", CV_LOAD_IMAGE_GRAYSCALE); // Read the file
	//sourceImage = imread("sample_images\\tonda.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//sourceImage = imread("sample_images\\pumpkin.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//sourceImage = imread("sample_images\\hungary.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//sourceImage = imread("sample_images\\lexa.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	
	if (!sourceImage.data) // Check for invalid input
	{
		cout << "Could not open or find the image" << endl;
		return -1;
	}

	Mat sourceImageF;
	sourceImage.convertTo(sourceImageF, CV_32FC1, 1 / 255.0); // convert image from uchar to float from 0.0 to 1.0 (due to negative values in Sobel operator)

	imshowInWindow("Original image", sourceImageF);

	// Gaussian blur convolution kernel 3x3 
	/*
	float dataGauss[9] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 }; // *1/16
	Mat gaussianKernel = Mat(3, 3, CV_32FC1, dataGauss);
	float kernelCoefficient = (float)1 / 16;
	*/
	
	// Gaussian blur convolution kernel 5x5
	float dataGauss[25] = { 1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36, 24, 6, 4, 16, 24, 16, 4, 1, 4, 6, 4, 1 }; // *1/256
	Mat gaussianKernel = Mat(5, 5, CV_32FC1, dataGauss);
	float kernelCoefficient = 1 / 256.0;
	
	Mat blurredImage = applyConvolution(sourceImageF, gaussianKernel, kernelCoefficient);
	//imshowInWindow("Blurred", blurredImage);
	
	/*
	Mat Gx = blurredImage.clone();

	for (int row = 0; row < blurredImage.rows; row++)
	{
		for (int col = 1; col < blurredImage.cols; col++)
		{
			Gx.at<uchar>(row, col) = abs(sourceImage.at<uchar>(row, col - 1) - sourceImage.at<uchar>(row, col));
		}
	}
	imshowInWindow("Exact derivation x", Gx);

	Mat Gy = blurredImage.clone();

	for (int row = 1; row < blurredImage.rows; row++)
	{
		for (int col = 0; col < blurredImage.cols; col++)
		{
			Gy.at<uchar>(row, col) = abs(sourceImage.at<uchar>(row - 1, col) - sourceImage.at<uchar>(row, col));
		}
	}
	imshowInWindow("Exact derivation y", Gy);
	*/

	// Convolution kernel to get the image gradient in x direction (Sobel operator)
	float dataGx[9] = { 1, 0, -1, 2, 0, -2, 1, 0, -1 };
	Mat xGradientKernel = Mat(3, 3, CV_32FC1, dataGx);

	Mat xGradientImage = applyConvolution(blurredImage, xGradientKernel, 1);
	imshowInWindow("Gradient X", xGradientImage);

	// Convolution kernel to get the image gradient in y direction (Sobel operator)
	float dataGy[9] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };
	Mat yGradientKernel = Mat(3, 3, CV_32FC1, dataGy);

	Mat yGradientImage = applyConvolution(blurredImage, yGradientKernel, 1);
	imshowInWindow("Gradient Y", yGradientImage);
	
	// Get the gradient magnitude: G = sqrt(Gx^2 + Gy^2)
	Mat gradientMagnitudeImage = getGradientMagnitude(xGradientImage, yGradientImage);
	imshowInWindow("Gradient Magnitude", gradientMagnitudeImage);

	// Get the gradient direction: Theta = atan(Gy/Gx)
	//Mat gradientDirectionImage = getGradientDirection(xGradientImage, yGradientImage);
	//imshowInWindow("Gradient Direction", gradientDirectionImage);
	
	//Mat toWriteImage;
	//gradientMagnitudeImage.convertTo(toWriteImage, CV_8UC1, 255.0);
	//imwrite("sample_images\\hungary_grad_mag_result.jpg", toWriteImage);

	Mat finalResultImage = applyDoubleThresholdAndWeakEdgesSuppression(gradientMagnitudeImage);
	imshowInWindow("Final Result", finalResultImage);

	//finalResultImage.convertTo(toWriteImage, CV_8UC1, 255.0);
	//imwrite("sample_images\\hungary_fin_result.jpg", toWriteImage);

	waitKey(0);
	
	return 0;
}

