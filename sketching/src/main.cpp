#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>


void processImage(std::string inputFile, std::string outputFile);

using namespace std;
using namespace cv;

int main(int argc, char **argv) {

	for (int i=1; i<=6; ++i) {
		string fileId = std::to_string(i);
		const string inputFile = argc >= 2 ? argv[1] : "data/opensuse"+fileId+".jpg";
		const string outputFile = argc >= 3 ? argv[2] : "data/opensuse"+fileId+"_sketching_color_cpu.jpg";
		processImage(inputFile, outputFile);
	}

	return 0;
}

/**
 * This method computes the gradients of a grayscale image.
 * @param inputImage A grayscale image of type CV_8U
 * @param outputImage A grayscale image of type CV_8U
 */
void computeGradients(const cv::Mat& inputImage, cv::Mat& outputImage)
{
	if(inputImage.channels() == 1) { // grayscale

		// compute the gradients on both directions x and y
		cv::Mat grad_x, grad_y;
		cv::Mat abs_grad_x, abs_grad_y;
		int scale = 1;
		int ddepth = CV_16S; // use 16 bits unsigned to avoid overflow

		// gradient x direction
		//filter = cv::createSobelFilter(inputImage.type(), ddepth, 1, 0, 3, scale, BORDER_DEFAULT);
		cv::Scharr(inputImage, grad_x, ddepth, 1, 0, scale, BORDER_DEFAULT);
		cv::abs(grad_x);
		grad_x.convertTo(abs_grad_x, CV_8UC1); // CV_16S -> CV_8U

		// gradient y direction
		//filter = cv::createSobelFilter(inputImage.type(), ddepth, 0, 1, 3, scale, BORDER_DEFAULT);
		cv::Scharr(inputImage, grad_y, ddepth, 0, 1, scale, BORDER_DEFAULT);
		cv::abs(grad_y);
		grad_y.convertTo(abs_grad_y, CV_8UC1); // CV_16S -> CV_8U

		// create the output by adding the absolute gradient images of each x and y direction
		cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, outputImage);

	} else {
		printf("Image type not supported.\n");
		outputImage = inputImage;
	}
}

/**
 * This method extracts the inverted gradient image of a color or grayscale image.
 * @param inputImage A matrix of a color or grayscale image
 * @param outputImage The inverted gradient matrix
 */
void processImage(const cv::Mat& inputImage, cv::Mat& outputImage)
{
	if (inputImage.channels() == 3) { // color image

		// All transformations are done on the output image
		outputImage = inputImage;

		// Blur the input image to remove the noise
		cv::GaussianBlur(outputImage, outputImage, Size(9,9), 0);

		// Convert it to grayscale (CV_8UC3 -> CV_8UC1)
		cv::cvtColor(outputImage, outputImage, COLOR_RGB2GRAY);

		// Compute the gradient image
		computeGradients(outputImage, outputImage);
		//normalize(outputImage, outputImage, 0, 255, NORM_MINMAX, CV_8U);
		//cv::threshold(outputImage, outputImage, 50, 255, THRESH_TOZERO);

		// invert the gradient image
		cv::subtract(cv::Scalar::all(255), outputImage, outputImage);

		cv::cvtColor(outputImage, outputImage, COLOR_GRAY2BGR);

        cv::addWeighted(inputImage , 0.3, outputImage, 0.7, 0.1, outputImage);

	} else if (inputImage.channels() == 1) { // grayscale image
		computeGradients(inputImage, outputImage);
		// invert the gradient image
		cv::subtract(cv::Scalar::all(255), outputImage, outputImage);

	} else { // not supported image format
		printf("Image type not supported.\n");
		outputImage = inputImage;
	}
}

/**
 * This method extracts the gradient image of a color or grayscale image.
 * The output image is saved as grayscale.
 */
void processImage(std::string inputFile, std::string outputFile) {
	printf("CPU::Processing image: %s ...\n", inputFile.c_str());

	// Read the file
	Mat inputImage = cv::imread(inputFile, CV_LOAD_IMAGE_UNCHANGED);
	if (!inputImage.data) {
		printf("Could not open image file: %s\n", inputFile.c_str());
		return;
	}

	// Init the output image as grayscale with the same size as the input
	Mat outputImage (inputImage.size(), CV_8U);

	// Process the image
	auto started = std::chrono::high_resolution_clock::now();
	processImage(inputImage, outputImage);
	auto done = std::chrono::high_resolution_clock::now();
	printf("Method processImage() ran in: %f msecs, image size: %ux%u, msecs/pixel: %f .\n",
	 		 (double) std::chrono::duration_cast<std::chrono::microseconds>(done-started).count()/1000,
			 inputImage.cols, inputImage.rows,
			 ((double) std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count())/(inputImage.rows*inputImage.cols));

	// copy the result gradient from GPU to CPU and release GPU memory
	cv::imwrite(outputFile, outputImage);
}
