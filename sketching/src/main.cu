#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaarithm.hpp"
#include "timer.h"

void processImage(std::string inputFile, std::string outputFile);

using namespace std;
using namespace cv;

int main(int argc, char **argv) {

	for (int i=1; i<=4; ++i) {
		string fileId = std::to_string(i);
		const string inputFile = argc >= 2 ? argv[1] : "data/opensuse"+fileId+".jpg";
		const string outputFile = argc >= 3 ? argv[2] :
            "data/opensuse"+fileId+"_sketching_color.jpg";
		processImage(inputFile, outputFile);
	}

	return 0;
}

/**
 * This method computes the gradients of a grayscale image.
 * @param inputImage A grayscale image of type CV_8U
 * @param outputImage A grayscale image of type CV_8U
 */
void computeGradients(const cv::cuda::GpuMat& inputImage, cv::cuda::GpuMat& outputImage)
{
	if(inputImage.channels() == 1) { // grayscale

		// compute the gradients on both directions x and y
		cv::cuda::GpuMat grad_x, grad_y;
		cv::cuda::GpuMat abs_grad_x, abs_grad_y;
		int scale = 1;
		int ddepth = CV_16S; // use 16 bits unsigned to avoid overflow
		Ptr<cv::cuda::Filter> filter;

		// gradient x direction
		filter = cv::cuda::createScharrFilter(inputImage.type(), ddepth, 1, 0, scale, BORDER_DEFAULT);
		filter->apply(inputImage, grad_x);
		cv::cuda::abs(grad_x, grad_x);
		grad_x.convertTo(abs_grad_x, CV_8UC1); // CV_16S -> CV_8U

		// gradient y direction
		filter = cv::cuda::createScharrFilter(inputImage.type(), ddepth, 0, 1, scale, BORDER_DEFAULT);
		filter->apply(inputImage, grad_y);
		cv::cuda::abs(grad_y, grad_y);
		grad_y.convertTo(abs_grad_y, CV_8UC1); // CV_16S -> CV_8U

		// create the output by adding the absolute gradient images of each x and y direction
		cv::cuda::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, outputImage);

		// release GPU memory
		grad_x.release();
		grad_y.release();
		abs_grad_x.release();
		abs_grad_y.release();

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
void processImage(const cv::cuda::GpuMat& inputImage, cv::cuda::GpuMat& outputImage)
{
	if (inputImage.channels() == 3) { // color image

		// All transformations are done on the output image
		outputImage = inputImage;

		// Blur the input image to remove the noise
		Ptr<cv::cuda::Filter> filter = cv::cuda::createGaussianFilter(outputImage.type(), outputImage.type(), Size(9,9), 0);
		filter->apply(outputImage, outputImage);

		// Convert it to grayscale (CV_8UC3 -> CV_8UC1)
		cv::cuda::cvtColor(outputImage, outputImage, COLOR_RGB2GRAY);

		// Compute the gradient image
		computeGradients(outputImage, outputImage);

		// invert the gradient image
		cv::cuda::subtract(cv::Scalar::all(255), outputImage, outputImage);
        // Convert back to 3 channels for blending
		cv::cuda::cvtColor(outputImage, outputImage, COLOR_GRAY2BGR);

        cv::cuda:: addWeighted(inputImage , 0.3, outputImage, 0.7, 0.1, outputImage);

	} else if (inputImage.channels() == 1) { // grayscale image
		computeGradients(inputImage, outputImage);

		// invert the gradient image
		cv::cuda::subtract(cv::Scalar::all(255), outputImage, outputImage);

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
	printf("GPU::Processing image: %s ...\n", inputFile.c_str());

	// Read the file
	Mat inputImage = imread(inputFile, CV_LOAD_IMAGE_UNCHANGED);
	if (!inputImage.data) {
		printf("Could not open image file: %s\n", inputFile.c_str());
		return;
	}

	// Init the output image as grayscale with the same size as the input
	Mat outputImage (inputImage.size(), CV_8U);

	// copy the input image from CPU to GPU memory
	cuda::GpuMat gpuInputImage = cuda::GpuMat(inputImage);
	cuda::GpuMat gpuOutputImage = cuda::GpuMat(outputImage);

	// Process the image
	GpuTimer timer;
	timer.Start();
	processImage(gpuInputImage, gpuOutputImage);

	timer.Stop();

	printf("Method processImage() ran in: %f msecs, image size: %ux%u, msecs/pixel: %f .\n",
			timer.Elapsed(), inputImage.cols, inputImage.rows, timer.Elapsed()/(inputImage.rows*inputImage.cols));

	// copy the result gradient from GPU to CPU and release GPU memory
	gpuOutputImage.download(outputImage);
  	gpuInputImage.release();
	gpuOutputImage.release();

	imwrite(outputFile, outputImage);
}
