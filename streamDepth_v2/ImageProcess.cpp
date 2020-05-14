#include "ImageProcess.h"


ImageProcess::ImageProcess()
{

}

void ImageProcess::setPreprocess(cv::Mat rgb_image)
{
	imageToProcess = rgb_image;
}

cv::Mat ImageProcess::getPreprocess()
{
	//				POINT PROCESSING

	cv::Mat imEqClahe;

	if (cv::cuda::getCudaEnabledDeviceCount())
	{
		std::cout << "Enable CUDA image processing" << std::endl;
		

	}
	else
	{

		// Convert to HSV
		cv::Mat imhsv;
		cv::cvtColor(imageToProcess, imhsv, cv::COLOR_BGR2HSV);

		// Split channels
		std::vector<cv::Mat> imhsvChannels(3);
		cv::split(imhsv, imhsvChannels);

		// Create CLAHE
		double clipLimit = 2.0;
		cv::Size tileGridSize = cv::Size(8, 8);
		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, tileGridSize);

		// Apply CLAHE
		clahe->apply(imhsvChannels[2], imhsvChannels[2]);

		// Merge channels
		cv::Mat imhsvClahe;
		cv::merge(imhsvChannels, imhsvClahe);

		// Convert back to BGR
		cv::cvtColor(imhsvClahe, imEqClahe, cv::COLOR_HSV2BGR);
	}
	//-------------------------------------------------------------

	//					NOISE REDUCTION

	//-------------------------------------------------------------
	// Sharpen image using "unsharp mask" algorithm
	cv::Mat blurred; 
	double sigma = 1, threshold = 5, amount = 1;
	cv::GaussianBlur(imEqClahe, blurred, cv::Size(), sigma, sigma);
	
	cv::Mat lowContrastMask = abs(imEqClahe - blurred) < threshold;
	
	cv::Mat sharpened = imEqClahe * (1 + amount) + blurred * (-amount);
	imEqClahe.copyTo(sharpened, lowContrastMask);

	return sharpened;
}

ImageProcess::~ImageProcess()
{

}