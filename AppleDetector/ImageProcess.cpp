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
	//-------------------------------------------------------------
	//			POINT PROCESSING

	//-------------------------------------------------------------

	// Convert to GPU
	cv::cuda::GpuMat src, dst;
	src.upload(imageToProcess);	

	// Convert to HSV
	cv::cuda::cvtColor(src, dst, cv::COLOR_BGR2HSV);

	// Split channels
	std::vector<cv::cuda::GpuMat> imhsvChannels(3);
	cv::cuda::split(dst, imhsvChannels);

	// Create CLAHE
	double clipLimit = 2.0;
	cv::Size tileGridSize = cv::Size(8, 8);
	cv::Ptr<cv::cuda::CLAHE> clahe = cv::cuda::createCLAHE(clipLimit, tileGridSize);

	// Apply CLAHE
	clahe->apply(imhsvChannels[2], imhsvChannels[2]);

	// Merge channels
	cv::cuda::GpuMat imhsvClahe;
	cv::cuda::merge(imhsvChannels, imhsvClahe);

	// Convert back to BGR
	cv::cuda::GpuMat imEqClahe;
	cv::cuda::cvtColor(imhsvClahe, imEqClahe, cv::COLOR_HSV2BGR);
	//-------------------------------------------------------------

	//					NOISE REDUCTION

	//-------------------------------------------------------------
	// Sharpen image using "unsharp mask" algorithm
	cv::cuda::GpuMat blurred; 
	double sigma = 1, threshold = 5, amount = 1;
		
	cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createGaussianFilter(imEqClahe.type(), blurred.type(), cv::Size(5, 5), sigma);
	filter->apply(imEqClahe, blurred);	
	
	cv::cuda::GpuMat sharpened;
	cv::cuda::addWeighted(imEqClahe, 2.0, blurred, -1.0, 0, sharpened );	
	
	
	cv::Mat imgCleaned;
	sharpened.download(imgCleaned);

	return imgCleaned;
}

ImageProcess::~ImageProcess()
{

}
