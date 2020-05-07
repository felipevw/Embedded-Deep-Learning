#pragma once

#include <opencv2/opencv.hpp>

class ImageProcess
{
public:
	ImageProcess();

	void setPreprocess(cv::Mat rgb_image);
	cv::Mat getPreprocess();

	~ImageProcess();

private:

	cv::Mat imageToProcess;

};

