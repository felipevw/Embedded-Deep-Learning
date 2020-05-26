#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>

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

