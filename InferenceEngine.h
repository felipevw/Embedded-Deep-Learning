#pragma once

#include <opencv2/opencv.hpp>

class InferenceEngine
{
public:
	InferenceEngine();

	cv::dnn::Net loadNetwork(std::string modelConfiguration, std::string modelWeights);
	std::vector<std::string> getClasses(std::string classesFile);
	
	~InferenceEngine();

private:
	cv::dnn::Net net;

	std::string classesFile;
	std::vector<std::string> classes;
	
	cv::Mat blob;
	std::vector<cv::Mat> outs; 

};

