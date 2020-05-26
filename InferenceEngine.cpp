#include "InferenceEngine.h"
#include <opencv2/opencv.hpp>

#include <fstream>
#include <sstream>



InferenceEngine::InferenceEngine()
{

}


cv::dnn::Net InferenceEngine::loadNetwork(std::string modelConfiguration, std::string modelWeights)
{
	cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfiguration, modelWeights);

	/*		SET BACKEND AND TARGET OF DEVICE		*/	
	if (cv::cuda::getCudaEnabledDeviceCount())
	{
		net.setPreferableBackend(cv::dnn::Backend::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::Target::DNN_TARGET_CUDA);

		std::cout << "CUDA backend and target enabled for inference." << std::endl;
	}
	else
	{
		net.setPreferableBackend(cv::dnn::Backend::DNN_BACKEND_DEFAULT);
		net.setPreferableTarget(cv::dnn::Target::DNN_TARGET_OPENCL_FP16);

		std::cout << "OpenCL_FP16 backend and target enabled for inference." << std::endl;
	}
	/*----------------------------------*/
	return net;
}


std::vector<std::string> InferenceEngine::getClasses(std::string classesFile)
{
	std::ifstream ifs(classesFile.c_str());
	std::string line;

	while (getline(ifs, line))
		classes.push_back(line);

	return classes;
}


InferenceEngine::~InferenceEngine()
{

}