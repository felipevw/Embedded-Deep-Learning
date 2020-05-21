// Super Resolution

#include <jetson-inference/superResNet.h>

#include <jetson-utils/loadImage.h>
#include <jetson-utils/commandLine.h>
#include <jetson-utils/cudaMappedMemory.h>
#include <jetson-utils/cudaRGB.h>


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>


class Superres
{
public:
	Superres();

	superResNet* setupSuperres(cv::Mat rgb_image_test, int w, int h);
	void setSuperres(cv::Mat rgb_image);
	cv::Mat getSuperres(superResNet* net);
	
	~Superres();	

private:
	cv::Mat imageToProcess;
	//superResNet* net;

	int inputWidth;
	int inputHeight;

	uchar3* imgBufferRGB;
        float4* imgBufferRGBAf;

	float* outputCPU;
        float* outputCUDA;

	int outputWidth;
        int outputHeight;


};
