#include "Superres.h"

Superres::Superres()
{

}

superResNet* Superres::setupSuperres(cv::Mat rgb_image_test, int w, int h)
{
	/*
         * load super resolution network
         */
        superResNet* net = superResNet::Create();

        if( !net )
        {
                printf("superres-console:  failed to load superResNet\n");
        }
	

	// Initial resolution
        inputWidth = w;
        inputHeight = h;


	// Allocate memory input
        imgBufferRGB = NULL;
        imgBufferRGBAf = NULL;


        cudaMalloc((void**) &imgBufferRGB, rgb_image_test.cols * rgb_image_test.rows * sizeof(uchar3));
        cudaMalloc((void**) &imgBufferRGBAf, rgb_image_test.cols * rgb_image_test.rows * sizeof(float4));


        outputCPU = NULL;
        outputCUDA = NULL;


        // Allocate memory output
        outputWidth = inputWidth * net->GetScaleFactor();
        outputHeight = inputHeight * net->GetScaleFactor();


        if( !cudaAllocMapped((void**)&outputCPU, (void**)&outputCUDA, outputWidth * outputHeight * sizeof(float4)) )
        {
                printf("superres-console:  failed to allocate memory for %ix%i output image\n", outputWidth, outputHeight);
        }

        printf("superres-console:  input image size - %ix%i\n", inputWidth, inputHeight);
        printf("superres-console:  output image size - %ix%i\n", outputWidth, outputHeight);

	return net;
}

void Superres::setSuperres(cv::Mat rgb_image)
{
	imageToProcess = rgb_image;
}

cv::Mat Superres::getSuperres(superResNet* net)
{
	// Convert from BGR to RGB
	cv::Mat inputImg;
        cv::cvtColor(imageToProcess, inputImg, cv::COLOR_BGR2RGB);


        // Copy 
        cudaMemcpy2D((void*) imgBufferRGB, inputImg.cols * sizeof(uchar3), (void*) inputImg.data, inputImg.step, inputImg.cols * sizeof(uchar3), inputImg.rows, cudaMemcpyHostToDevice);


        // Convert to RGBA float 32
        cudaRGB8ToRGBA32(imgBufferRGB, imgBufferRGBAf, inputImg.cols, inputImg.rows);


        /*
         * upscale image with network
         */

        if( !net->UpscaleRGBA((float*)imgBufferRGBAf, inputWidth, inputHeight,
                                        outputCUDA, outputWidth, outputHeight) )
        {
                printf("superres-console:  failed to process super resolution network\n");
        }


        /*
         * Return output image
         */
        cv::cuda::GpuMat infer = cv::cuda::GpuMat(outputHeight, outputWidth, CV_32FC4, outputCUDA);

        cv::Rect region(outputWidth / 2 - 320, outputHeight / 2 - 240, 640, 480);
        cv::cuda::GpuMat roi(infer, region);

        cv::Mat result;
        roi.download(result);


        // Convert the result
        result /= 255;
        cv::cvtColor(result, result, cv::COLOR_RGB2BGR);

	return result;
}


Superres::~Superres()
{
	
}
