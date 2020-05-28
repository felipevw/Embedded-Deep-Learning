// streamDepth_v2.cpp 

#include <iostream>
#include <vector>
#include <thread>
#include <string>

// include the librealsense C++ header file
#include <librealsense2/rs.hpp>

// include OpenCV header file
#include <opencv2/opencv.hpp>
#include <opencv2/xphoto.hpp>


// Camera parameters
const size_t FRAME_HEIGHT{ 480 };
const size_t FRAME_WIDTH{ 640 };
const size_t FRAME_RATE{ 60 };
const float WHRatio = static_cast<float>(FRAME_WIDTH) / static_cast<float>(FRAME_HEIGHT);
cv::Mat color_mat(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3);
cv::Mat depth_mat(FRAME_HEIGHT, FRAME_WIDTH, CV_16UC1);

std::string classesFile{ "model/coco.names" };
std::string modelConfiguration{ "model/yolov3.cfg" };
std::string modelWeights{ "model/yolov3.weights" };


// Model inference parameters
float objectnessThreshold = 0.4; // Objectness threshold
float confThreshold = 0.4; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;  // Width of network's input image Add.: 320, 416, 512 or 608
int inpHeight = 416; // Height of network's input image
std::vector<std::string> classes;


// Depth parameters
float scale{};                  // Depth sensor scale
std::vector<cv::Mat> outs;      // Layer outputs


bool enableImageProcess{ false }, enableInference{ false }, enableSuperres{ false };



#include "VideoStream.h"
#include "cv-helpers.hpp"
#include "ImageProcess.h"
#include "videostab.h"
#include "InferenceEngine.h"
#include "PredictionVisual.h"
#include "Superres.h"

std::vector<std::string> getOutputsNames(const cv::dnn::Net& net)
{
    static std::vector<std::string> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        std::vector<int> outLayers = net.getUnconnectedOutLayers();

        //get the names of all the layers in the network
        std::vector<std::string> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());

        for (size_t i = 0; i < outLayers.size(); ++i)
        {
            names[i] = layersNames[outLayers[i] - 1];
        }

    }
    return names;
}



VideoStream videostream;
ImageProcess imageprocess;
Superres superimage;

int main() try
{

    rs2::align align_to = videostream.setStreamRGBD(RS2_STREAM_COLOR, RS2_STREAM_DEPTH, 
                                                FRAME_WIDTH, 
                                                FRAME_HEIGHT, 
                                                RS2_FORMAT_BGR8, RS2_FORMAT_Z16, 
                                                FRAME_RATE);

    rs2::pipeline pipe = videostream.getPipeline();
    rs2::depth_sensor ds = videostream.getDepthSensor();
    

    const auto CAPACITY = 5; // allow max latency of 5 frames
    rs2::frame_queue queue_rgb(CAPACITY);
    rs2::frame_queue queue_depth(CAPACITY);

    // Warm Up camera
    videostream.cameraWarmUp(pipe);

    // Main frame name
    const auto window_rgb{ "RGB Stream 640 x 480" };
    const auto window_depth{ "Depth Stream 640 x 480" };
    VideoStream::getWindowRGBD(window_rgb, window_depth);


    //------------------------------------------------------
    //                   NETWORK SETUP

    InferenceEngine visualEngine;
    cv::dnn::Net net = visualEngine.loadNetwork(modelConfiguration, modelWeights);
    classes = visualEngine.getClasses(classesFile);

    superResNet* net_superres = superimage.setupSuperres(color_mat, FRAME_WIDTH, FRAME_HEIGHT);
    //------------------------------------------------------

    
    int key{};
    //------------------------------------------------------
    //                    INITIAL SCREEN

    cv::Mat apple_init = cv::imread("apple_init.png");
    cv::xphoto::oilPainting(apple_init, apple_init, 10, 1, cv::ColorConversionCodes::COLOR_BGR2Lab);

    while (true)
    {
        key = cv::waitKey(10);

        videostream.getNextFrame(align_to, pipe, color_mat);
        
        // Blend
        double alpha = 0.5; double beta; double input;
        beta = (1.0 - alpha);
        cv::Mat blended_img;
        cv::addWeighted(color_mat, alpha, apple_init, beta, 0.0, blended_img);

        std::string initmsg{ "Apple Detector" };
        std::string authormsg{ "Felipe VW" };
        std::string keymsg1{ "Press ESC to continue and exit" };
        std::string keymsg2{ "Press E to enhance frames" };
        std::string keymsg3{ "Press I to perform inference" };
        std::string keymsg4{ "Press S to enable superresolution" };

        int xPos{ 210 }, yPos{ 100 }, offset1{ -60 }, offset2{ 50 };
        cv::putText(blended_img, initmsg, cv::Point(xPos, yPos), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        cv::putText(blended_img, authormsg, cv::Point(xPos + 40, yPos + 40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        cv::putText(blended_img, keymsg1, cv::Point(xPos + offset1, yPos + 40 + offset2), cv::FONT_HERSHEY_SIMPLEX, 0.7f, cv::Scalar(152, 255, 152), 2);
        cv::putText(blended_img, keymsg2, cv::Point(xPos + offset1, yPos + 40 + offset2 * 2), cv::FONT_HERSHEY_SIMPLEX, 0.7f, cv::Scalar(152, 255, 152), 2);
        cv::putText(blended_img, keymsg3, cv::Point(xPos + offset1, yPos + 40 + offset2 * 3), cv::FONT_HERSHEY_SIMPLEX, 0.7f, cv::Scalar(152, 255, 152), 2);
        cv::putText(blended_img, keymsg4, cv::Point(xPos + offset1, yPos + 40 + offset2 * 4), cv::FONT_HERSHEY_SIMPLEX, 0.7f, cv::Scalar(152, 255, 152), 2);


        cv::imshow(window_rgb, blended_img);

        // Press ESC to exit loop
        if (key == 27)
        {
            break;
        }
    }


    //------------------------------------------------------

    //------------------------------------------------------
    //                   CAMERA THREAD
    // Enqueue frames
    std::thread t([&]() {
        while (true)
        {
            auto frames = pipe.wait_for_frames();

            // Make sure the frames are spatially aligned
            frames = align_to.process(frames);

            rs2::frame depth_frame = frames.get_depth_frame();
            rs2::frame filtered = videostream.FilterDepthFrame(depth_frame);
            queue_depth.enqueue(filtered);


            //queue_depth.enqueue(frames.get_depth_frame());
            queue_rgb.enqueue(frames.get_color_frame());

            // Press ESC to exit loop
            if (key == 27)
            {
                break;
            }
        }
        });
        
    //------------------------------------------------------


   //------------------------------------------------------
   //                    MAIN LOOP

    rs2::frame depth_frame;
    rs2::frame rgb_frame;


    // Videostab
    cv::Mat frame_2, frame2;
    cv::Mat frame_1, frame1;

    // Convert to opencv Mat format
    frame_1 = color_mat;
    cv::cvtColor(frame_1, frame1, cv::ColorConversionCodes::COLOR_BGR2GRAY);
    cv::Mat smoothedMat(2, 3, CV_64F);

    //Create a object of stabilization class
    VideoStab stab;



    while (true)
    {
        videostream.timeStart();
        key = cv::waitKey(10);

        if (queue_rgb.poll_for_frame(&rgb_frame))
        {
            // Get the frame
            rgb_frame.get_data();

            // Convert to opencv Mat format
            color_mat = frame_to_mat(rgb_frame);
            
        }
        
        // Depth frames dequeue
        if (queue_depth.poll_for_frame(&depth_frame) && enableSuperres == false)
        {
            // Get the frame and scale data
            depth_frame.get_data();
            scale = ds.get_depth_scale();

            // Create OpenCV matrix of size (w,h) from the colorized depth data
            cv::Mat depth_frame_mat(cv::Size(FRAME_WIDTH, FRAME_HEIGHT), CV_16UC1, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);
            depth_frame_mat.copyTo(depth_mat);

            // Convert 16bit image to 8bit image
            depth_frame_mat.convertTo(depth_frame_mat, CV_8UC1, 15 / 256.0);
            //cv::imshow(window_depth, depth_frame_mat);
        }
        

        //------------------------------------------------------
        //                 IMAGE ENHANCEMENT
        // Press 'e' for image enhancement and 'E' for quit
        if (key == 101)
        {
            enableImageProcess = true;
        }
        else if (key == 69)
        {
            enableImageProcess = false;

        }
        if (enableImageProcess == true)
        {
            // Image preprocessing
            imageprocess.setPreprocess(color_mat);
            cv::Mat color_mat_processed = imageprocess.getPreprocess();
            color_mat_processed.copyTo(color_mat);
        }
        //------------------------------------------------------


        //------------------------------------------------------
        //                 IMAGE STABILIZATION

        cv::Mat smoothedFrame;
        frame_2 = color_mat;
        smoothedFrame = stab.stabilize(frame_1, frame_2);

        cv::cvtColor(frame_2, frame2, cv::ColorConversionCodes::COLOR_BGR2GRAY);

        frame_1 = frame_2.clone();
        frame2.copyTo(frame1);
        //------------------------------------------------------


	//------------------------------------------------------
        //                 IMAGE SUPERRESOLUTION
	// Press 's' for image enhancement and 'S' for quit
	if (key == 115)
	{
		enableSuperres = true;
	}
	else if (key == 83)
	{
		enableSuperres = false;
	}
	if (enableSuperres == true)
	{
		superimage.setSuperres(smoothedFrame);

		cv::Mat color_mat_superres = superimage.getSuperres(net_superres);
		color_mat_superres.copyTo(smoothedFrame);
	}

	//------------------------------------------------------	

        //------------------------------------------------------
        //                    INFERENCE
        // Press 'i' for image enhancement and 'I' for quit
        if (key == 105)
        {
            enableInference = true;
        }
        else if (key == 73)
        {
            enableInference = false;

        }
        if (enableInference == true)
        {
            // Create a 4D blob from a frame.
            cv::Mat blob;
            cv::dnn::blobFromImage(smoothedFrame, blob, 1 / 255.0, cv::Size(inpWidth, inpHeight), cv::Scalar(0, 0, 0), true, false);

            //Sets the input to the network
            net.setInput(blob);

            // Runs the forward pass to get output of the output layers
            net.forward(outs, getOutputsNames(net));


            // Remove the bounding boxes with low confidence
            postprocess(smoothedFrame, outs, enableSuperres);
        }

        //cv::imshow(window_rgb, smoothedFrame);
        
        // Press ESC to exit loop
        if (key == 27)
        {
            t.join();
            break;
        }

        videostream.timeEnd();
        videostream.getLatency();
    }
    
    
    return 0; 
}


catch (const rs2::error& e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function()
        << "(" << e.get_failed_args() << "):\n   " << e.what() << std::endl;

    return EXIT_FAILURE;
}

catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;

    return EXIT_FAILURE;
}
