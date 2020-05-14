// streamDepth_v2.cpp 

#include <iostream>

// include the librealsense C++ header file
#include <librealsense2/rs.hpp>

// include OpenCV header file
#include <opencv2/opencv.hpp>

#include <opencv2/xphoto.hpp>

#include "VideoStream.h"
#include "cv-helpers.hpp"
#include "ImageProcess.h"
#include "videostab.h"


// Camera parameters
const size_t FRAME_HEIGHT{ 480 };
const size_t FRAME_WIDTH{ 640 };
const size_t FRAME_RATE{ 60 };
const float WHRatio = static_cast<float>(FRAME_WIDTH) / static_cast<float>(FRAME_HEIGHT);
cv::Mat color_mat(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3);
cv::Mat depth_mat(FRAME_HEIGHT, FRAME_WIDTH, CV_16UC1);

VideoStream videostream;
ImageProcess imageprocess;

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
    //                         THREAD
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
    bool enableImageProcess{ false };


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
        if (queue_depth.poll_for_frame(&depth_frame))
        {
            // Get the frame and scale data
            depth_frame.get_data();
            float scale = ds.get_depth_scale();

            // Create OpenCV matrix of size (w,h) from the colorized depth data
            cv::Mat depth_frame_mat(cv::Size(FRAME_WIDTH, FRAME_HEIGHT), CV_16UC1, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);
            depth_frame_mat.copyTo(depth_mat);

            // Convert 16bit image to 8bit image
            depth_frame_mat.convertTo(depth_frame_mat, CV_8UC1, 15 / 256.0);
            cv::imshow(window_depth, depth_frame_mat);
        }
        
        //------------------------------------------------------
        //                 IMAGE ENHANCEMENT
        // Press 'e' for image enhancement and 'E' for reverse
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


        cv::imshow(window_rgb, smoothedFrame);
        
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