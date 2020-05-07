// streamDepth_v2.cpp 

#include <iostream>

// include the librealsense C++ header file
#include <librealsense2/rs.hpp>

// include OpenCV header file
#include <opencv2/opencv.hpp>

#include "VideoStream.h"
#include "cv-helpers.hpp"
#include "ImageProcess.h"


// Camera parameters
const size_t FRAME_HEIGHT{ 480 };
const size_t FRAME_WIDTH{ 640 };
const size_t FRAME_RATE{ 60 };
const float WHRatio = static_cast<float>(FRAME_WIDTH) / static_cast<float>(FRAME_HEIGHT);
cv::Mat color_mat(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3);
cv::Mat color_mat_processed(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3);
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
    //                         THREAD
    // Enqueue frames
    std::thread t([&]() {
        while (true)
        {
            auto frames = pipe.wait_for_frames();

            // Make sure the frames are spatially aligned
            frames = align_to.process(frames);

            queue_depth.enqueue(frames.get_depth_frame());
            queue_rgb.enqueue(frames.get_color_frame());

            // Press ESC to exit loop
            if (key == 27)
            {
                break;
            }
        }
        });
        
    //------------------------------------------------------
    
    
    rs2::frame depth_frame;
    rs2::frame rgb_frame;
    while (true)
    {
        videostream.timeStart();
        key = cv::waitKey(15);

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
        

        if (queue_rgb.poll_for_frame(&rgb_frame))
        {
            // Get the frame
            rgb_frame.get_data();

            // Convert to opencv Mat format
            color_mat = frame_to_mat(rgb_frame);
            
        }
        // Image preprocessing
        imageprocess.setPreprocess(color_mat);
        color_mat_processed = imageprocess.getPreprocess();
        
        cv::imshow(window_rgb, color_mat_processed);
        
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