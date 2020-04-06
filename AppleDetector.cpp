//------------------------------------------------------------
//
//                      MASTER THESIS
//          Embedded Deep Learning for Fruit Detection
//
//------------------------------------------------------------

#include <iostream>
#include <chrono>

// include the librealsense C++ header file
#include <librealsense2/rs.hpp>

// include OpenCV header file
#include <opencv2/opencv.hpp>

// Helper file
#include "cv-helpers.hpp"


const size_t FRAME_HEIGHT{ 720 };
const size_t FRAME_WIDTH{ 1280 };
const size_t FRAME_RATE{ 30 };
const float WHRatio = FRAME_WIDTH / static_cast<float>(FRAME_HEIGHT);


int main() try
{
    using namespace cv;

    //Contruct a pipeline which abstracts the device
    rs2::pipeline pipe;

    //Create a configuration for configuring the pipeline with a non default profile
    rs2::config cfg;

    //Add desired streams to configuration
    cfg.enable_stream(RS2_STREAM_COLOR, FRAME_WIDTH, FRAME_HEIGHT, RS2_FORMAT_BGR8, FRAME_RATE);

    //Instruct pipeline to start streaming with the requested configuration
    pipe.start(cfg);

    // Spatial alignment stream of frames
    rs2::align align_to(RS2_STREAM_COLOR);

    // Camera warmup - dropping several first frames to let auto-exposure stabilize
    rs2::frameset frames;
    for (int i = 0; i < 30; i++)
    {
        //Wait for all configured streams to produce a frame
        frames = pipe.wait_for_frames();
    }

    // Main frame name
    const auto window_name = "RGB Stream 1280 x 720";
    namedWindow(window_name, WINDOW_AUTOSIZE);


    

    int key{};
    bool keyEnable{false};
    while (getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
    {
        auto timeStart = std::chrono::high_resolution_clock::now();

        // Wait for the next set of frames
        auto data = pipe.wait_for_frames();

        // Make sure the frames are spatially aligned
        data = align_to.process(data);
        
        // Find the color data
        rs2::frame color = data.get_color_frame();        
        
        // Convert to opencv Mat format
        auto color_mat = frame_to_mat(color);
        

        // Press ESC to exit loop
        if (key == 27)
            break;

        // Press 'e' or 'E'
        if (key == 101 || key == 69)
            keyEnable = !keyEnable;

        if (keyEnable == true)
            putText(color_mat, "Hello World", Point(10, 100),
                FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 0), 3);

        putText(color_mat, "Press ESC to exit", Point(10, 40), 
            FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 3);


        


        
        imshow(window_name, color_mat);
        key = waitKey(1) & 0xFF;

        auto timeEnd = std::chrono::high_resolution_clock::now();
        long long duration = std::chrono::duration_cast<std::chrono::milliseconds>(timeEnd - timeStart).count();
        std::cout << "Latency value: " << duration << " in ms." << std::endl;
    }
    destroyAllWindows();

    return 0;
}

catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() 
              << "(" << e.get_failed_args() << "):\n   " << e.what() << std::endl;

    return EXIT_FAILURE;
}

catch (const std::exception & e)
{
    std::cerr << e.what() << std::endl;

    return EXIT_FAILURE;
}