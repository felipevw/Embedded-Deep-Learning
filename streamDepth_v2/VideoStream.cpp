#include "VideoStream.h"
#include "cv-helpers.hpp"


VideoStream::VideoStream()
{

}


rs2::align VideoStream::setStreamRGB(rs2_stream stream_type, const size_t width,
    const size_t height, rs2_format format, const size_t frame_rate)
{
    //Add desired streams to configuration
    cfg.enable_stream(stream_type, width, height, format, frame_rate);

    //Instruct pipeline to start streaming with the requested configuration
    pipe.start(cfg);

    // Spatial alignment stream of frames
    rs2::align align_to(RS2_STREAM_COLOR);

    return align_to;
}

rs2::depth_sensor VideoStream::getDepthSensor()
{
    return dev.query_sensors().front().as<rs2::depth_sensor>();
}

rs2::align VideoStream::setStreamRGBD(rs2_stream stream_type1, rs2_stream stream_type2, const size_t width,
    const size_t height, rs2_format format1, rs2_format format2, const size_t frame_rate)
{
    //Add desired streams to configuration
    cfg.enable_stream(stream_type1, width, height, format1, frame_rate);
    cfg.enable_stream(stream_type2, width, height, format2, frame_rate);

    //Instruct pipeline to start streaming with the requested configuration
    profile = pipe.start(cfg);
    dev = profile.get_device();

    // Spatial alignment stream of frames
    rs2::align align_to(RS2_STREAM_COLOR);

    return align_to;
}

void VideoStream::cameraWarmUp(rs2::pipeline pipe)
{
    // Camera warmup - dropping several first frames to let auto-exposure stabilize
    rs2::frameset frames;
    std::cout << "Warm up camera." << std::endl;
    for (int i = 0; i < 30; i++)
    {
        //Wait for all configured streams to produce a frame
        frames = pipe.wait_for_frames();
    }
}

void VideoStream::getWindowRGB(const cv::String window_name)
{
    cv::namedWindow(window_name, cv::WindowFlags::WINDOW_AUTOSIZE);
}

void VideoStream::getWindowRGBD(const cv::String window_name_rgb, const cv::String window_name_depth)
{
    cv::namedWindow(window_name_rgb, cv::WindowFlags::WINDOW_AUTOSIZE);
    cv::namedWindow(window_name_depth, cv::WindowFlags::WINDOW_AUTOSIZE);
}

rs2::pipeline VideoStream::getPipeline()
{
    return pipe;
}

void VideoStream::getNextFrame(rs2::align align_to, rs2::pipeline pipe, cv::Mat& color_matrix)
{
    // Wait for the next set of frames
    auto data = pipe.wait_for_frames();

    // Make sure the frames are spatially aligned
    data = align_to.process(data);

    // Find the color data
    rs2::frame color = data.get_color_frame();

    // Convert to opencv Mat format
    color_matrix = frame_to_mat(color);
    
}

void VideoStream::timeStart()
{
    timeCount = std::chrono::high_resolution_clock::now();
}

void VideoStream::timeEnd()
{
    timeStopCount = std::chrono::high_resolution_clock::now();
}

void VideoStream::getLatency()
{
    globalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(timeStopCount - timeCount).count();
    std::cout << "Latency value: " << globalDuration << " [ms].  " << 1000.0f / globalDuration << "[fps]" << "\n";
}

VideoStream::~VideoStream()
{
    cv::destroyAllWindows();
}

