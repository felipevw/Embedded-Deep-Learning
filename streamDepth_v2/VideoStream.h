// VideoStream class
// This is the RealSense camera video streamer handler


#pragma once
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>


class VideoStream
{
public:
	VideoStream();

	rs2::align setStreamRGB(rs2_stream stream_type, const size_t width,
		const size_t height, rs2_format format, const size_t frame_rate);

	rs2::align setStreamRGBD(rs2_stream stream_type1, rs2_stream stream_type2, const size_t width,
		const size_t height, rs2_format format1, rs2_format format2, const size_t frame_rate);

	void cameraWarmUp(rs2::pipeline pipe);

	rs2::pipeline getPipeline();
	static void getWindowRGB(const cv::String window_name);
	static void getWindowRGBD(const cv::String window_name_rgb, const cv::String window_name_depth);

	static void getNextFrame(rs2::align align_to, rs2::pipeline pipe, cv::Mat& color_matrix);

	rs2::depth_sensor getDepthSensor();
	rs2::frame FilterDepthFrame(rs2::frame frame_to_filter);


	void timeStart();
	void timeEnd();
	void getLatency();
	
	~VideoStream();

private:
	// Main pipeline and configuration
	rs2::pipeline pipe;
	rs2::config cfg;

	// Depth sensor
	rs2::pipeline_profile profile;
	rs2::device dev;
	
	// Detph filter
	rs2::spatial_filter spatial_filter;
	rs2::hole_filling_filter hole_filter;

	std::chrono::time_point<std::chrono::high_resolution_clock> timeCount;
	std::chrono::time_point<std::chrono::high_resolution_clock> timeStopCount;
	long long globalDuration;
};

