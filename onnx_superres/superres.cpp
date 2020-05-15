#include <opencv2/dnn_superres.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <chrono>
#include <opencv2/dnn.hpp>

using namespace std;
using namespace cv;


int main()
{
	std::string onnxFile = "super-resolution-10.onnx";

	// Load the network
	cv::dnn::Net net = cv::dnn::readNetFromONNX(onnxFile);

	cv::Mat frame = imread("apple_init.png");	
	cvtColor(frame, frame, COLOR_BGR2YCrCb );

	cout << "Init size: " << frame.size() << endl;

	// Create a 4D blob from a frame.

	vector<Mat> images;
	split(frame, images);

	cv::Mat blob;
	//dnn::blobFromImage(images[0], blob, 1.0 / 255.0, Size(224, 224), Scalar(), true, false);
	
	images[0].convertTo(blob, CV_32FC1);

	// Sets the input to the network
	net.setInput(blob);

	cv::Mat prob = net.forward();

//	prob.convertTo(prob, CV_8UC1, 1.0/ 1.0/ 255.0, 0);
	
	//images[0] = prob;

	//cv::Mat result;
	//merge(images, result);
	cout << "Size is: " << prob.size() << endl;
	//imshow("resulting", prob);

	cout << "All ok" << endl;
	return 0;
}
