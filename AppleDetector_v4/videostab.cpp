#include "videostab.h"
#include <cmath>

//Parameters for Kalman Filter
#define Q1 0.004
#define R1 0.5

//To see the results of before and after stabilization simultaneously
#define test 1

VideoStab::VideoStab()
{

    smoothedMat.create(2, 3, CV_64F);

    k = 1;

    errscaleX = 1;
    errscaleY = 1;
    errthetha = 1;
    errtransX = 1;
    errtransY = 1;

    Q_scaleX = Q1;
    Q_scaleY = Q1;
    Q_thetha = Q1;
    Q_transX = Q1;
    Q_transY = Q1;

    R_scaleX = R1;
    R_scaleY = R1;
    R_thetha = R1;
    R_transX = R1;
    R_transY = R1;

    sum_scaleX = 0;
    sum_scaleY = 0;
    sum_thetha = 0;
    sum_transX = 0;
    sum_transY = 0;

    scaleX = 0;
    scaleY = 0;
    thetha = 0;
    transX = 0;
    transY = 0;

}

//The main stabilization function
Mat VideoStab::stabilize(Mat frame_1, Mat frame_2)
{
    cvtColor(frame_1, frame1, cv::ColorConversionCodes::COLOR_BGR2GRAY);
    cvtColor(frame_2, frame2, cv::ColorConversionCodes::COLOR_BGR2GRAY);

    int vert_border = HORIZONTAL_BORDER_CROP * 480 / 640;

    vector <Point2f> features1, features2;
    vector <Point2f> goodFeatures1, goodFeatures2;
    vector <uchar> status;
    vector <float> err;

    //Estimating the features in frame1 and frame2
    goodFeaturesToTrack(frame1, features1, 200, 0.01, 30);
    std::cout << "Num features: " << features1.size() << std::endl;
    calcOpticalFlowPyrLK(frame1, frame2, features1, features2, status, err);

    for (size_t i = 0; i < status.size(); i++)
    {
        if (status[i])
        {
            goodFeatures1.push_back(features1[i]);
            goodFeatures2.push_back(features2[i]);
        }
    }

    //All the parameters scale, angle, and translation are stored in affine
    affine = estimateAffinePartial2D(goodFeatures1, goodFeatures2);

    //cout<<affine;
    //flush(cout);


    dx = affine.at<double>(0, 2);
    dy = affine.at<double>(1, 2);
    da = atan2(affine.at<double>(1, 0), affine.at<double>(0, 0));
    ds_x = affine.at<double>(0, 0) / cos(da);
    ds_y = affine.at<double>(1, 1) / cos(da);

    sx = ds_x;
    sy = ds_y;

    sum_transX += dx;
    sum_transY += dy;
    sum_thetha += da;
    sum_scaleX += ds_x;
    sum_scaleY += ds_y;


    //Don't calculate the predicted state of Kalman Filter on 1st iteration
    if (k == 1)
    {
        k++;
    }
    else
    {
        Kalman_Filter(&scaleX, &scaleY, &thetha, &transX, &transY);

    }

    diff_scaleX = scaleX - sum_scaleX;
    diff_scaleY = scaleY - sum_scaleY;
    diff_transX = transX - sum_transX;
    diff_transY = transY - sum_transY;
    diff_thetha = thetha - sum_thetha;

    ds_x = ds_x + diff_scaleX;
    ds_y = ds_y + diff_scaleY;
    dx = dx + diff_transX;
    dy = dy + diff_transY;
    da = da + diff_thetha;

    //Creating the smoothed parameters matrix
    smoothedMat.at<double>(0, 0) = sx * cos(da);
    smoothedMat.at<double>(0, 1) = sx * -sin(da);
    smoothedMat.at<double>(1, 0) = sy * sin(da);
    smoothedMat.at<double>(1, 1) = sy * cos(da);

    smoothedMat.at<double>(0, 2) = dx;
    smoothedMat.at<double>(1, 2) = dy;


    //Warp the new frame using the smoothed parameters CUDA
    cv::cuda::GpuMat frame_1_gpu, smoothedFrame_gpu, smoothedMat_gpu;
    frame_1_gpu.upload(frame_1);
    smoothedMat_gpu.upload(smoothedMat);
    cv::cuda::warpAffine(frame_1_gpu, smoothedFrame_gpu, smoothedMat, frame_2.size());
    smoothedFrame_gpu.download(smoothedFrame);

    //Crop the smoothed frame a little to eliminate black region due to Kalman Filter
    smoothedFrame_gpu = smoothedFrame_gpu(Range(vert_border, smoothedFrame.rows - vert_border), Range(HORIZONTAL_BORDER_CROP, smoothedFrame.cols - HORIZONTAL_BORDER_CROP));

    cv::cuda::resize(smoothedFrame_gpu, smoothedFrame_gpu, frame_2.size());
    smoothedFrame_gpu.download(smoothedFrame);

    return smoothedFrame;
}


//Kalman Filter implementation
void VideoStab::Kalman_Filter(double* scaleX, double* scaleY, double* thetha, double* transX, double* transY)
{
    double frame_1_scaleX = *scaleX;
    double frame_1_scaleY = *scaleY;
    double frame_1_thetha = *thetha;
    double frame_1_transX = *transX;
    double frame_1_transY = *transY;

    double frame_1_errscaleX = errscaleX + Q_scaleX;
    double frame_1_errscaleY = errscaleY + Q_scaleY;
    double frame_1_errthetha = errthetha + Q_thetha;
    double frame_1_errtransX = errtransX + Q_transX;
    double frame_1_errtransY = errtransY + Q_transY;

    double gain_scaleX = frame_1_errscaleX / (frame_1_errscaleX + R_scaleX);
    double gain_scaleY = frame_1_errscaleY / (frame_1_errscaleY + R_scaleY);
    double gain_thetha = frame_1_errthetha / (frame_1_errthetha + R_thetha);
    double gain_transX = frame_1_errtransX / (frame_1_errtransX + R_transX);
    double gain_transY = frame_1_errtransY / (frame_1_errtransY + R_transY);

    *scaleX = frame_1_scaleX + gain_scaleX * (sum_scaleX - frame_1_scaleX);
    *scaleY = frame_1_scaleY + gain_scaleY * (sum_scaleY - frame_1_scaleY);
    *thetha = frame_1_thetha + gain_thetha * (sum_thetha - frame_1_thetha);
    *transX = frame_1_transX + gain_transX * (sum_transX - frame_1_transX);
    *transY = frame_1_transY + gain_transY * (sum_transY - frame_1_transY);

    errscaleX = (1 - gain_scaleX) * frame_1_errscaleX;
    errscaleY = (1 - gain_scaleY) * frame_1_errscaleX;
    errthetha = (1 - gain_thetha) * frame_1_errthetha;
    errtransX = (1 - gain_transX) * frame_1_errtransX;
    errtransY = (1 - gain_transY) * frame_1_errtransY;
}
