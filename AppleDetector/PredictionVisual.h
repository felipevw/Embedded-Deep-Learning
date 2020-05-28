using namespace cv;

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

    //Get the label for the class name and its confidence
    std::string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }


    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}


// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const std::vector<Mat>& outs, bool superresEnabler)
{
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;


    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j{}; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);


    for (size_t i{}; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];

        //Get the label for the class name and its confidence
        std::string label = format("%.2f", confidences[idx]);
        if (!classes.empty())
        {
            CV_Assert(classIds[idx] < (int)classes.size());
            label = classes[classIds[idx]] + ":" + label;
        }

	if (superresEnabler == false)
	{
        	// Convert bbox depth to meters
        	float obj_in_meters = bbox_to_dist(box, scale, depth_mat);
        	std::cout << "Object: " << label << " at " << obj_in_meters << " meters." << std::endl;
	}
	else if (superresEnabler == true)
	{
		std::cout << "Object: " << label << " detected." << std::endl;
	}

        // Draw rectangles and class id's
        //drawPred(classIds[idx], confidences[idx], box.x, box.y,
        //    box.x + box.width, box.y + box.height, frame);
    }
}
