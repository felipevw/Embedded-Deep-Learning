

// TensorRT parameters
#define CHECK(status)                                   \
{                                                       \
    if (status != 0)                                    \
    {                                                   \
        std::cout << "Cuda failure: " << status;        \
        abort();                                        \
    }                                                   \
}

const char* DEPLOY_FILE = "model/yolov3-10.onnx"; // Deploy model file

