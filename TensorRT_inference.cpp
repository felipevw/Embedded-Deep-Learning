// TensorRT_inference.cpp 
//


#include <assert.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvOnnxConfig.h"
#include "NvUffParser.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "SystemParam.h"

using namespace nvinfer1;



// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger
{
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity == Severity::kINFO) return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
        case Severity::kERROR: std::cerr << "ERROR: "; break;
        case Severity::kWARNING: std::cerr << "WARNING: "; break;
        case Severity::kINFO: std::cerr << "INFO: "; break;
        default: std::cerr << "UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }
} gLogger;

/*
// Gets layer execution time
struct Profiler : public IProfiler
{
    typedef std::pair<std::string, float> Record;
    std::vector<Record> mProfile;

    virtual void reportLayerTime(const char* layerName, float ms)
    {
        auto record = std::find_if(mProfile.begin(), mProfile.end(),
            [&](const Record& r) { return r.first == layerName; });
        if (record == mProfile.end())
            mProfile.push_back(std::make_pair(layerName, ms));
        else
            record->second += ms;
    }

    void printLayerTimes()
    {
        float totalTime = 0;
        for (size_t i = 0; i < mProfile.size(); i++)
        {
            printf("%-40.40s %4.3fms\n",
                mProfile[i].first.c_str(), mProfile[i].second / TIMING_ITERATIONS);
            totalTime += mProfile[i].second;
        }
        printf("Time over all layers: %4.3f\n", totalTime / TIMING_ITERATIONS);
    }

} gProfiler;
*/

/*
struct PPM
{
    std::string magic, fileName;
    int h, w, max;
    uint8_t buffer[INPUT_C * INPUT_H * INPUT_W];
};
*/

struct BBOX
{
    float x1, y1, x2, y2;
    int cls;
};

ICudaEngine* onnxToTRTModel()
{
    // 1) Create a network definition, import the model

    // 1.1.- Create the inference builder
    IBuilder* builder = createInferBuilder(gLogger);

    // Create the network definition
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

    // 1.2.- Create onnx parser
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

    // 1.3.- Parse the model
    size_t verbosity{4};
    parser->parseFromFile(DEPLOY_FILE, verbosity);

    if (!parser->parse(DEPLOY_FILE, verbosity))
    {
        std::cout << "Failed to parse onnx file." << std::endl;
        return nullptr;
    }
    


    return nullptr;
}



int TRTsession()
{
    IHostMemory* trtModelStream{ nullptr };
    ICudaEngine* engine;

    // <1> Parse the onnx model into TensorRT recognized stream
    engine = onnxToTRTModel();


    return 1;
}



int main()
{
    int verify{};
    verify = TRTsession();

}