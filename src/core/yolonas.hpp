
#ifndef YOLONAS_H
#define YOLONAS_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/features2d.hpp>

#include <unistd.h>
#include <time.h>

#include <cstdlib>
#include <string>
#include <vector>
#include <thread>

#include <DlContainer/IDlContainer.hpp>
#include <DlSystem/RuntimeList.hpp>
#include <SNPE/SNPE.hpp>
#include <SNPE/SNPEBuilder.hpp>
#include <DlSystem/ITensorFactory.hpp>
#include <SNPE/SNPEFactory.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <ctime>
#include "SnpeCommLib.hpp"
#include "TrackProcess.hpp"

// #include "qcsnpe.hpp"

#define OUTPUT_LAYER_1 "/model.24/Concat_15"

class yolonas {
private:
    std::unique_ptr<zdl::SNPE::SNPE> snpeYolonas;

public:
    yolonas();
    ~yolonas();
    int load(std::string model_path, zdl::DlSystem::Runtime_t targetDevice);
    cv::Mat execDetect(cv::Mat mat, std::vector<BoxInfo> &result);
    void nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH);
    std::vector<std::string> output_layers{"/heads/Sigmoid", "/heads/Mul"};

    const int yolonas_size = 320;
    const float yolonas_threshold = 0.5f;
    const int yolonas_classes = 80;
};

#endif
