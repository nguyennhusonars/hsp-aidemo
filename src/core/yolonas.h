
#ifndef PERSONDETECTION_CAMERAAPP_H
#define PERSONDETECTION_CAMERAAPP_H

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

// #include "qcsnpe.hpp"

#define OUTPUT_LAYER_1 "/model.24/Concat_15"

typedef struct BoxInfo {
    int x1;
    int y1;
    int x2;
    int y2;
    float score;
    int label;
} BoxInfo;

typedef struct TrackingBox {
    int trackID = -1;
    cv::Rect box;
    int status = 0;  // 1: true , 0: false,
    std::string mappedID;
    std::string mappedName;
} TrackingBox;

class yolonas {
private:
    std::unique_ptr<zdl::SNPE::SNPE> snpeYolonas;

public:
    yolonas();
    ~yolonas();
    // yolonas(const yolonas &other) = delete;
    // yolonas &operator=(const yolonas &other) = delete;
    int load(std::string model_path, zdl::DlSystem::Runtime_t targetDevice);
    cv::Mat execDetect(cv::Mat mat);
    std::vector<BoxInfo>postprocess(float *dataSource, float *dataSource_1, int yoloSize,
                                  int left, int top,
                                  int num_classes, float threshold);
    void nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH);
    std::vector<std::string> output_layers{"/heads/Sigmoid", "/heads/Mul"};

    const int yolonas_size = 320;
    const float yolonas_threshold = 0.3f;
    const int yolonas_classes = 80;
};

#endif
