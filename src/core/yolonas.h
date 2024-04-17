
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
#include "qcsnpe.hpp"

#define OUTPUT_LAYER_1 "/model.24/Concat_15"

typedef struct {
    int width;
    int height;
} YoloSize;

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

typedef struct {
    std::string index;
    int stride;
    std::vector<YoloSize> anchors;
    int grid_size;
} YoloLayerData;

class yolonas {
public:
    yolonas();
    ~yolonas();
    yolonas(const yolonas &other) = delete;
    yolonas &operator=(const yolonas &other) = delete;
    void nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH);
    int load(std::string model_path, zdl::DlSystem::Runtime_t targetDevice);
    cv::Mat execDetect(cv::Mat mat);
    std::vector<BoxInfo>postprocess(float *dataSource, float *dataSource_1, const YoloSize &frame_size,
                                  int left, int top,
                                  int num_classes, float threshold);

private:
    // OpenCV values
    cv::Mat img_mat;
    cv::Mat bgr_img;
    cv::Mat grey_img;
    cv::Mat rgb_img;
    cv::Mat out_img;
    Qcsnpe *qc;
    cv::VideoWriter video_writer;
    std::string model_path = "/home/demo/hsp-aidemo-master/models/yolo_nas_s_v213_quantized.dlc";
    std::vector<std::string> output_layers{"/heads/Sigmoid", "/heads/Mul"};
    std::vector<std::vector<float>> pred_out;
    const float INPUT_WIDTH = 320.0;
    const float INPUT_HEIGHT = 320.0;
    const float NMS_THRESHOLD = 0.5;
    const float CONFIDENCE_THRESHOLD = 0.4;

    struct Detection {
        int class_id;
        float confidence;
        cv::Rect box;
    };

    float im_scale;
};

#endif
