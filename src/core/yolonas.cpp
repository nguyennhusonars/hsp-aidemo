#include "yolonas.h"
#include <unistd.h>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <string>
#include <cstdlib>
#include <mutex>
#include <glob.h>
#include <dirent.h>
#include <stdio.h>
#include <opencv2/imgproc/types_c.h>

yolonas::yolonas() {
    qc = new Qcsnpe(model_path, 2, output_layers);
}

yolonas::~yolonas() {}

cv::Mat yolonas::execDetect(cv::Mat mat) {
    img_mat = mat;
    std::vector<Detection> output;
    cv::Mat input_mat;
    im_scale = std::min((float)INPUT_WIDTH / img_mat.cols, (float)INPUT_HEIGHT / img_mat.rows);
    int new_w = int(img_mat.cols * im_scale);
    int new_h = int(img_mat.rows * im_scale);
    cv::resize(img_mat, input_mat, cv::Size(new_w, new_h));
    int p_w = INPUT_WIDTH - new_w;
    int p_h = INPUT_WIDTH - new_h;
    int top = p_h / 2;
    int bottom = p_h - top;
    int left = p_w / 2;
    int right = p_w - left;
    cv::copyMakeBorder(input_mat, input_mat,
                       top, bottom,
                       left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // cv::resize(img_mat, input_mat, cv::Size(640, 640));
    // cv::imwrite("/home/input_mat.png", input_mat);
    zdl::DlSystem::TensorMap output_tensor_map = qc->predict(input_mat);
    zdl::DlSystem::StringList out_tensors = output_tensor_map.getTensorNames();
    out_tensors = output_tensor_map.getTensorNames();
    std::map<std::string, std::vector<float>> out_itensor_map;
    for (size_t i = 0; i < out_tensors.size(); i++) {
        zdl::DlSystem::ITensor *out_itensor = output_tensor_map.getTensor(out_tensors.at(i));
        std::vector<float> out_vec{reinterpret_cast<float *>(&(*out_itensor->begin())), reinterpret_cast<float *>(&(*out_itensor->end()))};
        out_itensor_map.insert(std::make_pair(std::string(out_tensors.at(i)), out_vec));
    }
    std::vector<BoxInfo> result;
    zdl::DlSystem::ITensor *out_itensor = output_tensor_map.getTensor(out_tensors.at(0));
    zdl::DlSystem::ITensor *out_itensor_1 = output_tensor_map.getTensor(out_tensors.at(1));
    std::cout << out_itensor->getSize()<< std::endl;
    auto boxes = yolonas::postprocess(out_itensor->begin().dataPointer(), out_itensor_1->begin().dataPointer(), {(int)img_mat.cols, (int)img_mat.rows}, left, top, 80, CONFIDENCE_THRESHOLD);
    // auto boxes = yolonas::postprocess(out_itensor->begin().dataPointer(), {(int)img_mat.cols, (int)img_mat.rows}, 0, 0, 80, CONFIDENCE_THRESHOLD);

    result.insert(result.begin(), boxes.begin(), boxes.end());
    yolonas::nms(result, NMS_THRESHOLD);
    for (int i = 0; i < result.size(); ++i) {
        auto detection = result[i];
        cv::Scalar color = cv::Scalar(255, 255, 0);
        cv::rectangle(img_mat, cv::Point(detection.x1, detection.y1), cv::Point(detection.x2, detection.y2), color, 2);
        cv::rectangle(img_mat, cv::Point(detection.x1, detection.y1 - 20), cv::Point(detection.x2, detection.y1), color, -1);
        std::stringstream ss;
        ss << detection.label << " " << detection.score;
        cv::putText(img_mat, ss.str(), cv::Point(detection.x1, detection.y1), cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    }
    cv::imwrite("/home/yolov8_out.png", img_mat);
    pred_out.clear();
    return img_mat;
}

inline float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x) { return 1.0f / (1.0f + fast_exp(-x)); }

std::vector<BoxInfo> yolonas::postprocess(float *dataSource, float *dataSource_1, const YoloSize &frame_size, int left, int top, int num_classes, float threshold) {
    float *data = dataSource;
    std::vector<BoxInfo> result;

    float cx, cy, w, h;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // YOLOV8 post process
    // for (int i = 0; i < 8400; i++) {
    //     std::vector<int> class_ids;
    //     float maxScore = 0;
    //     int maxClass = -1;
    //     printf("%d: ", i);
    //     float cx = data[0 * 8400 + i];
    //     float cy = data[1 * 8400 + i];
    //     float w = data[2 * 8400 + i];
    //     float h = data[3 * 8400 + i];
    //     for (int j = 0; j < 80; j++) {
    //         // printf("%f ", data[i * 80 + j]);
    //         printf("%f ", data[j * 8400 + i]);
    //         float score = data[(j + 4) * 8400 + i];
    //         if (score > maxScore) {
    //             maxScore = score;
    //             maxClass = j;
    //         }
    //         if (maxScore > threshold) {
    //             confidences.push_back(maxScore);
    //             class_ids.push_back(maxClass);
    //             BoxInfo box;
    //             box.x1 = std::max(0, std::min(frame_size.width,
    //                                         int((cx - w / 2.f - left) / im_scale)));
    //             box.y1 = std::max(0, std::min(frame_size.height,
    //                                         int((cy - h / 2.f - top) / im_scale)));
    //             box.x2 = std::max(0, std::min(frame_size.width,
    //                                         int((cx + w / 2.f - left) / im_scale)));
    //             box.y2 = std::max(0, std::min(frame_size.height,
    //                                         int((cy + h / 2.f - top) / im_scale)));
    //             box.score = maxScore;
    //             box.label = maxClass;
    //             result.push_back(box);
    //         }
    //     }
    //     printf("\n");
    // }

    // YOLO NAS post process
    for (int i = 0; i < 2100; i++) {
        std::vector<int> class_ids;
        float maxScore = 0;
        int maxClass = -1;
        // printf("%d: ", i);
        float x1 = dataSource_1[i * 4 + 0];
        float y1 = dataSource_1[i * 4 + 1];
        float x2 = dataSource_1[i * 4 + 2];
        float y2 = dataSource_1[i * 4 + 3];
        for (int j = 0; j < 80; j++) {
            float score = data[i * 80 + j];
            if (score > maxScore) {
                maxScore = score;
                maxClass = j;
            }
            if (maxScore > threshold) {
                confidences.push_back(maxScore);
                class_ids.push_back(maxClass);
                BoxInfo box;
                box.x1 = std::max(0, std::min(frame_size.width,
                                            int((x1 - left) / im_scale)));
                box.y1 = std::max(0, std::min(frame_size.height,
                                            int((y1 - top) / im_scale)));
                box.x2 = std::max(0, std::min(frame_size.width,
                                            int((x2 - left) / im_scale)));
                box.y2 = std::max(0, std::min(frame_size.height,
                                            int((y2 - top) / im_scale)));
                box.score = maxScore;
                box.label = maxClass;
                result.push_back(box);
            }
        }
    }
    return result;
}

void yolonas::nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH) {
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            } else {
                j++;
            }
        }
    }
}
