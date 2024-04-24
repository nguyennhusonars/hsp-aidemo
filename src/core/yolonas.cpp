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

#include "yolonas.hpp"

yolonas::yolonas() {
    this->snpeYolonas = std::unique_ptr<zdl::SNPE::SNPE>();
}

yolonas::~yolonas() {
    this->snpeYolonas.release();
}

int yolonas::load(std::string containerPath, zdl::DlSystem::Runtime_t targetDevice) {
    std::unique_ptr<zdl::DlContainer::IDlContainer> container = loadContainerFromFile(containerPath);
    if (container == nullptr) {
        std::cerr << "Load yolonas model failed." << std::endl;
        return -1;
    }
    DlSystem::RuntimeList runtimeList(targetDevice);
    if (!zdl::SNPE::SNPEFactory::isRuntimeAvailable(targetDevice)) {
        std::cerr << "Selected runtime not present. Falling back to CPU." << std::endl;
        targetDevice = CPU_RUNTIME;
    }
    zdl::DlSystem::StringList outputs;
    for (auto &output_layer: output_layers) {
        outputs.append(output_layer.c_str());
    }
    this->snpeYolonas = setBuilderOptions(container, runtimeList, outputs);
    if (this->snpeYolonas == nullptr) {
        std::cerr << "Error while building yolonas object." << std::endl;
        return -1;
    }
    std::cout << "Load yolonas model successfully" << std::endl;
    return 0;
}

cv::Mat yolonas::execDetect(cv::Mat mat, std::vector<BoxInfo> &result) {
    cv::Mat img_mat = mat;
    cv::Mat input_mat;
    float im_scale = std::min((float)yolonas_size / img_mat.cols, (float)yolonas_size / img_mat.rows);
    int new_w = int(img_mat.cols * im_scale);
    int new_h = int(img_mat.rows * im_scale);
    cv::resize(img_mat, input_mat, cv::Size(new_w, new_h));
    int p_w = yolonas_size - new_w;
    int p_h = yolonas_size - new_h;
    int top = p_h / 2;
    int bottom = p_h - top;
    int left = p_w / 2;
    int right = p_w - left;
    cv::copyMakeBorder(input_mat, input_mat,
                       top, bottom,
                       left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // cv::imwrite("/home/input_mat.png", input_mat);
    unsigned long int in_size = 1;
    const zdl::DlSystem::TensorShape i_tensor_shape = snpeYolonas->getInputDimensions();
    const zdl::DlSystem::Dimension *shapes = i_tensor_shape.getDimensions();
    // int img_size = input_image.channels() * input_image.cols * input_image.rows;
    for (int i = 1; i < i_tensor_shape.rank(); i++) {
        in_size *= shapes[i];
    }
    std::unique_ptr<zdl::DlSystem::ITensor> input_tensor = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(snpeYolonas->getInputDimensions());
    zdl::DlSystem::ITensor *tensor_ptr = input_tensor.get();
    if (tensor_ptr == nullptr) {
        printf("%s\n", "Could not create SNPE input tensor");
    }
    float *tensor_ptr_fl = reinterpret_cast<float *>(&(*input_tensor->begin()));
    const int channels = input_mat.channels();
    const int rows = input_mat.rows;
    const int cols = input_mat.cols;

    for (int i = 0; i < rows; i++) {
        const uchar *row_ptr = input_mat.ptr<uchar>(i);
        for (int j = 0; j < cols; j++) {
            const uchar *pixel_ptr = row_ptr + j * channels;
            for (int k = 0; k < channels; k++) {
                tensor_ptr_fl[(i * cols + j) * channels + k] = static_cast<float>(pixel_ptr[k])/255;
            }
        }
    }
    zdl::DlSystem::TensorMap output_tensor_map;
    bool exec_status = snpeYolonas->execute(tensor_ptr, output_tensor_map);
    if (!exec_status) {
       printf("%s\n", "Error while executing the network.");
    }
    zdl::DlSystem::StringList out_tensors = output_tensor_map.getTensorNames();
    std::map<std::string, std::vector<float>> out_itensor_map;
    for (size_t i = 0; i < out_tensors.size(); i++) {
        zdl::DlSystem::ITensor *out_itensor = output_tensor_map.getTensor(out_tensors.at(i));
        std::vector<float> out_vec{reinterpret_cast<float *>(&(*out_itensor->begin())), reinterpret_cast<float *>(&(*out_itensor->end()))};
        out_itensor_map.insert(std::make_pair(std::string(out_tensors.at(i)), out_vec));
    }
    zdl::DlSystem::ITensor *out_itensor = output_tensor_map.getTensor(out_tensors.at(0));
    zdl::DlSystem::ITensor *out_itensor_1 = output_tensor_map.getTensor(out_tensors.at(1));

    // std::vector<BoxInfo> result;
    float *data = out_itensor->begin().dataPointer();
    float *dataSource_1 = out_itensor_1->begin().dataPointer();
    std::vector<float> confidences;
    // std::cout << out_itensor_1->getShape()[0] << " " << out_itensor_1->getShape()[1] << " " << out_itensor_1->getShape()[2] << std::endl;
   
   // YOLO NAS post process
    int num_class =  out_itensor_1->getShape()[2];
    int num_pp = out_itensor_1->getShape()[1];
    for (int i = 0; i < num_pp; i++) {
        std::vector<int> class_ids;
        float maxScore = 0;
        int maxClass = -1;
        // printf("%d: ", i);
        float x1 = data[i * 4 + 0];
        float y1 = data[i * 4 + 1];
        float x2 = data[i * 4 + 2];
        float y2 = data[i * 4 + 3];
        for (int j = 0; j < num_class; j++) {
            // printf("%f ", dataSource_1[i * 80 + j]);
            float score = dataSource_1[i * num_class + j];
            if (score > maxScore) {
                maxScore = score;
                maxClass = j;
            }
            if (maxScore > yolonas_threshold) {
                confidences.push_back(maxScore);
                class_ids.push_back(maxClass);
                BoxInfo box;
                box.x1 = std::max(0, std::min(img_mat.cols, int((x1 - left) / im_scale)));
                box.y1 = std::max(0, std::min(img_mat.rows, int((y1 - top) / im_scale)));
                box.x2 = std::max(0, std::min(img_mat.cols, int((x2 - left) / im_scale)));
                box.y2 = std::max(0, std::min(img_mat.rows, int((y2 - top) / im_scale)));
                box.score = maxScore;
                box.label = maxClass;
                result.push_back(box);
            }
        }
        // printf("\n");
    }
    yolonas::nms(result, 0.5);
    // for (int i = 0; i < result.size(); ++i) {
    //     auto detection = result[i];
    //     cv::Scalar color = cv::Scalar(255, 255, 0);
    //     cv::rectangle(img_mat, cv::Point(detection.x1, detection.y1), cv::Point(detection.x2, detection.y2), color, 2);
    //     cv::rectangle(img_mat, cv::Point(detection.x1, detection.y1 - 20), cv::Point(detection.x2, detection.y1), color, -1);
    //     std::stringstream ss;
    //     ss << detection.label << " " << detection.score;
    //     cv::putText(img_mat, ss.str(), cv::Point(detection.x1, detection.y1), cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    // }
    // cv::imwrite("/home/yolov8_out.png", img_mat);
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
