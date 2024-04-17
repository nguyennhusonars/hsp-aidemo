#include <cstdio>
#include <functional>
#include <thread>
#include "qcsnpe.hpp"
#include <stdlib.h>

Qcsnpe::Qcsnpe(std::string &dlc, int system_type, std::vector<std::string> &output_layers) {
    std::ifstream dlc_file(dlc);
    zdl::DlSystem::Runtime_t runtime_cpu = zdl::DlSystem::Runtime_t::CPU;
    zdl::DlSystem::Runtime_t runtime_gpu = zdl::DlSystem::Runtime_t::GPU_FLOAT16;
    zdl::DlSystem::Runtime_t runtime_dsp = zdl::DlSystem::Runtime_t::DSP;
    zdl::DlSystem::Runtime_t runtime_aip = zdl::DlSystem::Runtime_t::AIP_FIXED8_TF;
    zdl::DlSystem::PerformanceProfile_t perf = zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE;

    static zdl::DlSystem::Version_t Version = zdl::SNPE::SNPEFactory::getLibraryVersion();
    container = zdl::DlContainer::IDlContainer::open(zdl::DlSystem::String(dlc.c_str()));

    runtime_list.add(runtime_dsp);
               
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
    for (auto &output_layer: output_layers) {
        outputs.append(output_layer.c_str());
    }
    zdl::DlSystem::PlatformConfig platform_config;
    model_handler = snpeBuilder.setOutputLayers(outputs)
            .setRuntimeProcessorOrder(runtime_list)
            .setPlatformConfig(platform_config)
            .build();
}

Qcsnpe::Qcsnpe(const Qcsnpe &qc) {
    model_handler = std::move(qc.model_handler);
    container = std::move(qc.container);
    runtime_list = qc.runtime_list;
    outputs = qc.outputs;
    output_tensor_map = qc.output_tensor_map;
    out_tensors = qc.out_tensors;
}

zdl::DlSystem::TensorMap Qcsnpe::predict(cv::Mat input_image) {
    unsigned long int in_size = 1;
    const zdl::DlSystem::TensorShape i_tensor_shape = model_handler->getInputDimensions();
    const zdl::DlSystem::Dimension *shapes = i_tensor_shape.getDimensions();
    int img_size = input_image.channels() * input_image.cols * input_image.rows;
    for (int i = 1; i < i_tensor_shape.rank(); i++) {
        in_size *= shapes[i];
    }
    std::unique_ptr<zdl::DlSystem::ITensor> input_tensor = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(model_handler->getInputDimensions());
    zdl::DlSystem::ITensor *tensor_ptr = input_tensor.get();
    if (tensor_ptr == nullptr) {
        printf("%s\n", "Could not create SNPE input tensor");
    }
    float *tensor_ptr_fl = reinterpret_cast<float *>(&(*input_tensor->begin()));
    const int channels = input_image.channels();
    const int rows = input_image.rows;
    const int cols = input_image.cols;

    for (int i = 0; i < rows; i++) {
        const uchar *row_ptr = input_image.ptr<uchar>(i);
        for (int j = 0; j < cols; j++) {
            const uchar *pixel_ptr = row_ptr + j * channels;
            for (int k = 0; k < channels; k++) {
                tensor_ptr_fl[(i * cols + j) * channels + k] = static_cast<float>(pixel_ptr[k])/255;
            }
        }
    }
    bool exec_status = model_handler->execute(tensor_ptr, output_tensor_map);
    
    if (!exec_status) {
       printf("%s\n", "Error while executing the network.");
    }
    return output_tensor_map;
}