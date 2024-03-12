#include "gstObject.hpp"

gstObject::gstObject(std::string url, int inputType, int i) {
    threadID = i;
    gst_init(nullptr, nullptr);
    if (inputType == INPUT_TYPE::VIDEO) {
        std::string tmp = "uridecodebin uri=file://" + url +
                          " ! videoconvert ! video/x-raw,format=I420 ! appsink name=sink sync=false";
        pipeline_ = gst_parse_launch(tmp.c_str(), nullptr);
    } else if (inputType == INPUT_TYPE::RTSP) {
        std::string tmp = "rtspsrc location=" + url +
                          " latency=0 ! rtph264depay ! avdec_h264 ! videoconvert ! appsink name=sink sync=false";
        pipeline_ = gst_parse_launch(tmp.c_str(), nullptr);
    } else {
    }
    appsink_ = GST_APP_SINK(gst_bin_get_by_name(GST_BIN(pipeline_), "sink"));
    g_object_set(G_OBJECT(appsink_), "emit-signals", TRUE, nullptr);
    g_signal_connect(appsink_, "new-sample", G_CALLBACK(onNewSampleStatic), this);
    bus_ = gst_pipeline_get_bus(GST_PIPELINE(pipeline_));
    gst_bus_add_signal_watch(bus_);
    g_signal_connect(bus_, "message::eos", G_CALLBACK(onEosStatic), this);
}

gstObject::~gstObject() {
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_object_unref(pipeline_);
    gst_object_unref(bus_);
}

GstFlowReturn gstObject::onNewSample(GstElement* appsink) {
    std::unique_lock<std::mutex> lock(mutex_);
    GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink));
    GstCaps* caps = gst_sample_get_caps(sample);
    int width = 0, height = 0;
    if (caps) {
        GstStructure* structure = gst_caps_get_structure(caps, 0);
        gst_structure_get_int(structure, "width", &width);
        gst_structure_get_int(structure, "height", &height);
        // g_print("Received frame with width: %d, height: %d.\n", width, height);
    }
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstMapInfo map_info;
    if (gst_buffer_map(buffer, &map_info, GST_MAP_READ)) {
        const guint8* raw_data = map_info.data;
        const gchar* format_string = gst_structure_get_string(gst_caps_get_structure(caps, 0), "format");
        std::string pixel_format(format_string ? format_string : "");
        cv::Mat frame;
        if (pixel_format == "BGR") {
            frame = cv::Mat(height, width, CV_8UC3, const_cast<guint8*>(raw_data));
        } else if (pixel_format == "RGB") {
            frame = cv::Mat(height, width, CV_8UC3, const_cast<guint8*>(raw_data));
        } else if (pixel_format == "I420") {
            cv::Mat yuv(height + height / 2, width, CV_8UC1, const_cast<guint8*>(raw_data));
            cv::cvtColor(yuv, frame, cv::COLOR_YUV2BGR_I420);
        } else if (pixel_format == "NV12") {
            cv::Mat yuv(height + height / 2, width, CV_8UC1, const_cast<guint8*>(raw_data));
            cv::cvtColor(yuv, frame, cv::COLOR_YUV2BGR_NV12);
        } else {
            g_printerr("Unsupported pixel format: %s\n", pixel_format.c_str());
            gst_buffer_unmap(buffer, &map_info);
            gst_sample_unref(sample);
            return GST_FLOW_OK;
        }
        lastFrame_ = frame;
        frameAvailable_ = true;
        lock.unlock();
        condition_.notify_one();
        gst_buffer_unmap(buffer, &map_info);
    }
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

cv::Mat gstObject::getLastFrame() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return frameAvailable_; });
    frameAvailable_ = false;
    return lastFrame_;
}

void gstObject::decode() {
    mainLoop_ = g_main_loop_new(nullptr, FALSE);
    gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    g_main_loop_run(mainLoop_);
}