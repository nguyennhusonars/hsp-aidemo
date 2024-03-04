// TODO
// Add USE_GST macro
// Add X86 or AARCH64 macro

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <chrono>
#include <unistd.h>
#include <filesystem>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "json.hpp"
#include "FaceDet.hpp"
#include "FaceAlign.hpp"
#include "FaceRec.hpp"

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>

using json = nlohmann::json;

#define BUILD_X86
#define TEST_VIDEO
#define NUM_THREADS 8

#if defined BUILD_X86
#define FACEDET_MODEL_PATH "../models/det_500m_quantized.dlc"
#define FACEREC_MODEL_PATH "../models/w600k_r50_quantized.dlc"
#define DB_IMAGE_PATH "../models/images/"
#define DB_PATH "../models/db.txt"
#elif defined BUILD_AARCH64
#define FACEDET_MODEL_PATH "/home/sonnn/aipackage/det_500m_quantized.dlc"
#define FACEREC_MODEL_PATH "/home/sonnn/aipackage/w600k_r50_quantized.dlc"
#define DB_IMAGE_PATH "/home/sonnn/aipackage/images/"
#define DB_PATH "/home/sonnn/aipackage/db.txt"
#endif

typedef enum INPUT_TYPE {
    NONE = 0,
    VIDEO = 1,
    RTSP = 2
} INPUT_TYPE;

class gstObject {
   public:
    gstObject(std::string url, int inputType);
    ~gstObject();
    void startThread() {
        decodeThread_ = std::thread(&gstObject::decode, this);
    }
    void joinThread() {
        decodeThread_.join();
    }
    void detachThread() {
        decodeThread_.detach();
    }
    static GstFlowReturn onNewSampleStatic(GstElement* appsink, gpointer user_data) {
        return reinterpret_cast<gstObject*>(user_data)->onNewSample(appsink);
    }
    static void onEosStatic(GstBus* bus, GstMessage* msg, gpointer user_data) {
        g_print("End-of-Stream reached\n");
        g_main_loop_quit(reinterpret_cast<gstObject*>(user_data)->mainLoop_);
    }
    cv::Mat getLastFrame() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return frameAvailable_; });
        frameAvailable_ = false;
        return lastFrame_;
    }

   private:
    GstElement* pipeline_;
    GstAppSink* appsink_;
    GstBus* bus_;
    std::thread decodeThread_;
    std::mutex mutex_;
    std::condition_variable condition_;
    bool frameAvailable_ = false;
    cv::Mat lastFrame_;
    GMainLoop* mainLoop_;
    GstFlowReturn onNewSample(GstElement* appsink) {
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

    void decode() {
        mainLoop_ = g_main_loop_new(nullptr, FALSE);
        gst_element_set_state(pipeline_, GST_STATE_PLAYING);
        g_main_loop_run(mainLoop_);
    }
};

gstObject::gstObject(std::string url, int inputType) {
    gst_init(nullptr, nullptr);
    if (inputType == INPUT_TYPE::VIDEO) {
        std::string tmp = "uridecodebin uri=file://" + url +
                          " ! videoconvert ! appsink name=sink sync=false";
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

class aiObject {
   public:
    aiObject();
    ~aiObject();
    void startThread(gstObject& gstObj) {
        processThread_ = std::thread(&aiObject::run, this, std::ref(gstObj));
    }
    void joinThread() {
        processThread_.join();
    }
    void detachThread() {
        processThread_.detach();
    }
    int loadDB(std::string jsonFilePath);
    int addDB(std::string imgFilePath);
    int run(gstObject& gstObj);
    std::unique_ptr<SCRFD> det = nullptr;
    std::unique_ptr<SnpeInsightface> rec = nullptr;

   private:
    std::vector<std::string> ids;
    cv::Mat feat;
    std::thread processThread_;
};

aiObject::aiObject() {
    SetAdspLibraryPath();
    det = std::make_unique<SCRFD>();
    rec = std::make_unique<SnpeInsightface>();
    zdl::DlSystem::Runtime_t runtime = DSP_RUNTIME;
    runtime = checkRuntime(runtime);
    det->load(FACEDET_MODEL_PATH, runtime);
    rec->load(FACEREC_MODEL_PATH, runtime);
}

aiObject::~aiObject() {
}

int aiObject::loadDB(std::string jsonFilePath) {
    std::ifstream jsonFile(jsonFilePath);
    if (!jsonFile.is_open()) {
        std::cout << "Database was not exist." << std::endl;
        return 1;
    }
    std::vector<std::vector<float>> emb;
    std::string line;
    while (std::getline(jsonFile, line)) {
        json jsonData = json::parse(line);
        std::string id = jsonData["id"];
        ids.push_back(id);
        std::vector<float> embVector = jsonData["emb"];
        emb.push_back(embVector);
    }
    for (uint32_t i = 0; i < emb.size(); i++) {
        feat.push_back(emb[i]);
    }
    feat = feat.reshape(1, emb.size());
    jsonFile.close();
    std::cout << "Database has " << ids.size() << " faces" << std::endl;
    return 0;
}

int aiObject::run(gstObject& gstObj) {
    XInitThreads();
    while (true) {
        cv::Mat img = gstObj.getLastFrame().clone();
        std::vector<FaceObject> faces;
        std::cout << "=============================================" << std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();
        det->execDetect(img, faces);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        std::cout << "FaceDet takes " << ms_int.count() << "ms\n";
        std::cout << "Frame has " << faces.size() << " faces" << std::endl;

        for (uint32_t i = 0; i < faces.size(); i++) {
            auto t1 = std::chrono::high_resolution_clock::now();
            FaceObject face = faces[i];
            float v2[5][2] =
                {
                    {float(face.point[0].x), float(face.point[0].y)},
                    {float(face.point[1].x), float(face.point[1].y)},
                    {float(face.point[2].x), float(face.point[2].y)},
                    {float(face.point[3].x), float(face.point[3].y)},
                    {float(face.point[4].x), float(face.point[4].y)}};
            cv::Mat src(5, 2, CV_32FC1, norm_face);
            cv::Mat dst(5, 2, CV_32FC1, v2);
            cv::Mat m = similarTransform(dst, src);
            cv::Size size(112, 112);
            cv::Mat aligned(112, 112, CV_32FC3);
            cv::Mat transfer = m(cv::Rect(0, 0, 3, 2));
            cv::warpAffine(img, aligned, transfer, size, 1, 0, 0);
            cv::Mat output = rec->execRecog(aligned);
            if (!feat.empty()) {
                class_info result = rec->classify(output, feat);
                std::cout << (result.min_distance < 0.95 ? ids[result.index] : "Unknown") << " " << result.min_distance << std::endl;
            }

            for (int j = 0; j < 5; j++) {
                cv::Point p(face.point[j].x, face.point[j].y);
                cv::circle(img, p, 2, cv::Scalar(0, 0, 255), -1);
            }
            cv::rectangle(img, face.rect, cv::Scalar(0, 255, 0), 2);

            // cv::imshow("aidemo", img);
            // cv::waitKey(10);
            auto t2 = std::chrono::high_resolution_clock::now();
            auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
            std::cout << "FaceRec takes " << ms_int.count() << "ms\n";
        }
    }
    // }
    return 0;
}

int aiObject::addDB(std::string imgFilePath) {
    SetAdspLibraryPath();
    zdl::DlSystem::Runtime_t runtime = DSP_RUNTIME;
    runtime = checkRuntime(runtime);

    std::string folderPath = DB_IMAGE_PATH;
    std::ofstream jsonFile;
    jsonFile.open(DB_PATH, std::ios::app);

    SCRFD* det = new SCRFD();
    SnpeInsightface* rec = new SnpeInsightface();

    det->load(FACEDET_MODEL_PATH, runtime);
    rec->load(FACEREC_MODEL_PATH, runtime);

    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        std::vector<FaceObject> faces;
        std::string filePath = entry.path().string();
        std::string fileName = entry.path().filename().string();
        cv::Mat img = cv::imread(filePath);
        copyMakeBorder(img, img, img.rows / 10, img.rows / 10, img.cols / 5, img.cols / 5, cv::BORDER_CONSTANT, cv::Scalar(0));
        det->execDetect(img, faces, 0.5, 0.45);

        if (faces.size() != 1) {
            std::cout << "-- " << fileName << " currently has " << faces.size() << " face. Must has exactly 1 face" << std::endl;
            std::filesystem::remove(filePath);
            continue;
        } else {
            FaceObject face = faces[0];
            float v2[5][2] =
                {
                    {float(face.point[0].x), float(face.point[0].y)},
                    {float(face.point[1].x), float(face.point[1].y)},
                    {float(face.point[2].x), float(face.point[2].y)},
                    {float(face.point[3].x), float(face.point[3].y)},
                    {float(face.point[4].x), float(face.point[4].y)}};
            cv::Mat src(5, 2, CV_32FC1, norm_face);
            cv::Mat dst(5, 2, CV_32FC1, v2);
            cv::Mat m = similarTransform(dst, src);
            cv::Size size(112, 112);
            cv::Mat aligned(112, 112, CV_32FC3);
            cv::Mat transfer = m(cv::Rect(0, 0, 3, 2));
            cv::warpAffine(img, aligned, transfer, size, 1, 0, 0);
            cv::Mat output = rec->execRecog(aligned);

            json resultJson;
            std::vector<float> vec;
            vec.assign(output.begin<float>(), output.end<float>());
            resultJson["emb"] = vec;
            resultJson["id"] = fileName.substr(0, fileName.find_last_of('.'));
            jsonFile << resultJson.dump() << std::endl;
            std::cout << "-- " << fileName << " added to db successfully" << std::endl;
        }
    }
    jsonFile.close();
    return 0;
}

class threadHandler {
   public:
    aiObject* aiObj = nullptr;
    gstObject* gstObj = nullptr;
    void startThread() {
        aiObj->startThread(*gstObj);
        gstObj->startThread();
    }
    void joinThread() {
        aiObj->joinThread();
        gstObj->joinThread();
    }
    void detachThread() {
        aiObj->detachThread();
        gstObj->detachThread();
    }

   private:
};

int main(int argc, char* argv[]) {
    if (argc > 1) {
        std::string command = argv[1];
        if (command == "--add") {
            threadHandler singleThread;
            singleThread.aiObj = new aiObject();
            singleThread.aiObj->addDB(DB_IMAGE_PATH);
            return 0;
        } else if (command == "--remove") {
            return 0;
        } else {
            return 0;
        }
    } else {
        std::vector<std::thread> streamThreads;
        for (int i = 0; i < NUM_THREADS; i++) {
            threadHandler singleThread;

            singleThread.aiObj = new aiObject();
            singleThread.aiObj->loadDB(DB_PATH);
            singleThread.gstObj = new gstObject("/home/vboxuser/Documents/aipackage/models/frtest.mp4", INPUT_TYPE::VIDEO);
            // streamThreads.emplace_back(std::thread(&threadHandler::startThread, singleThread));
            // streamThreads.emplace_back(std::thread(&threadHandler::joinThread, singleThread));
            singleThread.startThread();
            // singleThread.joinThread();
        }
        for (auto& thread : streamThreads) {
            // thread.detach();
            thread.join();
        }
    }
    // sleep(2000);
    return 0;
}