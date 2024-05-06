#ifndef GST_OBJ_HPP
#define GST_OBJ_HPP

#include "FaceAlign.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "FaceDet.hpp"
#include "FaceRec.hpp"
#include "yolonas.hpp"

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>
#include "cairo/cairo.h"

#include <thread>
#include <condition_variable>

#define FACEDET_MODEL_PATH "../models/det_500m_quantized.dlc"
#define FACEREC_MODEL_PATH "../models/w600k_r50_quantized.dlc"
#define YOLOV8_MODEL_PATH "../models/yolo_nas_s_v213_quantized.dlc"
#define DB_IMAGE_PATH "../models/images/"
#define DB_PATH "../models/db.txt"
#define NUM_THREADS 1

// Use below flag to run aidemo app for Face Recognition Application
#define TEST_FR

// or use below flag to run aidemo app for Object Detection Application
// #define TEST_YOLO

typedef enum INPUT_TYPE { NONE = 0, VIDEO = 1, RTSP = 2 } INPUT_TYPE;

class gstObject : public TrackProcess {
   public:
    gstObject(std::string url, int inputType, int threadID);
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
    static gboolean onDrawingStatic(GstElement* overlay, cairo_t *cr, gpointer user_data) {
        return reinterpret_cast<gstObject*>(user_data)->onDrawing(overlay, cr);
    }
    static void onEosStatic(GstBus* bus, GstMessage* msg, gpointer user_data) {
        g_print("End-of-Stream reached\n");
        g_main_loop_quit(reinterpret_cast<gstObject*>(user_data)->mainLoop_);
    }
    cv::Mat getLastFrame();
    int threadID;

    GstElement* pipeline_;
    GstAppSink* appsink_;
    GstElement* overlay_;
    GstBus* bus_;
    std::thread decodeThread_;
    GMainLoop* mainLoop_;
    gboolean onDrawing(GstElement* overlay, cairo_t* cr);
    GstFlowReturn onNewSample(GstElement* appsink);
    void decode();
    GstBuffer* buffer;
    GstMapInfo map_info;

    int loadDB(std::string jsonFilePath);
    int addDB(std::string imgFilePath);
    std::unique_ptr<SCRFD> det = std::make_unique<SCRFD>();
    std::unique_ptr<SnpeInsightface> rec = std::make_unique<SnpeInsightface>();
    std::unique_ptr<yolonas> objDet = std::make_unique<yolonas>();
    // std::vector<FaceObject> faceObjs;
    // FaceObject currentface;
    std::vector<TrackingBox> faceObjs;
    TrackingBox currentface;
    cv::Mat img;

   private:
    std::vector<std::string> ids;
    cv::Mat feat;
};

#endif