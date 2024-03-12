#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <thread>
#include <condition_variable>

typedef enum INPUT_TYPE {
    NONE = 0,
    VIDEO = 1,
    RTSP = 2
} INPUT_TYPE;

class gstObject {
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
    static void onEosStatic(GstBus* bus, GstMessage* msg, gpointer user_data) {
        g_print("End-of-Stream reached\n");
        g_main_loop_quit(reinterpret_cast<gstObject*>(user_data)->mainLoop_);
    }
    cv::Mat getLastFrame();
    int threadID;

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
    GstFlowReturn onNewSample(GstElement* appsink);
    void decode();
};