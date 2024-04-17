#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <chrono>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "gstObject.hpp"

#define MAIN

#define BUILD_X86
#define TEST_VIDEO

std::vector<std::string> rtspLists = {"rtsp://192.169.1.53/stream1", "rtsp://192.169.1.53/stream1", "rtsp://192.169.1.53/stream1", "rtsp://192.169.1.53/stream1",
                                      "rtsp://192.169.1.53/stream1", "rtsp://192.169.1.53/stream1", "rtsp://192.169.1.53/stream1", "rtsp://192.169.1.53/stream1"};

std::vector<std::string> videoLists = {"/home/demo/hsp-aidemo-master/models/frtest.mp4", "/home/demo/hsp-aidemo-master/models/testFR.mp4", "/home/demo/hsp-aidemo-master/models/frtest.mp4", "/home/demo/hsp-aidemo-master/models/frtest.mp4",
                                       "/home/demo/hsp-aidemo-master/models/frtest.mp4", "/home/demo/hsp-aidemo-master/models/frtest.mp4", "/home/demo/hsp-aidemo-master/models/frtest.mp4", "/home/demo/hsp-aidemo-master/models/frtest.mp4"};

#ifdef MAIN
int main(int argc, char* argv[]) {
    if (argc > 1) {
        std::string command = argv[1];
        if (command == "--add") {
            gstObject* gstObj = new gstObject("/home/demo/hsp-aidemo-master/models/testFR.mp4", INPUT_TYPE::VIDEO, 0);
            gstObj->addDB(DB_IMAGE_PATH);
            return 0;
        } else if (command == "--remove") {
            return 0;
        } else {
            return 0;
        }
    } else {
        std::vector<gstObject*> gstObj(NUM_THREADS);
        for (int i = 0; i < NUM_THREADS; i++) {
            gstObj[i] = new gstObject(videoLists[i], INPUT_TYPE::VIDEO, i);
            gstObj[i]->loadDB(DB_PATH);
            gstObj[i]->startThread();
        }
        for (int i = 0; i < NUM_THREADS; i++) {
            gstObj[i]->joinThread();
        }
    }
    sleep(2000);
    return 0;
}

#elif defined BENCHMARK_FD
int main(int argc, char* argv[]) {
    std::unique_ptr<SCRFD> det = std::make_unique<SCRFD>();
    det->load(FACEDET_MODEL_PATH, DSP_RUNTIME);
    cv::Mat img = cv::imread("../models/testfd.jpg");
    cv::resize(img, img, cv::Size(640, 640));
    std::vector<FaceObject> faces;
    int total_time = 0;
    int loop_count = 0;
    while (true) {
        auto t1 = std::chrono::high_resolution_clock::now();
        det->execDetect(img, faces);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        total_time += ms_int.count();
        loop_count++;
        // std::cout << "Average fd time cost: " << (float)total_time / loop_count << " ms\n";
    }
    return 0;
}

#elif defined BENCHMARK_FR
int main(int argc, char* argv[]) {
    std::unique_ptr<SnpeInsightface> rec = std::make_unique<SnpeInsightface>();
    rec->load(FACEREC_MODEL_PATH, DSP_RUNTIME);
    cv::Mat img = cv::imread("../models/testfr.jpg");
    int total_time = 0;
    int loop_count = 0;
    while (true) {
        // auto t1 = std::chrono::high_resolution_clock::now();
        // cv::Mat output = rec->execRecog(img);
        // auto t2 = std::chrono::high_resolution_clock::now();
        // auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        // total_time += ms_int.count();
        // loop_count++;
        // // std::cout << "Average fr time cost: " << (float)total_time / loop_count << " ms\n";
        sleep(2);
    }
    return 0;
}
#endif