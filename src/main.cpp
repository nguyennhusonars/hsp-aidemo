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

#include "drawObject.hpp"

#define BUILD_X86
#define TEST_VIDEO
#define NUM_THREADS 1

int main(int argc, char* argv[]) {
    if (argc > 1) {
        std::string command = argv[1];
        if (command == "--add") {
            aiObject* aiObj = new aiObject(0);
            aiObj->addDB(DB_IMAGE_PATH);
            return 0;
        } else if (command == "--remove") {
            return 0;
        } else {
            return 0;
        }
    } else {
        std::vector<aiObject*> aiObj(NUM_THREADS);
        std::vector<gstObject*> gstObj(NUM_THREADS);
        std::vector<drawObject*> drawObj(NUM_THREADS);
        // std::unique_ptr<SCRFD> det = nullptr;
        // det = std::make_unique<SCRFD>();

        for (int i = 0; i < NUM_THREADS; i++) {
            gstObj[i] = new gstObject("/home/vboxuser/hsp-aidemo/models/frtest.mp4", INPUT_TYPE::VIDEO, i);
            aiObj[i] = new aiObject(i);
            aiObj[i]->loadDB(DB_PATH);
            // drawObj[i] = new  drawObject(i);

            gstObj[i]->startThread();
            aiObj[i]->startThread(*gstObj[i]);
            // drawObj[i]->startThread(*aiObj[i]);
        }
        for (int i = 0; i < NUM_THREADS; i++) {
            gstObj[i]->joinThread();
            aiObj[i]->joinThread();
            // drawObj[i]->joinThread();
        }
    }
    sleep(2000);
    return 0;
}