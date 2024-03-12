#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <memory.h>

#include "FaceDet.hpp"
#include "FaceAlign.hpp"
#include "FaceRec.hpp"

#include "gstObject.hpp"

#include "json.hpp"
using json = nlohmann::json;

#define FACEDET_MODEL_PATH "../models/det_500m_quantized.dlc"
#define FACEREC_MODEL_PATH "../models/w600k_r50_quantized.dlc"
#define DB_IMAGE_PATH "../models/images/"
#define DB_PATH "../models/db.txt"

class aiObject {
   public:
    aiObject(int threadID);
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
    int threadID;
    std::vector<FaceObject> faceObjs;
    cv::Mat img;

   private:
    std::vector<std::string> ids;
    cv::Mat feat;
    std::thread processThread_;
};