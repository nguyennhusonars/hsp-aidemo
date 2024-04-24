#ifndef SCRFD_H
#define SCRFD_H

#include <vector>
#include <memory>
#include <string>

#include <opencv2/core/core.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "SnpeCommLib.hpp"
#include "DlSystem/ITensor.hpp"
#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/RuntimeList.hpp"
#include "DlSystem/StringList.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "TrackProcess.hpp"

#define input_width 640
#define input_height 640

enum { UNKNOWN,
       USERBUFFER_FLOAT,
       USERBUFFER_TF8,
       ITENSOR,
       USERBUFFER_TF16 };

typedef struct scrfd_params {
    std::unique_ptr<zdl::SNPE::SNPE> scrfd;
} scrfd_params;

class SCRFD {
   public:
    SCRFD();
    ~SCRFD();
    int load(std::string model_path, zdl::DlSystem::Runtime_t targetDevice);
    int execDetect(cv::Mat rgb, std::vector<FaceObject>& faceobjects, float scores_threshold = 0.5f, float nms_threshold = 0.45f);

   private:
    scrfd_params* s;
    bool has_kps = true;
};

#endif  // SCRFD_H
