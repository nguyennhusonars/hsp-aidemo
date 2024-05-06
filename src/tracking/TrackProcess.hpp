#ifndef TRACKPROCESS_HPP
#define TRACKPROCESS_HPP

#include <opencv2/opencv.hpp>
#include <set>
#include "Hungarian.hpp"
#include "KalmanTracker.hpp"

struct FaceObject {
    cv::Rect_<float> rect;
    cv::Point2f point[5];
    float scores, vert_ratio_1, vert_ratio_2, eye_ratio_1, eye_ratio_2, nose_ratio;
    bool quality = true;
    std::string label;
};

typedef struct BoxInfo {
    int x1;
    int y1;
    int x2;
    int y2;
    float score;
    int label;
} BoxInfo;

typedef struct TrackingBox {
    int trackID = -1;
    cv::Rect box;
    int status = 0;  // 1: true , 0: false,
    std::string mappedID;
    std::string mappedName;

    std::string label;
    std::vector<cv::Point2f> points;
} TrackingBox;

class TrackProcess {
   private:
    int max_age = 30;
    int min_hits = 3;
    double iouThreshold = 0.15;

    int latestTrackId = -1;
    std::vector<KalmanTracker> trackers;
    std::vector<cv::Rect> predictedBoxes;
    vector<vector<double>> iouMatrix;
    vector<int> assignment;
    unsigned int trkNum = 0;
    unsigned int detNum = 0;
    set<int> unmatchedDetections;
    set<int> unmatchedTrajectories;
    set<int> allItems;
    set<int> matchedItems;
    vector<cv::Point> matchedPairs;

   public:
    std::vector<KalmanTracker> getKalmanTrackers() { return trackers; };
    std::vector<cv::Rect> trackers_ori;
    std::map<int, std::vector<cv::Point>> idToPointHistory;
    std::vector<TrackingBox> frameTrackingResult;
    std::vector<TrackingBox> frameTrackingResult_ori;
    void sortTracking(std::vector<TrackingBox> &detData);
	std::map<int, std::string> listTr;
    int totalObjs = 0;
    // void mappingKalmanObj(std::vector<TrackingBox> &detData);
};

std::vector<TrackingBox> faceToTracking(std::vector<FaceObject> objs);
std::vector<TrackingBox> objToTracking(std::vector<BoxInfo> objs);

#endif