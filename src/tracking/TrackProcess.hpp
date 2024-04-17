#ifndef TRACKPROCESS_HPP
#define TRACKPROCESS_HPP

#include <opencv2/opencv.hpp>
#include <set>
#include "Hungarian.hpp"
#include "KalmanTracker.hpp"

typedef struct TrackingBox {
    int trackID = -1;
    cv::Rect box;
    int status = 0;  // 1: true , 0: false,
    std::string mappedID;
    std::string mappedName;
} TrackingBox;

class TrackProcess {
   private:
    int max_age = 60;
    int min_hits = 3;
    double iouThreshold = 0.3;

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
    // void mappingKalmanObj(std::vector<TrackingBox> &detData);
};

#endif