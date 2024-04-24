#include "TrackProcess.hpp"

std::vector<TrackingBox> faceToTracking(std::vector<FaceObject> objs) {
    std::vector<TrackingBox> convertVector;
    for (unsigned int i = 0; i < objs.size(); i++) {
        TrackingBox tempTrackingBox;
        tempTrackingBox.box.x = objs[i].rect.x;
        tempTrackingBox.box.y = objs[i].rect.y;
        tempTrackingBox.box.width = (objs[i].rect.width);
        tempTrackingBox.box.height = (objs[i].rect.height);
        convertVector.push_back(tempTrackingBox);
    }
    return convertVector;
}

std::vector<TrackingBox> objToTracking(std::vector<BoxInfo> objs) {
    std::vector<TrackingBox> convertVector;
    for (unsigned int i = 0; i < objs.size(); i++) {
        TrackingBox tempTrackingBox;
        tempTrackingBox.box.x = objs[i].x1;
        tempTrackingBox.box.y = objs[i].y1;
        tempTrackingBox.box.width = (objs[i].x2 - objs[i].x1);
        tempTrackingBox.box.height = (objs[i].y2 - objs[i].y1);
        convertVector.push_back(tempTrackingBox);
    }
    return convertVector;
}

double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt) {
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;

    if (un < DBL_EPSILON) return 0;

    return (double)(in / un);
}

void mapDetObj(TrackingBox frameTrackingResult, std::vector<TrackingBox> &dets) {
    int pos = -1;
    for (uint32_t i = 0; i < dets.size(); i++) {
        double maxIOU = 0.3;
        cv::Rect bestMatch(0, 0, 0, 0);
        double iou = GetIOU(dets[i].box, frameTrackingResult.box);
        if (iou > maxIOU) {
            maxIOU = iou;
            pos = i;
        }
        if (pos != -1) {
            dets[pos].trackID = frameTrackingResult.trackID;
        }
    }
}

void TrackProcess::sortTracking(std::vector<TrackingBox> &detData) {
    if (trackers.size() == 0) {
        for (unsigned int i = 0; i < detData.size(); i++) {
            KalmanTracker trk = KalmanTracker(detData[i].box);
            trackers.push_back(trk);
            // trackers_ori.push_back(detData[i].box);
        }
    }
    predictedBoxes.clear();
    for (auto it = trackers.begin(); it != trackers.end();) {
        cv::Rect pBox = (*it).predict();
        if (pBox.x >= 0 && pBox.y >= 0) {
            predictedBoxes.push_back(pBox);
            it++;
        } else {
            it = trackers.erase(it);
        }
    }
    trkNum = predictedBoxes.size();
    if (trkNum == 0) return;
    detNum = detData.size();
    iouMatrix.clear();
    iouMatrix.resize(trkNum, vector<double>(detNum, 0));
    for (unsigned int i = 0; i < trkNum; i++)  // compute iou matrix as a distance matrix
    {
        for (unsigned int j = 0; j < detNum; j++) {
            // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
            iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detData[j].box);
        }
    }
    // solve the assignment problem using hungarian algorithm.
    // the resulting assignment is [track(prediction) : detection], with len=preNum
    HungarianAlgorithm HungAlgo;
    assignment.clear();
    HungAlgo.Solve(iouMatrix, assignment);
    unmatchedTrajectories.clear();
    unmatchedDetections.clear();
    allItems.clear();
    matchedItems.clear();
    if (detNum > trkNum)  //	there are unmatched detections
    {
        for (unsigned int n = 0; n < detNum; n++) allItems.insert(n);

        for (unsigned int i = 0; i < trkNum; ++i) matchedItems.insert(assignment[i]);

        set_difference(allItems.begin(), allItems.end(), matchedItems.begin(), matchedItems.end(), insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
    } else if (detNum < trkNum)  // there are unmatched trajectory/predictions
    {
        for (unsigned int i = 0; i < trkNum; ++i)
            if (assignment[i] == -1)  // unassigned label will be set as -1 in the assignment algorithm
                unmatchedTrajectories.insert(i);
    } else
        ;
    // filter out matched with low IOU
    matchedPairs.clear();
    for (unsigned int i = 0; i < trkNum; ++i) {
        if (assignment[i] < 0 || assignment[i] > trkNum)  // pass over invalid values
            continue;
        if (1 - iouMatrix[i][assignment[i]] < iouThreshold) {
            unmatchedTrajectories.insert(i);
            unmatchedDetections.insert(assignment[i]);
        } else
            matchedPairs.push_back(cv::Point(i, assignment[i]));
    }
    // // 3.3. updating trackers
    // // update matched trackers with assigned detections.
    // // each prediction is corresponding to a tracker
    int detIdx, trkIdx;
    for (unsigned int i = 0; i < matchedPairs.size(); i++) {
        trkIdx = matchedPairs[i].x;
        detIdx = matchedPairs[i].y;

        trackers[trkIdx].update(detData[detIdx].box);
    }
    // // create and initialise new trackers for unmatched detections
    for (auto umd : unmatchedDetections) {
        KalmanTracker tracker = KalmanTracker(detData[umd].box);
        trackers.push_back(tracker);
    }
    // // get trackers' output
    frameTrackingResult.clear();
    for (auto it = trackers.begin(); it != trackers.end();) {
        if (((*it).m_time_since_update < 1) && ((*it).m_hit_streak >= min_hits)) {
            TrackingBox res;
            res.box = (*it).get_state();
            res.trackID = (*it).m_id;
            frameTrackingResult.push_back(res);
            it++;
        } else
            it++;

        // remove dead tracklet
        if (it != trackers.end() && (*it).m_time_since_update > max_age) it = trackers.erase(it);
    }
    for (auto tb : frameTrackingResult)
        printf("TrackID %d, x: %d, y: %d, w: %d, h: %d\n", tb.trackID, tb.box.x, tb.box.y, tb.box.width, tb.box.height);
}