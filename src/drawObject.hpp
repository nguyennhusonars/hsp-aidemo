#include <unistd.h>
#include "aiObject.hpp"

class drawObject {
    public:
    drawObject(int i);
    ~drawObject();
    void startThread(aiObject& aiObj) {
        drawThread_ = std::thread(&drawObject::cairodraw, this, std::ref(aiObj));
    }
    void joinThread() {
        drawThread_.join();
    }
    int draw(aiObject& aiObj);
    int threadID;

    private:
    std::thread drawThread_;
    void cairodraw(aiObject &aiObj);
};

drawObject::drawObject(int i) {
    threadID = i;
}

drawObject::~drawObject() {

}

void drawObject::cairodraw(aiObject &aiObj) {
    while (true) {
        std::cout << "Drawing " <<aiObj.faceObjs.size()<<std::endl;
        for (int i = 0; i < aiObj.faceObjs.size(); i++) {
            for (int j = 0; j < 5; j++) {
                cv::Point p(aiObj.faceObjs[i].point[j].x, aiObj.faceObjs[i].point[j].y);
                cv::circle(aiObj.img, p, 2, cv::Scalar(0, 0, 255), -1);
            }
            cv::rectangle(aiObj.img, aiObj.faceObjs[i].rect, cv::Scalar(0, 255, 0), 2);
            // cv::imshow("aidemo", aiObj.img);
            // cv::waitKey(10);
        }
        usleep(30000);
    }
}