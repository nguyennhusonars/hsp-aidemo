#include <filesystem>
#include "aiObject.hpp"

#define RECOGNITION_THRESHOLD 0.95

aiObject::aiObject(int i) {
    threadID = i;
    SetAdspLibraryPath();
    det = std::make_unique<SCRFD>();
    rec = std::make_unique<SnpeInsightface>();
    zdl::DlSystem::Runtime_t runtime = DSP_RUNTIME;
    runtime = checkRuntime(runtime);
    det->load(FACEDET_MODEL_PATH, runtime);
    rec->load(FACEREC_MODEL_PATH, runtime);
}

aiObject::~aiObject() {}

int aiObject::loadDB(std::string jsonFilePath) {
    std::ifstream jsonFile(jsonFilePath);
    if (!jsonFile.is_open()) {
        std::cout << "Database was not exist." << std::endl;
        return 1;
    }
    std::vector<std::vector<float>> emb;
    std::string line;
    while (std::getline(jsonFile, line)) {
        json jsonData = json::parse(line);
        std::string id = jsonData["id"];
        ids.push_back(id);
        std::vector<float> embVector = jsonData["emb"];
        emb.push_back(embVector);
    }
    for (uint32_t i = 0; i < emb.size(); i++) {
        feat.push_back(emb[i]);
    }
    feat = feat.reshape(1, emb.size());
    jsonFile.close();
    std::cout << "Database has " << ids.size() << " faces" << std::endl;
    return 0;
}

int aiObject::run(gstObject& gstObj) {
    // XInitThreads();
    while (true) {
        img = gstObj.getLastFrame().clone();
        std::vector<FaceObject> faces;
        std::cout << "=============================================" << std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();
        det->execDetect(img, faces);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        std::cout << "Thread " << threadID << ": FaceDet takes " << ms_int.count() << "ms\n";
        std::cout << "Thread " << threadID << ": Frame has " << faces.size() << " faces" << std::endl;

        for (uint32_t i = 0; i < faces.size(); i++) {
            faceObjs.clear();
            auto t1 = std::chrono::high_resolution_clock::now();
            FaceObject face = faces[i];
            float v2[5][2] = {{float(face.point[0].x), float(face.point[0].y)},
                              {float(face.point[1].x), float(face.point[1].y)},
                              {float(face.point[2].x), float(face.point[2].y)},
                              {float(face.point[3].x), float(face.point[3].y)},
                              {float(face.point[4].x), float(face.point[4].y)}};
            cv::Mat src(5, 2, CV_32FC1, norm_face);
            cv::Mat dst(5, 2, CV_32FC1, v2);
            cv::Mat m = similarTransform(dst, src);
            cv::Size size(112, 112);
            cv::Mat aligned(112, 112, CV_32FC3);
            cv::Mat transfer = m(cv::Rect(0, 0, 3, 2));
            cv::warpAffine(img, aligned, transfer, size, 1, 0, 0);
            cv::Mat output = rec->execRecog(aligned);
            if (!feat.empty()) {
                class_info result = rec->classify(output, feat);
                if (result.min_distance < RECOGNITION_THRESHOLD) {
                    face.label = ids[result.index];
                } else {
                    face.label = "Unknown";
                }
                std::cout << face.label << " " << result.min_distance << std::endl;
            }
            faceObjs.push_back(face);
            auto t2 = std::chrono::high_resolution_clock::now();
            auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
            std::cout << "Thread " << threadID << ": FaceRec takes " << ms_int.count() << "ms\n";
        }
    }
    // }
    return 0;
}

int aiObject::addDB(std::string imgFilePath) {
    SetAdspLibraryPath();
    zdl::DlSystem::Runtime_t runtime = DSP_RUNTIME;
    runtime = checkRuntime(runtime);

    std::string folderPath = DB_IMAGE_PATH;
    std::ofstream jsonFile;
    jsonFile.open(DB_PATH, std::ios::app);

    SCRFD* det = new SCRFD();
    SnpeInsightface* rec = new SnpeInsightface();

    det->load(FACEDET_MODEL_PATH, runtime);
    rec->load(FACEREC_MODEL_PATH, runtime);

    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        std::vector<FaceObject> faces;
        std::string filePath = entry.path().string();
        std::string fileName = entry.path().filename().string();
        cv::Mat img = cv::imread(filePath);
        copyMakeBorder(img, img, img.rows / 10, img.rows / 10, img.cols / 5, img.cols / 5, cv::BORDER_CONSTANT,
                       cv::Scalar(0));
        det->execDetect(img, faces, 0.5, 0.45);

        if (faces.size() != 1) {
            std::cout << "-- " << fileName << " currently has " << faces.size() << " face. Must has exactly 1 face"
                      << std::endl;
            std::filesystem::remove(filePath);
            continue;
        } else {
            FaceObject face = faces[0];
            float v2[5][2] = {{float(face.point[0].x), float(face.point[0].y)},
                              {float(face.point[1].x), float(face.point[1].y)},
                              {float(face.point[2].x), float(face.point[2].y)},
                              {float(face.point[3].x), float(face.point[3].y)},
                              {float(face.point[4].x), float(face.point[4].y)}};
            cv::Mat src(5, 2, CV_32FC1, norm_face);
            cv::Mat dst(5, 2, CV_32FC1, v2);
            cv::Mat m = similarTransform(dst, src);
            cv::Size size(112, 112);
            cv::Mat aligned(112, 112, CV_32FC3);
            cv::Mat transfer = m(cv::Rect(0, 0, 3, 2));
            cv::warpAffine(img, aligned, transfer, size, 1, 0, 0);
            cv::Mat output = rec->execRecog(aligned);

            json resultJson;
            std::vector<float> vec;
            vec.assign(output.begin<float>(), output.end<float>());
            resultJson["emb"] = vec;
            resultJson["id"] = fileName.substr(0, fileName.find_last_of('.'));
            jsonFile << resultJson.dump() << std::endl;
            std::cout << "-- " << fileName << " added to db successfully" << std::endl;
        }
    }
    jsonFile.close();
    return 0;
}
