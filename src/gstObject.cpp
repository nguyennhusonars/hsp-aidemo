#include <gst/video/videooverlay.h>
#include <filesystem>
#include <unistd.h>

#include "gstObject.hpp"
#include "json.hpp"

using json = nlohmann::json;
#define RECOGNITION_THRESHOLD 0.95

#ifdef TEST_FR
int gstObject::loadDB(std::string jsonFilePath) {
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

cv::Mat resizeKeepAspectRatio(const cv::Mat& input, const cv::Size& dstSize, const cv::Scalar& bgcolor) {
    cv::Mat output;
    double h1 = dstSize.width * (input.rows / (double)input.cols);
    double w2 = dstSize.height * (input.cols / (double)input.rows);
    if (h1 <= dstSize.height) {
        cv::resize(input, output, cv::Size(dstSize.width, h1));
    } else {
        cv::resize(input, output, cv::Size(w2, dstSize.height));
    }
    int top = (dstSize.height - output.rows) / 2;
    int down = (dstSize.height - output.rows + 1) / 2;
    int left = (dstSize.width - output.cols) / 2;
    int right = (dstSize.width - output.cols + 1) / 2;
    cv::copyMakeBorder(output, output, top, down, left, right, cv::BORDER_CONSTANT, bgcolor);
    return output;
}

int gstObject::addDB(std::string imgFilePath) {
    SetAdspLibraryPath();
    zdl::DlSystem::Runtime_t runtime = DSP_RUNTIME;
    runtime = checkRuntime(runtime);
    std::string folderPath = DB_IMAGE_PATH;
    std::ofstream jsonFile;
    jsonFile.open(DB_PATH, std::ios::app);
    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        std::vector<FaceObject> faces;
        std::string filePath = entry.path().string();
        std::string fileName = entry.path().filename().string();
        cv::Mat img = cv::imread(filePath);
        img = resizeKeepAspectRatio(img, cv::Size(640, 640), cv::Scalar(0, 0, 0));
        // copyMakeBorder(img, img, img.rows / 10, img.rows / 10, img.cols / 4, img.cols / 4, cv::BORDER_CONSTANT,
        //                cv::Scalar(0));
        det->execDetect(img, faces, 0.3, 0.45);
        if (faces.size() != 1) {
            std::cout << "-- " << fileName << " currently has " << faces.size() << " face. Must has exactly 1 face" << std::endl;
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
#endif

gstObject::gstObject(std::string url, int inputType, int i) {
    SetAdspLibraryPath();
    zdl::DlSystem::Runtime_t runtime = DSP_RUNTIME;
    runtime = checkRuntime(runtime);
    det->load(FACEDET_MODEL_PATH, runtime);
    rec->load(FACEREC_MODEL_PATH, runtime);
    objDet->load(YOLOV8_MODEL_PATH, runtime);
    threadID = i;
    gst_init(nullptr, nullptr);
    if (inputType == INPUT_TYPE::VIDEO) {
        std::string tmp = "filesrc location=" + url +
                          " ! qtdemux ! queue ! h264parse ! qtivdec skip-frames=yes turbo=yes ! tee name=t \
                            t. ! queue ! qtivtransform ! video/x-raw,format=NV12 ! appsink name=sink sync=false \
                            t. ! queue ! qtivtransform ! cairooverlay name=overlay ! waylandsink sync=false x=" +
                          std::to_string(i % 3 * 480) + " y=" + std::to_string(i / 3 * 240) + " width=480 height=240";
        pipeline_ = gst_parse_launch(tmp.c_str(), nullptr);
    } else if (inputType == INPUT_TYPE::RTSP) {
        std::string tmp = "rtspsrc location=" + url +
                          " ! rtph264depay ! h264parse ! qtivdec skip-frames=yes turbo=yes ! tee name=t \
                            t. ! queue ! qtivtransform ! video/x-raw,format=NV12 ! appsink name=sink sync=false \
                            t. ! queue ! qtivtransform ! cairooverlay name=overlay ! waylandsink sync=false x=" +
                          std::to_string(i % 3 * 480) + " y=" + std::to_string(i / 3 * 240) + " width=480 height=240";
        pipeline_ = gst_parse_launch(tmp.c_str(), nullptr);
    }

    appsink_ = GST_APP_SINK(gst_bin_get_by_name(GST_BIN(pipeline_), "sink"));
    g_object_set(G_OBJECT(appsink_), "emit-signals", TRUE, nullptr);
    g_signal_connect(appsink_, "new-sample", G_CALLBACK(onNewSampleStatic), this);

    // gstObject * tmp = this;
    // overlay_ = gst_bin_get_by_name(GST_BIN(pipeline_), "overlay");
    // g_signal_connect(overlay_, "draw", G_CALLBACK(onDrawingStatic), NULL);

    bus_ = gst_pipeline_get_bus(GST_PIPELINE(pipeline_));
    gst_bus_add_signal_watch(bus_);
    g_signal_connect(bus_, "message::eos", G_CALLBACK(onEosStatic), this);
}

gstObject::~gstObject() {
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_object_unref(pipeline_);
    gst_object_unref(bus_);
}

int tid;
std::vector<std::vector<FaceObject>> gfaces(NUM_THREADS);
gboolean gstObject::onDrawing(GstElement* overlay, cairo_t* cr) {
    // std::cout << threadID << std::endl;
    std::vector<FaceObject> tmp = gfaces[tid];
    cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
    cairo_select_font_face(cr, "@cairo:Georgia", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_antialias(cr, CAIRO_ANTIALIAS_BEST);
    {
        cairo_font_options_t* options = cairo_font_options_create();
        cairo_font_options_set_antialias(options, CAIRO_ANTIALIAS_BEST);
        cairo_set_font_options(cr, options);
        cairo_font_options_destroy(options);
        cairo_set_font_size(cr, 50);
    }
    cairo_set_source_rgba(cr, 0.0, 1.0, 0.0, 1.0);
    for (auto curFace : tmp) {
        cairo_set_line_width(cr, 10);
        cairo_rectangle(cr, curFace.rect.x, curFace.rect.y, curFace.rect.width, curFace.rect.height);
        cairo_move_to(cr, curFace.rect.x, curFace.rect.y);
        cairo_show_text(cr, curFace.label.c_str());
        cairo_stroke(cr);
    }
    return true;
}

GstFlowReturn gstObject::onNewSample(GstElement* appsink) {
    GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink));
    GstCaps* caps = gst_sample_get_caps(sample);
    int width = 0, height = 0;
    if (caps) {
        GstStructure* structure = gst_caps_get_structure(caps, 0);
        gst_structure_get_int(structure, "width", &width);
        gst_structure_get_int(structure, "height", &height);
    }
    buffer = gst_sample_get_buffer(sample);
    if (gst_buffer_map(buffer, &map_info, GST_MAP_READ)) {
        const guint8* raw_data = map_info.data;
        const gchar* format_string = gst_structure_get_string(gst_caps_get_structure(caps, 0), "format");
        std::string pixel_format(format_string ? format_string : "");
        cv::Mat frame;
        if (pixel_format == "BGR") {
            frame = cv::Mat(height, width, CV_8UC3, const_cast<guint8*>(raw_data));
        } else if (pixel_format == "RGB") {
            frame = cv::Mat(height, width, CV_8UC3, const_cast<guint8*>(raw_data));
        } else if (pixel_format == "I420") {
            cv::Mat yuv(height + height / 2, width, CV_8UC1, const_cast<guint8*>(raw_data));
            cv::cvtColor(yuv, frame, cv::COLOR_YUV2BGR_I420);
        } else if (pixel_format == "NV12") {
            cv::Mat yuv(height + height / 2, width, CV_8UC1, const_cast<guint8*>(raw_data));
            cv::cvtColor(yuv, frame, cv::COLOR_YUV2BGR_NV12);
        } else if (pixel_format == "BGRx") {
            frame = cv::Mat(height, width, CV_8UC4, const_cast<guint8*>(raw_data));
            cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
        } else {
            g_printerr("Unsupported pixel format: %s\n", pixel_format.c_str());
            gst_buffer_unmap(buffer, &map_info);
            gst_sample_unref(sample);
            return GST_FLOW_OK;
        }
        cv::Mat img = frame.clone();

        std::cout << "=============================================" << std::endl;
#ifdef TEST_FR
        auto t1 = std::chrono::high_resolution_clock::now();
        std::vector<FaceObject> faces;
        det->execDetect(img, faces);
        std::vector<TrackingBox> ftracks = faceToTracking(faces);
        sortTracking(ftracks);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        // std::cout << "Thread " << threadID << ": FaceDet takes " << ms_int.count() << "ms\n";
        // std::cout << "Thread " << threadID << ": Frame has " << faces.size() << " faces" << std::endl;
        faceObjs.clear();
        for (uint32_t i = 0; i < faces.size(); i++) {
            auto t3 = std::chrono::high_resolution_clock::now();
            currentface = faces[i];
            float v2[5][2] = {{float(currentface.point[0].x), float(currentface.point[0].y)},
                              {float(currentface.point[1].x), float(currentface.point[1].y)},
                              {float(currentface.point[2].x), float(currentface.point[2].y)},
                              {float(currentface.point[3].x), float(currentface.point[3].y)},
                              {float(currentface.point[4].x), float(currentface.point[4].y)}};
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
                    currentface.label = ids[result.index];
                } else {
                    currentface.label = "Unknown";
                }
                printf("%s %f, x: %d, y: %d, w: %d, h: %d\n", currentface.label.c_str(), result.min_distance, 
                    (int)currentface.rect.x, (int)currentface.rect.y, (int)currentface.rect.width, (int)currentface.rect.height);
            }
            faceObjs.push_back(currentface);
            auto t4 = std::chrono::high_resolution_clock::now();
            auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3);
            // std::cout << "Thread " << threadID << ": FaceRec takes " << ms_int.count() << "ms\n";

            tid = threadID;
            gfaces[tid] = faceObjs;

            overlay_ = gst_bin_get_by_name(GST_BIN(pipeline_), "overlay");
            g_signal_connect(overlay_, "draw", G_CALLBACK(onDrawingStatic), NULL);
            // usleep(30000);
        }
        
#elif defined TEST_YOLO 
        auto t5 = std::chrono::high_resolution_clock::now();
        std::vector<BoxInfo> result;
        objDet->execDetect(img, result);
        std::vector<TrackingBox> tracks = objToTracking(result);
        sortTracking(tracks);
        auto t6 = std::chrono::high_resolution_clock::now();
        auto ms_int_1 = std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5);
        std::cout << "Thread " << threadID << ": yolo takes " << ms_int_1.count() << "ms\n";
        std::cout << "Thread " << threadID << ": Frame has " << result.size() << " objs" << std::endl;
#endif
        gst_buffer_unmap(buffer, &map_info);
    }
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

void gstObject::decode() {
    mainLoop_ = g_main_loop_new(nullptr, FALSE);
    gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    g_main_loop_run(mainLoop_);
}