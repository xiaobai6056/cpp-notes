#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <termios.h>
#include <unistd.h>

#include "BlockingQueue.hpp"
#include "LocalizationBridge.hpp"
#include "PathPlanner.hpp"
#include "Serial.hpp"
#include "common.hpp"
#include "detection.hpp"
#include "json.hpp"
#include "shared_memory.h"
#include "servo.hpp"

using namespace std;
using namespace cv;

enum SearchMode {
    NORMAL = 0,
    CROSS,
    RING_BEGIN,
    RING_IN,
    RING_RUNNING,
    RING_OUT,
    BRIDGE_Up,
    BRIDGE_Down,
    Start,
    HAMBURGER,
    CHARGE,
    CHARGE_UP,
    WORKPLACE,
};

enum TrackLine {
    LEFT = 1,
    DEFAULT = 0,
    RIGHT = -1,
    NONE = -2
};

enum CalMode {
    SINGLE = 0,
    CUPPLE
};

class PathSearch {
public:
    Params params;
    Mat homoMatrix;

    float ROAD_WIDTH = 0.30f;
    float pixel_per_meter = 222.5f;
    int edgeMaximum = 200;
    float kl = 1.17f;
    float bl = 129.0f;
    float kr = -1.26f;
    float br = 523.0f;

    vector<Point2f> centerLine;
    bool pathFound = true;
    CalMode calMode = CalMode::SINGLE;
    TrackLine trackLine = TrackLine::DEFAULT;

    vector<Point2f>& center() { return centerLine; }
};

class ImgProcess {
public:
    Params params;
    string calibrationPath;
    string ipmMatrixPath;
};

#include "Recognition.hpp"

struct DetectionPacket {
    int frame_id = 0;
    std::vector<PredictResult> results;
    cv::Mat frame_640;
};

struct VideoSavePacket {
    int frame_id = 0;
    cv::Mat raw_frame;
    cv::Mat result_frame;
    cv::Mat world_frame;
};

struct LowerTelemetry {
    float speed_mps = 0.0f;
    float yaw_deg = 0.0f;
    uint64_t timestamp_ms = 0;
    bool valid = false;
};

class Car {
private:
    Params params;
    int colsSet = 320;
    int rowsSet = 240;

    Serial uart;
    LocalizationBridge localizationBridge;
    const char* Serial_dev = "/dev/ttyCH341USB0";

    BlockingQueue<DetectionPacket> detectionPackets;
    void BlockingQueueClear(BlockingQueue<DetectionPacket>& BQ);
    BlockingQueue<VideoSavePacket> videoSavePackets;
    void BlockingQueueClear(BlockingQueue<VideoSavePacket>& BQ);

    string workfolderPath = "/home/orangepi/workspace/Dragon";
    string paramsPath = workfolderPath + "/include/config/config.json";
    string modelPath = workfolderPath + "/res/model";
    string aiViewPath = workfolderPath + "/res/Videos/aiView.avi";
    string debugViewPath = workfolderPath + "/res/Videos/debugView.avi";
    string worldPathViewPath = workfolderPath + "/res/Videos/worldPathView.avi";

    std::atomic<bool> loadSuccess{false};
    std::atomic<bool> initSuccess{false};

    PathSearch pathSearcher;
    ImgProcess imgProcessor;
    Detection detection;
    servo servor;
    std::unique_ptr<ObjectTracker> objectTracker;

    std::unique_ptr<yoloe::SharedMemoryReader> shmReader_;
    std::atomic<bool> shmReaderRunning_{false};
    int stale_path_count = 0;

    LowerTelemetry lower_telemetry_;
    std::mutex lower_telemetry_mutex_;
    std::ofstream imu_time_align_log_;
    size_t imu_time_align_pending_lines_ = 0;

    Mat map1, map2;
    Mat homoMatrix, cameraMatrix, distCoeffs;
    Size imageSize;

    bool EnableAI = false;

public:
    bool isLocalizationForwardOnlyMode() const { return params.localizationForwardOnlyMode; }
    void imageProcessor();
    void videoSaver();
    void serialPort();
    void shmDetectionReader();

    float orange_data[10] = {0};
    std::mutex orange_data_mutex;

    void getOrangeData(float (&out)[4]) {
        std::lock_guard<std::mutex> lock(orange_data_mutex);
        for (int i = 0; i < 4; ++i) {
            out[i] = orange_data[i];
        }
    }

    void getOrangeDataFull(float (&out)[10]) {
        std::lock_guard<std::mutex> lock(orange_data_mutex);
        for (int i = 0; i < 10; ++i) {
            out[i] = orange_data[i];
        }
    }

    bool getLowerTelemetry(LowerTelemetry &out) {
        std::lock_guard<std::mutex> lock(lower_telemetry_mutex_);
        if (!lower_telemetry_.valid) {
            return false;
        }
        out = lower_telemetry_;
        return true;
    }

    std::atomic<bool> startDelayActive{false};
    std::chrono::steady_clock::time_point startTime;
    static constexpr int START_DELAY_MS = 2000;
    bool prevStartFlag = false;
    bool initialStartDelayConsumed = false;

    void Car_init();
    bool loadParams(string path);

    bool debug1 = false;
};
