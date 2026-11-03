#pragma once

#include <cstdint>
#include <deque>
#include <limits>
#include <vector>

#include <opencv2/opencv.hpp>

#include "detection.hpp"

struct ArrowGuide {
    cv::Point2f pos;
    cv::Point2f world_pos = cv::Point2f(0.0f, 0.0f);
    bool has_world_pos = false;
    float half_width = 8.0f;
    float half_width_world_m = std::numeric_limits<float>::quiet_NaN();
    int tracker_id = -1;
};

struct ArrowGuideDebugInfo {
    int max_arrows = 0;
    int active_arrows = 0;
    int invalid_pos = 0;
    int accepted = 0;
    bool limit_hit = false;
};

enum class ObjectType {
    ARROW = 0,
    PERSON = 1,
    CAR = 2,
    GOLD = 3,
    UNKNOWN = 255
};

struct TrackedObject {
    int id;
    ObjectType type;
    cv::Point2f pos;
    cv::Point2f velocity;
    std::deque<cv::Point2f> history;
    int first_seen;
    int last_seen;
    int64_t last_timestamp;
    int disappear_count;
    bool observed_this_frame;
    bool is_active;
    cv::KalmanFilter kf;
    float bbox_w;
    float bbox_h;

    TrackedObject(int id_,
                  ObjectType type_,
                  const cv::Point2f& p,
                  int frame_idx,
                  int64_t timestamp_us,
                  float w,
                  float h);
    void initKalmanFilter(float dt);
    void updateProcessNoise(float dt);
    float clampDt(int64_t current_timestamp);
    void updateTransitionMatrix(float dt);
    cv::Point2f predict(int64_t current_timestamp);
    void update(const cv::Point2f& m, int frame_idx, int64_t timestamp_us, float w, float h);
    void predictOnly(int frame_idx, int64_t timestamp_us);
    void applyGlobalShift(const cv::Point2f& delta);
};

// Demo-compatible simple tracked object for ARROW-only tracking.
// Non-ARROW objects continue to use the full TrackedObject with Kalman filter.
struct ArrowTrackedObject {
    int id = -1;
    ObjectType type = ObjectType::ARROW;
    cv::Point2f pos;
    float bbox_w = 0.0f;
    float bbox_h = 0.0f;
    bool is_active = true;
};

class ObjectTracker {
private:
    std::vector<TrackedObject> objects;
    int next_id = 0;
    int frame_w;
    int frame_h;

    // ARROW-only demo-compatible tracking (no Kalman, no IOU, no disappear_count).
    std::vector<ArrowTrackedObject> arrow_objects_;
    int next_arrow_id_ = 1;

    float calculateIOU(const cv::Rect& a, const cv::Rect& b);

    // ARROW-only update using demo-style simple pixel-distance association.
    void updateArrowsDemo(const std::vector<PredictResult>& detections);
    std::vector<ArrowGuide> collectArrowGuidesDemoCompatible(
        int max_arrows,
        float min_y_ratio,
        ArrowGuideDebugInfo* debug_info);

public:
    ObjectTracker(int w, int h);

    std::vector<TrackedObject*> update(const std::vector<PredictResult>& detections,
                                       int frame_idx,
                                       int64_t timestamp_us,
                                       const cv::Point2f& camera_shift = cv::Point2f(0.f, 0.f));
    std::vector<TrackedObject*> getObjectsByType(ObjectType t);
    std::vector<cv::Point2f> getArrowPoints(int max_arrows = 12, float min_y_ratio = 0.05f);
    std::vector<ArrowGuide> getArrowGuides(int max_arrows = 12, float min_y_ratio = 0.05f);
    std::vector<ArrowGuide> getArrowGuides(int max_arrows,
                                           ArrowGuideDebugInfo* debug_info);
    void draw(cv::Mat& frame);
};
