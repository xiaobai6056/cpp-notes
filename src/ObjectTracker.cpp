#include "ObjectTracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace {

bool isFinitePointLocal(const cv::Point2f& p) {
    return std::isfinite(p.x) && std::isfinite(p.y);
}

}  // namespace

TrackedObject::TrackedObject(int id_,
                             ObjectType type_,
                             const cv::Point2f& p,
                             int frame_idx,
                             int64_t timestamp_us,
                             float w,
                             float h)
    : id(id_),
      type(type_),
      pos(p),
      velocity(0.f, 0.f),
      first_seen(frame_idx),
      last_seen(frame_idx),
      last_timestamp(timestamp_us),
      disappear_count(0),
      observed_this_frame(true),
      is_active(true),
      kf(4, 2, 0),
      bbox_w(std::max(1.0f, w)),
      bbox_h(std::max(1.0f, h)) {
    initKalmanFilter(1.0f);
    history.push_back(p);
}

void TrackedObject::initKalmanFilter(float dt) {
    kf.transitionMatrix =
        (cv::Mat_<float>(4, 4) << 1, 0, dt, 0, 0, 1, 0, dt, 0, 0, 1, 0, 0, 0, 0, 1);
    kf.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0);
    updateProcessNoise(dt);
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(5e-2f));
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(0.5f));
    kf.statePost = (cv::Mat_<float>(4, 1) << pos.x, pos.y, 0, 0);
}

void TrackedObject::updateProcessNoise(float dt) {
    const float base_noise = 5e-2f;
    const float pos_noise = base_noise * dt;
    const float vel_noise = pos_noise * 5.0f;

    kf.processNoiseCov = cv::Mat::zeros(4, 4, CV_32F);
    kf.processNoiseCov.at<float>(0, 0) = pos_noise;
    kf.processNoiseCov.at<float>(1, 1) = pos_noise;
    kf.processNoiseCov.at<float>(2, 2) = vel_noise;
    kf.processNoiseCov.at<float>(3, 3) = vel_noise;
}

float TrackedObject::clampDt(int64_t current_timestamp) {
    float dt = static_cast<float>(current_timestamp - last_timestamp) / 1000000.0f;
    dt = std::max(0.001f, std::min(0.2f, dt));
    return dt;
}

void TrackedObject::updateTransitionMatrix(float dt) {
    kf.transitionMatrix.at<float>(0, 2) = dt;
    kf.transitionMatrix.at<float>(1, 3) = dt;
    updateProcessNoise(dt);
}

cv::Point2f TrackedObject::predict(int64_t current_timestamp) {
    const float dt = clampDt(current_timestamp);
    updateTransitionMatrix(dt);

    cv::Mat pred = kf.predict();
    pos = cv::Point2f(pred.at<float>(0), pred.at<float>(1));
    velocity = cv::Point2f(pred.at<float>(2), pred.at<float>(3));
    last_timestamp = current_timestamp;
    return pos;
}

void TrackedObject::update(const cv::Point2f& m, int frame_idx, int64_t timestamp_us, float w, float h) {
    const float dt = clampDt(timestamp_us);
    updateTransitionMatrix(dt);

    kf.predict();
    cv::Mat meas = (cv::Mat_<float>(2, 1) << m.x, m.y);
    cv::Mat est = kf.correct(meas);
    pos = cv::Point2f(est.at<float>(0), est.at<float>(1));
    velocity = cv::Point2f(est.at<float>(2), est.at<float>(3));
    last_seen = frame_idx;
    disappear_count = 0;
    observed_this_frame = true;
    last_timestamp = timestamp_us;
    bbox_w = bbox_w * 0.7f + std::max(1.0f, w) * 0.3f;
    bbox_h = bbox_h * 0.7f + std::max(1.0f, h) * 0.3f;
    history.push_back(pos);
    if (history.size() > 10) {
        history.pop_front();
    }
}

void TrackedObject::predictOnly(int frame_idx, int64_t timestamp_us) {
    const float dt = clampDt(timestamp_us);
    updateTransitionMatrix(dt);

    cv::Mat pred = kf.predict();
    pos = cv::Point2f(pred.at<float>(0), pred.at<float>(1));
    velocity = cv::Point2f(pred.at<float>(2), pred.at<float>(3));
    disappear_count++;
    observed_this_frame = false;
    last_seen = frame_idx;
    last_timestamp = timestamp_us;
}

void TrackedObject::applyGlobalShift(const cv::Point2f& delta) {
    if (std::abs(delta.x) < 1e-4f && std::abs(delta.y) < 1e-4f) {
        return;
    }

    pos += delta;
    kf.statePost.at<float>(0) += delta.x;
    kf.statePost.at<float>(1) += delta.y;
    kf.statePre.at<float>(0) += delta.x;
    kf.statePre.at<float>(1) += delta.y;
    for (auto& p : history) {
        p += delta;
    }
}

ObjectTracker::ObjectTracker(int w, int h) : frame_w(w), frame_h(h) {}

// ARROW-only demo-compatible update:
// - Simple pixel-distance association (max_dist = 120 px for ARROW).
// - No Kalman filter, no IOU cost, no disappear_count grace period.
// - Unmatched arrow objects are immediately deactivated (is_active = false).
// - No boundary-based deactivation.
// - Id starts at 1 and increments monotonically.
void ObjectTracker::updateArrowsDemo(const std::vector<PredictResult>& detections) {
    std::vector<bool> matched(arrow_objects_.size(), false);

    for (const auto& det : detections) {
        if (det.type != static_cast<int>(ObjectType::ARROW)) {
            continue;
        }
        if (det.width <= 0 || det.height <= 0) {
            continue;
        }

        const cv::Point2f measurement(
            static_cast<float>(det.x) + static_cast<float>(det.width) * 0.5f,
            static_cast<float>(det.y) + static_cast<float>(det.height) * 0.5f);

        int best_idx = -1;
        float best_dist = std::numeric_limits<float>::max();
        const float max_dist = 120.0f;
        for (size_t i = 0; i < arrow_objects_.size(); ++i) {
            ArrowTrackedObject& obj = arrow_objects_[i];
            if (!obj.is_active || matched[i]) {
                continue;
            }
            const float dist = static_cast<float>(cv::norm(obj.pos - measurement));
            if (dist < best_dist && dist <= max_dist) {
                best_dist = dist;
                best_idx = static_cast<int>(i);
            }
        }

        if (best_idx >= 0) {
            ArrowTrackedObject& obj = arrow_objects_[static_cast<size_t>(best_idx)];
            obj.pos = measurement;
            obj.bbox_w = static_cast<float>(det.width);
            obj.bbox_h = static_cast<float>(det.height);
            obj.is_active = true;
            matched[static_cast<size_t>(best_idx)] = true;
        } else {
            ArrowTrackedObject obj;
            obj.id = next_arrow_id_++;
            obj.type = ObjectType::ARROW;
            obj.pos = measurement;
            obj.bbox_w = static_cast<float>(det.width);
            obj.bbox_h = static_cast<float>(det.height);
            arrow_objects_.push_back(obj);
            matched.push_back(true);
        }
    }

    // Immediate deactivation for unmatched ARROW objects (demo behavior).
    for (size_t i = 0; i < arrow_objects_.size(); ++i) {
        if (i < matched.size() && matched[i]) {
            continue;
        }
        arrow_objects_[i].is_active = false;
    }
}

float ObjectTracker::calculateIOU(const cv::Rect& a, const cv::Rect& b) {
    if (a.width <= 0 || a.height <= 0 || b.width <= 0 || b.height <= 0) {
        return 0.0f;
    }
    const cv::Rect inter = a & b;
    const float inter_area = static_cast<float>(inter.area());
    const float union_area = static_cast<float>(a.area() + b.area() - inter.area());
    return union_area > 0.0f ? inter_area / union_area : 0.0f;
}

std::vector<TrackedObject*> ObjectTracker::update(const std::vector<PredictResult>& detections,
                                                  int frame_idx,
                                                  int64_t timestamp_us,
                                                  const cv::Point2f& camera_shift) {
    // ARROW-only: use demo-compatible simple pixel-distance tracking.
    // Non-ARROW (PERSON, CAR, etc.): continue using Dragon Kalman/IOU tracker.
    updateArrowsDemo(detections);

    for (auto& obj : objects) {
        obj.observed_this_frame = false;
    }

    // Split detections: non-ARROW go to the Kalman/IOU path.
    std::vector<PredictResult> non_arrow_detections;
    non_arrow_detections.reserve(detections.size());
    for (const auto& det : detections) {
        if (det.type != static_cast<int>(ObjectType::ARROW)) {
            non_arrow_detections.push_back(det);
        }
    }

    const float max_dist = 100.0f;
    const float cost_threshold = 2.0f;

    if (objects.empty()) {
        for (const auto& det : non_arrow_detections) {
            if (det.width <= 0 || det.height <= 0) {
                continue;
            }

            ObjectType dtype = static_cast<ObjectType>(det.type);
            objects.emplace_back(next_id++,
                                 dtype,
                                 cv::Point2f(static_cast<float>(det.x) + static_cast<float>(det.width) * 0.5f,
                                             static_cast<float>(det.y) + static_cast<float>(det.height) * 0.5f),
                                 frame_idx,
                                 timestamp_us,
                                 static_cast<float>(det.width),
                                 static_cast<float>(det.height));
        }

        std::vector<TrackedObject*> active;
        for (auto& obj : objects) {
            if (obj.is_active) {
                active.push_back(&obj);
            }
        }
        return active;
    }

    for (auto& obj : objects) {
        if (obj.is_active) {
            obj.applyGlobalShift(camera_shift);
        }
    }

    for (auto& obj : objects) {
        if (obj.is_active) {
            obj.predict(timestamp_us);
        }
    }

    const size_t existing_object_count = objects.size();
    std::vector<std::vector<float>> cost(existing_object_count, std::vector<float>(non_arrow_detections.size(), 1e6f));
    for (size_t i = 0; i < existing_object_count; ++i) {
        if (!objects[i].is_active) {
            continue;
        }

        for (size_t j = 0; j < non_arrow_detections.size(); ++j) {
            if (non_arrow_detections[j].width <= 0 || non_arrow_detections[j].height <= 0) {
                continue;
            }

            ObjectType dtype = static_cast<ObjectType>(non_arrow_detections[j].type);
            if (objects[i].type != dtype) {
                continue;
            }

            cv::Point2f meas(static_cast<float>(non_arrow_detections[j].x) + static_cast<float>(non_arrow_detections[j].width) * 0.5f,
                             static_cast<float>(non_arrow_detections[j].y) + static_cast<float>(non_arrow_detections[j].height) * 0.5f);
            const float dist = static_cast<float>(cv::norm(objects[i].pos - meas));

            cv::Rect dbbox(non_arrow_detections[j].x, non_arrow_detections[j].y, non_arrow_detections[j].width, non_arrow_detections[j].height);
            cv::Rect pbbox(static_cast<int>(objects[i].pos.x - non_arrow_detections[j].width * 0.5f),
                           static_cast<int>(objects[i].pos.y - non_arrow_detections[j].height * 0.5f),
                           non_arrow_detections[j].width,
                           non_arrow_detections[j].height);
            const float iou = calculateIOU(pbbox, dbbox);

            if (dist < max_dist) {
                cost[i][j] = dist / max_dist + (1.0f - iou) * 2.0f;
            }
        }
    }

    std::vector<bool> obj_matched(existing_object_count, false);
    std::vector<bool> det_matched(non_arrow_detections.size(), false);

    for (int iter = 0; iter < 3; ++iter) {
        for (size_t i = 0; i < existing_object_count; ++i) {
            if (!objects[i].is_active || obj_matched[i]) {
                continue;
            }

            int best = -1;
            float best_cost = cost_threshold;
            for (size_t j = 0; j < non_arrow_detections.size(); ++j) {
                if (det_matched[j]) {
                    continue;
                }
                if (cost[i][j] < best_cost) {
                    best_cost = cost[i][j];
                    best = static_cast<int>(j);
                }
            }

            if (best >= 0) {
                cv::Point2f meas(static_cast<float>(non_arrow_detections[best].x) +
                                     static_cast<float>(non_arrow_detections[best].width) * 0.5f,
                                 static_cast<float>(non_arrow_detections[best].y) +
                                     static_cast<float>(non_arrow_detections[best].height) * 0.5f);
                objects[i].update(meas,
                                  frame_idx,
                                  timestamp_us,
                                  static_cast<float>(non_arrow_detections[best].width),
                                  static_cast<float>(non_arrow_detections[best].height));
                obj_matched[i] = true;
                det_matched[best] = true;
            }
        }
    }

    for (size_t i = 0; i < existing_object_count; ++i) {
        if (!objects[i].is_active || obj_matched[i]) {
            continue;
        }

        const float border_ratio = 0.08f;
        const float left_border = frame_w * border_ratio;
        const float right_border = frame_w * (1.0f - border_ratio);
        const float top_border = frame_h * border_ratio;
        const float bottom_border = frame_h * (1.0f - border_ratio);
        const cv::Point2f& pos = objects[i].pos;
        const bool near_border = (pos.x < left_border || pos.x > right_border ||
                                  pos.y < top_border || pos.y > bottom_border);

        if (near_border) {
            objects[i].is_active = false;
            continue;
        }

        objects[i].predictOnly(frame_idx, timestamp_us);
        if (objects[i].disappear_count > 5) {
            objects[i].is_active = false;
        }
    }

    for (size_t j = 0; j < non_arrow_detections.size(); ++j) {
        if (det_matched[j] || non_arrow_detections[j].width <= 0 || non_arrow_detections[j].height <= 0) {
            continue;
        }

        ObjectType dtype = static_cast<ObjectType>(non_arrow_detections[j].type);
        objects.emplace_back(next_id++,
                             dtype,
                             cv::Point2f(static_cast<float>(non_arrow_detections[j].x) +
                                             static_cast<float>(non_arrow_detections[j].width) * 0.5f,
                                         static_cast<float>(non_arrow_detections[j].y) +
                                             static_cast<float>(non_arrow_detections[j].height) * 0.5f),
                             frame_idx,
                             timestamp_us,
                             static_cast<float>(non_arrow_detections[j].width),
                             static_cast<float>(non_arrow_detections[j].height));
    }

    std::vector<TrackedObject*> active;
    for (auto& obj : objects) {
        if (obj.is_active) {
            active.push_back(&obj);
        }
    }
    return active;
}

std::vector<TrackedObject*> ObjectTracker::getObjectsByType(ObjectType t) {
    std::vector<TrackedObject*> result;
    for (auto& obj : objects) {
        if (obj.is_active && obj.type == t) {
            result.push_back(&obj);
        }
    }
    return result;
}

std::vector<cv::Point2f> ObjectTracker::getArrowPoints(int max_arrows, float min_y_ratio) {
    std::vector<cv::Point2f> pts;
    auto guides = getArrowGuides(max_arrows, min_y_ratio);
    for (const auto& guide : guides) {
        pts.push_back(guide.pos);
    }
    return pts;
}

namespace {

cv::Point2f pixelToWorldMeters(const cv::Point2f& pixel, bool* valid = nullptr) {
    static constexpr double h00 = -0.0005731584;
    static constexpr double h01 = 0.0000185282;
    static constexpr double h02 = 0.1769570949;
    static constexpr double h10 = 0.0000257719;
    static constexpr double h11 = 0.0000116754;
    static constexpr double h12 = -0.2010946013;
    static constexpr double h20 = -0.0000148160;
    static constexpr double h21 = -0.0039996717;
    static constexpr double h22 = 1.0000000000;

    if (valid) *valid = false;
    if (!std::isfinite(pixel.x) || !std::isfinite(pixel.y)) {
        return cv::Point2f(0.0f, 0.0f);
    }

    const double denom = h20 * static_cast<double>(pixel.x) +
                         h21 * static_cast<double>(pixel.y) + h22;
    if (!std::isfinite(denom) || std::abs(denom) < 1e-9) {
        return cv::Point2f(0.0f, 0.0f);
    }

    const double wx = (h00 * static_cast<double>(pixel.x) +
                       h01 * static_cast<double>(pixel.y) + h02) / denom;
    const double wy = (h10 * static_cast<double>(pixel.x) +
                       h11 * static_cast<double>(pixel.y) + h12) / denom;
    if (!std::isfinite(wx) || !std::isfinite(wy)) {
        return cv::Point2f(0.0f, 0.0f);
    }

    if (valid) *valid = true;
    return cv::Point2f(static_cast<float>(wx), static_cast<float>(wy));
}

float invalidWorldHalfWidth() {
    return std::numeric_limits<float>::quiet_NaN();
}

float halfWidthPxToWorld(float half_width_px,
                         const cv::Point2f& guide_center_px,
                         const cv::Point2f& guide_world_center) {
    if (!std::isfinite(half_width_px) || half_width_px <= 0.0f ||
        !isFinitePointLocal(guide_center_px) || !isFinitePointLocal(guide_world_center)) {
        return invalidWorldHalfWidth();
    }

    bool left_valid = false;
    const cv::Point2f px_left(guide_center_px.x - half_width_px, guide_center_px.y);
    const cv::Point2f world_left = pixelToWorldMeters(px_left, &left_valid);
    if (!left_valid || !isFinitePointLocal(world_left)) {
        return invalidWorldHalfWidth();
    }

    const float lateral = std::abs(world_left.x - guide_world_center.x);
    if (!std::isfinite(lateral) || lateral < 0.01f) {
        return invalidWorldHalfWidth();
    }
    return std::min(0.35f, std::max(0.03f, lateral));
}

}  // namespace

std::vector<ArrowGuide> ObjectTracker::collectArrowGuidesDemoCompatible(
    int max_arrows,
    float min_y_ratio,
    ArrowGuideDebugInfo* debug_info) {
    if (debug_info) {
        *debug_info = ArrowGuideDebugInfo{};
        debug_info->max_arrows = max_arrows;
    }

    // ARROW-only demo-compatible path: use arrow_objects_ (no Kalman, no disappear_count).
    std::vector<ArrowGuide> guides;

    // Collect active ARROW objects, sort by y descending (near-to-far).
    std::vector<const ArrowTrackedObject*> active_arrows;
    active_arrows.reserve(arrow_objects_.size());
    for (const auto& obj : arrow_objects_) {
        if (obj.is_active) {
            active_arrows.push_back(&obj);
        }
    }
    if (debug_info) {
        debug_info->active_arrows = static_cast<int>(active_arrows.size());
    }

    std::sort(active_arrows.begin(), active_arrows.end(),
              [](const ArrowTrackedObject* a, const ArrowTrackedObject* b) {
                  return a->pos.y > b->pos.y;
              });

    const float min_y = static_cast<float>(frame_h) * min_y_ratio;
    const float x_margin = static_cast<float>(frame_w) * 0.02f;
    for (const auto* obj : active_arrows) {
        if (!isFinitePointLocal(obj->pos)) {
            if (debug_info) debug_info->invalid_pos++;
            continue;
        }
        if (obj->pos.y < min_y ||
            obj->pos.x < -x_margin ||
            obj->pos.x > static_cast<float>(frame_w) + x_margin) {
            continue;
        }
        const float hw = std::max(8.0f, 0.55f * std::max(1.0f, obj->bbox_w));
        ArrowGuide guide;
        guide.pos = obj->pos;
        guide.half_width = hw;
        guide.world_pos = pixelToWorldMeters(guide.pos, &guide.has_world_pos);
        if (guide.has_world_pos) {
            guide.half_width_world_m =
                halfWidthPxToWorld(guide.half_width, guide.pos, guide.world_pos);
        }
        guide.tracker_id = obj->id;
        guides.push_back(guide);
        if (debug_info) debug_info->accepted++;
        if (max_arrows > 0 && guides.size() >= static_cast<size_t>(max_arrows)) {
            if (debug_info) debug_info->limit_hit = true;
            break;
        }
    }

    return guides;
}

std::vector<ArrowGuide> ObjectTracker::getArrowGuides(int max_arrows, float min_y_ratio) {
    return collectArrowGuidesDemoCompatible(max_arrows, min_y_ratio, nullptr);
}

std::vector<ArrowGuide> ObjectTracker::getArrowGuides(int max_arrows,
                                                       ArrowGuideDebugInfo* debug_info) {
    return collectArrowGuidesDemoCompatible(max_arrows, 0.02f, debug_info);
}

void ObjectTracker::draw(cv::Mat& frame) {
    for (auto& obj : objects) {
        if (!obj.is_active) {
            continue;
        }

        cv::Scalar color;
        switch (obj.type) {
            case ObjectType::PERSON: color = cv::Scalar(0, 0, 255); break;
            case ObjectType::CAR: color = cv::Scalar(255, 0, 0); break;
            case ObjectType::GOLD: color = cv::Scalar(0, 255, 255); break;
            case ObjectType::ARROW: color = cv::Scalar(0, 255, 0); break;
            default: color = cv::Scalar(255, 255, 255); break;
        }

        cv::circle(frame, obj.pos, 5, color, -1);

        std::string label;
        switch (obj.type) {
            case ObjectType::PERSON: label = "P"; break;
            case ObjectType::CAR: label = "C"; break;
            case ObjectType::GOLD: label = "G"; break;
            case ObjectType::ARROW: label = "A"; break;
            default: label = "?"; break;
        }
        label += std::to_string(obj.id);
        cv::putText(frame,
                    label,
                    obj.pos + cv::Point2f(10.f, -10.f),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2);

        if (obj.history.size() > 1) {
            for (size_t i = 1; i < obj.history.size(); ++i) {
                cv::line(frame, obj.history[i - 1], obj.history[i], color, 1, cv::LINE_AA);
            }
        }

        cv::Point2f vel_tip = obj.pos + obj.velocity * 5.0f;
        cv::arrowedLine(frame, obj.pos, vel_tip, color, 1, cv::LINE_AA, 0, 0.2);
    }
}
