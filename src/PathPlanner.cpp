#include "PathPlanner.hpp"
#include "common.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace {

static constexpr float kWorldPlanScale = 300.0f;

static constexpr size_t kRightForkAnchorCandidateMaxIndex = 4;
static constexpr size_t kRightForkSearchBranchSensitiveMaxGuideCount = 3;
static constexpr size_t kRightForkAnchorMinMainTailGuideCount = 1;
static constexpr size_t kRightForkConfirmedSuffixMinGuideCount = 4;
static constexpr float kRightForkMinPostSplitLateralTrend = 0.05f;

static constexpr float kSharedForkHandoffMaxLateralM = 0.24f;
static constexpr float kSharedForkHandoffMinForwardM = -0.035f;
static constexpr float kSharedForkHandoffMaxForwardM = 0.30f;
static constexpr float kSharedForkHandoffMaxSlope = 2.00f;
static constexpr float kSharedForkHandoffMaxAngleDeg = 60.0f;
static constexpr float kSharedForkHandoffBridgeMaxLateralM = 0.20f;

static constexpr int kDetachedRightForkMinIndependentIds = 3;
static constexpr int kDetachedRightForkMinGuideCount = 3;
static constexpr int kDetachedRightForkMinWidthSamples = 2;
static constexpr float kDetachedRightForkMaxMainReuseRatio = 0.35f;
static constexpr float kDetachedRightForkMaxRightMainOverlap = 0.35f;
static constexpr float kDetachedRightForkMinCommonForwardOverlapM = 0.12f;
static constexpr float kDetachedRightForkMinForwardSpanM = 0.12f;
static constexpr float kDetachedRightForkMinWidthM = 0.08f;
static constexpr float kDetachedRightForkMaxWidthM = 1.05f;
static constexpr float kDetachedRightForkMaxWidthVariationM = 0.80f;
static constexpr float kDetachedRightForkMinMeanWidthM = 0.14f;
static constexpr float kDetachedRightForkMinContinuityScore = 0.18f;

static constexpr float kSplitMainSwitchMinScoreAdvantage = 0.10f;
static constexpr int kSplitMainSwitchConfirmFrames = 2;
static constexpr int kSplitMainReverseLockFrames = 2;
static constexpr float kSplitMainIdentityMinOverlapRatio = 0.50f;
static constexpr int kSplitMainIdentityMinCommonIds = 2;
static constexpr float kSplitMainIdentityMinForwardSpanM = 0.60f;
static constexpr int kStableDoubleBoundaryRecoveryRequiredFrames = 3;
static constexpr int kStableDoubleBoundaryRecoveryMaxAgeFrames = 8;
static constexpr float kStableDoubleBoundaryRecoveryMinLateralSeparationM = 0.20f;
static constexpr float kStableDoubleBoundaryRecoveryMinCommonForwardOverlapM = 0.12f;
static constexpr float kStableDoubleBoundaryRecoveryMaxWidthM = 1.13f;

static constexpr float planMeters(float meters) {
    return meters * kWorldPlanScale;
}

static bool normalizeVectorImpl(const cv::Point2f& v, cv::Point2f& out_unit, float* out_length = nullptr) {
    const float len = static_cast<float>(cv::norm(v));
    if (!std::isfinite(len) || len <= 1e-4f) {
        out_unit = cv::Point2f(0.0f, 0.0f);
        if (out_length) *out_length = 0.0f;
        return false;
    }
    out_unit = v * (1.0f / len);
    if (out_length) *out_length = len;
    return isFinitePoint(out_unit);
}

static bool estimateTailDirectionImpl(const std::vector<ArrowGuide>& guides,
                                      cv::Point2f& out_unit,
                                      float* out_length = nullptr,
                                      size_t max_back = 3) {
    if (guides.size() < 2) return false;
    const auto& tail = guides.back();
    const size_t usable_back = std::min(max_back, guides.size() - 1);
    for (size_t back = 1; back <= usable_back; ++back) {
        const auto& anchor = guides[guides.size() - 1 - back];
        if (!isFinitePoint(anchor.pos) || !isFinitePoint(tail.pos)) continue;
        float delta_len = 0.0f;
        if (normalizeVectorImpl(tail.pos - anchor.pos, out_unit, &delta_len)) {
            if (out_length) *out_length = delta_len;
            return true;
        }
        if (back == 1) break;
    }
    return false;
}

static float angleBetweenUnitVectorsDegImpl(const cv::Point2f& a, const cv::Point2f& b) {
    const float dot = clampValue(a.x * b.x + a.y * b.y, -1.0f, 1.0f);
    return static_cast<float>(std::acos(dot) * 180.0 / CV_PI);
}

static float forwardScoreFromUnitDirectionImpl(const cv::Point2f& unit_dir) {
    return -unit_dir.y;
}

static cv::Point2f worldToPixel(const cv::Point2f& world, bool* valid = nullptr) {
    static constexpr double H00 = 110558.8010839990;
    static constexpr double H01 = 101305.7633155219;
    static constexpr double H02 = 807.8778285337;
    static constexpr double H10 = 3179.1451623942;
    static constexpr double H11 = 79579.7393497304;
    static constexpr double H12 = 15440.4836603204;
    static constexpr double H20 = 14.3535713144;
    static constexpr double H21 = 319.7937747534;
    static constexpr double H22 = 1.0000000000;
    if (valid) *valid = false;
    if (!isFinitePoint(world)) return cv::Point2f(0.0f, 0.0f);
    const double denom = H20 * static_cast<double>(world.x) + H21 * static_cast<double>(world.y) + H22;
    if (!std::isfinite(denom) || std::abs(denom) < 1e-9) return cv::Point2f(0.0f, 0.0f);
    const double u = (H00 * static_cast<double>(world.x) + H01 * static_cast<double>(world.y) + H02) / denom;
    const double v = (H10 * static_cast<double>(world.x) + H11 * static_cast<double>(world.y) + H12) / denom;
    if (!std::isfinite(u) || !std::isfinite(v)) return cv::Point2f(0.0f, 0.0f);
    if (valid) *valid = true;
    return cv::Point2f(static_cast<float>(u), static_cast<float>(v));
}

static float worldLateralM(const ArrowGuide& guide) { return guide.world_pos.x; }
static float worldForwardM(const ArrowGuide& guide) { return guide.world_pos.y; }
static bool hasValidWorldGuide(const ArrowGuide& guide) { return guide.has_world_pos && isFinitePoint(guide.world_pos); }

static cv::Point2f guidePlanPoint(const ArrowGuide& guide) {
    if (hasValidWorldGuide(guide)) {
        return cv::Point2f(worldLateralM(guide) * kWorldPlanScale,
                           worldForwardM(guide) * kWorldPlanScale);
    }
    const float nan = std::numeric_limits<float>::quiet_NaN();
    return cv::Point2f(nan, nan);
}
static float guidePlanLateral(const ArrowGuide& guide) { return guidePlanPoint(guide).x; }
static float guidePlanForward(const ArrowGuide& guide) { return guidePlanPoint(guide).y; }
static float guidePlanHalfWidth(const ArrowGuide& guide) {
    if (isFiniteFloat(guide.half_width_world_m) && guide.half_width_world_m > 0.0f) {
        return guide.half_width_world_m * kWorldPlanScale;
    }
    return guide.half_width;
}
static cv::Point2f planPointToWorld(const cv::Point2f& plan) {
    return cv::Point2f(plan.x / kWorldPlanScale, plan.y / kWorldPlanScale);
}
static cv::Point2f planPointToPixel(const cv::Point2f& plan, bool* valid = nullptr) {
    return worldToPixel(planPointToWorld(plan), valid);
}

static bool interpolateGuidePlanLateralAtForward(const std::vector<ArrowGuide>& guides,
                                                 float target_forward,
                                                 float& out_lateral);
static float computeGuideIdMatchRatio(const std::vector<ArrowGuide>& guides,
                                      const std::vector<ArrowGuide>& reference);
static std::vector<CenterlineSample> buildCenterlineSamplesFromPolyImpl(const PolyFit2& fit,
                                                                        float y_max,
                                                                        float y_min,
                                                                        int frame_w,
                                                                        int frame_h);
static int detectSingleBranchPreviewTailStart(const std::vector<ArrowGuide>& guides,
                                              float median_hw);
static std::vector<CenterlineSample> buildCenterlineSamplesFromPolyTailBlend(
    const std::vector<ArrowGuide>& guides,
    int tail_start_idx,
    int frame_w,
    int frame_h);

static std::string formatArrowGuideForDebug(const ArrowGuide& guide) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2)
        << "id=" << guide.tracker_id
        << ":px=(" << guide.pos.x << "," << guide.pos.y << ")"
        << ":w=(" << guide.world_pos.x << "," << guide.world_pos.y << ")";
    return oss.str();
}

static std::string formatArrowGuidesForDebug(const std::vector<ArrowGuide>& guides) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < guides.size(); ++i) {
        if (i > 0) oss << ";";
        oss << formatArrowGuideForDebug(guides[i]);
    }
    oss << "]";
    return oss.str();
}

static std::string formatArrowGuideWorldSummaryForDebug(const std::vector<ArrowGuide>& guides) {
    int valid = 0;
    for (const auto& guide : guides) if (hasValidWorldGuide(guide)) ++valid;
    std::ostringstream oss;
    oss << "world_valid=" << valid << "/" << guides.size();
    return oss.str();
}

static std::string formatArrowGuidesWorldListForDebug(const std::vector<ArrowGuide>& guides) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < guides.size(); ++i) {
        if (i > 0) oss << ";";
        oss << "(" << guides[i].world_pos.x << "," << guides[i].world_pos.y << ")";
    }
    oss << "]";
    return oss.str();
}

static bool shouldEmitSelectedFrameDebug(uint32_t) {
    return false;
}

struct TwoStepForkConfig {
    float forward_cluster_threshold = 18.0f;
    float max_continuation_distance = 105.0f;
    float single_lateral_shift = 48.0f;
};


struct ForkTailTrendMetrics {
    bool valid = false;
    bool previous_residual_valid = false;
    float expected_x = 0.0f;
    float candidate_residual = 0.0f;
    float previous_residual = 0.0f;
    float residual_sum = 0.0f;
    float residual_growth = 0.0f;
};

struct SingleAnchorGrowTransition {
    bool feasible = false;
    float cost = std::numeric_limits<float>::infinity();
    float gap_m = 0.0f;
    float lateral_step_m = 0.0f;
    float physical_distance_m = 0.0f;
    float slope = 0.0f;
    float heading_delta_deg = 0.0f;
    float turn_delta_deg = 0.0f;
    float prediction_residual_m = 0.0f;
    bool near_same_layer_continuation = false;
    std::string near_same_layer_accept_reason;
    bool near_same_layer_lateral_jump_like = false;
    float near_same_layer_soft_penalty = 0.0f;
    float cost_before_near_penalty = 0.0f;
    float cost_after_near_penalty = 0.0f;
    bool demoted_to_normal_cost = false;
    std::string reject_reason = "not_evaluated";
};

struct SingleAnchorGrowPath {
    std::vector<ArrowGuide> guides;
    std::vector<float> transition_costs;
    float score = std::numeric_limits<float>::infinity();
    float continuity_cost = 0.0f;
    size_t core_count = 0;
    size_t tail_count = 0;
    bool credible = false;
};

struct SingleRecentTurnTrend {
    bool stable = false;
    int sign = 0;
    int valid_turn_count = 0;
};

static RoadCorridorSet buildRoadCorridorSetForBranches(
    const std::vector<ArrowBranch>& final_branches,
    int frame_w,
    int frame_h,
    float half_width_m);
static std::string formatArrowGuideBandsForDebug(const std::vector<std::vector<ArrowGuide>>& bands) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    oss << "[";
    for (size_t i = 0; i < bands.size(); ++i) {
        if (i > 0) {
            oss << ";";
        }
        float y_max = 0.0f;
        float y_min = 0.0f;
        if (!bands[i].empty()) {
            y_max = bands[i].front().pos.y;
            y_min = bands[i].front().pos.y;
            for (const auto& guide : bands[i]) {
                y_max = std::max(y_max, guide.pos.y);
                y_min = std::min(y_min, guide.pos.y);
            }
        }
        oss << "b" << i
            << "(n=" << bands[i].size()
            << ",y_max=" << y_max
            << ",y_min=" << y_min
            << ")=" << formatArrowGuidesForDebug(bands[i]);
    }
    oss << "]";
    return oss.str();
}

static std::string formatArrowGuideBandsWorldForDebug(const std::vector<std::vector<ArrowGuide>>& bands) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "[";
    for (size_t i = 0; i < bands.size(); ++i) {
        if (i > 0) {
            oss << ";";
        }
        int valid_count = 0;
        float fwd_min = std::numeric_limits<float>::max();
        float fwd_max = -std::numeric_limits<float>::max();
        float lat_min = std::numeric_limits<float>::max();
        float lat_max = -std::numeric_limits<float>::max();
        std::vector<float> lats;
        for (const auto& guide : bands[i]) {
            if (!guide.has_world_pos || !isFinitePoint(guide.world_pos)) {
                continue;
            }
            const float lat = worldLateralM(guide);
            const float fwd = worldForwardM(guide);
            fwd_min = std::min(fwd_min, fwd);
            fwd_max = std::max(fwd_max, fwd);
            lat_min = std::min(lat_min, lat);
            lat_max = std::max(lat_max, lat);
            lats.push_back(lat);
            ++valid_count;
        }
        std::sort(lats.begin(), lats.end());
        oss << "b" << i << "(n=" << bands[i].size()
            << ",valid=" << valid_count;
        if (valid_count > 0) {
            oss << ",fwd=(" << fwd_min << "," << fwd_max << ")"
                << ",lat=(" << lat_min << "," << lat_max << ")"
                << ",width=" << (lat_max - lat_min)
                << ",lats=[";
            for (size_t j = 0; j < lats.size(); ++j) {
                if (j > 0) {
                    oss << ",";
                }
                oss << lats[j];
            }
            oss << "]";
        }
        oss << ")";
    }
    oss << "]";
    return oss.str();
}

static std::string formatArrowGuideDeltasForDebug(const std::vector<ArrowGuide>& guides) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    oss << "[";
    for (size_t i = 1; i < guides.size(); ++i) {
        if (i > 1) {
            oss << ",";
        }
        const float dx = guides[i].pos.x - guides[i - 1].pos.x;
        const float dy = guides[i].pos.y - guides[i - 1].pos.y;
        oss << i - 1 << "->" << i << ":(dx=" << dx << ",dy=" << dy << ")";
        if (guides[i - 1].has_world_pos && guides[i].has_world_pos) {
            const float wdx = guides[i].world_pos.x - guides[i - 1].world_pos.x;
            const float wdy = guides[i].world_pos.y - guides[i - 1].world_pos.y;
            const float wdist = std::sqrt(wdx * wdx + wdy * wdy);
            oss << std::setprecision(3)
                << ":(dmx=" << wdx << ",dmy=" << wdy << ",dm=" << wdist << ")"
                << std::setprecision(1);
        }
    }
    oss << "]";
    return oss.str();
}

static std::string formatCenterlineSamplesForDebug(const std::vector<CenterlineSample>& centerline) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    oss << "[";
    for (size_t i = 0; i < centerline.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << i << ":(" << centerline[i].x << "," << centerline[i].y << ")";
    }
    oss << "]";
    return oss.str();
}

static std::string formatCenterlineSamplesWorldForDebug(const std::vector<CenterlineSample>& centerline) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "[";
    for (size_t i = 0; i < centerline.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        bool valid = false;
        const cv::Point2f world =
            pixelToWorldMeters(cv::Point2f(centerline[i].x, centerline[i].y), &valid);
        oss << i << ":";
        if (valid && isFinitePoint(world)) {
            oss << "(lat=" << world.x << ",fwd=" << world.y << ")";
        } else {
            oss << "(invalid)";
        }
    }
    oss << "]";
    return oss.str();
}



static float computeBranchAnchorX(const std::vector<ArrowGuide>& guides) {
    if (guides.empty()) {
        return 0.0f;
    }

    float weighted_sum = 0.0f;
    float weight_total = 0.0f;
    const size_t count = std::min<size_t>(guides.size(), 3);
    for (size_t i = 0; i < count; ++i) {
        const float weight = 1.5f - 0.35f * static_cast<float>(i);
        weighted_sum += guides[i].pos.x * weight;
        weight_total += weight;
    }
    return (weight_total > 1e-3f) ? (weighted_sum / weight_total) : guides.front().pos.x;
}

static size_t clampedSharedPrefixCount(size_t guide_count, int shared_prefix_count) {
    if (shared_prefix_count <= 0) {
        return 0;
    }
    return std::min(guide_count, static_cast<size_t>(shared_prefix_count));
}

static std::vector<ArrowGuide> guidesAfterSharedPrefix(const std::vector<ArrowGuide>& guides,
                                                       int shared_prefix_count) {
    const size_t start = clampedSharedPrefixCount(guides.size(), shared_prefix_count);
    if (start >= guides.size()) {
        return {};
    }
    return std::vector<ArrowGuide>(guides.begin() + static_cast<std::ptrdiff_t>(start), guides.end());
}

static bool isLateralForkEvidence(const std::vector<ArrowGuide>& fork_guides) {
    if (fork_guides.size() < 2) {
        return false;
    }
    float x_min = fork_guides.front().pos.x;
    float x_max = fork_guides.front().pos.x;
    float y_min = fork_guides.front().pos.y;
    float y_max = fork_guides.front().pos.y;
    for (const auto& guide : fork_guides) {
        x_min = std::min(x_min, guide.pos.x);
        x_max = std::max(x_max, guide.pos.x);
        y_min = std::min(y_min, guide.pos.y);
        y_max = std::max(y_max, guide.pos.y);
    }
    const float x_span = x_max - x_min;
    const float y_span = y_max - y_min;
    return x_span >= 42.0f && y_span <= std::max(14.0f, 0.22f * x_span);
}

static bool hasUsableForkRoot(const std::vector<ArrowGuide>& guides,
                              int shared_prefix_count) {
    const size_t prefix_count = clampedSharedPrefixCount(guides.size(), shared_prefix_count);
    const auto fork_guides = guidesAfterSharedPrefix(guides, shared_prefix_count);
    if (fork_guides.empty()) {
        return false;
    }

    if (prefix_count > 0) {
        const auto& last_shared = guides[prefix_count - 1];
        const auto& first_fork = fork_guides.front();
        const float handoff_dx = std::abs(guidePlanLateral(first_fork) - guidePlanLateral(last_shared));
        const float handoff_dfwd = guidePlanForward(first_fork) - guidePlanForward(last_shared);
        const float plan_hw_sum = guidePlanHalfWidth(last_shared) + guidePlanHalfWidth(first_fork);
        const float handoff_dx_limit =
            std::max(planMeters(0.30f), 1.35f * plan_hw_sum);
        const bool handoff_fwd_ok = handoff_dfwd >= planMeters(0.025f) && handoff_dfwd <= planMeters(0.24f);
        return handoff_fwd_ok && handoff_dx <= handoff_dx_limit;
    }

    float fwd_min = guidePlanForward(fork_guides.front());
    float fwd_max = fwd_min;
    for (const auto& guide : fork_guides) {
        const float fwd = guidePlanForward(guide);
        fwd_min = std::min(fwd_min, fwd);
        fwd_max = std::max(fwd_max, fwd);
    }
    return (fwd_max - fwd_min) >= planMeters(0.06f);
}

static bool hasStableForkNearRootChain(const std::vector<ArrowGuide>& guides,
                                       int shared_prefix_count,
                                       float split_center_x,
                                       bool left_side) {
    const auto fork_guides = guidesAfterSharedPrefix(guides, shared_prefix_count);
    if (fork_guides.size() < 2) {
        return false;
    }

    const ArrowGuide& first_fork = fork_guides[0];
    const ArrowGuide& second_fork = fork_guides[1];
    const float plan_hw_max = std::max(guidePlanHalfWidth(first_fork), guidePlanHalfWidth(second_fork));
    const float center_slack = std::max(planMeters(0.04f), 0.28f * plan_hw_max);
    const float first_lat = guidePlanLateral(first_fork);
    const float second_lat = guidePlanLateral(second_fork);
    const bool first_side_ok =
        left_side ? (first_lat < split_center_x + center_slack)
                  : (first_lat > split_center_x - center_slack);
    const bool second_side_ok =
        left_side ? (second_lat < split_center_x + center_slack)
                  : (second_lat > split_center_x - center_slack);
    if (!(first_side_ok && second_side_ok)) {
        return false;
    }

    const float root_dfwd = guidePlanForward(second_fork) - guidePlanForward(first_fork);
    if (root_dfwd < planMeters(0.025f) || root_dfwd > planMeters(0.28f)) {
        return false;
    }

    const float root_dlat = second_lat - first_lat;
    const float plan_hw_sum = guidePlanHalfWidth(first_fork) + guidePlanHalfWidth(second_fork);
    const float handoff_dx_limit =
        std::max(planMeters(0.37f), 1.45f * plan_hw_sum);
    if (std::abs(root_dlat) > handoff_dx_limit) {
        return false;
    }

    const float inward_fold_limit =
        std::max(planMeters(0.12f), 0.65f * plan_hw_sum);
    const bool severe_inward_fold =
        left_side ? (root_dlat > inward_fold_limit)
                  : (-root_dlat > inward_fold_limit);
    if (severe_inward_fold) {
        return false;
    }

    return true;
}

static bool sharedSuffixCreatesBranchReversal(const std::vector<ArrowGuide>& guides,
                                              size_t prefix_count) {
    if (prefix_count < 2 || prefix_count >= guides.size()) {
        return false;
    }

    const auto& prev_shared = guides[prefix_count - 2];
    const auto& last_shared = guides[prefix_count - 1];
    const auto& first_fork = guides[prefix_count];
    cv::Point2f shared_dir;
    cv::Point2f fork_dir;
    if (!normalizeVector(guidePlanPoint(last_shared) - guidePlanPoint(prev_shared), shared_dir) ||
        !normalizeVector(guidePlanPoint(first_fork) - guidePlanPoint(last_shared), fork_dir)) {
        return false;
    }

    const float shared_dlat = guidePlanLateral(last_shared) - guidePlanLateral(prev_shared);
    const float handoff_dlat = guidePlanLateral(first_fork) - guidePlanLateral(last_shared);
    const float handoff_dfwd = guidePlanForward(first_fork) - guidePlanForward(last_shared);
    const bool forward_handoff = handoff_dfwd >= planMeters(0.02f) && handoff_dfwd <= planMeters(0.25f);
    const bool strong_x_reversal =
        shared_dlat * handoff_dlat < 0.0f &&
        std::abs(handoff_dlat) >= std::max(planMeters(0.09f), 0.38f * guidePlanHalfWidth(first_fork)) &&
        std::abs(shared_dlat) >= std::max(planMeters(0.12f), 0.45f * guidePlanHalfWidth(last_shared));
    const float angle_deg = angleBetweenUnitVectorsDeg(shared_dir, fork_dir);
    return forward_handoff && strong_x_reversal && angle_deg >= 105.0f;
}

static void pruneIncompatibleSharedSuffix(std::vector<ArrowGuide>& guides,
                                          int& shared_prefix_count) {
    size_t prefix_count = clampedSharedPrefixCount(guides.size(), shared_prefix_count);
    while (sharedSuffixCreatesBranchReversal(guides, prefix_count)) {
        guides.erase(guides.begin() + static_cast<std::ptrdiff_t>(prefix_count - 1));
        --prefix_count;
    }
    shared_prefix_count = static_cast<int>(prefix_count);
}

static void trimBranchSharedPrefixToCount(std::vector<ArrowGuide>& guides,
                                          int& shared_prefix_count,
                                          int keep_prefix_count) {
    size_t prefix_count = clampedSharedPrefixCount(guides.size(), shared_prefix_count);
    const size_t target_prefix = std::min(
        prefix_count, static_cast<size_t>(std::max(0, keep_prefix_count)));
    while (prefix_count > target_prefix) {
        guides.erase(guides.begin() + static_cast<std::ptrdiff_t>(prefix_count - 1));
        --prefix_count;
    }
    shared_prefix_count = static_cast<int>(prefix_count);
}

static void synchronizeSplitCandidateSharedTrunk(BoundaryPairCandidate& candidate) {
    const int common_prefix = std::max(
        0,
        std::min({candidate.left_shared_prefix_count,
                  candidate.right_shared_prefix_count,
                  static_cast<int>(candidate.left_branch.size()),
                  static_cast<int>(candidate.right_branch.size())}));
    trimBranchSharedPrefixToCount(
        candidate.left_branch, candidate.left_shared_prefix_count, common_prefix);
    trimBranchSharedPrefixToCount(
        candidate.right_branch, candidate.right_shared_prefix_count, common_prefix);
    candidate.left_shared_prefix_count = common_prefix;
    candidate.right_shared_prefix_count = common_prefix;

    candidate.shared_trunk.clear();
    if (common_prefix > 0 &&
        static_cast<int>(candidate.left_branch.size()) >= common_prefix) {
        candidate.shared_trunk.insert(
            candidate.shared_trunk.end(),
            candidate.left_branch.begin(),
            candidate.left_branch.begin() + static_cast<std::ptrdiff_t>(common_prefix));
    }
}

static bool sharedRootFitsForkCorridor(const BoundaryPairCandidate& candidate,
                                       float median_hw) {
    const int shared_prefix_count = std::min(
        candidate.left_shared_prefix_count, candidate.right_shared_prefix_count);
    if (shared_prefix_count <= 0) {
        return true;
    }
    if (static_cast<size_t>(shared_prefix_count) >= candidate.left_branch.size() ||
        static_cast<size_t>(shared_prefix_count) >= candidate.right_branch.size()) {
        return false;
    }

    const auto left_fork = guidesAfterSharedPrefix(
        candidate.left_branch, shared_prefix_count);
    const auto right_fork = guidesAfterSharedPrefix(
        candidate.right_branch, shared_prefix_count);
    if (left_fork.empty() || right_fork.empty()) {
        return false;
    }

    const auto& root =
        candidate.left_branch[static_cast<size_t>(shared_prefix_count - 1)];
    const auto& left_first = left_fork.front();
    const auto& right_first = right_fork.front();
    const float left_handoff_dy = guidePlanForward(left_first) - guidePlanForward(root);
    const float right_handoff_dy = guidePlanForward(right_first) - guidePlanForward(root);
    if (left_handoff_dy < 4.0f || right_handoff_dy < 4.0f ||
        left_handoff_dy > 88.0f || right_handoff_dy > 88.0f) {
        return false;
    }

    const float fork_gap = std::abs(guidePlanLateral(right_first) - guidePlanLateral(left_first));
    const float fork_center_x =
        0.5f * (guidePlanLateral(left_first) + guidePlanLateral(right_first));
    const float center_offset_limit = std::max(
        22.0f,
        std::min(56.0f, 0.34f * std::max(fork_gap, median_hw * 1.25f)));
    const float center_offset = std::abs(guidePlanLateral(root) - fork_center_x);
    if (center_offset > center_offset_limit) {
        return false;
    }

    const float left_handoff_dx = std::abs(guidePlanLateral(root) - guidePlanLateral(left_first));
    const float right_handoff_dx = std::abs(guidePlanLateral(right_first) - guidePlanLateral(root));
    const float max_handoff_dx = std::max(left_handoff_dx, right_handoff_dx);
    const float min_handoff_dx = std::min(left_handoff_dx, right_handoff_dx);
    const float handoff_dx_soft_limit = std::max(
        72.0f,
        std::min(148.0f, 0.58f * fork_gap + 0.55f * median_hw));
    if (shared_prefix_count >= 2 &&
        max_handoff_dx > handoff_dx_soft_limit) {
        return false;
    }
    const float handoff_balance =
        max_handoff_dx > 1e-3f ? (min_handoff_dx / max_handoff_dx) : 1.0f;
    const float min_balance_ratio =
        shared_prefix_count >= 2 ? 0.46f : 0.24f;
    return !(handoff_balance < min_balance_ratio &&
             center_offset > center_offset_limit * 0.35f);
}

static void normalizeCandidateSharedTrunk(BoundaryPairCandidate& candidate,
                                          float median_hw) {
    synchronizeSplitCandidateSharedTrunk(candidate);
    while (candidate.left_shared_prefix_count > 0 &&
           candidate.right_shared_prefix_count > 0) {
        if (sharedRootFitsForkCorridor(candidate, median_hw)) {
            break;
        }

        const int next_prefix =
            std::max(0, candidate.left_shared_prefix_count - 1);
        trimBranchSharedPrefixToCount(
            candidate.left_branch, candidate.left_shared_prefix_count, next_prefix);
        trimBranchSharedPrefixToCount(
            candidate.right_branch, candidate.right_shared_prefix_count, next_prefix);
        pruneIncompatibleSharedSuffix(
            candidate.left_branch, candidate.left_shared_prefix_count);
        pruneIncompatibleSharedSuffix(
            candidate.right_branch, candidate.right_shared_prefix_count);
        synchronizeSplitCandidateSharedTrunk(candidate);
    }
}

static void absorbIdenticalForkPrefixIntoSharedTrunk(BoundaryPairCandidate& candidate) {
    synchronizeSplitCandidateSharedTrunk(candidate);

    int common_prefix = std::max(
        0,
        std::min({candidate.left_shared_prefix_count,
                  candidate.right_shared_prefix_count,
                  static_cast<int>(candidate.left_branch.size()),
                  static_cast<int>(candidate.right_branch.size())}));
    while (common_prefix < static_cast<int>(candidate.left_branch.size()) &&
           common_prefix < static_cast<int>(candidate.right_branch.size()) &&
           std::abs(candidate.left_branch[static_cast<size_t>(common_prefix)].pos.x -
                    candidate.right_branch[static_cast<size_t>(common_prefix)].pos.x) < 1e-3f &&
           std::abs(candidate.left_branch[static_cast<size_t>(common_prefix)].pos.y -
                    candidate.right_branch[static_cast<size_t>(common_prefix)].pos.y) < 1e-3f) {
        ++common_prefix;
    }

    if (common_prefix != candidate.left_shared_prefix_count ||
        common_prefix != candidate.right_shared_prefix_count) {
        candidate.left_shared_prefix_count = common_prefix;
        candidate.right_shared_prefix_count = common_prefix;
        synchronizeSplitCandidateSharedTrunk(candidate);
    }
}

static bool sharedForkHandoffPhysicallyOk(const ArrowGuide& last_shared,
                                          const std::vector<ArrowGuide>& fork_guides) {
    if (fork_guides.empty()) {
        return false;
    }
    const cv::Point2f shared_plan = guidePlanPoint(last_shared);
    const cv::Point2f first_plan = guidePlanPoint(fork_guides.front());
    const float handoff_lat = std::abs(first_plan.x - shared_plan.x);
    const float handoff_fwd = first_plan.y - shared_plan.y;
    const float lat_limit =
        std::max(planMeters(kSharedForkHandoffMaxLateralM),
                 0.75f * std::max(guidePlanHalfWidth(last_shared),
                                   guidePlanHalfWidth(fork_guides.front())));
    if (handoff_lat > lat_limit) {
        return false;
    }
    if (handoff_fwd < planMeters(kSharedForkHandoffMinForwardM) ||
        handoff_fwd > planMeters(kSharedForkHandoffMaxForwardM)) {
        return false;
    }
    const float slope = handoff_lat / std::max(planMeters(0.03f),
                                               std::abs(handoff_fwd));
    if (slope > kSharedForkHandoffMaxSlope) {
        return false;
    }

    if (fork_guides.size() >= 2) {
        cv::Point2f handoff_dir;
        cv::Point2f fork_dir;
        if (normalizeVector(first_plan - shared_plan, handoff_dir) &&
            normalizeVector(guidePlanPoint(fork_guides[1]) - first_plan, fork_dir)) {
            const float angle_deg = angleBetweenUnitVectorsDeg(handoff_dir, fork_dir);
            if (angle_deg > kSharedForkHandoffMaxAngleDeg) {
                return false;
            }
        }
    }
    return true;
}

static std::vector<ArrowGuide> guidesForBranchCenterline(const ArrowBranch& branch,
                                                         CenterlineDebugInfo* debug_info = nullptr) {
    if (debug_info) {
        debug_info->guide_selection_mode = "full_branch";
        debug_info->fork_guides_available = static_cast<int>(branch.guides.size());
        debug_info->selected_fork_guides = static_cast<int>(branch.guides.size());
    }
    if (branch.shared_prefix_count <= 0) {
        return branch.guides;
    }

    const size_t prefix_count =
        clampedSharedPrefixCount(branch.guides.size(), branch.shared_prefix_count);
    auto fork_guides = guidesAfterSharedPrefix(branch.guides, branch.shared_prefix_count);
    if (debug_info) {
        debug_info->shared_guides_available = static_cast<int>(prefix_count);
        debug_info->fork_guides_available = static_cast<int>(fork_guides.size());
        debug_info->selected_shared_guides = static_cast<int>(prefix_count);
        debug_info->selected_fork_guides = static_cast<int>(fork_guides.size());
        debug_info->guide_selection_mode = "shared_plus_fork";
    }
    if (fork_guides.size() >= 2) {
        if (prefix_count > 0) {
            const auto& last_shared = branch.guides[prefix_count - 1];
            const auto& first_fork = fork_guides.front();
            const float handoff_dx = std::abs(guidePlanLateral(first_fork) - guidePlanLateral(last_shared));
            const float handoff_dy = guidePlanForward(first_fork) - guidePlanForward(last_shared);
            const float plan_hw_sum = guidePlanHalfWidth(last_shared) + guidePlanHalfWidth(first_fork);
            const float handoff_dx_limit =
                std::max(planMeters(kSharedForkHandoffMaxLateralM),
                         0.92f * plan_hw_sum);
            const bool handoff_y_ok =
                handoff_dy >= planMeters(kSharedForkHandoffMinForwardM) &&
                handoff_dy <= planMeters(kSharedForkHandoffMaxForwardM);
            const bool handoff_world_ok =
                sharedForkHandoffPhysicallyOk(last_shared, fork_guides);
            if (debug_info) {
                debug_info->handoff_dx = handoff_dx;
                debug_info->handoff_dy = handoff_dy;
                debug_info->handoff_dx_limit = handoff_dx_limit;
            }
            if (handoff_dx <= handoff_dx_limit && handoff_y_ok && handoff_world_ok) {
                if (debug_info) {
                    debug_info->guide_selection_mode = "shared_plus_fork_direct";
                }
                return branch.guides;
            }
            if (handoff_dx <= handoff_dx_limit && handoff_y_ok) {
                std::vector<ArrowGuide> connected;
                connected.reserve(fork_guides.size() + 1);
                connected.push_back(last_shared);
                connected.insert(connected.end(), fork_guides.begin(), fork_guides.end());
                if (debug_info) {
                    debug_info->guide_selection_mode =
                        "shared_anchor_plus_fork_fallback";
                }
                return connected;
            }
        }
        // handoff 澶辫触锛歴hared鈫抐ork x 璺冲彉澶с€?
        // 鎸?x 璺?fork 鍧囧€肩殑鍋忕Щ杩囨护 shared锛屽彧淇濈暀涓?fork 鍚屽悜鐨?shared guides銆?
        float fork_sum_x = 0.0f, fork_sum_hw = 0.0f;
        for (const auto& fg : fork_guides) {
            fork_sum_x += guidePlanLateral(fg);
            fork_sum_hw += guidePlanHalfWidth(fg);
        }
        const float fork_mean_x = fork_sum_x / static_cast<float>(fork_guides.size());
        const float fork_avg_hw = fork_sum_hw / static_cast<float>(fork_guides.size());
        const float max_x_offset = std::max(planMeters(0.11f), fork_avg_hw * 1.2f);
        if (debug_info) {
            debug_info->fork_mean_x = fork_mean_x;
            debug_info->shared_filter_x_limit = max_x_offset;
        }

        std::vector<ArrowGuide> compatible_shared;
        compatible_shared.reserve(prefix_count);
        for (size_t i = 0; i < prefix_count; ++i) {
            if (std::abs(guidePlanLateral(branch.guides[i]) - fork_mean_x) <= max_x_offset) {
                compatible_shared.push_back(branch.guides[i]);
            }
        }
        if (debug_info) {
            debug_info->compatible_shared_guides = static_cast<int>(compatible_shared.size());
        }
        // Rescue: when no shared guide passes the offset filter and the fork
        // part is short (鈮? guides), check whether the topmost shared guide's
        // arrow extent physically covers the fork region.  A wide trunk arrow
        // near the camera base naturally spans both branches, so including it
        // as a root anchor gives the centerline enough y-coverage.
        if (compatible_shared.empty() && prefix_count > 0 && fork_guides.size() <= 4) {
            const auto& root = branch.guides[0];
            const float extent_margin =
                std::max(planMeters(0.06f), guidePlanHalfWidth(root) * 0.25f);
            const bool root_covers_fork =
                fork_mean_x >= (guidePlanLateral(root) - guidePlanHalfWidth(root) - extent_margin) &&
                fork_mean_x <= (guidePlanLateral(root) + guidePlanHalfWidth(root) + extent_margin);
            if (root_covers_fork) {
                compatible_shared.push_back(root);
            }
        }
        if (!compatible_shared.empty()) {
            const float bridge_dx =
                std::abs(guidePlanLateral(compatible_shared.back()) -
                         guidePlanLateral(fork_guides.front()));
            if (debug_info) {
                debug_info->bridge_dx = bridge_dx;
            }
            if (bridge_dx < planMeters(kSharedForkHandoffBridgeMaxLateralM) &&
                sharedForkHandoffPhysicallyOk(compatible_shared.back(), fork_guides)) {
                std::vector<ArrowGuide> combined;
                combined.reserve(compatible_shared.size() + fork_guides.size());
                combined.insert(combined.end(), compatible_shared.begin(),
                                compatible_shared.end());
                combined.insert(combined.end(), fork_guides.begin(), fork_guides.end());
                if (debug_info) {
                    debug_info->guide_selection_mode = "shared_filtered_plus_fork";
                    debug_info->selected_shared_guides =
                        static_cast<int>(compatible_shared.size());
                    debug_info->selected_fork_guides =
                        static_cast<int>(fork_guides.size());
                }
                return combined;
            }
        }
        if (debug_info) {
            debug_info->guide_selection_mode = "fork_only";
            debug_info->selected_shared_guides = 0;
            debug_info->selected_fork_guides = static_cast<int>(fork_guides.size());
        }
        return fork_guides;
    }

    if (fork_guides.size() == 1) {
        std::vector<ArrowGuide> selected;
        selected.reserve(2);
        if (prefix_count > 0) {
            selected.push_back(branch.guides[prefix_count - 1]);
        }
        selected.push_back(fork_guides.front());
        if (debug_info) {
            debug_info->guide_selection_mode = "last_shared_plus_single_fork";
            debug_info->selected_shared_guides = prefix_count > 0 ? 1 : 0;
            debug_info->selected_fork_guides = 1;
        }
        return selected;
    }

    if (debug_info) {
        debug_info->guide_selection_mode = "shared_only";
        debug_info->selected_shared_guides = static_cast<int>(branch.guides.size());
        debug_info->selected_fork_guides = 0;
    }
    return branch.guides;
}

static size_t guideCountAfterSharedPrefix(const std::vector<ArrowGuide>& guides,
                                          int shared_prefix_count) {
    return guides.size() - clampedSharedPrefixCount(guides.size(), shared_prefix_count);
}

static float computeBranchAnchorXAfterSharedPrefix(const std::vector<ArrowGuide>& guides,
                                                   int shared_prefix_count) {
    const auto fork_guides = guidesAfterSharedPrefix(guides, shared_prefix_count);
    if (!fork_guides.empty()) {
        if (isLateralForkEvidence(fork_guides)) {
            return fork_guides.front().pos.x;
        }
        return computeBranchAnchorX(fork_guides);
    }
    return computeBranchAnchorX(guides);
}

static std::vector<ArrowGuide> selectSharedTrunkGuides(const std::vector<ArrowGuide>& parent_guides,
                                                       float split_forward) {
    constexpr size_t kMaxSharedTrunkGuides = 4;
    constexpr float kMaxSharedTrunkForwardSpan = 96.0f;
    constexpr float kAllowNearSplitSlack = 10.0f;
    constexpr float kTrimNearSplitGap = 18.0f;

    std::vector<ArrowGuide> trunk_reversed;
    trunk_reversed.reserve(kMaxSharedTrunkGuides);
    for (auto it = parent_guides.rbegin(); it != parent_guides.rend(); ++it) {
        if (!isFinitePoint(it->pos)) {
            continue;
        }
        const float guide_forward = guidePlanForward(*it);
        if (guide_forward > split_forward + kAllowNearSplitSlack) {
            continue;
        }
        if (split_forward - guide_forward > kMaxSharedTrunkForwardSpan) {
            break;
        }
        trunk_reversed.push_back(*it);
        if (trunk_reversed.size() >= kMaxSharedTrunkGuides) {
            break;
        }
    }

    std::reverse(trunk_reversed.begin(), trunk_reversed.end());
    // Do not let a shared guide sit almost on top of the split seeds.  Keep a
    // moderate near-split root: in oblique Y entries it is often the only
    // usable handoff from the common trunk to the short lateral right fork.
    if (trunk_reversed.size() >= 2) {
        const auto& last_shared = trunk_reversed.back();
        if (split_forward - guidePlanForward(last_shared) <= kTrimNearSplitGap) {
            trunk_reversed.pop_back();
        }
    }
    return trunk_reversed;
}

static std::vector<std::vector<ArrowGuide>> buildArrowGuideBands(const std::vector<ArrowGuide>& guides,
                                                                 float forward_band_threshold) {
    std::vector<ArrowGuide> sorted = guides;
    std::sort(sorted.begin(), sorted.end(), [](const ArrowGuide& a, const ArrowGuide& b) {
        const float af = guidePlanForward(a);
        const float bf = guidePlanForward(b);
        if (std::abs(af - bf) > 1e-3f) {
            return af < bf;
        }
        return guidePlanLateral(a) < guidePlanLateral(b);
    });

    std::vector<std::vector<ArrowGuide>> bands;
    for (const auto& guide : sorted) {
        if (bands.empty()) {
            bands.push_back({guide});
            continue;
        }
        const float ref_forward = guidePlanForward(bands.back().front());
        if (std::abs(guidePlanForward(guide) - ref_forward) <= forward_band_threshold) {
            bands.back().push_back(guide);
        } else {
            bands.push_back({guide});
        }
    }

    for (auto& band : bands) {
        std::sort(band.begin(), band.end(), [](const ArrowGuide& a, const ArrowGuide& b) {
            return guidePlanLateral(a) < guidePlanLateral(b);
        });
    }
    return bands;
}

static int twoStepSelectSinglePoint(const std::vector<ArrowGuide>& band,
                                    const std::vector<ArrowGuide>& main_branch,
                                    const TwoStepForkConfig& cfg);
static bool isReliableSingleGuideSet(const std::vector<ArrowGuide>& guides,
                                     float min_y_span);
static float averageGuideStepCost(const std::vector<ArrowGuide>& guides);
static float maxGuideStepDistance(const std::vector<ArrowGuide>& guides);
static float maxGuideDxJump(const std::vector<ArrowGuide>& guides);
static float normalizeAngleDeltaDeg(float delta);
static float canonicalBoundaryPhysicalLateral(float plan_lateral);
static float twoStepExtrapolateXAtY(const ArrowGuide& a,
                                    const ArrowGuide& b,
                                    float target_y);

static void appendUniqueGuides(const std::vector<ArrowGuide>& src,
                               std::vector<ArrowGuide>& dst) {
    for (const auto& guide : src) {
        bool exists = false;
        for (const auto& saved : dst) {
            if (sameGuidePoint(saved, guide)) {
                exists = true;
                break;
            }
        }
        if (!exists) {
            dst.push_back(guide);
        }
    }
}

static bool fitForkPrefixTrendXAtY(const std::vector<ArrowGuide>& guides,
                                   size_t begin_idx,
                                   size_t end_idx,
                                   float target_y,
                                   float* out_x) {
    if (!out_x || end_idx <= begin_idx || end_idx - begin_idx < 2) {
        return false;
    }

    const size_t sample_count = end_idx - begin_idx;
    if (sample_count == 2) {
        *out_x = twoStepExtrapolateXAtY(guides[begin_idx], guides[begin_idx + 1], target_y);
        return std::isfinite(*out_x);
    }

    double sum_y = 0.0;
    double sum_x = 0.0;
    double sum_yy = 0.0;
    double sum_yx = 0.0;
    for (size_t i = begin_idx; i < end_idx; ++i) {
        const cv::Point2f plan = guidePlanPoint(guides[i]);
        const double y = static_cast<double>(plan.y);
        const double x = static_cast<double>(plan.x);
        sum_y += y;
        sum_x += x;
        sum_yy += y * y;
        sum_yx += y * x;
    }

    const double n = static_cast<double>(sample_count);
    const double denom = n * sum_yy - sum_y * sum_y;
    if (std::abs(denom) < 1e-6) {
        *out_x =
            twoStepExtrapolateXAtY(guides[end_idx - 2], guides[end_idx - 1], target_y);
        return std::isfinite(*out_x);
    }

    const double slope = (n * sum_yx - sum_y * sum_x) / denom;
    const double intercept = (sum_x - slope * sum_y) / n;
    *out_x = static_cast<float>(slope * static_cast<double>(target_y) + intercept);
    return std::isfinite(*out_x);
}

static ForkTailTrendMetrics evaluateForkTailTrendMetrics(
    const std::vector<ArrowGuide>& guides,
    size_t prev_idx,
    const ArrowGuide& candidate) {
    ForkTailTrendMetrics metrics;
    if (guides.size() < 3 || prev_idx >= guides.size()) {
        return metrics;
    }

    size_t prefix_end = prev_idx + 1;
    if (prefix_end >= 4) {
        prefix_end -= 1;
    }
    const size_t prefix_begin = (prefix_end > 3) ? (prefix_end - 3) : 0;
    if (prefix_end <= prefix_begin || prefix_end - prefix_begin < 2) {
        return metrics;
    }

    float expected_x = 0.0f;
    if (!fitForkPrefixTrendXAtY(
            guides, prefix_begin, prefix_end, guidePlanForward(candidate), &expected_x)) {
        return metrics;
    }

    metrics.valid = true;
    metrics.expected_x = expected_x;
    metrics.candidate_residual = std::abs(guidePlanLateral(candidate) - expected_x);

    if (prefix_end <= prev_idx) {
        float expected_prev_x = 0.0f;
        if (fitForkPrefixTrendXAtY(
                guides, prefix_begin, prefix_end, guidePlanForward(guides[prev_idx]), &expected_prev_x)) {
            metrics.previous_residual_valid = true;
            metrics.previous_residual =
                std::abs(guidePlanLateral(guides[prev_idx]) - expected_prev_x);
        }
    }

    metrics.residual_sum = metrics.candidate_residual + metrics.previous_residual;
    metrics.residual_growth =
        metrics.candidate_residual - metrics.previous_residual;
    return metrics;
}

static bool twoStepIsPlausibleContinuation(const ArrowGuide& prev,
                                           const ArrowGuide& curr,
                                           float max_distance,
                                           float max_lateral_shift,
                                           float max_forward_reach = -1.0f) {
    if (!isFinitePoint(prev.pos) || !isFinitePoint(curr.pos)) {
        return false;
    }
    const cv::Point2f prev_plan = guidePlanPoint(prev);
    const cv::Point2f curr_plan = guidePlanPoint(curr);
    const float dx = curr_plan.x - prev_plan.x;
    const float dy = curr_plan.y - prev_plan.y;
    const float abs_dx = std::abs(dx);
    const float dist = std::sqrt(dx * dx + dy * dy);
    if (curr_plan.y < prev_plan.y - planMeters(0.065f)) {
        return false;
    }
    if (abs_dx >= max_lateral_shift) {
        return false;
    }
    if (dist < max_distance) {
        return true;
    }

    if (max_forward_reach <= max_distance || dy <= 0.0f || dy >= max_forward_reach) {
        return false;
    }

    const float strict_forward_reach_lateral =
        std::min(max_lateral_shift,
                 std::max(planMeters(0.10f), max_lateral_shift * 0.82f));
    return abs_dx < strict_forward_reach_lateral;
}

static float twoStepContinuationCost(const ArrowGuide& prev, const ArrowGuide& curr) {
    const cv::Point2f prev_plan = guidePlanPoint(prev);
    const cv::Point2f curr_plan = guidePlanPoint(curr);
    const float dx = curr_plan.x - prev_plan.x;
    const float dy = std::abs(curr_plan.y - prev_plan.y);
    const float dist = std::sqrt(dx * dx + dy * dy);
    return dist + 0.65f * std::abs(dx);
}

static int twoStepSelectSinglePoint(const std::vector<ArrowGuide>& band,
                                    const std::vector<ArrowGuide>& main_branch,
                                    const TwoStepForkConfig& cfg) {
    int best_idx = -1;
    float best_cost = 1e9f;
    if (main_branch.empty()) {
        float lateral_sum = 0.0f;
        int lateral_count = 0;
        for (const auto& guide : band) {
            if (hasValidWorldGuide(guide)) {
                lateral_sum += guidePlanLateral(guide);
                ++lateral_count;
            }
        }
        const float band_lateral_center =
            (lateral_count > 0) ? (lateral_sum / static_cast<float>(lateral_count)) : 0.0f;
        for (size_t i = 0; i < band.size(); ++i) {
            const float cost = std::abs(guidePlanLateral(band[i]) - band_lateral_center);
            if (cost < best_cost) {
                best_cost = cost;
                best_idx = static_cast<int>(i);
            }
        }
        return best_idx;
    }

    const auto& tail = main_branch.back();
    for (size_t i = 0; i < band.size(); ++i) {
        if (!twoStepIsPlausibleContinuation(tail, band[i], cfg.max_continuation_distance, cfg.single_lateral_shift)) {
            continue;
        }
        const float cost = twoStepContinuationCost(tail, band[i]);
        if (cost < best_cost) {
            best_cost = cost;
            best_idx = static_cast<int>(i);
        }
    }
    return best_idx;
}

// Least-squares refit on a larger inlier set (normal equations for degree-2)
static PolyFit2 refitPoly2LeastSquares(const std::vector<ArrowGuide>& guides,
                                        const std::vector<bool>& inlier_mask) {
    // Accumulate normal equation entries:  A^T A x = A^T b
    // A row = [y^2, y, 1], b = x
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0;
    double r0 = 0, r1 = 0, r2 = 0;
    int n = 0;
    for (size_t i = 0; i < guides.size(); ++i) {
        if (!inlier_mask[i]) continue;
        const cv::Point2f plan = guidePlanPoint(guides[i]);
        const double y = static_cast<double>(plan.y);
        const double x = static_cast<double>(plan.x);
        const double y2 = y * y;
        s0 += 1.0;
        s1 += y;
        s2 += y2;
        s3 += y2 * y;
        s4 += y2 * y2;
        r0 += x;
        r1 += x * y;
        r2 += x * y2;
        ++n;
    }

    PolyFit2 fit;
    if (n < 3) return fit;

    // Solve 3x3 symmetric system via Cramer
    //   | s4 s3 s2 | | a |   | r2 |
    //   | s3 s2 s1 | | b | = | r1 |
    //   | s2 s1 s0 | | c |   | r0 |
    const double det = s4 * (s2 * s0 - s1 * s1)
                     - s3 * (s3 * s0 - s1 * s2)
                     + s2 * (s3 * s1 - s2 * s2);
    if (std::abs(det) < 1e-12) return fit;
    const double inv = 1.0 / det;

    const double a = (r2 * (s2 * s0 - s1 * s1)
                    - s3 * (r1 * s0 - s1 * r0)
                    + s2 * (r1 * s1 - s2 * r0)) * inv;
    const double b = (s4 * (r1 * s0 - s1 * r0)
                    - r2 * (s3 * s0 - s1 * s2)
                    + s2 * (s3 * r0 - r1 * s2)) * inv;
    const double c = (s4 * (s2 * r0 - r1 * s1)
                    - s3 * (s3 * r0 - r1 * s2)
                    + r2 * (s3 * s1 - s2 * s2)) * inv;

    fit.a = static_cast<float>(a);
    fit.b = static_cast<float>(b);
    fit.c = static_cast<float>(c);
    fit.valid = std::isfinite(fit.a) && std::isfinite(fit.b) && std::isfinite(fit.c);
    return fit;
}

static float twoStepExtrapolateXAtY(const ArrowGuide& a,
                                    const ArrowGuide& b,
                                    float target_y) {
    const cv::Point2f pa = guidePlanPoint(a);
    const cv::Point2f pb = guidePlanPoint(b);
    const float dy = pb.y - pa.y;
    if (std::abs(dy) < 1e-3f) {
        return pa.x;
    }
    const float t = (target_y - pa.y) / dy;
    return pa.x + t * (pb.x - pa.x);
}

static float twoStepMedianJump(std::vector<float> jumps) {
    if (jumps.empty()) {
        return 0.0f;
    }
    std::sort(jumps.begin(), jumps.end());
    return jumps[jumps.size() / 2];
}

static float twoStepEndpointContaminationScore(const std::vector<ArrowGuide>& branch,
                                               bool front_endpoint,
                                               float median_hw,
                                               float branch_lateral_ref) {
    if (branch.size() < 4) {
        return -1.0f;
    }

    const size_t point_count = branch.size();
    const size_t idx0 = front_endpoint ? 0 : point_count - 1;
    const size_t idx1 = front_endpoint ? 1 : point_count - 2;
    const size_t idx2 = front_endpoint ? 2 : point_count - 3;
    const size_t idx3 = front_endpoint ? 3 : point_count - 4;

    const ArrowGuide& endpoint = branch[idx0];
    const ArrowGuide& neighbor1 = branch[idx1];
    const ArrowGuide& neighbor2 = branch[idx2];
    const ArrowGuide& neighbor3 = branch[idx3];

    const float endpoint_jump = std::abs(guidePlanLateral(neighbor1) - guidePlanLateral(endpoint));
    const float ref_jump = twoStepMedianJump({
        std::abs(guidePlanLateral(neighbor2) - guidePlanLateral(neighbor1)),
        std::abs(guidePlanLateral(neighbor3) - guidePlanLateral(neighbor2))
    });
    const float extrapolated_x =
        twoStepExtrapolateXAtY(neighbor1, neighbor2, guidePlanForward(endpoint));
    const float local_residual = std::abs(guidePlanLateral(endpoint) - extrapolated_x);

    const float endpoint_ref_residual =
        std::abs(guidePlanLateral(endpoint) - branch_lateral_ref);
    const float neighbor_ref_residual =
        std::abs(guidePlanLateral(neighbor1) - branch_lateral_ref);

    const float jump_thresh = std::max(planMeters(0.08f), median_hw * 0.90f);
    const float local_residual_thresh = std::max(planMeters(0.08f), median_hw * 1.00f);
    const float ref_gap_thresh = std::max(planMeters(0.04f), median_hw * 0.45f);

    const bool local_spike = endpoint_jump > jump_thresh && endpoint_jump > ref_jump * 1.8f;
    const bool local_mismatch = local_residual > local_residual_thresh;
    const bool reference_mismatch = endpoint_ref_residual > neighbor_ref_residual + ref_gap_thresh;
    if (!(local_spike && local_mismatch && reference_mismatch)) {
        return -1.0f;
    }

    return endpoint_jump + local_residual + (endpoint_ref_residual - neighbor_ref_residual);
}

static float twoStepEndpointShockScore(const std::vector<ArrowGuide>& branch,
                                       bool front_endpoint,
                                       float median_hw) {
    if (branch.size() < 4) {
        return 0.0f;
    }

    const size_t point_count = branch.size();
    const size_t idx0 = front_endpoint ? 0 : point_count - 1;
    const size_t idx1 = front_endpoint ? 1 : point_count - 2;
    const size_t idx2 = front_endpoint ? 2 : point_count - 3;
    const size_t idx3 = front_endpoint ? 3 : point_count - 4;

    const ArrowGuide& endpoint = branch[idx0];
    const ArrowGuide& neighbor1 = branch[idx1];
    const ArrowGuide& neighbor2 = branch[idx2];
    const ArrowGuide& neighbor3 = branch[idx3];

    const float endpoint_dx = std::abs(guidePlanLateral(neighbor1) - guidePlanLateral(endpoint));
    const float endpoint_dy = std::abs(guidePlanForward(neighbor1) - guidePlanForward(endpoint));
    const float ref_jump = twoStepMedianJump({
        std::abs(guidePlanLateral(neighbor2) - guidePlanLateral(neighbor1)),
        std::abs(guidePlanLateral(neighbor3) - guidePlanLateral(neighbor2))
    });
    const float extrapolated_x =
        twoStepExtrapolateXAtY(neighbor1, neighbor2, guidePlanForward(endpoint));
    const float local_residual = std::abs(guidePlanLateral(endpoint) - extrapolated_x);

    const float base_dx_thresh = std::max(planMeters(0.14f), median_hw * 1.05f);
    const float relative_dx_thresh = std::max(base_dx_thresh, ref_jump * 2.25f);
    const float local_residual_thresh = std::max(planMeters(0.07f), median_hw * 0.80f);
    const float short_hop_dy_limit = std::max(planMeters(0.11f), median_hw * 0.90f);

    const float dx_excess = std::max(0.0f, endpoint_dx - relative_dx_thresh);
    const float residual_excess = std::max(0.0f, local_residual - local_residual_thresh);
    const bool short_hop = endpoint_dy <= short_hop_dy_limit;

    if (dx_excess <= 0.0f && residual_excess <= 0.0f) {
        return 0.0f;
    }

    float shock = dx_excess + 0.65f * residual_excess;
    if (short_hop && endpoint_dx > base_dx_thresh) {
        shock += 0.25f * (endpoint_dx - base_dx_thresh);
    }
    return shock;
}

static float twoStepBackEndpointPreviewSupport(
    const std::vector<ArrowGuide>& branch,
    float median_hw) {
    if (branch.size() < 4) {
        return 0.0f;
    }

    const size_t point_count = branch.size();
    const ArrowGuide& endpoint = branch[point_count - 1];
    const ArrowGuide& neighbor1 = branch[point_count - 2];
    const ArrowGuide& neighbor2 = branch[point_count - 3];
    const ArrowGuide& neighbor3 = branch[point_count - 4];

    const float dx0 = guidePlanLateral(neighbor2) - guidePlanLateral(neighbor3);
    const float dx1 = guidePlanLateral(neighbor1) - guidePlanLateral(neighbor2);
    const float dx2 = guidePlanLateral(endpoint) - guidePlanLateral(neighbor1);
    const float abs_dx1 = std::abs(dx1);
    const float abs_dx2 = std::abs(dx2);
    const float tail_dy = std::abs(guidePlanForward(neighbor1) - guidePlanForward(endpoint));
    const bool same_turn =
        (dx1 * dx2 > 0.0f) &&
        (dx0 * dx1 >= 0.0f || abs_dx1 <= std::max(planMeters(0.04f), median_hw * 0.24f));
    if (!same_turn) {
        return 0.0f;
    }

    const float tail_dy_floor = 0.5f;
    const float tail_dy_hard_limit = std::max(planMeters(0.06f), median_hw * 0.45f);
    if (tail_dy < tail_dy_floor || tail_dy > tail_dy_hard_limit) {
        return 0.0f;
    }

    const float local_extrap_x =
        twoStepExtrapolateXAtY(neighbor2, neighbor1, guidePlanForward(endpoint));
    const float local_residual = std::abs(guidePlanLateral(endpoint) - local_extrap_x);
    const float local_residual_hard_limit =
        std::max(planMeters(0.185f), median_hw * 1.35f);
    if (local_residual > local_residual_hard_limit) {
        return 0.0f;
    }

    const float dx_growth_hard_limit =
        abs_dx1 + std::max(planMeters(0.20f), median_hw * 1.30f);
    if (abs_dx2 > dx_growth_hard_limit) {
        return 0.0f;
    }

    const float reach_score = clampValue(
        (abs_dx2 - 0.60f * abs_dx1) /
            std::max(planMeters(0.065f), median_hw * 0.55f),
        0.0f,
        1.0f);
    const float residual_score = clampValue(
        1.0f - local_residual / std::max(planMeters(0.13f), median_hw * 1.05f),
        0.0f,
        1.0f);
    const float short_hop_score = clampValue(
        (tail_dy_hard_limit - tail_dy) /
            std::max(1.0f, tail_dy_hard_limit - tail_dy_floor),
        0.0f,
        1.0f);
    const float lateral_reach_score = clampValue(
        (abs_dx2 - std::max(planMeters(0.06f), median_hw * 0.35f)) /
            std::max(planMeters(0.065f), median_hw * 0.50f),
        0.0f,
        1.0f);

    float support =
        0.40f * reach_score +
        0.25f * residual_score +
        0.20f * short_hop_score +
        0.15f * lateral_reach_score;

    return clampValue(support, 0.0f, 1.0f);
}

static bool twoStepBackEndpointLooksLikePreviewExtension(
    const std::vector<ArrowGuide>& branch,
    float median_hw) {
    return twoStepBackEndpointPreviewSupport(branch, median_hw) >= 0.52f;
}

static bool twoStepPruneEndpointContamination(std::vector<ArrowGuide>& branch,
                                              float median_hw,
                                              float branch_lateral_ref,
                                              PolyFit2* refit_out) {
    if (branch.size() < 4) {
        return false;
    }

    bool changed = false;
    for (int iter = 0; iter < 2 && branch.size() >= 4; ++iter) {
        float front_score = twoStepEndpointContaminationScore(
            branch, true, median_hw, branch_lateral_ref);
        float back_score = twoStepEndpointContaminationScore(
            branch, false, median_hw, branch_lateral_ref);
        const float front_prune_thresh = std::max(planMeters(0.11f), median_hw * 0.72f);
        const float back_prune_thresh = std::max(planMeters(0.23f), median_hw * 1.35f);
        if (front_score >= 0.0f && front_score < front_prune_thresh) {
            front_score = -1.0f;
        }
        if (back_score >= 0.0f) {
            if (twoStepBackEndpointLooksLikePreviewExtension(
                    branch, median_hw)) {
                back_score = -1.0f;
            } else if (back_score < back_prune_thresh) {
                back_score = -1.0f;
            }
        }
        if (front_score < 0.0f && back_score < 0.0f) {
            break;
        }
        if (front_score >= back_score) {
            branch.erase(branch.begin());
        } else {
            branch.pop_back();
        }
        changed = true;
    }

    if (changed && refit_out && branch.size() >= 3) {
        std::vector<bool> inlier_mask(branch.size(), true);
        const auto refit = refitPoly2LeastSquares(branch, inlier_mask);
        if (refit.valid) {
            *refit_out = refit;
        }
    }
    return changed;
}

static float twoStepInteriorSideReversalScore(const std::vector<ArrowGuide>& branch,
                                              size_t idx,
                                              float median_hw) {
    if (branch.size() < 5 || idx == 0 || idx + 1 >= branch.size()) {
        return -1.0f;
    }

    const ArrowGuide& prev = branch[idx - 1];
    const ArrowGuide& curr = branch[idx];
    const ArrowGuide& next = branch[idx + 1];

    const float dx_prev = guidePlanLateral(curr) - guidePlanLateral(prev);
    const float dx_next = guidePlanLateral(next) - guidePlanLateral(curr);
    if (dx_prev * dx_next >= 0.0f) {
        return -1.0f;
    }

    const float jump_prev = std::abs(dx_prev);
    const float jump_next = std::abs(dx_next);
    const float min_jump = std::min(jump_prev, jump_next);
    const float jump_thresh = std::max(planMeters(0.07f), median_hw * 0.75f);
    if (min_jump < jump_thresh) {
        return -1.0f;
    }

    const float interp_x = twoStepExtrapolateXAtY(prev, next, guidePlanForward(curr));
    const float local_residual = std::abs(guidePlanLateral(curr) - interp_x);
    const float local_residual_thresh = std::max(planMeters(0.065f), median_hw * 0.85f);
    if (local_residual < local_residual_thresh) {
        return -1.0f;
    }

    return min_jump + local_residual;
}

static bool twoStepPruneInternalSideReversal(std::vector<ArrowGuide>& branch,
                                             float median_hw,
                                             PolyFit2* refit_out) {
    if (branch.size() < 5) {
        return false;
    }

    bool changed = false;
    for (int iter = 0; iter < 2 && branch.size() >= 5; ++iter) {
        size_t best_idx = branch.size();
        float best_score = -1.0f;
        for (size_t i = 1; i + 1 < branch.size(); ++i) {
            const float score =
                twoStepInteriorSideReversalScore(branch, i, median_hw);
            if (score > best_score) {
                best_score = score;
                best_idx = i;
            }
        }
        if (best_idx >= branch.size() || best_score < 0.0f) {
            break;
        }
        branch.erase(branch.begin() + static_cast<std::ptrdiff_t>(best_idx));
        changed = true;
    }

    if (changed && refit_out && branch.size() >= 3) {
        std::vector<bool> inlier_mask(branch.size(), true);
        const auto refit = refitPoly2LeastSquares(branch, inlier_mask);
        if (refit.valid) {
            *refit_out = refit;
        }
    }
    return changed;
}

static void sortGuidesByDescendingForward(std::vector<ArrowGuide>& guides) {
    std::sort(guides.begin(), guides.end(), [](const ArrowGuide& a, const ArrowGuide& b) {
        const float af = guidePlanForward(a);
        const float bf = guidePlanForward(b);
        if (std::abs(af - bf) > 1e-3f) {
            return af < bf;
        }
        return guidePlanLateral(a) < guidePlanLateral(b);
    });
}

static PolyFit2 refitPoly2OnAllGuides(const std::vector<ArrowGuide>& guides) {
    if (guides.size() < 3) {
        return {};
    }
    std::vector<bool> inlier_mask(guides.size(), true);
    return refitPoly2LeastSquares(guides, inlier_mask);
}

static bool finalizeSingleBranchGuides(std::vector<ArrowGuide>& guides,
                                       float median_hw,
                                       float branch_lateral_ref,
                                       PolyFit2* fit_out) {
    if (guides.empty()) {
        if (fit_out) {
            *fit_out = PolyFit2{};
        }
        return false;
    }

    sortGuidesByDescendingForward(guides);
    bool changed = false;
    if (twoStepPruneEndpointContamination(
            guides, median_hw, branch_lateral_ref, fit_out)) {
        sortGuidesByDescendingForward(guides);
        changed = true;
    }
    if (twoStepPruneInternalSideReversal(
            guides, median_hw, fit_out)) {
        sortGuidesByDescendingForward(guides);
        changed = true;
    }

    if (fit_out && !fit_out->valid) {
        *fit_out = refitPoly2OnAllGuides(guides);
    }
    return changed;
}

static void attachPolyFitToSingleBranch(ArrowBranch& branch,
                                        const PolyFit2& fit) {
    if (!fit.valid || branch.guides.empty()) {
        return;
    }
    branch.has_poly_fit = true;
    branch.poly_a = fit.a;
    branch.poly_b = fit.b;
    branch.poly_c = fit.c;
    branch.poly_y_min = guidePlanForward(branch.guides.front());
    branch.poly_y_max = guidePlanForward(branch.guides.back());
}

static ArrowBranch buildSingleMainBranch(std::vector<ArrowGuide> guides,
                                         const PolyFit2& fit = PolyFit2{}) {
    sortGuidesByDescendingForward(guides);
    ArrowBranch main_out;
    main_out.guides = std::move(guides);
    main_out.anchor_x = computeBranchAnchorX(main_out.guides);
    main_out.label = "main";
    main_out.color = cv::Scalar(0, 0, 255);
    attachPolyFitToSingleBranch(main_out, fit);
    return main_out;
}

static ArrowBranch buildBoundaryVisibleBranch(std::vector<ArrowGuide> guides,
                                              const std::string& label,
                                              int shared_prefix_count) {
    sortGuidesByDescendingForward(guides);
    ArrowBranch branch;
    branch.guides = std::move(guides);
    branch.label = label;
    branch.shared_prefix_count =
        static_cast<int>(clampedSharedPrefixCount(branch.guides.size(), shared_prefix_count));
    branch.anchor_x =
        computeBranchAnchorXAfterSharedPrefix(branch.guides, branch.shared_prefix_count);
    if (label == "main") {
        branch.color = cv::Scalar(0, 0, 255);
    } else if (label == "right") {
        branch.color = cv::Scalar(0, 165, 255);
    } else {
        branch.color = cv::Scalar(255, 255, 255);
    }
    return branch;
}

static float guideWorldHalfWidthM(const ArrowGuide& guide) {
    if (isFiniteFloat(guide.half_width_world_m) && guide.half_width_world_m > 0.0f) {
        return guide.half_width_world_m;
    }
    const float plan_hw = guidePlanHalfWidth(guide);
    if (isFiniteFloat(plan_hw) && plan_hw > 0.0f) {
        return plan_hw / kWorldPlanScale;
    }
    return 0.10f;
}

static bool sameSingleGuideIdentity(const ArrowGuide& a, const ArrowGuide& b) {
    if (a.tracker_id >= 0 && a.tracker_id == b.tracker_id) {
        return true;
    }
    return std::abs(a.pos.x - b.pos.x) <= 0.5f &&
           std::abs(a.pos.y - b.pos.y) <= 0.5f;
}

static std::vector<ArrowGuide> collectValidSingleGuidesNearToFar(
    const std::vector<ArrowGuide>& guides) {
    std::vector<ArrowGuide> sorted;
    sorted.reserve(guides.size());
    for (const auto& guide : guides) {
        if (isFinitePoint(guide.pos) && hasValidWorldGuide(guide)) {
            sorted.push_back(guide);
        }
    }
    std::sort(sorted.begin(), sorted.end(), [](const ArrowGuide& a, const ArrowGuide& b) {
        const float af = worldForwardM(a);
        const float bf = worldForwardM(b);
        if (std::abs(af - bf) > 1e-4f) {
            return af < bf;
        }
        return worldLateralM(a) < worldLateralM(b);
    });

    std::vector<ArrowGuide> unique_guides;
    unique_guides.reserve(sorted.size());
    for (const auto& guide : sorted) {
        bool duplicate = false;
        for (const auto& kept : unique_guides) {
            if (sameSingleGuideIdentity(guide, kept)) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            unique_guides.push_back(guide);
        }
    }
    return unique_guides;
}

static float medianSingleGuideHalfWidthM(const std::vector<ArrowGuide>& guides) {
    if (guides.empty()) {
        return 0.10f;
    }
    std::vector<float> hws;
    hws.reserve(guides.size());
    for (const auto& guide : guides) {
        hws.push_back(guideWorldHalfWidthM(guide));
    }
    const size_t mid = hws.size() / 2;
    std::nth_element(hws.begin(), hws.begin() + static_cast<std::ptrdiff_t>(mid), hws.end());
    return clampValue(hws[mid], 0.04f, 0.24f);
}

static float singleSignedTurnDeg(const cv::Point2f& prev_unit,
                                 const cv::Point2f& next_unit) {
    const float cross = prev_unit.x * next_unit.y - prev_unit.y * next_unit.x;
    const float sign = cross < 0.0f ? -1.0f : 1.0f;
    return sign * angleBetweenUnitVectorsDegImpl(prev_unit, next_unit);
}

static bool chainContainsSingleGuide(const std::vector<ArrowGuide>& chain,
                                     const ArrowGuide& candidate) {
    for (const auto& guide : chain) {
        if (sameSingleGuideIdentity(guide, candidate)) {
            return true;
        }
    }
    return false;
}

static bool singleGuideWorldDirection(const ArrowGuide& from,
                                      const ArrowGuide& to,
                                      cv::Point2f& out_dir) {
    const cv::Point2f delta(worldLateralM(to) - worldLateralM(from),
                            worldForwardM(to) - worldForwardM(from));
    return normalizeVectorImpl(delta, out_dir, nullptr);
}

static SingleRecentTurnTrend computeRecentSingleTurnTrend(
    const std::vector<ArrowGuide>& path) {
    SingleRecentTurnTrend trend;
    if (path.size() < 4) {
        return trend;
    }

    static constexpr float kMeaningfulTurnDeg = 5.0f;
    int positive_count = 0;
    int negative_count = 0;
    size_t checked_turns = 0;
    for (size_t end = path.size() - 1; end >= 2 && checked_turns < 3; --end) {
        cv::Point2f prev_dir;
        cv::Point2f next_dir;
        if (singleGuideWorldDirection(path[end - 2], path[end - 1], prev_dir) &&
            singleGuideWorldDirection(path[end - 1], path[end], next_dir)) {
            const float turn_deg = singleSignedTurnDeg(prev_dir, next_dir);
            if (std::abs(turn_deg) >= kMeaningfulTurnDeg) {
                if (turn_deg > 0.0f) {
                    ++positive_count;
                } else {
                    ++negative_count;
                }
                ++trend.valid_turn_count;
            }
        }
        ++checked_turns;
        if (end == 2) {
            break;
        }
    }

    if (positive_count >= 2 || negative_count >= 2) {
        trend.stable = true;
        trend.sign = positive_count > negative_count ? 1 : -1;
    }
    return trend;
}

static constexpr size_t kSingleCoreScoreGuideCount = 5;
static constexpr size_t kSingleMinHealthyGuideCount = 3;
static constexpr float kSingleEarlyCoreMaxForwardGapM = 0.58f;
static constexpr float kSingleMinForwardStepM = 0.025f;
static constexpr float kSingleNearSameLayerMinForwardStepM = 0.008f;
static constexpr float kSingleNearSameLayerMaxForwardStepM = 0.090f;
static constexpr float kSingleNearSameLayerHeadingLimitDeg = 30.0f;
static constexpr float kSingleNearSameLayerTurnLimitDeg = 25.0f;
static constexpr float kSingleNormalSlopeLimit = 2.00f;
static constexpr float kForkContextTailBonusScale = 0.65f;
static constexpr float kAbruptChainSwitchPenalty = 0.12f;
static constexpr float kAbruptChainSwitchRootLateralJumpM = 0.25f;
static constexpr int kAbruptChainSwitchMaxSharedSuffixCount = 2;
static constexpr int kAbruptChainSwitchMinPrevSharedIdCount = 3;
static constexpr float kAbruptChainSwitchMinPrevSharedRatio = 0.50f;

static size_t singleCoreGuideCount(const std::vector<ArrowGuide>& guides) {
    return std::min(guides.size(), kSingleCoreScoreGuideCount);
}

static float singleGuideForwardSpanM(const std::vector<ArrowGuide>& guides,
                                     size_t begin_index,
                                     size_t end_index) {
    if (guides.empty() || begin_index >= guides.size() || end_index >= guides.size() ||
        end_index <= begin_index) {
        return 0.0f;
    }
    return std::max(0.0f, worldForwardM(guides[end_index]) - worldForwardM(guides[begin_index]));
}

static bool singleCandidateBreaksReferenceTurnTrend(
    const std::vector<ArrowGuide>& path,
    const ArrowGuide& candidate,
    const SingleRecentTurnTrend& reference_trend) {
    if (!reference_trend.stable || reference_trend.valid_turn_count < 2 || path.size() < 2) {
        return false;
    }

    cv::Point2f prev_dir;
    cv::Point2f curr_dir;
    if (!singleGuideWorldDirection(path[path.size() - 2], path.back(), prev_dir)) {
        return false;
    }
    const cv::Point2f curr_delta(worldLateralM(candidate) - worldLateralM(path.back()),
                                 worldForwardM(candidate) - worldForwardM(path.back()));
    if (!normalizeVectorImpl(curr_delta, curr_dir, nullptr)) {
        return false;
    }

    static constexpr float kCoreTrendFlipRejectDeg = 14.0f;
    const float curr_turn_deg = singleSignedTurnDeg(prev_dir, curr_dir);
    return reference_trend.sign * curr_turn_deg < 0.0f &&
           std::abs(curr_turn_deg) >= kCoreTrendFlipRejectDeg;
}

static bool singleTailBreaksCoreTrend(const std::vector<ArrowGuide>& path,
                                      const ArrowGuide& candidate) {
    if (path.size() < kSingleCoreScoreGuideCount) {
        return false;
    }
    const size_t core_count = singleCoreGuideCount(path);
    std::vector<ArrowGuide> core(path.begin(), path.begin() + static_cast<std::ptrdiff_t>(core_count));
    const SingleRecentTurnTrend core_trend = computeRecentSingleTurnTrend(core);
    return singleCandidateBreaksReferenceTurnTrend(path, candidate, core_trend);
}

static float singleCoreTransitionCostAverage(const std::vector<float>& transition_costs,
                                             size_t core_count) {
    if (core_count < 2 || transition_costs.empty()) {
        return 1.45f;
    }
    const size_t core_step_count = std::min(core_count - 1, transition_costs.size());
    if (core_step_count == 0) {
        return 1.45f;
    }
    float sum = 0.0f;
    for (size_t i = 0; i < core_step_count; ++i) {
        sum += transition_costs[i];
    }
    return sum / static_cast<float>(core_step_count);
}

static float singleTailCoverageBonus(const std::vector<ArrowGuide>& guides,
                                     size_t core_count) {
    if (guides.size() <= core_count || core_count == 0) {
        return 0.0f;
    }
    const size_t tail_count = guides.size() - core_count;
    const float tail_span =
        singleGuideForwardSpanM(guides, core_count - 1, guides.size() - 1);
    const float count_bonus = 0.035f * std::min<size_t>(tail_count, 3);
    const float span_bonus = 0.12f * clampValue(tail_span, 0.0f, 0.55f);
    return std::min(0.17f, count_bonus + span_bonus);
}

static bool singleEarlyCoreForwardGapTooLarge(const std::vector<ArrowGuide>& path,
                                              const ArrowGuide& candidate) {
    if (path.empty() || path.size() >= kSingleMinHealthyGuideCount) {
        return false;
    }
    const float gap_m = worldForwardM(candidate) - worldForwardM(path.back());
    return isFiniteFloat(gap_m) && gap_m > kSingleEarlyCoreMaxForwardGapM;
}

static SingleAnchorGrowTransition evaluateSingleAnchorGrowTransition(
    const std::vector<ArrowGuide>& path,
    const ArrowGuide& candidate,
    float median_hw_m) {
    SingleAnchorGrowTransition out;
    if (path.empty() || !hasValidWorldGuide(candidate)) {
        out.reject_reason = path.empty() ? "empty_path" : "invalid_candidate_world";
        return out;
    }

    const ArrowGuide& tail = path.back();
    const float gap_m = worldForwardM(candidate) - worldForwardM(tail);
    const float dx_m = worldLateralM(candidate) - worldLateralM(tail);
    out.gap_m = gap_m;
    out.lateral_step_m = dx_m;
    out.physical_distance_m = std::sqrt(gap_m * gap_m + dx_m * dx_m);
    out.slope = dx_m / std::max(gap_m, 1e-3f);

    if (!isFiniteFloat(gap_m) || !isFiniteFloat(dx_m)) {
        out.reject_reason = "invalid_delta";
        return out;
    }
    if (gap_m <= kSingleNearSameLayerMinForwardStepM) {
        out.reject_reason = "forward_gap_too_small";
        return out;
    }

    cv::Point2f prev_dir;
    cv::Point2f curr_dir;
    float heading_delta_deg = 0.0f;
    float curr_turn_deg = 0.0f;
    bool have_prev_curr_dirs = false;
    bool have_curr_turn = false;
    if (path.size() >= 2) {
        const ArrowGuide& prev = path[path.size() - 2];
        const cv::Point2f curr_delta(dx_m, gap_m);
        if (!singleGuideWorldDirection(prev, tail, prev_dir) ||
            !normalizeVectorImpl(curr_delta, curr_dir, nullptr)) {
            out.reject_reason = "invalid_direction";
            return out;
        }
        have_prev_curr_dirs = true;
        heading_delta_deg = angleBetweenUnitVectorsDegImpl(prev_dir, curr_dir);
        out.heading_delta_deg = heading_delta_deg;
        curr_turn_deg = singleSignedTurnDeg(prev_dir, curr_dir);
        have_curr_turn = true;
    }

    // [ALIGN-DEMO] kTrendContinueMaxTurnDeg removed — demo keeps trend_break
    // but does not have trend_continue / near_same_layer_trend fast-accept.
    static constexpr float kFlipRejectMinDeg = 14.0f;
    const SingleRecentTurnTrend recent_trend = computeRecentSingleTurnTrend(path);
    const bool has_stable_turn_trend =
        recent_trend.stable && recent_trend.valid_turn_count >= 2 && have_curr_turn;
    const bool trend_break =
        has_stable_turn_trend &&
        recent_trend.sign * curr_turn_deg < 0.0f &&
        std::abs(curr_turn_deg) >= kFlipRejectMinDeg;
    if (trend_break) {
        out.reject_reason = "trend_break";
        return out;
    }

    // [ALIGN-DEMO] trend_continue removed — demo does not fast-accept
    // candidates based on turn-trend continuity.

    const float local_hw_m = std::max(median_hw_m, guideWorldHalfWidthM(tail));
    const float abs_dx_m = std::abs(dx_m);
    const float slope = dx_m / std::max(gap_m, 1e-3f);
    out.slope = slope;

    float pred_err_m = 0.0f;
    bool have_pred_err = false;
    if (path.size() >= 2 && have_prev_curr_dirs) {
        const ArrowGuide& prev = path[path.size() - 2];
        const float prev_gap_m = worldForwardM(tail) - worldForwardM(prev);
        const float prev_slope =
            (worldLateralM(tail) - worldLateralM(prev)) / std::max(prev_gap_m, 1e-3f);
        const float predicted_lat_m = worldLateralM(tail) + prev_slope * gap_m;
        pred_err_m = std::abs(worldLateralM(candidate) - predicted_lat_m);
        out.prediction_residual_m = pred_err_m;
        have_pred_err = true;
    }

    float turn_delta_deg = 0.0f;
    bool have_turn_delta = false;
    if (path.size() >= 3 && have_prev_curr_dirs && have_curr_turn) {
        const ArrowGuide& prev = path[path.size() - 2];
        const ArrowGuide& before_prev = path[path.size() - 3];
        cv::Point2f before_dir;
        if (!singleGuideWorldDirection(before_prev, prev, before_dir)) {
            out.reject_reason = "invalid_turn_direction";
            return out;
        }
        const float prev_turn_deg = singleSignedTurnDeg(before_dir, prev_dir);
        turn_delta_deg = std::abs(curr_turn_deg - prev_turn_deg);
        out.turn_delta_deg = turn_delta_deg;
        have_turn_delta = true;
    }

    auto recentForwardLikeMainTrend = [&]() -> bool {
        if (path.size() < 4) {
            return false;
        }
        float max_abs_slope = 0.0f;
        float slope_sum = 0.0f;
        size_t valid_edges = 0;
        const size_t first = path.size() - 4;
        for (size_t i = first + 1; i < path.size(); ++i) {
            const float edge_gap_m =
                worldForwardM(path[i]) - worldForwardM(path[i - 1]);
            const float edge_dx_m =
                worldLateralM(path[i]) - worldLateralM(path[i - 1]);
            if (!isFiniteFloat(edge_gap_m) || !isFiniteFloat(edge_dx_m) ||
                edge_gap_m <= kSingleNearSameLayerMinForwardStepM) {
                return false;
            }
            const float edge_slope = edge_dx_m / std::max(edge_gap_m, 1e-3f);
            const float abs_edge_slope = std::abs(edge_slope);
            max_abs_slope = std::max(max_abs_slope, abs_edge_slope);
            slope_sum += abs_edge_slope;
            ++valid_edges;
        }
        if (valid_edges < 3) {
            return false;
        }
        const float avg_abs_slope = slope_sum / static_cast<float>(valid_edges);
        return max_abs_slope <= 1.20f && avg_abs_slope <= 0.95f;
    };

    auto acceptNearSameLayerIfContinuous = [&](const char* gate_reason) -> bool {
        if (path.size() < 2 || !have_prev_curr_dirs || !have_pred_err) {
            return false;
        }
        if (gap_m > kSingleNearSameLayerMaxForwardStepM) {
            return false;
        }
        const float max_dist_m = 0.075f + 1.55f * local_hw_m + 0.75f * gap_m;
        if (out.physical_distance_m > max_dist_m) {
            out.reject_reason = gate_reason;
            return false;
        }
        const float max_abs_dx_m = 0.050f + 1.45f * local_hw_m + 0.80f * gap_m;
        if (abs_dx_m > max_abs_dx_m) {
            out.reject_reason = gate_reason;
            return false;
        }
        if (heading_delta_deg > kSingleNearSameLayerHeadingLimitDeg) {
            out.reject_reason = gate_reason;
            return false;
        }
        const float pred_limit_m = std::max(0.11f, 0.035f + 1.10f * local_hw_m + 0.35f * gap_m);
        if (pred_err_m > pred_limit_m) {
            out.reject_reason = gate_reason;
            return false;
        }
        if (have_turn_delta && turn_delta_deg > kSingleNearSameLayerTurnLimitDeg) {
            out.reject_reason = gate_reason;
            return false;
        }

        const float near_cost =
            0.24f * out.physical_distance_m +
            0.85f * abs_dx_m +
            0.012f * heading_delta_deg +
            1.20f * pred_err_m +
            (have_turn_delta ? 0.010f * turn_delta_deg : 0.0f);
        float final_cost = near_cost;
        out.cost_before_near_penalty = near_cost;
        out.cost_after_near_penalty = near_cost;

        const bool lateral_jump_like =
            std::string(gate_reason) == "near_same_layer_slope" &&
            recentForwardLikeMainTrend() &&
            gap_m > 0.04f &&
            std::abs(slope) > 2.0f &&
            heading_delta_deg > 20.0f &&
            have_turn_delta &&
            turn_delta_deg > 15.0f;
        if (lateral_jump_like) {
            const float slope_excess =
                clampValue((std::abs(slope) - 2.0f) / 1.50f, 0.0f, 1.0f);
            const float heading_excess =
                clampValue((heading_delta_deg - 20.0f) / 25.0f, 0.0f, 1.0f);
            const float turn_excess =
                clampValue((turn_delta_deg - 15.0f) / 25.0f, 0.0f, 1.0f);
            const float soft_penalty =
                0.45f + 0.10f * slope_excess +
                0.10f * heading_excess +
                0.10f * turn_excess;
            const float normal_cost =
                0.35f * gap_m +
                1.60f * abs_dx_m +
                0.18f * std::abs(slope) +
                0.021f * heading_delta_deg +
                1.80f * pred_err_m +
                (have_turn_delta ? 0.016f * turn_delta_deg : 0.0f);
            const float normal_floor = 0.85f * normal_cost;
            final_cost = std::max(near_cost + soft_penalty, normal_floor);
            out.near_same_layer_lateral_jump_like = true;
            out.near_same_layer_soft_penalty = soft_penalty;
            out.cost_after_near_penalty = final_cost;
            out.demoted_to_normal_cost = final_cost >= normal_floor - 1e-4f;
        }

        out.feasible = true;
        out.cost = final_cost;
        out.near_same_layer_continuation = true;
        out.near_same_layer_accept_reason = gate_reason;
        out.reject_reason = "accepted";
        return true;
    };

    if (gap_m <= kSingleMinForwardStepM &&
        acceptNearSameLayerIfContinuous("near_same_layer_forward_gap")) {
        return out;
    }
    if (gap_m <= kSingleMinForwardStepM) {
        out.reject_reason = "forward_gap_too_small";
        return out;
    }

    if (gap_m > 0.72f) {
        out.reject_reason = "gap_too_large";
        return out;
    }
    const float max_lateral_step_m =
        0.10f + 0.62f * gap_m + 1.35f * local_hw_m;
    if (abs_dx_m > max_lateral_step_m) {
        out.reject_reason = "lateral_step";
        return out;
    }
    if (std::abs(slope) > kSingleNormalSlopeLimit) {
        if (acceptNearSameLayerIfContinuous("near_same_layer_slope")) {
            return out;
        }
        out.reject_reason = "slope";
        return out;
    }

    float cost = 0.35f * gap_m + 1.60f * abs_dx_m + 0.18f * std::abs(slope);
    if (path.size() >= 2) {
        if (!have_prev_curr_dirs) {
            out.reject_reason = "missing_heading";
            return out;
        }

        const float heading_limit_deg =
            gap_m < 0.09f ? 72.0f : 64.0f;
        if (heading_delta_deg > heading_limit_deg) {
            out.reject_reason = "heading";
            return out;
        }

        const float pred_limit_m = 0.10f + 0.58f * gap_m + 0.90f * local_hw_m;
        if (pred_err_m > pred_limit_m && heading_delta_deg > 42.0f) {
            out.reject_reason = "prediction";
            return out;
        }

        cost += 0.021f * heading_delta_deg + 1.80f * pred_err_m;

        if (path.size() >= 3) {
            if (turn_delta_deg > 70.0f) {
                out.reject_reason = "turn_delta";
                return out;
            }
            cost += 0.016f * turn_delta_deg;
        }
    }

    out.feasible = cost <= 2.85f;
    out.cost = cost;
    out.reject_reason = out.feasible ? "accepted" : "cost";
    return out;
}

static std::string formatSingleGrowGuideIds(const std::vector<ArrowGuide>& guides) {
    std::ostringstream oss;
    for (size_t i = 0; i < guides.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << guides[i].tracker_id;
    }
    return oss.str();
}

static std::string formatFirstGuideIdsCompact(const std::vector<ArrowGuide>& guides,
                                              size_t max_count) {
    std::ostringstream oss;
    const size_t count = std::min(guides.size(), max_count);
    for (size_t i = 0; i < count; ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << guides[i].tracker_id;
    }
    return oss.str();
}

static std::string formatTrackerIdsCompact(const std::vector<int>& ids) {
    if (ids.empty()) {
        return "none";
    }
    std::ostringstream oss;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << ids[i];
    }
    return oss.str();
}

static bool evaluateSingleGrowFeasibleCandidate(
    const std::vector<ArrowGuide>& guides,
    const std::vector<ArrowGuide>& path,
    size_t candidate_index,
    float median_hw_m,
    SingleAnchorGrowTransition* out_transition) {
    if (out_transition != nullptr) {
        *out_transition = SingleAnchorGrowTransition{};
    }
    if (candidate_index >= guides.size() || path.empty() ||
        chainContainsSingleGuide(path, guides[candidate_index])) {
        if (out_transition != nullptr) out_transition->reject_reason = "invalid_candidate";
        return false;
    }
    if (worldForwardM(guides[candidate_index]) <=
        worldForwardM(path.back()) + kSingleNearSameLayerMinForwardStepM) {
        if (out_transition != nullptr) out_transition->reject_reason = "forward_gap_too_small";
        return false;
    }
    if (singleEarlyCoreForwardGapTooLarge(path, guides[candidate_index])) {
        if (out_transition != nullptr) out_transition->reject_reason = "early_core_forward_gap";
        return false;
    }
    SingleAnchorGrowTransition transition =
        evaluateSingleAnchorGrowTransition(path, guides[candidate_index], median_hw_m);
    if (!transition.feasible) {
        if (out_transition != nullptr) *out_transition = transition;
        return false;
    }
    if (singleTailBreaksCoreTrend(path, guides[candidate_index])) {
        transition.feasible = false;
        transition.reject_reason = "tail_breaks_core_trend";
        if (out_transition != nullptr) *out_transition = transition;
        return false;
    }
    if (out_transition != nullptr) *out_transition = transition;
    return true;
}

static float singleGrowSelectionCostWithLookahead(
    const std::vector<ArrowGuide>& guides,
    const std::vector<ArrowGuide>& path,
    size_t candidate_index,
    const SingleAnchorGrowTransition& transition,
    float median_hw_m) {
    float selection_cost = transition.cost;
    std::vector<ArrowGuide> lookahead_path = path;
    lookahead_path.push_back(guides[candidate_index]);
    bool has_farther_guide = false;
    float best_next_cost = std::numeric_limits<float>::infinity();
    for (size_t j = 0; j < guides.size(); ++j) {
        if (candidate_index == j || chainContainsSingleGuide(lookahead_path, guides[j])) {
            continue;
        }
        if (worldForwardM(guides[j]) <=
            worldForwardM(guides[candidate_index]) + kSingleNearSameLayerMinForwardStepM) {
            continue;
        }
        if (singleEarlyCoreForwardGapTooLarge(lookahead_path, guides[j])) {
            continue;
        }
        has_farther_guide = true;
        const SingleAnchorGrowTransition next_transition =
            evaluateSingleAnchorGrowTransition(lookahead_path, guides[j], median_hw_m);
        if (next_transition.feasible &&
            !singleTailBreaksCoreTrend(lookahead_path, guides[j])) {
            best_next_cost = std::min(best_next_cost, next_transition.cost);
        }
    }
    if (std::isfinite(best_next_cost)) {
        selection_cost += 0.18f * best_next_cost;
    } else if (has_farther_guide) {
        selection_cost += 0.42f;
    }
    return selection_cost;
}

static void finalizeSingleAnchorGrowPathScore(
    SingleAnchorGrowPath& path,
    float nearest_forward_m,
    float fork_tail_scale) {
    if (path.guides.empty()) {
        return;
    }
    path.core_count = singleCoreGuideCount(path.guides);
    path.tail_count = path.guides.size() - path.core_count;
    const float core_span_m =
        path.core_count >= 2
            ? singleGuideForwardSpanM(path.guides, 0, path.core_count - 1)
            : 0.0f;
    const float core_avg_continuity =
        singleCoreTransitionCostAverage(path.transition_costs, path.core_count);
    const float root_forward_penalty =
        std::max(0.0f, worldForwardM(path.guides.front()) - nearest_forward_m);
    const float root_forward_score_term = 1.35f * root_forward_penalty;
    const float root_lateral_soft_penalty =
        0.12f * std::abs(worldLateralM(path.guides.front()));
    const float core_count_bonus =
        0.10f * static_cast<float>(std::min(path.core_count, kSingleCoreScoreGuideCount));
    const float core_span_bonus =
        path.core_count >= kSingleMinHealthyGuideCount
            ? 0.24f * clampValue(core_span_m, 0.0f, 0.75f)
            : 0.0f;
    const float tail_bonus =
        singleTailCoverageBonus(path.guides, path.core_count) * fork_tail_scale;
    path.credible = path.guides.size() >= kSingleMinHealthyGuideCount;
    path.score = core_avg_continuity + root_forward_score_term +
                 root_lateral_soft_penalty - core_count_bonus -
                 core_span_bonus - tail_bonus;
}

static SingleAnchorGrowPath growSinglePathFromRoot(const std::vector<ArrowGuide>& guides,
                                                   size_t root_index,
                                                   float median_hw_m,
                                                   float nearest_forward_m,
                                                   uint32_t frame_id,
                                                   float fork_tail_scale = 1.0f) {
    (void)frame_id;
    SingleAnchorGrowPath path;
    if (root_index >= guides.size()) {
        return path;
    }

    path.guides.push_back(guides[root_index]);
    while (true) {
        size_t best_index = guides.size();
        SingleAnchorGrowTransition best_transition;
        float best_selection_cost = std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < guides.size(); ++i) {
            SingleAnchorGrowTransition transition;
            if (!evaluateSingleGrowFeasibleCandidate(
                    guides, path.guides, i, median_hw_m, &transition)) {
                continue;
            }
            const float selection_cost = singleGrowSelectionCostWithLookahead(
                guides, path.guides, i, transition, median_hw_m);

            if (best_index >= guides.size() || selection_cost < best_selection_cost) {
                best_index = i;
                best_transition = transition;
                best_selection_cost = selection_cost;
            }
        }
        if (best_index >= guides.size()) {
            break;
        }
        path.continuity_cost += best_transition.cost;
        path.transition_costs.push_back(best_transition.cost);
        path.guides.push_back(guides[best_index]);
    }

    finalizeSingleAnchorGrowPathScore(path, nearest_forward_m, fork_tail_scale);
    return path;
}

static int historicalIdOrderIndex(const std::vector<int>& ids, int tracker_id) {
    if (tracker_id < 0) return -1;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] == tracker_id) return static_cast<int>(i);
    }
    return -1;
}

static int visibleHistoricalIdCount(const std::vector<ArrowGuide>& guides,
                                    const std::vector<int>& ids) {
    int count = 0;
    for (const int id : ids) {
        for (const auto& guide : guides) {
            if (guide.tracker_id == id) {
                ++count;
                break;
            }
        }
    }
    return count;
}

static SingleAnchorGrowPath growHistoricalMainGuidedPath(
    const std::vector<ArrowGuide>& guides,
    const std::vector<int>& historical_main_ids,
    float median_hw_m,
    float nearest_forward_m,
    float fork_tail_scale,
    std::string* reject_reason) {
    if (reject_reason != nullptr) *reject_reason = "not_evaluated";
    SingleAnchorGrowPath path;
    if (guides.empty() || historical_main_ids.empty()) {
        if (reject_reason != nullptr) *reject_reason = "empty_input";
        return path;
    }
    size_t root_index = guides.size();
    int root_history_pos = -1;
    for (size_t h = 0; h < historical_main_ids.size(); ++h) {
        for (size_t i = 0; i < guides.size(); ++i) {
            if (guides[i].tracker_id == historical_main_ids[h]) {
                root_index = i;
                root_history_pos = static_cast<int>(h);
                break;
            }
        }
        if (root_index < guides.size()) break;
    }
    if (root_index >= guides.size()) {
        if (reject_reason != nullptr) *reject_reason = "no_visible_historical_root";
        return path;
    }

    path.guides.push_back(guides[root_index]);
    int next_history_pos = root_history_pos + 1;
    while (true) {
        size_t best_index = guides.size();
        int best_history_order = std::numeric_limits<int>::max();
        float best_selection_cost = std::numeric_limits<float>::infinity();
        SingleAnchorGrowTransition best_transition;
        std::string historical_reject = "no_remaining_historical_candidate";
        for (size_t i = 0; i < guides.size(); ++i) {
            SingleAnchorGrowTransition transition;
            const bool feasible = evaluateSingleGrowFeasibleCandidate(
                guides, path.guides, i, median_hw_m, &transition);
            const int order = historicalIdOrderIndex(
                historical_main_ids, guides[i].tracker_id);
            if (!feasible) {
                if (order >= next_history_pos &&
                    historical_reject == "no_remaining_historical_candidate") {
                    historical_reject = transition.reject_reason;
                }
                continue;
            }
            if (order < next_history_pos) continue;
            const float selection_cost = singleGrowSelectionCostWithLookahead(
                guides, path.guides, i, transition, median_hw_m);
            if (best_index >= guides.size() || order < best_history_order ||
                (order == best_history_order && selection_cost < best_selection_cost)) {
                best_index = i;
                best_history_order = order;
                best_selection_cost = selection_cost;
                best_transition = transition;
            }
        }
        if (best_index >= guides.size()) {
            if (path.guides.size() < kSingleMinHealthyGuideCount &&
                reject_reason != nullptr) {
                *reject_reason = historical_reject;
            }
            break;
        }
        path.continuity_cost += best_transition.cost;
        path.transition_costs.push_back(best_transition.cost);
        path.guides.push_back(guides[best_index]);
        next_history_pos = best_history_order + 1;
    }
    finalizeSingleAnchorGrowPathScore(path, nearest_forward_m, fork_tail_scale);
    if (reject_reason != nullptr && path.credible) *reject_reason = "none";
    else if (reject_reason != nullptr && *reject_reason == "not_evaluated") {
        *reject_reason = "not_credible";
    }
    return path;
}

static std::vector<size_t> selectSingleAnchorRootCandidates(
    const std::vector<ArrowGuide>& guides,
    float median_hw_m) {
    std::vector<size_t> roots;
    if (guides.empty()) {
        return roots;
    }

    const float nearest_forward_m = worldForwardM(guides.front());
    const float near_window_m = clampValue(2.60f * median_hw_m, 0.18f, 0.38f);
    struct RootCandidate {
        size_t index = 0;
        float score = 0.0f;
    };
    std::vector<RootCandidate> candidates;
    for (size_t i = 0; i < guides.size(); ++i) {
        const float forward_delta = worldForwardM(guides[i]) - nearest_forward_m;
        if (forward_delta > near_window_m && candidates.size() >= 3) {
            break;
        }
        if (forward_delta > near_window_m && i >= 5) {
            break;
        }

        float best_first_step_cost = 1.10f;
        const std::vector<ArrowGuide> singleton{guides[i]};
        for (size_t j = 0; j < guides.size(); ++j) {
            if (i == j) {
                continue;
            }
            const SingleAnchorGrowTransition transition =
                evaluateSingleAnchorGrowTransition(singleton, guides[j], median_hw_m);
            if (transition.feasible) {
                best_first_step_cost = std::min(best_first_step_cost, transition.cost);
            }
        }

        RootCandidate candidate;
        candidate.index = i;
        candidate.score = 1.80f * std::max(0.0f, forward_delta) +
                          0.14f * std::abs(worldLateralM(guides[i])) +
                          best_first_step_cost;
        candidates.push_back(candidate);
    }

    if (candidates.empty()) {
        roots.push_back(0);
        return roots;
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const RootCandidate& a, const RootCandidate& b) {
                  if (std::abs(a.score - b.score) > 1e-4f) {
                      return a.score < b.score;
                  }
                  return a.index < b.index;
              });

    roots.push_back(0);
    for (const auto& candidate : candidates) {
        if (roots.size() >= 3) {
            break;
        }
        if (std::find(roots.begin(), roots.end(), candidate.index) == roots.end()) {
            roots.push_back(candidate.index);
        }
    }
    std::sort(roots.begin(), roots.end());
    return roots;
}

struct ForkContextMainTieBreakDecision {
    bool fork_context = false;
    float score_margin = std::numeric_limits<float>::infinity();
    std::string primary_ids = "none";
    std::string secondary_ids = "none";
    bool split_not_taken = false;
    bool primary_consumes_right_fork = false;
    bool secondary_preserves_right_fork = false;
    bool tail_bonus_limited_by_fork_context = false;
    bool core_span_bonus_limited_by_fork_context = false;
    bool single_right_diverge_soft_penalty_enabled = false;
    float penalty_value = 0.0f;
    float adjusted_score = 0.0f;
    bool selected_secondary = false;
    std::string selected_ids = "none";
    std::string reason = "not_evaluated";
};

static int sharedPrefixGuideCountForSingle(const std::vector<ArrowGuide>& a,
                                           const std::vector<ArrowGuide>& b) {
    int count = 0;
    while (static_cast<size_t>(count) < a.size() &&
           static_cast<size_t>(count) < b.size() &&
           a[static_cast<size_t>(count)].tracker_id ==
               b[static_cast<size_t>(count)].tracker_id) {
        ++count;
    }
    return count;
}

static bool findEarlySharedAnchorForSingle(const std::vector<ArrowGuide>& a,
                                           const std::vector<ArrowGuide>& b,
                                           int* out_a_idx,
                                           int* out_b_idx) {
    if (out_a_idx != nullptr) {
        *out_a_idx = -1;
    }
    if (out_b_idx != nullptr) {
        *out_b_idx = -1;
    }
    const size_t a_limit = std::min<size_t>(a.size(), 3);
    const size_t b_limit = std::min<size_t>(b.size(), 3);
    for (size_t ai = 0; ai < a_limit; ++ai) {
        for (size_t bi = 0; bi < b_limit; ++bi) {
            if (a[ai].tracker_id >= 0 && a[ai].tracker_id == b[bi].tracker_id) {
                if (out_a_idx != nullptr) {
                    *out_a_idx = static_cast<int>(ai);
                }
                if (out_b_idx != nullptr) {
                    *out_b_idx = static_cast<int>(bi);
                }
                return true;
            }
        }
    }
    return false;
}

static float averageLateralDeltaInOverlapM(const std::vector<ArrowGuide>& probe,
                                           const std::vector<ArrowGuide>& reference,
                                           int start_probe_idx,
                                           int* out_sample_count) {
    if (out_sample_count != nullptr) {
        *out_sample_count = 0;
    }
    if (probe.empty() || reference.size() < 2) {
        return 0.0f;
    }
    float delta_sum = 0.0f;
    int count = 0;
    const size_t start = static_cast<size_t>(std::max(0, start_probe_idx));
    for (size_t i = start; i < probe.size(); ++i) {
        const float forward = guidePlanForward(probe[i]);
        const float lateral = guidePlanLateral(probe[i]);
        float reference_lateral = 0.0f;
        if (!isFiniteFloat(forward) || !isFiniteFloat(lateral) ||
            !interpolateGuidePlanLateralAtForward(reference, forward, reference_lateral)) {
            continue;
        }
        delta_sum += (lateral - reference_lateral) / kWorldPlanScale;
        ++count;
    }
    if (out_sample_count != nullptr) {
        *out_sample_count = count;
    }
    return count > 0 ? delta_sum / static_cast<float>(count) : 0.0f;
}

static ForkContextMainTieBreakDecision evaluateForkContextMainTieBreak(
    const SingleAnchorGrowPath& primary,
    const SingleAnchorGrowPath& secondary,
    bool split_not_taken) {
    ForkContextMainTieBreakDecision decision;
    decision.primary_ids = formatSingleGrowGuideIds(primary.guides);
    decision.secondary_ids = formatSingleGrowGuideIds(secondary.guides);
    decision.selected_ids = decision.primary_ids;
    decision.split_not_taken = split_not_taken;
    decision.adjusted_score = primary.score;
    if (!primary.credible || !secondary.credible ||
        primary.guides.size() < 3 || secondary.guides.size() < 3 ||
        !isFiniteFloat(primary.score) || !isFiniteFloat(secondary.score)) {
        decision.reason = "insufficient_credible_paths";
        return decision;
    }

    decision.score_margin = secondary.score - primary.score;
    static constexpr float kForkContextTightMargin = 0.12f;
    static constexpr float kForkContextLateralSeparationM = 0.07f;
    static constexpr float kForkContextStrongLateralConsumptionM = 0.12f;
    static constexpr float kSingleRightDivergeSoftPenalty = 0.04f;

    int primary_anchor_idx = -1;
    int secondary_anchor_idx = -1;
    const int prefix_count = sharedPrefixGuideCountForSingle(primary.guides, secondary.guides);
    const bool has_shared_anchor =
        prefix_count > 0 ||
        findEarlySharedAnchorForSingle(primary.guides,
                                       secondary.guides,
                                       &primary_anchor_idx,
                                       &secondary_anchor_idx);
    if (prefix_count > 0) {
        primary_anchor_idx = prefix_count - 1;
        secondary_anchor_idx = prefix_count - 1;
    }
    if (!has_shared_anchor) {
        decision.reason = "no_shared_prefix_or_early_anchor";
        return decision;
    }

    int primary_samples = 0;
    int secondary_samples = 0;
    const float primary_minus_secondary =
        averageLateralDeltaInOverlapM(primary.guides,
                                      secondary.guides,
                                      primary_anchor_idx + 1,
                                      &primary_samples);
    const float secondary_minus_primary =
        averageLateralDeltaInOverlapM(secondary.guides,
                                      primary.guides,
                                      secondary_anchor_idx + 1,
                                      &secondary_samples);
    const float lateral_separation =
        0.5f * (std::abs(primary_minus_secondary) + std::abs(secondary_minus_primary));
    const bool separated_after_anchor =
        (primary_samples >= 2 || secondary_samples >= 2) &&
        lateral_separation >= kForkContextLateralSeparationM;
    if (!separated_after_anchor) {
        decision.reason = "no_post_anchor_lateral_separation";
        return decision;
    }

    const bool tight_margin =
        decision.score_margin >= 0.0f &&
        decision.score_margin <= kForkContextTightMargin;
    const bool primary_right_of_secondary =
        primary_minus_secondary >= kForkContextLateralSeparationM ||
        -secondary_minus_primary >= kForkContextLateralSeparationM;
    const bool strong_primary_right_consumption =
        primary_minus_secondary >= kForkContextStrongLateralConsumptionM ||
        -secondary_minus_primary >= kForkContextStrongLateralConsumptionM;

    decision.primary_consumes_right_fork =
        primary_right_of_secondary &&
        (tight_margin || strong_primary_right_consumption);
    decision.secondary_preserves_right_fork =
        decision.primary_consumes_right_fork && secondary.guides.size() >= 3;
    decision.fork_context =
        decision.primary_consumes_right_fork && decision.secondary_preserves_right_fork;

    if (!decision.fork_context) {
        decision.reason = "separated_but_margin_not_tight";
        return decision;
    }

    decision.tail_bonus_limited_by_fork_context =
        split_not_taken && tight_margin && decision.primary_consumes_right_fork;
    decision.core_span_bonus_limited_by_fork_context =
        split_not_taken && tight_margin && decision.primary_consumes_right_fork;
    if (!split_not_taken) {
        decision.reason = "split_state_active_or_confirmed";
        return decision;
    }
    if (!tight_margin) {
        decision.reason = "fork_context_margin_not_close";
        return decision;
    }
    decision.single_right_diverge_soft_penalty_enabled = true;
    decision.penalty_value = kSingleRightDivergeSoftPenalty;
    decision.adjusted_score = primary.score + decision.penalty_value;
    decision.selected_secondary = decision.adjusted_score > secondary.score;
    if (decision.selected_secondary) {
        decision.selected_ids = decision.secondary_ids;
        decision.reason = "secondary_selected_after_right_diverge_soft_penalty";
    } else {
        decision.reason = "primary_kept_after_right_diverge_soft_penalty";
    }
    return decision;
}

static float guideIdOverlapRatioWithConfirmedRightFork(
    const std::vector<ArrowGuide>& guides,
    const std::vector<int>& confirmed_right_fork_ids) {
    if (guides.empty() || confirmed_right_fork_ids.empty()) {
        return 0.0f;
    }
    std::unordered_set<int> confirmed_ids(confirmed_right_fork_ids.begin(),
                                          confirmed_right_fork_ids.end());
    int match_count = 0;
    for (const auto& guide : guides) {
        if (guide.tracker_id >= 0 && confirmed_ids.count(guide.tracker_id) > 0) {
            ++match_count;
        }
    }
    const float denominator = static_cast<float>(
        std::min(guides.size(), confirmed_right_fork_ids.size()));
    return denominator > 0.0f
        ? clampValue(static_cast<float>(match_count) / denominator, 0.0f, 1.0f)
        : 0.0f;
}

static void applyConfirmedRightForkMainIdentityGuard(
    const SingleAnchorGrowPath& primary,
    const SingleAnchorGrowPath& secondary,
    uint32_t frame_id,
    const BoundaryTrackManager* boundary_tracks,
    ForkContextMainTieBreakDecision* decision) {
    if (decision == nullptr || boundary_tracks == nullptr ||
        !primary.credible || !secondary.credible ||
        primary.guides.size() < 3 || secondary.guides.size() < 3 ||
        boundary_tracks->last_confirmed_right_fork_ids.empty()) {
        return;
    }

    static constexpr uint32_t kRecentConfirmedRightForkFrameWindow = 2;
    static constexpr float kConfirmedRightForkHighOverlap = 0.75f;
    static constexpr float kConfirmedRightForkLowOverlap = 0.35f;
    static constexpr float kConfirmedRightForkOverlapGap = 0.50f;
    static constexpr float kConfirmedRightForkTightScoreMargin = 0.12f;

    const bool split_active =
        boundary_tracks->visible_split_output_active ||
        boundary_tracks->visible_split_maintain_age > 0;
    const bool recently_split =
        boundary_tracks->last_confirmed_right_fork_frame > 0 &&
        frame_id >= boundary_tracks->last_confirmed_right_fork_frame &&
        frame_id - boundary_tracks->last_confirmed_right_fork_frame <=
            kRecentConfirmedRightForkFrameWindow;
    if (!split_active && !recently_split) {
        return;
    }

    const float primary_right_overlap =
        guideIdOverlapRatioWithConfirmedRightFork(
            primary.guides, boundary_tracks->last_confirmed_right_fork_ids);
    const float secondary_right_overlap =
        guideIdOverlapRatioWithConfirmedRightFork(
            secondary.guides, boundary_tracks->last_confirmed_right_fork_ids);
    const float score_margin = secondary.score - primary.score;
    const bool tight_score_margin =
        score_margin >= 0.0f &&
        score_margin <= kConfirmedRightForkTightScoreMargin;
    const bool primary_matches_confirmed_right =
        primary_right_overlap >= kConfirmedRightForkHighOverlap;
    const bool secondary_is_distinct_main_candidate =
        secondary_right_overlap <= kConfirmedRightForkLowOverlap &&
        primary_right_overlap - secondary_right_overlap >=
            kConfirmedRightForkOverlapGap;

    if (!tight_score_margin || !primary_matches_confirmed_right ||
        !secondary_is_distinct_main_candidate) {
        return;
    }

    decision->selected_secondary = true;
    decision->selected_ids = decision->secondary_ids;
    decision->reason =
        "secondary_selected_to_preserve_confirmed_right_fork_identity";
}

static std::vector<int> parseTrackerIdsText(const std::string& ids_text) {
    std::vector<int> ids;
    std::istringstream iss(ids_text);
    std::string token;
    while (std::getline(iss, token, ',')) {
        if (token.empty() || token == "none") {
            continue;
        }
        try {
            size_t parsed = 0;
            const int id = std::stoi(token, &parsed);
            if (parsed == token.size()) {
                ids.push_back(id);
            }
        } catch (...) {
        }
    }
    return ids;
}

static std::vector<int> collectIndependentRecoveryRightSeedIds(
    const std::string& primary_ids_text,
    const std::string& fallback_ids_text,
    const std::vector<int>& main_seed_ids) {
    const std::string& selected_ids_text =
        primary_ids_text != "none" ? primary_ids_text : fallback_ids_text;
    const std::vector<int> parsed_ids = parseTrackerIdsText(selected_ids_text);
    const std::unordered_set<int> main_ids(main_seed_ids.begin(),
                                           main_seed_ids.end());
    std::unordered_set<int> seen_ids;
    std::vector<int> independent_ids;
    independent_ids.reserve(parsed_ids.size());
    for (const int id : parsed_ids) {
        if (id < 0 || main_ids.count(id) > 0 || !seen_ids.insert(id).second) {
            continue;
        }
        independent_ids.push_back(id);
    }
    return independent_ids;
}

static int sharedIdCountWithIds(const std::vector<ArrowGuide>& guides,
                                const std::vector<int>& ids) {
    int count = 0;
    for (const auto& guide : guides) {
        for (const int id : ids) {
            if (guide.tracker_id >= 0 && guide.tracker_id == id) {
                ++count;
                break;
            }
        }
    }
    return count;
}

static int sharedSuffixCountWithIds(const std::vector<ArrowGuide>& guides,
                                    const std::vector<int>& ids) {
    int count = 0;
    while (static_cast<size_t>(count) < guides.size() &&
           static_cast<size_t>(count) < ids.size()) {
        const size_t guide_index = guides.size() - 1 - static_cast<size_t>(count);
        const size_t id_index = ids.size() - 1 - static_cast<size_t>(count);
        if (guides[guide_index].tracker_id != ids[id_index]) {
            break;
        }
        ++count;
    }
    return count;
}

static bool sameGuideIdsAsText(const std::vector<ArrowGuide>& guides,
                               const std::string& ids_text) {
    return formatSingleGrowGuideIds(guides) == ids_text;
}

static const SingleAnchorGrowPath* findPreviousWinnerCandidate(
    const std::vector<SingleAnchorGrowPath>& candidates,
    const std::vector<int>& prev_winner_ids) {
    if (prev_winner_ids.empty()) {
        return nullptr;
    }

    const SingleAnchorGrowPath* best = nullptr;
    int best_shared_count = 0;
    float best_shared_ratio = 0.0f;
    for (const auto& candidate : candidates) {
        const int shared_count =
            sharedIdCountWithIds(candidate.guides, prev_winner_ids);
        const float shared_ratio =
            static_cast<float>(shared_count) /
            static_cast<float>(prev_winner_ids.size());
        if (shared_count > best_shared_count ||
            (shared_count == best_shared_count &&
             shared_ratio > best_shared_ratio)) {
            best = &candidate;
            best_shared_count = shared_count;
            best_shared_ratio = shared_ratio;
        }
    }

    if (best == nullptr ||
        best_shared_count < kAbruptChainSwitchMinPrevSharedIdCount ||
        best_shared_ratio < kAbruptChainSwitchMinPrevSharedRatio) {
        return nullptr;
    }
    return best;
}

static void applyAbruptChainSwitchPenalty(
    std::vector<SingleAnchorGrowPath>& candidates,
    const std::string& prev_winner_ids_text,
    const BoundaryTrackManager* boundary_tracks,
    float fork_tail_scale) {
    if (candidates.size() < 2 || prev_winner_ids_text == "none") {
        return;
    }
    if (fork_tail_scale < 0.99f) {
        return;
    }
    if (boundary_tracks != nullptr &&
        (boundary_tracks->visible_split_output_active ||
         boundary_tracks->split_confirmed)) {
        return;
    }

    const std::vector<int> prev_winner_ids =
        parseTrackerIdsText(prev_winner_ids_text);
    const SingleAnchorGrowPath* prev_candidate =
        findPreviousWinnerCandidate(candidates, prev_winner_ids);
    if (prev_candidate == nullptr || prev_candidate->guides.empty()) {
        return;
    }

    const float prev_root_lateral_m = worldLateralM(prev_candidate->guides.front());
    for (auto& candidate : candidates) {
        if (candidate.guides.empty() ||
            sameGuideIdsAsText(candidate.guides, prev_winner_ids_text)) {
            continue;
        }
        const int shared_suffix_count =
            sharedSuffixCountWithIds(candidate.guides, prev_winner_ids);
        if (shared_suffix_count > kAbruptChainSwitchMaxSharedSuffixCount) {
            continue;
        }
        const float root_lateral_jump =
            std::abs(worldLateralM(candidate.guides.front()) - prev_root_lateral_m);
        if (root_lateral_jump <= kAbruptChainSwitchRootLateralJumpM) {
            continue;
        }

        candidate.score += kAbruptChainSwitchPenalty;
    }
}

static float guideForwardSpan(const std::vector<ArrowGuide>& guides);

static std::vector<int> trackerIdsFromGuidesForIdentity(
    const std::vector<ArrowGuide>& guides) {
    std::vector<int> ids;
    ids.reserve(guides.size());
    for (const auto& guide : guides) {
        if (guide.tracker_id >= 0) {
            ids.push_back(guide.tracker_id);
        }
    }
    return ids;
}

static bool guideChainMatchesIdentityIds(const std::vector<ArrowGuide>& guides,
                                         const std::vector<int>& ids,
                                         int* out_common_ids = nullptr,
                                         float* out_ratio = nullptr) {
    const int common_ids = sharedIdCountWithIds(guides, ids);
    const float denom =
        static_cast<float>(std::max<size_t>(1, std::min(guides.size(), ids.size())));
    const float ratio = common_ids / denom;
    if (out_common_ids != nullptr) {
        *out_common_ids = common_ids;
    }
    if (out_ratio != nullptr) {
        *out_ratio = ratio;
    }
    return common_ids >= kSplitMainIdentityMinCommonIds &&
           ratio >= kSplitMainIdentityMinOverlapRatio;
}

static const SingleAnchorGrowPath* findBestIdentityMatchedPath(
    const std::vector<SingleAnchorGrowPath>& candidates,
    const std::vector<int>& ids,
    int* out_common_ids,
    float* out_ratio) {
    if (out_common_ids != nullptr) *out_common_ids = 0;
    if (out_ratio != nullptr) *out_ratio = 0.0f;
    if (candidates.empty() || ids.empty()) {
        return nullptr;
    }
    const SingleAnchorGrowPath* best = nullptr;
    int best_common = 0;
    float best_ratio = 0.0f;
    for (const auto& candidate : candidates) {
        int common = 0;
        float ratio = 0.0f;
        guideChainMatchesIdentityIds(candidate.guides, ids, &common, &ratio);
        if (common > best_common ||
            (common == best_common && ratio > best_ratio) ||
            (common == best_common && std::abs(ratio - best_ratio) <= 1e-4f &&
             best != nullptr && candidate.score < best->score)) {
            best = &candidate;
            best_common = common;
            best_ratio = ratio;
        }
    }
    if (out_common_ids != nullptr) *out_common_ids = best_common;
    if (out_ratio != nullptr) *out_ratio = best_ratio;
    if (best_common < kSplitMainIdentityMinCommonIds ||
        best_ratio < kSplitMainIdentityMinOverlapRatio) {
        return nullptr;
    }
    return best;
}

static bool splitMainCandidateUsable(const SingleAnchorGrowPath& candidate) {
    if (!candidate.credible || candidate.guides.size() < 3 ||
        candidate.core_count < kSingleMinHealthyGuideCount ||
        !isFiniteFloat(candidate.score)) {
        return false;
    }
    return guideForwardSpan(candidate.guides) / kWorldPlanScale >=
           kSplitMainIdentityMinForwardSpanM;
}

static void clearStableDoubleBoundaryRecoveryContext(
    BoundaryTrackManager* manager,
    const std::string& reason) {
    if (manager == nullptr) return;
    manager->split_recovery_context_active = false;
    manager->split_recovery_context_age = 0;
    manager->stable_double_boundary_recovery_count = 0;
    manager->split_recovery_seed_main_ids.clear();
    manager->split_recovery_seed_right_ids.clear();
    manager->split_recovery_context_reason = "none";
    manager->stable_double_boundary_recovery_reject_reason = reason;
}

static bool hasSplitRecoverySearchContext(const BoundaryTrackManager* manager) {
    return manager != nullptr && manager->split_recovery_context_active &&
           manager->split_recovery_context_age <=
               kStableDoubleBoundaryRecoveryMaxAgeFrames &&
           !manager->split_recovery_seed_main_ids.empty() &&
           !manager->split_recovery_seed_right_ids.empty();
}

static bool sameHysteresisIdentity(const std::vector<ArrowGuide>& guides,
                                   const std::vector<int>& ids) {
    int common = 0;
    float ratio = 0.0f;
    return guideChainMatchesIdentityIds(guides, ids, &common, &ratio);
}

static SingleAnchorGrowPath applySplitMainIdentityHysteresis(
    const std::vector<SingleAnchorGrowPath>& candidates,
    const SingleAnchorGrowPath& selected_after_tiebreak,
    uint32_t frame_id,
    BoundaryTrackManager* boundary_tracks,
    BranchDebugInfo* debug_info) {
    (void)frame_id;
    if (candidates.empty()) {
        return selected_after_tiebreak;
    }
    const bool active_split_context =
        boundary_tracks != nullptr &&
        (boundary_tracks->visible_split_output_active ||
         boundary_tracks->visible_split_maintain_age > 0 ||
         boundary_tracks->split_confirmed ||
         !boundary_tracks->previous_output_main_ids.empty() ||
         hasSplitRecoverySearchContext(boundary_tracks));
    const std::vector<int>* main_identity_ids =
        boundary_tracks == nullptr
            ? nullptr
            : (!boundary_tracks->previous_output_main_ids.empty()
                   ? &boundary_tracks->previous_output_main_ids
                   : (hasSplitRecoverySearchContext(boundary_tracks)
                          ? &boundary_tracks->split_recovery_seed_main_ids
                          : nullptr));
    const std::vector<int>* right_identity_ids =
        boundary_tracks == nullptr
            ? nullptr
            : (!boundary_tracks->previous_output_right_ids.empty()
                   ? &boundary_tracks->previous_output_right_ids
                   : (hasSplitRecoverySearchContext(boundary_tracks)
                          ? &boundary_tracks->split_recovery_seed_right_ids
                          : nullptr));
    if (debug_info != nullptr) {
        debug_info->main_identity_active_split_context = active_split_context;
        debug_info->main_identity_previous_accepted_main_ids =
            boundary_tracks != nullptr
                ? formatTrackerIdsCompact(boundary_tracks->previous_accepted_main_ids)
                : "none";
        debug_info->main_identity_previous_main_ids =
            main_identity_ids != nullptr ? formatTrackerIdsCompact(*main_identity_ids) : "none";
        debug_info->main_identity_previous_right_ids =
            right_identity_ids != nullptr ? formatTrackerIdsCompact(*right_identity_ids) : "none";
        debug_info->main_identity_raw_top1_ids =
            formatSingleGrowGuideIds(candidates.front().guides);
        debug_info->main_identity_raw_top1_score = candidates.front().score;
        debug_info->main_identity_challenger_ids =
            formatSingleGrowGuideIds(selected_after_tiebreak.guides);
        debug_info->main_identity_challenger_score = selected_after_tiebreak.score;
        debug_info->main_identity_switch_required_confirm_frames =
            kSplitMainSwitchConfirmFrames;
    }
    if (!active_split_context || boundary_tracks == nullptr ||
        main_identity_ids == nullptr || main_identity_ids->empty()) {
        if (boundary_tracks != nullptr && !active_split_context) {
            boundary_tracks->pending_main_challenger_ids.clear();
            boundary_tracks->pending_main_challenger_confirm_count = 0;
            boundary_tracks->main_identity_reverse_lock_remaining = 0;
            boundary_tracks->last_switched_from_main_ids.clear();
        }
        if (debug_info != nullptr) {
            debug_info->main_identity_mode = "single_line_hold";
            debug_info->main_identity_switch_reason = "inactive_or_no_previous_main";
            debug_info->main_identity_final_selected_main_ids =
                formatSingleGrowGuideIds(selected_after_tiebreak.guides);
        }
        return selected_after_tiebreak;
    }
    if (boundary_tracks->main_identity_reverse_lock_remaining > 0) {
        --boundary_tracks->main_identity_reverse_lock_remaining;
    }

    int incumbent_common = 0;
    float incumbent_ratio = 0.0f;
    const SingleAnchorGrowPath* incumbent =
        findBestIdentityMatchedPath(candidates,
                                    *main_identity_ids,
                                    &incumbent_common,
                                    &incumbent_ratio);
    const bool incumbent_usable =
        incumbent != nullptr && splitMainCandidateUsable(*incumbent);
    const bool incumbent_credible = incumbent != nullptr && incumbent->credible;
    const bool incumbent_reliable =
        incumbent != nullptr && incumbent_usable && incumbent_credible &&
        incumbent_common >= kSplitMainIdentityMinCommonIds &&
        incumbent_ratio >= kSplitMainIdentityMinOverlapRatio;
    const bool selected_is_incumbent =
        incumbent != nullptr &&
        formatSingleGrowGuideIds(selected_after_tiebreak.guides) ==
            formatSingleGrowGuideIds(incumbent->guides);

    int previous_right_common = 0;
    float previous_right_ratio = 0.0f;
    const bool challenger_matches_previous_right =
        right_identity_ids != nullptr && !right_identity_ids->empty() &&
        guideChainMatchesIdentityIds(selected_after_tiebreak.guides,
                                     *right_identity_ids,
                                     &previous_right_common,
                                     &previous_right_ratio);
    bool previous_right_candidate_found = false;
    if (right_identity_ids != nullptr && !right_identity_ids->empty()) {
        for (const auto& candidate : candidates) {
            if (incumbent != nullptr && &candidate == incumbent) continue;
            if (guideChainMatchesIdentityIds(candidate.guides, *right_identity_ids)) {
                previous_right_candidate_found = true;
                break;
            }
        }
    }
    const bool stable_dual_line_identity =
        active_split_context && main_identity_ids != nullptr &&
        right_identity_ids != nullptr && !main_identity_ids->empty() &&
        !right_identity_ids->empty() && incumbent_reliable &&
        previous_right_candidate_found;
    const bool previous_right_veto =
        stable_dual_line_identity && incumbent_reliable &&
        challenger_matches_previous_right;
    const float challenger_advantage =
        incumbent != nullptr && isFiniteFloat(incumbent->score) &&
                isFiniteFloat(selected_after_tiebreak.score)
            ? incumbent->score - selected_after_tiebreak.score
            : 0.0f;
    if (debug_info != nullptr) {
        debug_info->main_identity_mode =
            stable_dual_line_identity ? "dual_line_identity" : "single_line_hold";
        debug_info->main_identity_stable_dual_line_identity =
            stable_dual_line_identity;
        debug_info->main_identity_incumbent_ids =
            incumbent != nullptr ? formatSingleGrowGuideIds(incumbent->guides) : "none";
        debug_info->main_identity_incumbent_score =
            incumbent != nullptr ? incumbent->score : std::numeric_limits<float>::infinity();
        debug_info->main_identity_incumbent_match_common_ids = incumbent_common;
        debug_info->main_identity_incumbent_match_ratio = incumbent_ratio;
        debug_info->main_identity_incumbent_usable = incumbent_usable;
        debug_info->main_identity_incumbent_credible = incumbent_credible;
        debug_info->main_identity_incumbent_main_reliable = incumbent_reliable;
        debug_info->main_identity_challenger_advantage = challenger_advantage;
        debug_info->main_identity_challenger_match_previous_right_ratio =
            previous_right_ratio;
        debug_info->main_identity_challenger_matches_previous_right =
            challenger_matches_previous_right;
        debug_info->main_identity_previous_right_veto_active = previous_right_veto;
    }

    SingleAnchorGrowPath final_selected = selected_after_tiebreak;
    bool switch_blocked = false;
    std::string block_reason = "none";
    bool switched = false;
    std::string switch_reason = "incumbent_kept";
    bool takeover_allowed = true;
    if (stable_dual_line_identity && incumbent_reliable) {
        final_selected = *incumbent;
        takeover_allowed = false;
        switch_blocked = true;
        block_reason = previous_right_veto
            ? "challenger_is_previous_right"
            : "reliable_main_identity";
        switch_reason = "incumbent_kept_reliable_dual_line_identity";
        boundary_tracks->pending_main_challenger_ids.clear();
        boundary_tracks->pending_main_challenger_confirm_count = 0;
    } else if (!incumbent_usable) {
        boundary_tracks->pending_main_challenger_ids.clear();
        boundary_tracks->pending_main_challenger_confirm_count = 0;
        boundary_tracks->main_identity_reverse_lock_remaining = 0;
        boundary_tracks->last_switched_from_main_ids.clear();
        switched = !selected_is_incumbent;
        switch_reason = "incumbent_invalid_immediate_takeover";
    } else if (selected_is_incumbent) {
        boundary_tracks->pending_main_challenger_ids.clear();
        boundary_tracks->pending_main_challenger_confirm_count = 0;
        final_selected = *incumbent;
        switch_reason = "incumbent_raw_selected";
    } else {
        const bool reverse_lock =
            boundary_tracks->main_identity_reverse_lock_remaining > 0 &&
            sameHysteresisIdentity(selected_after_tiebreak.guides,
                                   boundary_tracks->last_switched_from_main_ids);
        if (reverse_lock) {
            final_selected = *incumbent;
            switch_blocked = true;
            block_reason = "reverse_switch_lock";
            switch_reason = "incumbent_kept_reverse_switch_lock";
            boundary_tracks->pending_main_challenger_ids.clear();
            boundary_tracks->pending_main_challenger_confirm_count = 0;
        } else if (challenger_advantage < kSplitMainSwitchMinScoreAdvantage) {
            final_selected = *incumbent;
            switch_blocked = true;
            block_reason = "challenger_advantage_below_threshold";
            switch_reason = "incumbent_kept_score_margin";
            boundary_tracks->pending_main_challenger_ids.clear();
            boundary_tracks->pending_main_challenger_confirm_count = 0;
        } else {
            const std::vector<int> selected_ids =
                trackerIdsFromGuidesForIdentity(selected_after_tiebreak.guides);
            const bool same_pending =
                !boundary_tracks->pending_main_challenger_ids.empty() &&
                sameHysteresisIdentity(selected_after_tiebreak.guides,
                                       boundary_tracks->pending_main_challenger_ids);
            boundary_tracks->pending_main_challenger_ids = selected_ids;
            boundary_tracks->pending_main_challenger_confirm_count =
                same_pending
                    ? boundary_tracks->pending_main_challenger_confirm_count + 1
                    : 1;
            if (boundary_tracks->pending_main_challenger_confirm_count >=
                kSplitMainSwitchConfirmFrames) {
                boundary_tracks->last_switched_from_main_ids = *main_identity_ids;
                boundary_tracks->main_identity_reverse_lock_remaining =
                    kSplitMainReverseLockFrames;
                boundary_tracks->pending_main_challenger_ids.clear();
                boundary_tracks->pending_main_challenger_confirm_count = 0;
                final_selected = selected_after_tiebreak;
                switched = true;
                switch_reason = "challenger_confirmed_takeover";
            } else {
                final_selected = *incumbent;
                switch_blocked = true;
                block_reason = "challenger_pending_confirmation";
                switch_reason = "incumbent_kept_waiting_for_confirmation";
            }
        }
    }
    if (debug_info != nullptr) {
        debug_info->main_identity_main_takeover_allowed = takeover_allowed;
        debug_info->main_identity_pending_challenger_ids =
            formatTrackerIdsCompact(boundary_tracks->pending_main_challenger_ids);
        debug_info->main_identity_challenger_confirm_count =
            boundary_tracks->pending_main_challenger_confirm_count;
        debug_info->main_identity_reverse_switch_lock_remaining =
            boundary_tracks->main_identity_reverse_lock_remaining;
        debug_info->main_identity_switch_blocked = switch_blocked;
        debug_info->main_identity_switch_block_reason = block_reason;
        debug_info->main_identity_switched = switched;
        debug_info->main_identity_switch_reason = switch_reason;
        debug_info->main_identity_final_selected_main_ids =
            formatSingleGrowGuideIds(final_selected.guides);
    }
    return final_selected;
}

static std::vector<ArrowGuide> selectSingleMainGuidesAnchorGrow(
    const std::vector<ArrowGuide>& guides,
    uint32_t frame_id,
    BoundaryTrackManager* boundary_tracks = nullptr,
    float fork_tail_scale = 1.0f,
    LineLossTraceFields* line_loss = nullptr,
    BranchDebugInfo* debug_info = nullptr) {
    static bool has_pending_winner = false;
    static uint32_t pending_winner_frame = 0;
    static std::string pending_winner_ids = "none";
    static std::string prev_winner_ids = "none";
    if (has_pending_winner && pending_winner_frame != frame_id) {
        prev_winner_ids = pending_winner_ids;
        has_pending_winner = false;
    }

    const std::vector<ArrowGuide> valid_guides = collectValidSingleGuidesNearToFar(guides);
    if (line_loss != nullptr) {
        line_loss->line_single_anchor_count = static_cast<int>(valid_guides.size());
    }
    if (valid_guides.empty()) {
        return {};
    }

    const float median_hw_m = medianSingleGuideHalfWidthM(valid_guides);
    const std::vector<size_t> roots =
        selectSingleAnchorRootCandidates(valid_guides, median_hw_m);
    if (line_loss != nullptr) {
        line_loss->line_single_root_count = static_cast<int>(roots.size());
    }
    const float nearest_forward_m = worldForwardM(valid_guides.front());

    SingleAnchorGrowPath best_path;
    std::vector<SingleAnchorGrowPath> credible_paths;
    for (const size_t root_index : roots) {
        SingleAnchorGrowPath path =
            growSinglePathFromRoot(valid_guides,
                                   root_index,
                                   median_hw_m,
                                   nearest_forward_m,
                                   frame_id,
                                   fork_tail_scale);
        if (!path.credible) {
            continue;
        }
        credible_paths.push_back(path);
        if (!best_path.credible || path.score < best_path.score) {
            best_path = std::move(path);
        }
    }
    if (line_loss != nullptr) {
        line_loss->line_credible_single_count =
            static_cast<int>(credible_paths.size());
    }

    if (!best_path.credible) {
        return {};
    }
    const bool normal_historical_identity_available =
        boundary_tracks != nullptr &&
        !boundary_tracks->previous_accepted_main_ids.empty() &&
        !boundary_tracks->previous_output_main_ids.empty() &&
        !boundary_tracks->previous_output_right_ids.empty();
    const bool recovery_historical_identity_available =
        hasSplitRecoverySearchContext(boundary_tracks);
    const std::vector<int>* historical_main_identity_ids =
        boundary_tracks == nullptr
            ? nullptr
            : (normal_historical_identity_available
                   ? &boundary_tracks->previous_accepted_main_ids
                   : (recovery_historical_identity_available
                          ? &boundary_tracks->split_recovery_seed_main_ids
                          : nullptr));
    const bool historical_main_context_active =
        historical_main_identity_ids != nullptr &&
        !historical_main_identity_ids->empty() &&
        (boundary_tracks->visible_split_output_active ||
         boundary_tracks->visible_split_maintain_age > 0 ||
         boundary_tracks->split_confirmed ||
         recovery_historical_identity_available);
    if (debug_info != nullptr) {
        debug_info->historical_main_context_active = historical_main_context_active;
        debug_info->historical_main_ids =
            historical_main_identity_ids != nullptr
                ? formatTrackerIdsCompact(*historical_main_identity_ids)
                : "none";
        debug_info->historical_main_visible_id_count =
            historical_main_identity_ids != nullptr
                ? visibleHistoricalIdCount(valid_guides,
                                           *historical_main_identity_ids)
                : 0;
        debug_info->normal_candidate_ids =
            formatSingleGrowGuideIds(best_path.guides);
        int normal_common = 0;
        float normal_ratio = 0.0f;
        debug_info->normal_matches_historical_main =
            historical_main_identity_ids != nullptr &&
            guideChainMatchesIdentityIds(
                best_path.guides,
                *historical_main_identity_ids,
                &normal_common,
                &normal_ratio);
        const std::vector<int>* right_ids =
            boundary_tracks == nullptr
                ? nullptr
                : (!boundary_tracks->previous_output_right_ids.empty()
                       ? &boundary_tracks->previous_output_right_ids
                       : (hasSplitRecoverySearchContext(boundary_tracks)
                              ? &boundary_tracks->split_recovery_seed_right_ids
                              : nullptr));
        if (right_ids != nullptr) {
            int right_common = 0;
            float right_ratio = 0.0f;
            guideChainMatchesIdentityIds(best_path.guides,
                                         *right_ids,
                                         &right_common,
                                         &right_ratio);
            debug_info->normal_match_previous_right_ratio = right_ratio;
        }
    }
    if (historical_main_context_active) {
        std::string historical_reject_reason = "not_evaluated";
        SingleAnchorGrowPath historical_path = growHistoricalMainGuidedPath(
            valid_guides,
            *historical_main_identity_ids,
            median_hw_m,
            nearest_forward_m,
            fork_tail_scale,
            &historical_reject_reason);
        int historical_common = 0;
        float historical_ratio = 0.0f;
        const bool historical_matches = guideChainMatchesIdentityIds(
            historical_path.guides,
            *historical_main_identity_ids,
            &historical_common,
            &historical_ratio);
        const bool historical_usable =
            !historical_path.guides.empty() &&
            splitMainCandidateUsable(historical_path);
        const bool historical_reliable =
            historical_path.credible && historical_usable && historical_matches;
        const std::string historical_ids = historical_path.guides.empty()
            ? "none"
            : formatSingleGrowGuideIds(historical_path.guides);
        bool duplicate = false;
        for (const auto& candidate : credible_paths) {
            if (formatSingleGrowGuideIds(candidate.guides) == historical_ids) {
                duplicate = true;
                break;
            }
        }
        if (historical_reliable && !duplicate) {
            credible_paths.push_back(historical_path);
        } else if (!historical_path.guides.empty()) {
            if (!historical_path.credible) historical_reject_reason = "not_credible";
            else if (!historical_usable) historical_reject_reason = "not_usable";
            else if (!historical_matches) {
                historical_reject_reason = "identity_match_below_threshold";
            }
        }
        if (debug_info != nullptr) {
            debug_info->historical_main_guided_grow_attempted = true;
            debug_info->historical_main_candidate_ids = historical_ids;
            debug_info->historical_main_candidate_generated =
                !historical_path.guides.empty();
            debug_info->historical_main_candidate_credible = historical_path.credible;
            debug_info->historical_main_candidate_usable = historical_usable;
            debug_info->historical_main_match_common_ids = historical_common;
            debug_info->historical_main_match_ratio = historical_ratio;
            debug_info->historical_main_reliable = historical_reliable;
            debug_info->historical_main_reject_reason =
                historical_reliable ? (duplicate ? "duplicate_candidate" : "none")
                                    : historical_reject_reason;
            debug_info->historical_candidate_appended_to_credible_paths =
                historical_reliable && !duplicate;
        }
    }
    applyAbruptChainSwitchPenalty(
        credible_paths,
        prev_winner_ids,
        boundary_tracks,
        fork_tail_scale);
    std::sort(credible_paths.begin(),
              credible_paths.end(),
              [](const SingleAnchorGrowPath& a, const SingleAnchorGrowPath& b) {
                  if (std::abs(a.score - b.score) > 1e-4f) {
                      return a.score < b.score;
                  }
                  return formatSingleGrowGuideIds(a.guides) <
                         formatSingleGrowGuideIds(b.guides);
              });
    if (credible_paths.size() >= 2) {
        const bool split_not_taken =
            boundary_tracks == nullptr ||
            (!boundary_tracks->visible_split_output_active &&
             !boundary_tracks->split_confirmed);
        ForkContextMainTieBreakDecision fork_tiebreak =
            evaluateForkContextMainTieBreak(credible_paths[0],
                                            credible_paths[1],
                                            split_not_taken);
        applyConfirmedRightForkMainIdentityGuard(credible_paths[0],
                                                 credible_paths[1],
                                                 frame_id,
                                                 boundary_tracks,
                                                 &fork_tiebreak);
        best_path = fork_tiebreak.selected_secondary ? credible_paths[1] : credible_paths[0];
    } else {
        best_path = credible_paths[0];
    }
    best_path = applySplitMainIdentityHysteresis(credible_paths,
                                                 best_path,
                                                 frame_id,
                                                 boundary_tracks,
                                                 debug_info);
    pending_winner_frame = frame_id;
    pending_winner_ids = formatSingleGrowGuideIds(best_path.guides);
    has_pending_winner = true;
    if (line_loss != nullptr) {
        line_loss->line_best_single_guide_count =
            static_cast<int>(best_path.guides.size());
        line_loss->line_best_single_forward_span =
            guideForwardSpan(best_path.guides);
    }
    return best_path.guides;
}

static float guideForwardSpan(const std::vector<ArrowGuide>& guides) {
    if (guides.size() < 2) {
        return 0.0f;
    }
    return std::max(0.0f, guidePlanForward(guides.back()) - guidePlanForward(guides.front()));
}

static float maxAdjacentGuideForwardGap(const std::vector<ArrowGuide>& guides) {
    if (guides.size() < 2) {
        return 0.0f;
    }

    float max_gap = 0.0f;
    for (size_t i = 1; i < guides.size(); ++i) {
        max_gap = std::max(max_gap,
                           std::abs(guidePlanForward(guides[i]) -
                                    guidePlanForward(guides[i - 1])));
    }
    return max_gap;
}

static bool isReliableSingleGuideSet(const std::vector<ArrowGuide>& guides,
                                     float min_y_span = planMeters(0.14f)) {
    if (guides.size() < 4) {
        return false;
    }

    const float y_span = guideForwardSpan(guides);
    if (y_span < min_y_span) {
        return false;
    }

    const float max_gap = maxAdjacentGuideForwardGap(guides);
    const float max_allowed_gap = std::max(planMeters(0.15f), y_span * 0.50f);
    return max_gap <= max_allowed_gap;
}

static float guidePathMeanFitResidual(const std::vector<ArrowGuide>& guides);

static float averageGuideHalfWidthPlan(const std::vector<ArrowGuide>& guides) {
    if (guides.empty()) {
        return 0.0f;
    }
    float sum = 0.0f;
    for (const auto& guide : guides) {
        sum += guidePlanHalfWidth(guide);
    }
    return sum / static_cast<float>(guides.size());
}

static float averageGuideStepCost(const std::vector<ArrowGuide>& guides) {
    if (guides.size() < 2) {
        return 0.0f;
    }
    float total_cost = 0.0f;
    for (size_t i = 1; i < guides.size(); ++i) {
        total_cost += twoStepContinuationCost(guides[i - 1], guides[i]);
    }
    return total_cost / static_cast<float>(guides.size() - 1);
}

static float maxGuideStepDistance(const std::vector<ArrowGuide>& guides) {
    if (guides.size() < 2) {
        return 0.0f;
    }
    float max_dist = 0.0f;
    for (size_t i = 1; i < guides.size(); ++i) {
        const float dx = guidePlanLateral(guides[i]) - guidePlanLateral(guides[i - 1]);
        const float dy = guidePlanForward(guides[i]) - guidePlanForward(guides[i - 1]);
        max_dist = std::max(max_dist, std::sqrt(dx * dx + dy * dy));
    }
    return max_dist;
}

static float maxGuideDxJump(const std::vector<ArrowGuide>& guides) {
    if (guides.size() < 2) {
        return 0.0f;
    }
    float max_dx = 0.0f;
    for (size_t i = 1; i < guides.size(); ++i) {
        max_dx = std::max(max_dx,
                          std::abs(guidePlanLateral(guides[i]) -
                                   guidePlanLateral(guides[i - 1])));
    }
    return max_dx;
}

static float guidePathHeadingDiscontinuityDeg(const std::vector<ArrowGuide>& guides) {
    if (guides.size() < 3) {
        return 0.0f;
    }

    float total = 0.0f;
    int count = 0;
    for (size_t i = 2; i < guides.size(); ++i) {
        cv::Point2f prev_dir;
        cv::Point2f curr_dir;
        if (!normalizeVector(guidePlanPoint(guides[i - 1]) - guidePlanPoint(guides[i - 2]),
                             prev_dir) ||
            !normalizeVector(guidePlanPoint(guides[i]) - guidePlanPoint(guides[i - 1]),
                             curr_dir)) {
            continue;
        }
        total += angleBetweenUnitVectorsDeg(prev_dir, curr_dir);
        ++count;
    }
    return count > 0 ? total / static_cast<float>(count) : 0.0f;
}

static float guidePathMeanFitResidual(const std::vector<ArrowGuide>& guides) {
    if (guides.size() < 3) {
        return 0.0f;
    }

    const PolyFit2 fit = refitPoly2OnAllGuides(guides);
    if (!fit.valid) {
        return 0.0f;
    }

    float residual_sum = 0.0f;
    int residual_count = 0;
    for (const auto& guide : guides) {
        const float fit_x = fit.eval(guidePlanForward(guide));
        if (!std::isfinite(fit_x)) {
            continue;
        }
        residual_sum += std::abs(guidePlanLateral(guide) - fit_x);
        ++residual_count;
    }
    return residual_count > 0
        ? residual_sum / static_cast<float>(residual_count)
        : 0.0f;
}

static bool guidePathHasForwardContinuity(const std::vector<ArrowGuide>& guides) {
    if (guides.size() < 2) {
        return false;
    }

    constexpr float kReverseSlackM = 0.025f;
    for (size_t i = 1; i < guides.size(); ++i) {
        const float df =
            guidePlanForward(guides[i]) - guidePlanForward(guides[i - 1]);
        if (df < -planMeters(kReverseSlackM)) {
            return false;
        }
    }
    return true;
}




static bool sameGuidePointLoose(const ArrowGuide& a,
                                const ArrowGuide& b,
                                float x_tol = 10.0f,
                                float y_tol = 10.0f) {
    return std::abs(a.pos.x - b.pos.x) <= x_tol &&
           std::abs(a.pos.y - b.pos.y) <= y_tol;
}

static int findGuideBandIndexLoose(const std::vector<std::vector<ArrowGuide>>& bands,
                                   const ArrowGuide& guide,
                                   float x_tol = 10.0f,
                                   float y_tol = 10.0f) {
    for (size_t band_idx = 0; band_idx < bands.size(); ++band_idx) {
        for (const auto& band_guide : bands[band_idx]) {
            if (sameGuidePointLoose(band_guide, guide, x_tol, y_tol)) {
                return static_cast<int>(band_idx);
            }
        }
    }
    return -1;
}

static int countGuideMatchesLoose(const std::vector<ArrowGuide>& guides,
                                  const std::vector<ArrowGuide>& reference,
                                  float x_tol = 10.0f,
                                  float y_tol = 10.0f) {
    int match_count = 0;
    for (const auto& guide : guides) {
        bool matched = false;
        for (const auto& ref : reference) {
            if (sameGuidePointLoose(guide, ref, x_tol, y_tol)) {
                matched = true;
                break;
            }
        }
        if (matched) {
            ++match_count;
        }
    }
    return match_count;
}

static float computeGuideMatchRatioLoose(const std::vector<ArrowGuide>& guides,
                                         const std::vector<ArrowGuide>& reference,
                                         float x_tol = 10.0f,
                                         float y_tol = 10.0f) {
    if (guides.empty() || reference.empty()) {
        return 0.0f;
    }
    const int match_count = countGuideMatchesLoose(guides, reference, x_tol, y_tol);
    const float denom = static_cast<float>(std::min(guides.size(), reference.size()));
    if (denom <= 1e-3f) {
        return 0.0f;
    }
    return clampValue(static_cast<float>(match_count) / denom, 0.0f, 1.0f);
}

static float guideForwardOverlap(const std::vector<ArrowGuide>& a,
                                 const std::vector<ArrowGuide>& b) {
    if (a.empty() || b.empty()) {
        return 0.0f;
    }
    auto rangeFor = [](const std::vector<ArrowGuide>& guides, float& lo, float& hi) {
        lo = guidePlanForward(guides.front());
        hi = lo;
        for (const auto& guide : guides) {
            const float fwd = guidePlanForward(guide);
            lo = std::min(lo, fwd);
            hi = std::max(hi, fwd);
        }
    };

    float a_lo = 0.0f, a_hi = 0.0f, b_lo = 0.0f, b_hi = 0.0f;
    rangeFor(a, a_lo, a_hi);
    rangeFor(b, b_lo, b_hi);
    return std::max(0.0f, std::min(a_hi, b_hi) - std::max(a_lo, b_lo));
}

static bool interpolateGuidePlanLateralAtForward(const std::vector<ArrowGuide>& guides,
                                                 float target_forward,
                                                 float& out_lateral) {
    if (guides.empty() || !std::isfinite(target_forward)) {
        return false;
    }

    std::vector<cv::Point2f> samples;
    samples.reserve(guides.size());
    for (const auto& guide : guides) {
        const cv::Point2f plan = guidePlanPoint(guide);
        if (isFinitePoint(plan)) {
            samples.push_back(plan);
        }
    }
    if (samples.empty()) {
        return false;
    }
    std::sort(samples.begin(), samples.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.y < b.y;
    });
    if (samples.size() == 1) {
        out_lateral = samples.front().x;
        return true;
    }

    size_t best_idx = 0;
    float best_gap = std::numeric_limits<float>::infinity();
    for (size_t i = 1; i < samples.size(); ++i) {
        const float lo = std::min(samples[i - 1].y, samples[i].y);
        const float hi = std::max(samples[i - 1].y, samples[i].y);
        float gap = 0.0f;
        if (target_forward < lo) {
            gap = lo - target_forward;
        } else if (target_forward > hi) {
            gap = target_forward - hi;
        }
        if (gap < best_gap) {
            best_gap = gap;
            best_idx = i - 1;
        }
    }

    const cv::Point2f a = samples[best_idx];
    const cv::Point2f b = samples[best_idx + 1];
    const float dy = b.y - a.y;
    if (std::abs(dy) < 1e-3f) {
        out_lateral = 0.5f * (a.x + b.x);
        return true;
    }
    const float t = (target_forward - a.y) / dy;
    out_lateral = a.x + t * (b.x - a.x);
    return std::isfinite(out_lateral);
}

static int countUnusedConnectableGuides(const std::vector<ArrowGuide>& all_guides,
                                        const std::vector<ArrowGuide>& used_guides,
                                        const std::vector<ArrowGuide>& left_fork,
                                        const std::vector<ArrowGuide>& right_fork,
                                        float median_hw) {
    if (all_guides.empty() || (left_fork.size() < 2 && right_fork.size() < 2)) {
        return 0;
    }

    int count = 0;
    const float residual_limit = std::max(planMeters(0.085f), median_hw * 0.70f);
    const float forward_slack = std::max(planMeters(0.055f), median_hw * 0.35f);
    auto branchCouldUse = [&](const std::vector<ArrowGuide>& branch,
                              const ArrowGuide& guide) {
        if (branch.size() < 2) {
            return false;
        }
        float fwd_min = guidePlanForward(branch.front());
        float fwd_max = fwd_min;
        for (const auto& item : branch) {
            const float fwd = guidePlanForward(item);
            fwd_min = std::min(fwd_min, fwd);
            fwd_max = std::max(fwd_max, fwd);
        }
        const float target_fwd = guidePlanForward(guide);
        if (target_fwd < fwd_min - forward_slack ||
            target_fwd > fwd_max + forward_slack) {
            return false;
        }
        float expected_lat = 0.0f;
        if (!interpolateGuidePlanLateralAtForward(branch, target_fwd, expected_lat)) {
            return false;
        }
        return std::abs(guidePlanLateral(guide) - expected_lat) <= residual_limit;
    };

    for (const auto& guide : all_guides) {
        bool used = false;
        for (const auto& item : used_guides) {
            if (sameGuidePointLoose(guide, item, 10.0f, 10.0f)) {
                used = true;
                break;
            }
        }
        if (used) {
            continue;
        }
        if (branchCouldUse(left_fork, guide) || branchCouldUse(right_fork, guide)) {
            ++count;
        }
    }
    return count;
}

static float computeWeakForkFarFieldPenalty(const std::vector<ArrowGuide>& weaker_fork,
                                            bool weaker_root_ok,
                                            bool weaker_near_root_ok,
                                            float visible_min_forward,
                                            float median_hw) {
    if (weaker_fork.empty() || !std::isfinite(visible_min_forward)) {
        return 0.0f;
    }

    // A fork branch begins at the split, not at the bottom of the visible
    // route. Penalize bottom reach only for floating weak forks that have no
    // usable root evidence.
    if (weaker_root_ok || weaker_near_root_ok) {
        return 0.0f;
    }

    const float bottom_gap =
        std::max(0.0f, guidePlanForward(weaker_fork.front()) - visible_min_forward);
    const float allowed_bottom_gap = weaker_near_root_ok
        ? std::max(planMeters(0.26f), median_hw * 1.90f)
        : std::max(planMeters(0.17f), median_hw * 1.35f);
    if (bottom_gap <= allowed_bottom_gap) {
        return 0.0f;
    }

    const float gap_scale = clampValue(
        (bottom_gap - allowed_bottom_gap) /
            std::max(planMeters(0.08f), allowed_bottom_gap * 0.70f),
        0.0f,
        2.0f);
    const float short_branch_factor = clampValue(
        (4.0f - static_cast<float>(weaker_fork.size())) / 2.0f,
        0.0f,
        1.0f);
    const float span_target = std::max(planMeters(0.15f), median_hw * 1.35f);
    const float low_span_factor = clampValue(
        (span_target - guideForwardSpan(weaker_fork)) /
            std::max(planMeters(0.06f), span_target),
        0.0f,
        1.0f);
    const float weak_shape_factor =
        std::max(short_branch_factor, low_span_factor);

    float penalty = (weaker_near_root_ok ? 6.0f : 12.0f) * gap_scale;
    penalty += 10.0f * std::min(gap_scale, 1.5f) * weak_shape_factor;
    return penalty;
}

static float computeForkCoverageReward(const std::vector<ArrowGuide>& fork_guides,
                                       float visible_min_forward,
                                       float median_hw) {
    if (fork_guides.empty()) {
        return 0.0f;
    }

    float reward = 0.0f;
    reward += 2.0f * std::max(
        0, static_cast<int>(fork_guides.size()) - 2);
    reward += 0.03f * guideForwardSpan(fork_guides);

    if (std::isfinite(visible_min_forward)) {
        const float bottom_gap =
            std::max(0.0f, guidePlanForward(fork_guides.front()) - visible_min_forward);
        const float allowed_bottom_gap =
            std::max(planMeters(0.11f), median_hw * 0.85f);
        reward += 2.5f * clampValue(
            1.0f - bottom_gap / std::max(1.0f, allowed_bottom_gap * 1.8f),
            0.0f,
            1.0f);
    }

    return reward;
}

static float computeForkBottomReachScore(const std::vector<ArrowGuide>& fork_guides,
                                         float visible_min_forward,
                                         float median_hw) {
    if (fork_guides.empty()) {
        return 0.0f;
    }
    if (!std::isfinite(visible_min_forward)) {
        return clampValue(
            static_cast<float>(fork_guides.size()) / 4.0f,
            0.0f,
            1.0f);
    }

    const float bottom_gap =
        std::max(0.0f, guidePlanForward(fork_guides.front()) - visible_min_forward);
    const float allowed_bottom_gap = std::max(planMeters(0.13f), median_hw * 1.05f);
    return clampValue(
        1.0f - bottom_gap / std::max(planMeters(0.085f), allowed_bottom_gap * 2.1f),
        0.0f,
        1.0f);
}

static float computeForkDirectionalBacktrackPenalty(const std::vector<ArrowGuide>& fork_guides,
                                                    bool left_side,
                                                    float median_hw) {
    if (fork_guides.size() < 2) {
        return 0.0f;
    }

    const float inward_slack = std::max(planMeters(0.02f), median_hw * 0.18f);
    float penalty = 0.0f;
    for (size_t i = 1; i < fork_guides.size(); ++i) {
        const float dx = guidePlanLateral(fork_guides[i]) -
                         guidePlanLateral(fork_guides[i - 1]);
        const float inward_amount =
            left_side ? (dx - inward_slack) : (-dx - inward_slack);
        if (inward_amount <= 0.0f) {
            continue;
        }

        // In large-curvature / loop-like scenes a fork that first expands to
        // one side and then hooks back inward usually indicates that the tail
        // latched onto the wrong remote arrow.
        const float stage =
            static_cast<float>(i) / static_cast<float>(fork_guides.size() - 1);
        penalty += (0.42f + 0.28f * stage) * inward_amount;
    }

    return penalty;
}

static float computeForkTailLatchRisk(const std::vector<ArrowGuide>& fork_guides,
                                      bool left_side,
                                      float median_hw) {
    if (fork_guides.size() < 3) {
        return 0.0f;
    }

    const auto& prev = fork_guides[fork_guides.size() - 2];
    const auto& last = fork_guides.back();
    const float dx_last = guidePlanLateral(last) - guidePlanLateral(prev);
    const float dy_last = std::abs(guidePlanForward(last) - guidePlanForward(prev));
    const float step_last = std::hypot(dx_last, guidePlanForward(last) - guidePlanForward(prev));

    float prev_dx_sum = 0.0f;
    float prev_dy_sum = 0.0f;
    float prev_step_sum = 0.0f;
    int prev_step_count = 0;
    for (size_t i = 1; i + 1 < fork_guides.size(); ++i) {
        const float dx = guidePlanLateral(fork_guides[i]) -
                         guidePlanLateral(fork_guides[i - 1]);
        const float dy = std::abs(guidePlanForward(fork_guides[i]) -
                                  guidePlanForward(fork_guides[i - 1]));
        prev_dx_sum += std::abs(dx);
        prev_dy_sum += dy;
        prev_step_sum += std::hypot(dx,
                                    guidePlanForward(fork_guides[i]) -
                                        guidePlanForward(fork_guides[i - 1]));
        ++prev_step_count;
    }
    if (prev_step_count <= 0) {
        return 0.0f;
    }

    const float mean_prev_dx = prev_dx_sum / static_cast<float>(prev_step_count);
    const float mean_prev_dy = prev_dy_sum / static_cast<float>(prev_step_count);
    const float mean_prev_step = prev_step_sum / static_cast<float>(prev_step_count);

    float risk = 0.0f;
    const float inward_jump_limit =
        std::max(planMeters(0.06f), std::max(mean_prev_dx * 1.75f, mean_prev_dx + median_hw * 0.28f));
    const float dy_jump_limit =
        std::max(planMeters(0.065f), std::max(mean_prev_dy * 1.70f, mean_prev_dy + median_hw * 0.32f));
    const float step_jump_limit =
        std::max(planMeters(0.095f), std::max(mean_prev_step * 1.55f, mean_prev_step + median_hw * 0.42f));
    const float inward_amount =
        left_side ? std::max(0.0f, dx_last) : std::max(0.0f, -dx_last);
    if (inward_amount > inward_jump_limit) {
        risk += 0.42f * (inward_amount - inward_jump_limit);
    }
    if (inward_amount > std::max(planMeters(0.02f), mean_prev_dx * 0.60f) &&
        dy_last > dy_jump_limit) {
        risk += 0.26f * (dy_last - dy_jump_limit);
    }
    if (inward_amount > std::max(planMeters(0.02f), mean_prev_dx * 0.60f) &&
        step_last > step_jump_limit) {
        risk += 0.18f * (step_last - step_jump_limit);
    }

    if (fork_guides.size() >= 4) {
        const auto& prev2 = fork_guides[fork_guides.size() - 3];
        const cv::Point2f v0(guidePlanLateral(prev) - guidePlanLateral(prev2),
                             guidePlanForward(prev) - guidePlanForward(prev2));
        const cv::Point2f v1(guidePlanLateral(last) - guidePlanLateral(prev),
                             guidePlanForward(last) - guidePlanForward(prev));
        const float norm0 = std::sqrt(v0.x * v0.x + v0.y * v0.y);
        const float norm1 = std::sqrt(v1.x * v1.x + v1.y * v1.y);
        if (norm0 > 1e-3f && norm1 > 1e-3f) {
            const float cos_angle =
                clampValue((v0.x * v1.x + v0.y * v1.y) / (norm0 * norm1), -1.0f, 1.0f);
            if (inward_amount > std::max(planMeters(0.02f), mean_prev_dx * 0.45f) &&
                cos_angle < 0.35f) {
                risk += (0.35f - cos_angle) * 16.0f;
            }
        }
    }

    return risk;
}

static float computeForkTailFrontierPenalty(
    const std::vector<ArrowGuide>& fork_guides,
    const std::vector<std::vector<ArrowGuide>>& bands,
    bool left_side,
    float split_center_x,
    float median_hw) {
    if (fork_guides.size() < 3 || bands.empty()) {
        return 0.0f;
    }

    const auto& prev = fork_guides[fork_guides.size() - 2];
    const auto& last = fork_guides.back();
    const int prev_band_idx = findGuideBandIndexLoose(bands, prev, 12.0f, 12.0f);
    const int last_band_idx = findGuideBandIndexLoose(bands, last, 12.0f, 12.0f);
    if (prev_band_idx < 0 || last_band_idx < 0 || last_band_idx < prev_band_idx) {
        return 0.0f;
    }

    const float same_side_slack = std::max(planMeters(0.06f), median_hw * 0.32f);
    float frontier_x = guidePlanLateral(last);
    int local_support_count = 0;
    for (int band_idx = prev_band_idx;
         band_idx <= last_band_idx && band_idx < static_cast<int>(bands.size());
         ++band_idx) {
        for (const auto& guide : bands[static_cast<size_t>(band_idx)]) {
            const bool same_side =
                left_side ? (guidePlanLateral(guide) <= split_center_x + same_side_slack)
                          : (guidePlanLateral(guide) >= split_center_x - same_side_slack);
            if (!same_side) {
                continue;
            }
            frontier_x = left_side
                             ? std::min(frontier_x, guidePlanLateral(guide))
                             : std::max(frontier_x, guidePlanLateral(guide));
            if (std::abs(guidePlanLateral(guide) - guidePlanLateral(last)) <=
                std::max(planMeters(0.07f), median_hw * 0.45f)) {
                ++local_support_count;
            }
        }
    }

    const float frontier_gap =
        left_side ? (guidePlanLateral(last) - frontier_x)
                  : (frontier_x - guidePlanLateral(last));
    const float remote_drop =
        std::max(0.0f, guidePlanForward(last) - guidePlanForward(prev));
    const float outward_gain =
        left_side ? std::max(0.0f, guidePlanLateral(prev) - guidePlanLateral(last))
                  : std::max(0.0f, guidePlanLateral(last) - guidePlanLateral(prev));
    if (frontier_gap <= std::max(planMeters(0.05f), median_hw * 0.30f) ||
        remote_drop <= std::max(planMeters(0.06f), median_hw * 0.32f)) {
        return 0.0f;
    }

    float penalty = 0.0f;
    const float small_outward_gain_limit =
        std::max(planMeters(0.03f), median_hw * 0.18f) + 0.28f * frontier_gap;
    if (outward_gain < small_outward_gain_limit) {
        penalty += 0.34f * (frontier_gap - std::max(planMeters(0.04f), median_hw * 0.22f));
    }
    if (local_support_count <= 1) {
        penalty += 0.18f * (remote_drop - std::max(planMeters(0.05f), median_hw * 0.28f));
    }
    if (last_band_idx > prev_band_idx + 1 &&
        frontier_gap > std::max(planMeters(0.07f), median_hw * 0.38f)) {
        penalty += 0.12f * frontier_gap;
    }
    return std::max(0.0f, penalty);
}


static float normalizeAngleDeltaDeg(float delta) {
    while (delta > 180.0f) {
        delta -= 360.0f;
    }
    while (delta < -180.0f) {
        delta += 360.0f;
    }
    return delta;
}

static int inferRouteSide(const std::vector<ArrowGuide>& guides) {
    if (guides.empty()) {
        return 0;
    }
    const size_t sample_count = std::min<size_t>(guides.size(), 3);
    float lateral_sum = 0.0f;
    for (size_t i = 0; i < sample_count; ++i) {
        lateral_sum += guidePlanLateral(guides[i]);
    }
    const float avg_lateral = lateral_sum / static_cast<float>(sample_count);
    const float side_deadband = planMeters(0.04f);
    if (avg_lateral > side_deadband) {
        return 1;
    }
    if (avg_lateral < -side_deadband) {
        return -1;
    }
    return 0;
}

static float computeRouteCenterLateral(const std::vector<ArrowGuide>& guides) {
    if (guides.empty()) {
        return 0.0f;
    }
    float weighted_sum = 0.0f;
    float weight_total = 0.0f;
    const size_t count = std::min<size_t>(guides.size(), 4);
    for (size_t i = 0; i < count; ++i) {
        const float weight = 1.6f - 0.25f * static_cast<float>(i);
        weighted_sum += guidePlanLateral(guides[i]) * weight;
        weight_total += weight;
    }
    return weight_total > 1e-3f
        ? weighted_sum / weight_total
        : guidePlanLateral(guides.front());
}

static float computeRouteHeadingDeg(const std::vector<ArrowGuide>& guides) {
    if (guides.size() < 2) {
        return 0.0f;
    }
    const cv::Point2f delta = guidePlanPoint(guides.back()) - guidePlanPoint(guides.front());
    if (!isFinitePoint(delta) || std::abs(delta.y) < 1e-3f) {
        return 0.0f;
    }
    return static_cast<float>(std::atan2(delta.x, delta.y) * 180.0 / CV_PI);
}

struct PhysicalBoundaryGuidePoint {
    ArrowGuide guide;
    float forward = 0.0f;
    float lateral = 0.0f;
    int bucket_idx = 0;
};

struct PhysicalBoundaryChain {
    std::vector<ArrowGuide> guides;
    int side_hint = 0;
    float near_forward = 0.0f;
    float far_forward = 0.0f;
    float span = 0.0f;
    float median_lateral = 0.0f;
    float root_lateral = 0.0f;
    float root_forward = 0.0f;
    float heading_deg = 0.0f;
    float max_forward_gap = 0.0f;
    float continuity_score = 0.0f;
    float coverage_score = 0.0f;
    float confidence = 0.0f;
    int bucket_coverage = 0;
    bool used_extra_forward_alt = false;
    float grow_accumulated_cost = 0.0f;
    bool created_by_right_fork_alt = false;
    std::string right_fork_alt_source_path_ids = "none";
    int right_fork_alt_source_candidate_id = -1;
};

enum class PhysicalBoundaryPrimitiveType {
    Forward = 0,
    Transverse = 1,
    Diagonal = 2,
};

struct PhysicalBoundaryPrimitive {
    PhysicalBoundaryPrimitiveType type = PhysicalBoundaryPrimitiveType::Forward;
    PhysicalBoundaryChain chain;
    std::vector<int> guide_ids;
    float fwd_min = 0.0f;
    float fwd_max = 0.0f;
    float fwd_span = 0.0f;
    float lat_min = 0.0f;
    float lat_max = 0.0f;
    float lat_span = 0.0f;
    float principal_heading_deg = 0.0f;
    float forward_extent_m = 0.0f;
    float lateral_extent_m = 0.0f;
    bool root_ok = false;
    bool near_ok = false;
    int near_cov = 0;
    float quality_score = 0.0f;
    bool weak_short_forward = false;
    bool transverse_derived = false;
    bool near_root_short_branch = false;
    std::string reject_reason = "none";
};

struct PhysicalBoundaryPairCandidate {
    PhysicalBoundaryChain negative_chain;
    PhysicalBoundaryChain positive_chain;
    PhysicalBoundaryPrimitiveType negative_type = PhysicalBoundaryPrimitiveType::Forward;
    PhysicalBoundaryPrimitiveType positive_type = PhysicalBoundaryPrimitiveType::Forward;
    float overlap_forward = 0.0f;
    float width_median = 0.0f;
    float width_near = 0.0f;
    float width_mid = 0.0f;
    float width_far = 0.0f;
    float width_delta = 0.0f;
    float far_width_ratio = 0.0f;
    float width_slope_m_per_m = 0.0f;
    float left_lat_slope = 0.0f;
    float right_lat_slope = 0.0f;
    float relative_lat_slope = 0.0f;
    float width_stability = 0.0f;
    float heading_consistency = 0.0f;
    float pair_confidence = 0.0f;
    float far_divergence_penalty = 0.0f;
    float shared_guide_ratio = 0.0f;
    float stable_split_score = 0.0f;
    float opening_score = 0.0f;
    float closing_score = 0.0f;
    bool stable_split_like = false;
    bool opening_like = false;
    bool closing_like = false;
    bool closing_suspect = false;
    bool strong_closing_veto = false;
    int pair_unique_guide_count = 0;
    float pair_coverage_ratio = 0.0f;
    int pair_explained_transverse_count = 0;
    int pair_unexplained_transverse_count = 0;
    float pair_unexplained_transverse_risk = 0.0f;
    float pair_semantic_rank_adjust = 0.0f;
    float pair_fake_forward_penalty = 0.0f;
    float pair_forward_side_support_score = 0.0f;
    bool pair_has_transverse = false;
    float transverse_merge_band_score = 0.0f;
    float transverse_opening_branch_score = 0.0f;
    std::string transverse_role = "none";
    std::string transverse_role_reason = "none";
    bool ft_opening_candidate = false;
    std::string ft_opening_block_reason = "none";
    std::string transverse_t_ids;
    float transverse_near_forward_m = 0.0f;
    float transverse_far_forward_m = 0.0f;
    float transverse_lat_min_m = 0.0f;
    float transverse_lat_max_m = 0.0f;
    float transverse_lat_span_m = 0.0f;
    int transverse_guide_count = 0;
    bool transverse_near_ok = false;
    int transverse_near_cov = 0;
    std::string transverse_primitive_debug;
    std::string rank_reason = "none";
    std::string pair_type = "F+F";
    std::string semantic = "noise";
    std::string semantic_reason = "none";
    std::string opening_reject_reason = "none";
    std::string final_pair_source = "primitive_pair";
    std::string reject_reason = "none";
    std::string width_sampling_mode = "none";
    int width_sample_count = 0;
    int valid_width_count = 0;
    float min_overlap_used = 0.0f;
    std::string sampled_forward_values;
    std::string sampled_width_values;
    std::string width_stats_reject_reason = "none";
};

static constexpr float kPhysicalBoundaryProviderBucketWidth = planMeters(0.12f);
static constexpr float kPhysicalBoundaryProviderSideMargin = planMeters(0.05f);
static constexpr float kPhysicalBoundaryProviderMinSeparation = planMeters(0.04f);
static constexpr float kPhysicalBoundaryProviderMaxForwardGap = planMeters(0.32f);
static constexpr float kPhysicalBoundaryProviderMinSpan = planMeters(0.08f);
static constexpr float kPhysicalBoundaryProviderMinOverlap = planMeters(0.06f);
static constexpr float kPhysicalBoundaryProviderMaxWidth = planMeters(0.65f);
static constexpr float kPhysicalBoundaryProviderMinLateralGap = planMeters(0.06f);
static constexpr size_t kPhysicalBoundaryProviderMaxChainsPerSide = 3;
static constexpr size_t kPhysicalBoundaryProviderMaxChainsV2 = 12;
static constexpr size_t kPhysicalBoundaryProviderMaxPairs = 4;
static constexpr float kPhysicalBoundaryAppendMinLateralJumpM = 0.24f;
static constexpr float kPhysicalBoundaryAppendLateralJumpHalfWidthScale = 1.60f;
static constexpr float kPhysicalBoundaryAppendMaxAngleDeg = 88.0f;
static constexpr float kPhysicalBoundaryAppendMaxCurvatureResidualM = 0.22f;

struct PhysicalPairWidthStatsDebug {
    std::string width_sampling_mode = "none";
    float overlap_forward = 0.0f;
    float min_overlap_used = 0.0f;
    int sample_count = 0;
    int valid_width_count = 0;
    std::string sampled_forward_values;
    std::string sampled_width_values;
    std::string reject_reason = "none";
};

static std::string formatPlanMetersListForDebug(const std::vector<float>& values) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << (values[i] / kWorldPlanScale);
    }
    return oss.str();
}

static const char* physicalBoundaryPrimitiveTypeName(PhysicalBoundaryPrimitiveType type) {
    switch (type) {
        case PhysicalBoundaryPrimitiveType::Forward:
            return "forward";
        case PhysicalBoundaryPrimitiveType::Transverse:
            return "transverse";
        case PhysicalBoundaryPrimitiveType::Diagonal:
            return "diagonal";
        default:
            return "unknown";
    }
}

static std::string physicalBoundaryPrimitiveTypeTag(PhysicalBoundaryPrimitiveType type) {
    switch (type) {
        case PhysicalBoundaryPrimitiveType::Forward:
            return "F";
        case PhysicalBoundaryPrimitiveType::Transverse:
            return "T";
        case PhysicalBoundaryPrimitiveType::Diagonal:
            return "D";
        default:
            return "U";
    }
}

static bool physicalBoundaryPrimitiveSharesGuideId(const PhysicalBoundaryPrimitive& a,
                                                   const PhysicalBoundaryPrimitive& b,
                                                   int min_shared = 1) {
    if (min_shared <= 0) {
        return true;
    }
    std::unordered_set<int> ids;
    for (int id : a.guide_ids) {
        if (id >= 0) {
            ids.insert(id);
        }
    }
    int shared = 0;
    for (int id : b.guide_ids) {
        if (id >= 0 && ids.count(id) > 0) {
            ++shared;
            if (shared >= min_shared) {
                return true;
            }
        }
    }
    return false;
}

static int countUniqueGuideIds(const std::vector<ArrowGuide>& guides) {
    std::unordered_set<int> ids;
    for (const auto& guide : guides) {
        if (guide.tracker_id >= 0) {
            ids.insert(guide.tracker_id);
        }
    }
    return static_cast<int>(ids.size());
}

static bool physicalBoundaryPrimitiveIsStrongTransverse(const PhysicalBoundaryPrimitive& primitive) {
    return primitive.type == PhysicalBoundaryPrimitiveType::Transverse &&
        primitive.quality_score >= 0.70f &&
        primitive.lateral_extent_m >= 0.30f &&
        primitive.forward_extent_m <= 0.28f;
}

static bool physicalBoundaryPrimitiveIsWeakShortForward(const PhysicalBoundaryPrimitive& primitive) {
    return primitive.type == PhysicalBoundaryPrimitiveType::Forward &&
        primitive.chain.guides.size() <= 2 &&
        !primitive.root_ok &&
        !primitive.near_ok &&
        primitive.near_cov <= 0 &&
        primitive.quality_score <= 0.45f;
}

static float physicalBoundaryPrimitiveForwardSupportScore(const PhysicalBoundaryPrimitive& primitive) {
    if (primitive.type != PhysicalBoundaryPrimitiveType::Forward) {
        return 0.5f;
    }
    return clampValue(
        0.35f * clampValue(static_cast<float>(primitive.chain.guides.size()) / 4.0f, 0.0f, 1.0f) +
            0.20f * (primitive.root_ok ? 1.0f : 0.0f) +
            0.20f * (primitive.near_ok ? 1.0f : 0.0f) +
            0.15f * clampValue(static_cast<float>(primitive.near_cov) / 3.0f, 0.0f, 1.0f) +
            0.10f * clampValue(primitive.quality_score, 0.0f, 1.0f),
        0.0f,
        1.0f);
}

static float computeMedianFloat(std::vector<float> values) {
    if (values.empty()) {
        return 0.0f;
    }
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(),
                     values.begin() + static_cast<std::ptrdiff_t>(mid),
                     values.end());
    float median = values[mid];
    if ((values.size() % 2) == 0 && mid > 0) {
        const auto max_it =
            std::max_element(values.begin(),
                             values.begin() + static_cast<std::ptrdiff_t>(mid));
        if (max_it != values.begin() + static_cast<std::ptrdiff_t>(mid)) {
            median = 0.5f * (median + *max_it);
        }
    }
    return median;
}

static float findLateralSeparationHint(const std::vector<ArrowGuide>& valid_guides,
                                        float* out_gap_confidence) {
    if (out_gap_confidence) *out_gap_confidence = 0.0f;
    if (valid_guides.size() < 3) {
        std::vector<float> laterals;
        laterals.reserve(valid_guides.size());
        for (const auto& g : valid_guides) {
            laterals.push_back(guidePlanLateral(g));
        }
        return computeMedianFloat(laterals);
    }
    std::vector<float> laterals;
    laterals.reserve(valid_guides.size());
    for (const auto& g : valid_guides) {
        laterals.push_back(guidePlanLateral(g));
    }
    std::sort(laterals.begin(), laterals.end());
    float max_gap = 0.0f;
    size_t max_gap_idx = 0;
    for (size_t i = 0; i + 1 < laterals.size(); ++i) {
        float gap = laterals[i + 1] - laterals[i];
        if (gap > max_gap) {
            max_gap = gap;
            max_gap_idx = i;
        }
    }
    if (max_gap >= kPhysicalBoundaryProviderMinLateralGap) {
        float separation = 0.5f * (laterals[max_gap_idx] + laterals[max_gap_idx + 1]);
        if (out_gap_confidence) {
            *out_gap_confidence = clampValue(
                max_gap / planMeters(0.18f), 0.0f, 1.0f);
        }
        return separation;
    }
    return computeMedianFloat(laterals);
}

static void computeGuideForwardRange(const std::vector<ArrowGuide>& guides,
                                     float* out_min_forward,
                                     float* out_max_forward) {
    if (out_min_forward != nullptr) {
        *out_min_forward = 0.0f;
    }
    if (out_max_forward != nullptr) {
        *out_max_forward = 0.0f;
    }
    if (guides.empty()) {
        return;
    }

    float min_forward = guidePlanForward(guides.front());
    float max_forward = min_forward;
    for (const auto& guide : guides) {
        const float forward = guidePlanForward(guide);
        min_forward = std::min(min_forward, forward);
        max_forward = std::max(max_forward, forward);
    }
    if (out_min_forward != nullptr) {
        *out_min_forward = min_forward;
    }
    if (out_max_forward != nullptr) {
        *out_max_forward = max_forward;
    }
}

static float computeChainWidthStability(std::vector<float> widths) {
    if (widths.size() < 2) {
        return 0.0f;
    }
    std::sort(widths.begin(), widths.end());
    return widths.back() - widths.front();
}

static PhysicalBoundaryChain finalizePhysicalBoundaryChain(
    std::vector<ArrowGuide> guides,
    int side_hint,
    float min_forward_ref) {
    PhysicalBoundaryChain chain;
    if (guides.empty()) {
        return chain;
    }

    sortGuidesByDescendingForward(guides);
    chain.guides = std::move(guides);
    chain.side_hint = side_hint;
    chain.root_lateral = guidePlanLateral(chain.guides.front());
    chain.root_forward = guidePlanForward(chain.guides.front());
    computeGuideForwardRange(chain.guides, &chain.near_forward, &chain.far_forward);
    chain.span = std::max(0.0f, chain.far_forward - chain.near_forward);
    chain.heading_deg = computeRouteHeadingDeg(chain.guides);
    chain.max_forward_gap = maxAdjacentGuideForwardGap(chain.guides);

    std::vector<float> laterals;
    laterals.reserve(chain.guides.size());
    std::unordered_set<int> covered_buckets;
    for (const auto& guide : chain.guides) {
        laterals.push_back(guidePlanLateral(guide));
        const float rel_forward = guidePlanForward(guide) - min_forward_ref;
        const int bucket_idx = static_cast<int>(std::floor(
            rel_forward / std::max(1.0f, kPhysicalBoundaryProviderBucketWidth)));
        covered_buckets.insert(bucket_idx);
    }
    chain.median_lateral = computeMedianFloat(laterals);
    chain.bucket_coverage = static_cast<int>(covered_buckets.size());

    const int expected_buckets = std::max(
        1,
        static_cast<int>(std::floor(
            chain.span / std::max(1.0f, kPhysicalBoundaryProviderBucketWidth))) + 1);
    chain.coverage_score = clampValue(
        static_cast<float>(chain.bucket_coverage) / static_cast<float>(expected_buckets),
        0.0f,
        1.0f);

    const float gap_score = 1.0f - clampValue(
        chain.max_forward_gap / std::max(kPhysicalBoundaryProviderMaxForwardGap,
                                         chain.span + planMeters(0.02f)),
        0.0f,
        1.0f);
    const float span_score = clampValue(
        chain.span / std::max(planMeters(0.20f), kPhysicalBoundaryProviderMinSpan),
        0.0f,
        1.0f);
    const float count_score = clampValue(
        static_cast<float>(chain.guides.size()) / 5.0f,
        0.0f,
        1.0f);
    chain.continuity_score = clampValue(
        0.55f * gap_score + 0.45f * chain.coverage_score,
        0.0f,
        1.0f);
    chain.confidence = clampValue(
        0.30f * span_score +
            0.26f * count_score +
            0.24f * chain.continuity_score +
            0.20f * chain.coverage_score,
        0.0f,
        1.0f);
    return chain;
}

static std::vector<PhysicalBoundaryChain> buildPhysicalBoundaryChainsForSide(
    const std::vector<ArrowGuide>& valid_guides,
    float center_ref_lateral,
    bool negative_side,
    float median_hw) {
    std::vector<PhysicalBoundaryGuidePoint> points;
    points.reserve(valid_guides.size());

    float min_forward_ref = std::numeric_limits<float>::infinity();
    for (const auto& guide : valid_guides) {
        const cv::Point2f plan = guidePlanPoint(guide);
        if (!isFinitePoint(plan)) {
            continue;
        }
        const float lateral = plan.x;
        const bool side_match =
            negative_side ? (lateral < center_ref_lateral - kPhysicalBoundaryProviderSideMargin)
                          : (lateral > center_ref_lateral + kPhysicalBoundaryProviderSideMargin);
        if (!side_match) {
            continue;
        }
        min_forward_ref = std::min(min_forward_ref, plan.y);
        PhysicalBoundaryGuidePoint point;
        point.guide = guide;
        point.forward = plan.y;
        point.lateral = lateral;
        points.push_back(point);
    }
    if (points.empty()) {
        return {};
    }

    if (!std::isfinite(min_forward_ref)) {
        min_forward_ref = 0.0f;
    }
    for (auto& point : points) {
        const float rel_forward = point.forward - min_forward_ref;
        point.bucket_idx = static_cast<int>(std::floor(
            rel_forward / std::max(1.0f, kPhysicalBoundaryProviderBucketWidth)));
    }

    std::sort(points.begin(), points.end(), [](const PhysicalBoundaryGuidePoint& a,
                                               const PhysicalBoundaryGuidePoint& b) {
        if (std::abs(a.forward - b.forward) > 1e-3f) {
            return a.forward < b.forward;
        }
        return a.lateral < b.lateral;
    });

    std::vector<std::vector<ArrowGuide>> chain_guides;
    const float lateral_jump_limit =
        std::max(planMeters(0.18f), median_hw * 1.10f);
    for (const auto& point : points) {
        int best_chain_idx = -1;
        float best_cost = std::numeric_limits<float>::infinity();
        for (size_t chain_idx = 0; chain_idx < chain_guides.size(); ++chain_idx) {
            const auto& chain = chain_guides[chain_idx];
            if (chain.empty()) {
                continue;
            }
            const ArrowGuide& tail = chain.back();
            const float dfwd = point.forward - guidePlanForward(tail);
            if (dfwd < -planMeters(0.005f) ||
                dfwd > kPhysicalBoundaryProviderMaxForwardGap) {
                continue;
            }
            const float dlat = std::abs(point.lateral - guidePlanLateral(tail));
            if (dlat > lateral_jump_limit) {
                continue;
            }
            float heading_delta = 0.0f;
            if (chain.size() >= 2) {
                const ArrowGuide& prev = chain[chain.size() - 2];
                const cv::Point2f v0 = guidePlanPoint(tail) - guidePlanPoint(prev);
                const cv::Point2f v1 = guidePlanPoint(point.guide) - guidePlanPoint(tail);
                const float norm0 = std::sqrt(v0.x * v0.x + v0.y * v0.y);
                const float norm1 = std::sqrt(v1.x * v1.x + v1.y * v1.y);
                if (norm0 > 1e-3f && norm1 > 1e-3f) {
                    const float cos_angle = clampValue(
                        (v0.x * v1.x + v0.y * v1.y) / (norm0 * norm1), -1.0f, 1.0f);
                    heading_delta =
                        std::abs(static_cast<float>(std::acos(cos_angle) * 180.0 / CV_PI));
                }
            }
            if (heading_delta > 70.0f) {
                continue;
            }
            const float attach_cost = dlat + 0.35f * dfwd + 0.20f * heading_delta;
            if (attach_cost < best_cost) {
                best_cost = attach_cost;
                best_chain_idx = static_cast<int>(chain_idx);
            }
        }

        if (best_chain_idx >= 0) {
            chain_guides[static_cast<size_t>(best_chain_idx)].push_back(point.guide);
        } else {
            chain_guides.push_back({point.guide});
        }
    }

    std::vector<PhysicalBoundaryChain> chains;
    chains.reserve(chain_guides.size());
    for (auto& guides : chain_guides) {
        PhysicalBoundaryChain chain =
            finalizePhysicalBoundaryChain(std::move(guides),
                                          negative_side ? -1 : 1,
                                          min_forward_ref);
        if (chain.guides.size() < 2 || chain.span < kPhysicalBoundaryProviderMinSpan) {
            continue;
        }
        if (chain.max_forward_gap > kPhysicalBoundaryProviderMaxForwardGap) {
            chain.confidence *= 0.70f;
            chain.continuity_score *= 0.65f;
        }
        chains.push_back(std::move(chain));
    }

    std::sort(chains.begin(), chains.end(), [](const PhysicalBoundaryChain& a,
                                               const PhysicalBoundaryChain& b) {
        if (std::abs(a.confidence - b.confidence) > 1e-4f) {
            return a.confidence > b.confidence;
        }
        if (std::abs(a.span - b.span) > 1e-3f) {
            return a.span > b.span;
        }
        return a.guides.size() > b.guides.size();
    });
    if (chains.size() > kPhysicalBoundaryProviderMaxChainsPerSide) {
        chains.resize(kPhysicalBoundaryProviderMaxChainsPerSide);
    }
    return chains;
}

static std::vector<PhysicalBoundaryChain> buildPhysicalBoundaryChainsFromPoints(
    std::vector<PhysicalBoundaryGuidePoint> points,
    int side_hint,
    float median_hw,
    float min_forward_ref,
    bool relaxed_min_guides = false) {
    if (points.empty()) {
        return {};
    }
    if (!std::isfinite(min_forward_ref)) {
        min_forward_ref = 0.0f;
    }
    for (auto& point : points) {
        const float rel_forward = point.forward - min_forward_ref;
        point.bucket_idx = static_cast<int>(std::floor(
            rel_forward / std::max(1.0f, kPhysicalBoundaryProviderBucketWidth)));
    }
    std::sort(points.begin(), points.end(), [](const PhysicalBoundaryGuidePoint& a,
                                               const PhysicalBoundaryGuidePoint& b) {
        if (std::abs(a.forward - b.forward) > 1e-3f) {
            return a.forward < b.forward;
        }
        return a.lateral < b.lateral;
    });

    std::vector<std::vector<ArrowGuide>> chain_guides;
    const float lateral_jump_limit =
        std::max(planMeters(0.18f), median_hw * 1.10f);
    for (const auto& point : points) {
        int best_chain_idx = -1;
        float best_cost = std::numeric_limits<float>::infinity();
        for (size_t chain_idx = 0; chain_idx < chain_guides.size(); ++chain_idx) {
            const auto& chain = chain_guides[chain_idx];
            if (chain.empty()) continue;
            const ArrowGuide& tail = chain.back();
            const float dfwd = point.forward - guidePlanForward(tail);
            if (dfwd < -planMeters(0.005f) ||
                dfwd > kPhysicalBoundaryProviderMaxForwardGap) continue;
            const float dlat = std::abs(point.lateral - guidePlanLateral(tail));
            if (dlat > lateral_jump_limit) continue;
            float heading_delta = 0.0f;
            if (chain.size() >= 2) {
                const ArrowGuide& prev = chain[chain.size() - 2];
                const cv::Point2f v0 = guidePlanPoint(tail) - guidePlanPoint(prev);
                const cv::Point2f v1 = guidePlanPoint(point.guide) - guidePlanPoint(tail);
                const float norm0 = std::sqrt(v0.x * v0.x + v0.y * v0.y);
                const float norm1 = std::sqrt(v1.x * v1.x + v1.y * v1.y);
                if (norm0 > 1e-3f && norm1 > 1e-3f) {
                    const float cos_angle = clampValue(
                        (v0.x * v1.x + v0.y * v1.y) / (norm0 * norm1), -1.0f, 1.0f);
                    heading_delta =
                        std::abs(static_cast<float>(std::acos(cos_angle) * 180.0 / CV_PI));
                }
            }
            if (heading_delta > 70.0f) continue;
            const float attach_cost = dlat + 0.35f * dfwd + 0.20f * heading_delta;
            if (attach_cost < best_cost) {
                best_cost = attach_cost;
                best_chain_idx = static_cast<int>(chain_idx);
            }
        }
        if (best_chain_idx >= 0) {
            chain_guides[static_cast<size_t>(best_chain_idx)].push_back(point.guide);
        } else {
            chain_guides.push_back({point.guide});
        }
    }

    std::vector<PhysicalBoundaryChain> chains;
    chains.reserve(chain_guides.size());
    const size_t min_guides = relaxed_min_guides ? 1 : 2;
    for (auto& guides : chain_guides) {
        PhysicalBoundaryChain chain =
            finalizePhysicalBoundaryChain(std::move(guides), side_hint, min_forward_ref);
        if (chain.guides.size() < min_guides) continue;
        if (!relaxed_min_guides && chain.span < kPhysicalBoundaryProviderMinSpan) continue;
        if (chain.max_forward_gap > kPhysicalBoundaryProviderMaxForwardGap) {
            chain.confidence *= 0.70f;
            chain.continuity_score *= 0.65f;
        }
        if (relaxed_min_guides && chain.guides.size() < 3) {
            chain.confidence *= 0.55f;
            chain.continuity_score *= 0.50f;
        }
        chains.push_back(std::move(chain));
    }

    std::sort(chains.begin(), chains.end(), [](const PhysicalBoundaryChain& a,
                                               const PhysicalBoundaryChain& b) {
        if (std::abs(a.confidence - b.confidence) > 1e-4f) {
            return a.confidence > b.confidence;
        }
        if (std::abs(a.span - b.span) > 1e-3f) {
            return a.span > b.span;
        }
        return a.guides.size() > b.guides.size();
    });
    if (chains.size() > kPhysicalBoundaryProviderMaxChainsPerSide) {
        chains.resize(kPhysicalBoundaryProviderMaxChainsPerSide);
    }
    return chains;
}

static bool chainContainsTrackerId(const std::vector<ArrowGuide>& guides, int tracker_id) {
    if (tracker_id < 0) {
        return false;
    }
    for (const auto& guide : guides) {
        if (guide.tracker_id == tracker_id) {
            return true;
        }
    }
    return false;
}

struct PhysicalChainAppendEval {
    bool feasible = false;
    float cost = std::numeric_limits<float>::infinity();
    float dfwd = 0.0f;
    float dlat = 0.0f;
    float slope = 0.0f;
    float angle_delta = 0.0f;
    float turn_delta = 0.0f;
    float curvature_residual = 0.0f;
    bool near_same_layer_continuation = false;
    bool right_fork_search_near_same_layer_lateral_jump_like = false;
    float right_fork_search_near_same_layer_soft_penalty = 0.0f;
    float right_fork_search_cost_before_penalty = 0.0f;
    float right_fork_search_cost_after_penalty = 0.0f;
    bool right_fork_search_demoted_to_normal_cost = false;
    std::string reject_reason = "not_evaluated";
};

static bool recentPhysicalChainForwardLike(const std::vector<ArrowGuide>& chain,
                                           size_t edge_count) {
    if (chain.size() < edge_count + 1) {
        return false;
    }
    float slope_sum = 0.0f;
    float max_abs_slope = 0.0f;
    size_t valid_edges = 0;
    const size_t first = chain.size() - edge_count - 1;
    for (size_t i = first + 1; i < chain.size(); ++i) {
        const float dfwd = guidePlanForward(chain[i]) - guidePlanForward(chain[i - 1]);
        const float dlat = guidePlanLateral(chain[i]) - guidePlanLateral(chain[i - 1]);
        if (!isFiniteFloat(dfwd) || !isFiniteFloat(dlat) ||
            dfwd <= planMeters(0.012f)) {
            return false;
        }
        const float abs_slope = std::abs(dlat / std::max(dfwd, planMeters(0.001f)));
        slope_sum += abs_slope;
        max_abs_slope = std::max(max_abs_slope, abs_slope);
        ++valid_edges;
    }
    if (valid_edges == 0) {
        return false;
    }
    const float avg_abs_slope = slope_sum / static_cast<float>(valid_edges);
    return max_abs_slope <= 1.20f && avg_abs_slope <= 0.95f;
}

static PhysicalChainAppendEval evaluatePhysicalChainAppend(
    const std::vector<ArrowGuide>& chain,
    const PhysicalBoundaryGuidePoint& point,
    float median_hw,
    bool apply_right_fork_search_near_penalty = false) {
    PhysicalChainAppendEval eval;
    if (chain.empty()) {
        eval.feasible = true;
        eval.cost = 0.0f;
        eval.reject_reason = "accepted";
        return eval;
    }
    if (chainContainsTrackerId(chain, point.guide.tracker_id)) {
        eval.reject_reason = "duplicate_tracker";
        return eval;
    }

    const ArrowGuide& tail = chain.back();
    const float dfwd = point.forward - guidePlanForward(tail);
    eval.dfwd = dfwd;
    if (dfwd <= planMeters(0.012f) || dfwd > kPhysicalBoundaryProviderMaxForwardGap) {
        eval.reject_reason = "forward_gap";
        return eval;
    }

    const float dlat = std::abs(point.lateral - guidePlanLateral(tail));
    eval.dlat = dlat;
    eval.slope = dlat / std::max(dfwd, planMeters(0.001f));
    const float lateral_jump_limit =
        std::max(planMeters(kPhysicalBoundaryAppendMinLateralJumpM),
                 median_hw * kPhysicalBoundaryAppendLateralJumpHalfWidthScale);
    if (dlat > lateral_jump_limit) {
        eval.reject_reason = "lateral_step";
        return eval;
    }

    float angle_delta = 0.0f;
    float turn_delta = 0.0f;
    bool have_turn_delta = false;
    float curvature_residual = 0.0f;
    if (chain.size() >= 2) {
        const ArrowGuide& prev = chain[chain.size() - 2];
        const cv::Point2f v0 = guidePlanPoint(tail) - guidePlanPoint(prev);
        const cv::Point2f v1 = guidePlanPoint(point.guide) - guidePlanPoint(tail);
        const float norm0 = std::sqrt(v0.x * v0.x + v0.y * v0.y);
        const float norm1 = std::sqrt(v1.x * v1.x + v1.y * v1.y);
        if (norm0 > 1e-3f && norm1 > 1e-3f) {
            const float cos_angle = clampValue(
                (v0.x * v1.x + v0.y * v1.y) / (norm0 * norm1), -1.0f, 1.0f);
            angle_delta =
                std::abs(static_cast<float>(std::acos(cos_angle) * 180.0 / CV_PI));
        }
        const float prev_dfwd = std::max(planMeters(0.02f),
                                         guidePlanForward(tail) - guidePlanForward(prev));
        const float prev_slope =
            (guidePlanLateral(tail) - guidePlanLateral(prev)) / prev_dfwd;
        const float predicted_lateral =
            guidePlanLateral(tail) + prev_slope * dfwd;
        curvature_residual = std::abs(point.lateral - predicted_lateral);
        if (chain.size() >= 3 && norm0 > 1e-3f && norm1 > 1e-3f) {
            const ArrowGuide& before_prev = chain[chain.size() - 3];
            const cv::Point2f before_dir =
                guidePlanPoint(prev) - guidePlanPoint(before_prev);
            const float before_norm =
                std::sqrt(before_dir.x * before_dir.x +
                          before_dir.y * before_dir.y);
            if (before_norm > 1e-3f) {
                const cv::Point2f before_unit(before_dir.x / before_norm,
                                              before_dir.y / before_norm);
                const cv::Point2f prev_unit(v0.x / norm0, v0.y / norm0);
                const cv::Point2f curr_unit(v1.x / norm1, v1.y / norm1);
                const float prev_turn = singleSignedTurnDeg(before_unit, prev_unit);
                const float curr_turn = singleSignedTurnDeg(prev_unit, curr_unit);
                turn_delta = std::abs(curr_turn - prev_turn);
                have_turn_delta = true;
            }
        }
    }
    eval.angle_delta = angle_delta;
    eval.turn_delta = turn_delta;
    eval.curvature_residual = curvature_residual;

    if (angle_delta > kPhysicalBoundaryAppendMaxAngleDeg) {
        eval.reject_reason = "angle_delta";
        return eval;
    }
    if (curvature_residual >
        std::max(planMeters(kPhysicalBoundaryAppendMaxCurvatureResidualM),
                 median_hw * 1.10f)) {
        eval.reject_reason = "curvature_residual";
        return eval;
    }

    eval.feasible = true;
    eval.near_same_layer_continuation = dfwd <= planMeters(0.10f);
    eval.reject_reason = "accepted";
    const float near_cost = dlat +
                            0.26f * dfwd +
                            planMeters(0.0025f) * angle_delta +
                            0.80f * curvature_residual;
    eval.cost = near_cost;
    eval.right_fork_search_cost_before_penalty = near_cost;
    eval.right_fork_search_cost_after_penalty = near_cost;

    const float dfwd_m = dfwd / kWorldPlanScale;
    const float abs_slope = std::abs(eval.slope);
    const bool high_slope_jump = abs_slope > 2.0f;
    const bool high_turn_near_layer_jump =
        abs_slope > 0.30f && angle_delta > 45.0f &&
        have_turn_delta && turn_delta > 35.0f;
    const bool lateral_jump_like =
        apply_right_fork_search_near_penalty &&
        eval.near_same_layer_continuation &&
        chain.size() >= 4 &&
        recentPhysicalChainForwardLike(chain, 3) &&
        dfwd_m > 0.04f &&
        angle_delta > 20.0f &&
        have_turn_delta &&
        turn_delta > 15.0f &&
        (high_slope_jump || high_turn_near_layer_jump);
    if (lateral_jump_like) {
        const float dlat_m = dlat / kWorldPlanScale;
        const float curvature_residual_m = curvature_residual / kWorldPlanScale;
        const float slope_excess =
            clampValue((abs_slope - 2.0f) / 1.50f, 0.0f, 1.0f);
        const float angle_excess =
            clampValue((angle_delta - 20.0f) / 35.0f, 0.0f, 1.0f);
        const float turn_excess =
            clampValue((turn_delta - 15.0f) / 45.0f, 0.0f, 1.0f);
        const float soft_penalty_m =
            0.50f + 0.10f * slope_excess +
            0.08f * angle_excess +
            0.07f * turn_excess;
        const float near_cost_m = near_cost / kWorldPlanScale;
        const float normal_like_cost_m =
            0.35f * dfwd_m +
            1.60f * dlat_m +
            0.18f * abs_slope +
            0.021f * angle_delta +
            1.80f * curvature_residual_m +
            0.016f * turn_delta;
        const float penalized_near_cost_m = near_cost_m + soft_penalty_m;
        const float normal_floor_m = 0.85f * normal_like_cost_m;
        const float final_cost_m =
            std::max(penalized_near_cost_m, normal_floor_m);
        eval.cost = planMeters(final_cost_m);
        eval.right_fork_search_near_same_layer_lateral_jump_like = true;
        eval.right_fork_search_near_same_layer_soft_penalty =
            planMeters(soft_penalty_m);
        eval.right_fork_search_cost_after_penalty = eval.cost;
        eval.right_fork_search_demoted_to_normal_cost =
            normal_floor_m > penalized_near_cost_m;
    }
    return eval;
}

static float physicalChainAppendCost(const std::vector<ArrowGuide>& chain,
                                     const PhysicalBoundaryGuidePoint& point,
                                     float median_hw) {
    return evaluatePhysicalChainAppend(chain, point, median_hw).cost;
}

static std::vector<PhysicalBoundaryChain> buildPhysicalBoundaryChainsV2(
    const std::vector<PhysicalBoundaryGuidePoint>& input_points,
    float median_hw,
    float min_forward_ref) {
    if (input_points.empty()) {
        return {};
    }

    std::vector<PhysicalBoundaryGuidePoint> points = input_points;
    std::sort(points.begin(), points.end(), [](const PhysicalBoundaryGuidePoint& a,
                                               const PhysicalBoundaryGuidePoint& b) {
        if (std::abs(a.forward - b.forward) > 1e-3f) {
            return a.forward < b.forward;
        }
        return a.lateral < b.lateral;
    });

    std::vector<PhysicalBoundaryChain> chains;
    chains.reserve(points.size());
    for (size_t seed_idx = 0; seed_idx < points.size(); ++seed_idx) {
        std::vector<ArrowGuide> chain_guides;
        chain_guides.push_back(points[seed_idx].guide);

        size_t last_idx = seed_idx;
        while (true) {
            int best_idx = -1;
            float best_cost = std::numeric_limits<float>::infinity();
            for (size_t i = last_idx + 1; i < points.size(); ++i) {
                const float cost = physicalChainAppendCost(chain_guides, points[i], median_hw);
                if (cost < best_cost) {
                    best_cost = cost;
                    best_idx = static_cast<int>(i);
                }
            }
            if (best_idx < 0) {
                break;
            }
            chain_guides.push_back(points[static_cast<size_t>(best_idx)].guide);
            last_idx = static_cast<size_t>(best_idx);
        }

        PhysicalBoundaryChain chain =
            finalizePhysicalBoundaryChain(std::move(chain_guides), 0, min_forward_ref);
        if (chain.guides.size() < 2 || chain.span < kPhysicalBoundaryProviderMinSpan) {
            continue;
        }
        if (chain.max_forward_gap > kPhysicalBoundaryProviderMaxForwardGap) {
            chain.confidence *= 0.72f;
            chain.continuity_score *= 0.68f;
        }
        chains.push_back(std::move(chain));
    }

    std::sort(chains.begin(), chains.end(), [](const PhysicalBoundaryChain& a,
                                               const PhysicalBoundaryChain& b) {
        if (std::abs(a.confidence - b.confidence) > 1e-4f) {
            return a.confidence > b.confidence;
        }
        if (std::abs(a.span - b.span) > 1e-3f) {
            return a.span > b.span;
        }
        return a.guides.size() > b.guides.size();
    });

    std::vector<PhysicalBoundaryChain> unique_chains;
    unique_chains.reserve(chains.size());
    for (auto& chain : chains) {
        bool duplicate = false;
        for (const auto& kept : unique_chains) {
            if (computeGuideIdMatchRatio(chain.guides, kept.guides) >= 0.80f ||
                computeGuideMatchRatioLoose(chain.guides, kept.guides, 10.0f, 10.0f) >= 0.85f) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            unique_chains.push_back(std::move(chain));
        }
        if (unique_chains.size() >= kPhysicalBoundaryProviderMaxChainsV2) {
            break;
        }
    }
    return unique_chains;
}

struct RightForkSearchGrowState {
    std::vector<ArrowGuide> guides;
    size_t last_idx = 0;
    bool used_extra_forward_alt = false;
    float accumulated_cost = 0.0f;
    bool created_by_right_fork_alt = false;
    int right_fork_alt_depth = 0;
    std::string right_fork_alt_source_path_ids = "none";
    int right_fork_alt_source_candidate_id = -1;
};

struct RightForkSearchGrowOption {
    PhysicalChainAppendEval eval;
    size_t point_idx = 0;
};

struct RightForkAltDecision {
    bool considered = false;
    bool created = false;
    std::string reason = "not_considered";
    size_t candidate_option_idx = 0;
    int expected_tail_len = 0;
    float post_split_lateral_trend = 0.0f;
    bool best_reconnect_main_tail = false;
    std::string keep_reason = "none";
    std::string drop_reason = "none";
};

static bool isNearSameLayerSideJump(const PhysicalChainAppendEval& eval) {
    if (!eval.near_same_layer_continuation) {
        return false;
    }
    const float dfwd_m = eval.dfwd / kWorldPlanScale;
    const float dlat_m = eval.dlat / kWorldPlanScale;
    return dfwd_m <= 0.10f &&
           (dlat_m >= 0.055f || std::abs(eval.slope) >= 0.75f ||
            eval.angle_delta >= 24.0f);
}

static bool isForwardLikeAlternative(const PhysicalChainAppendEval& best,
                                     const PhysicalChainAppendEval& alt) {
    if (!alt.feasible || alt.near_same_layer_continuation) {
        return false;
    }
    const float forward_gain_m = (alt.dfwd - best.dfwd) / kWorldPlanScale;
    if (forward_gain_m < 0.055f) {
        return false;
    }
    const bool steadier_slope = std::abs(alt.slope) <=
        std::max(1.05f, std::abs(best.slope) * 0.72f);
    const bool steadier_angle = alt.angle_delta + 6.0f <= best.angle_delta ||
        alt.angle_delta <= 18.0f;
    if (!steadier_slope && !steadier_angle) {
        return false;
    }
    const float allowed_cost_delta = planMeters(0.36f);
    return alt.cost <= best.cost + allowed_cost_delta;
}

static bool rightForkSearchCandidateReconnectsMainTail(
    const std::vector<ArrowGuide>& path,
    const PhysicalBoundaryGuidePoint& point,
    const PhysicalChainAppendEval& eval,
    const std::vector<PhysicalBoundaryGuidePoint>& points) {
    if (path.empty() || !eval.feasible || eval.near_same_layer_continuation) {
        return false;
    }
    const float dfwd_m = eval.dfwd / kWorldPlanScale;
    const float dlat_m = eval.dlat / kWorldPlanScale;
    if (dfwd_m < 0.12f || dlat_m > 0.14f || std::abs(eval.slope) > 0.95f) {
        return false;
    }

    std::vector<ArrowGuide> probe = path;
    probe.push_back(point.guide);
    int forward_like_tail = recentPhysicalChainForwardLike(
        probe, std::min<size_t>(2, probe.size() - 1)) ? 1 : 0;
    size_t last_idx = 0;
    for (size_t i = 0; i < points.size(); ++i) {
        if (points[i].guide.tracker_id == point.guide.tracker_id) {
            last_idx = i;
            break;
        }
    }
    for (size_t i = last_idx + 1; i < points.size() && forward_like_tail < 3; ++i) {
        const PhysicalChainAppendEval next_eval =
            evaluatePhysicalChainAppend(probe, points[i], planMeters(0.12f), false);
        if (!next_eval.feasible || next_eval.near_same_layer_continuation ||
            std::abs(next_eval.slope) > 0.95f) {
            continue;
        }
        probe.push_back(points[i].guide);
        ++forward_like_tail;
    }
    return forward_like_tail >= 2;
}

static int estimateRightForkAltExpectedTailLen(
    const std::vector<ArrowGuide>& path,
    size_t candidate_idx,
    const std::vector<PhysicalBoundaryGuidePoint>& points,
    float median_hw,
    float* out_post_split_lateral_trend) {
    if (out_post_split_lateral_trend != nullptr) {
        *out_post_split_lateral_trend = 0.0f;
    }
    if (path.empty() || candidate_idx >= points.size()) {
        return 0;
    }

    std::vector<ArrowGuide> probe = path;
    const float split_lateral = guidePlanLateral(path.back());
    probe.push_back(points[candidate_idx].guide);
    if (out_post_split_lateral_trend != nullptr) {
        *out_post_split_lateral_trend =
            (guidePlanLateral(probe.back()) - split_lateral) / kWorldPlanScale;
    }

    int tail_len = 1;
    size_t last_idx = candidate_idx;
    while (tail_len < 3) {
        int best_idx = -1;
        float best_cost = std::numeric_limits<float>::infinity();
        const float tail_lateral = guidePlanLateral(probe.back());
        for (size_t i = last_idx + 1; i < points.size(); ++i) {
            const PhysicalChainAppendEval eval =
                evaluatePhysicalChainAppend(probe, points[i], median_hw, true);
            if (!eval.feasible || !std::isfinite(eval.cost)) {
                continue;
            }
            const float lateral_step_m =
                (points[i].lateral - tail_lateral) / kWorldPlanScale;
            if (lateral_step_m < -0.025f) {
                continue;
            }
            if (eval.cost < best_cost) {
                best_cost = eval.cost;
                best_idx = static_cast<int>(i);
            }
        }
        if (best_idx < 0) {
            break;
        }
        probe.push_back(points[static_cast<size_t>(best_idx)].guide);
        last_idx = static_cast<size_t>(best_idx);
        ++tail_len;
    }
    return tail_len;
}

static RightForkAltDecision chooseRightForkLikeAlternate(
    const std::vector<ArrowGuide>& current_path,
    const std::vector<RightForkSearchGrowOption>& next_options,
    const std::vector<PhysicalBoundaryGuidePoint>& points,
    float median_hw) {
    RightForkAltDecision decision;
    if (current_path.empty() || next_options.size() < 2) {
        decision.drop_reason = "insufficient_options";
        return decision;
    }
    decision.considered = true;

    const RightForkSearchGrowOption& best = next_options.front();
    const PhysicalBoundaryGuidePoint& best_point = points[best.point_idx];
    decision.best_reconnect_main_tail =
        rightForkSearchCandidateReconnectsMainTail(
            current_path, best_point, best.eval, points);
    const float best_lateral_step_m =
        (best_point.lateral - guidePlanLateral(current_path.back())) /
        kWorldPlanScale;
    const bool best_near_layer_crossback =
        best.eval.near_same_layer_continuation && best_lateral_step_m <= 0.020f &&
        current_path.size() >= 3;
    if (!decision.best_reconnect_main_tail && !best_near_layer_crossback) {
        decision.reason = "best_not_mixed_like";
        decision.drop_reason = "best_not_mixed_like";
        return decision;
    }

    float best_alt_score = -1.0f;
    for (size_t option_idx = 1; option_idx < next_options.size(); ++option_idx) {
        const RightForkSearchGrowOption& option = next_options[option_idx];
        const PhysicalBoundaryGuidePoint& candidate = points[option.point_idx];
        const float cost_gap_m = (option.eval.cost - best.eval.cost) / kWorldPlanScale;
        if (cost_gap_m > 0.12f) {
            continue;
        }
        if (rightForkSearchCandidateReconnectsMainTail(
                current_path, candidate, option.eval, points)) {
            continue;
        }

        float post_split_trend = 0.0f;
        const int expected_tail_len =
            estimateRightForkAltExpectedTailLen(
                current_path, option.point_idx, points, median_hw,
                &post_split_trend);
        const float lateral_step_m =
            (candidate.lateral - guidePlanLateral(current_path.back())) /
            kWorldPlanScale;
        const bool right_fork_like =
            lateral_step_m >= 0.055f &&
            post_split_trend >= 0.055f &&
            (expected_tail_len >= 2 || std::abs(option.eval.slope) >= 1.25f);
        if (!right_fork_like) {
            continue;
        }

        const float alt_score =
            0.55f * clampValue(post_split_trend / 0.18f, 0.0f, 1.0f) +
            0.30f * clampValue(static_cast<float>(expected_tail_len) / 3.0f,
                               0.0f,
                               1.0f) -
            0.15f * clampValue(cost_gap_m / 0.12f, 0.0f, 1.0f);
        if (alt_score > best_alt_score) {
            best_alt_score = alt_score;
            decision.created = true;
            decision.reason = decision.best_reconnect_main_tail
                ? "best_reconnect_main_tail"
                : "best_near_layer_crossback";
            decision.candidate_option_idx = option_idx;
            decision.expected_tail_len = expected_tail_len;
            decision.post_split_lateral_trend = post_split_trend;
            decision.keep_reason = "right_fork_like_tail";
            decision.drop_reason = "none";
        }
    }

    if (!decision.created) {
        decision.reason = decision.best_reconnect_main_tail
            ? "best_reconnect_main_tail"
            : "best_near_layer_crossback";
        decision.drop_reason = "no_right_fork_like_candidate";
    }
    return decision;
}

static bool guideIdsAreStrictSuffix(const std::vector<ArrowGuide>& maybe_suffix,
                                    const std::vector<ArrowGuide>& full) {
    if (maybe_suffix.empty() || maybe_suffix.size() >= full.size()) {
        return false;
    }
    const size_t offset = full.size() - maybe_suffix.size();
    for (size_t i = 0; i < maybe_suffix.size(); ++i) {
        if (maybe_suffix[i].tracker_id != full[offset + i].tracker_id) {
            return false;
        }
    }
    return true;
}

static std::string choosePreferredSimilarChain(const PhysicalBoundaryChain& existing,
                                              const PhysicalBoundaryChain& candidate,
                                              bool* prefer_candidate) {
    *prefer_candidate = false;
    if (guideIdsAreStrictSuffix(existing.guides, candidate.guides)) {
        *prefer_candidate = true;
        return "prefer_prefix";
    }
    if (guideIdsAreStrictSuffix(candidate.guides, existing.guides)) {
        return "duplicate_short_suffix";
    }
    if (candidate.guides.size() > existing.guides.size()) {
        *prefer_candidate = true;
        return "prefer_longer_chain";
    }
    if (existing.guides.size() > candidate.guides.size()) {
        return "prefer_longer_chain";
    }
    if (candidate.span > existing.span + planMeters(0.05f)) {
        *prefer_candidate = true;
        return "prefer_forward_span";
    }
    if (existing.span > candidate.span + planMeters(0.05f)) {
        return "prefer_forward_span";
    }
    if (candidate.near_forward + planMeters(0.05f) < existing.near_forward) {
        *prefer_candidate = true;
        return "prefer_prefix";
    }
    if (existing.near_forward + planMeters(0.05f) < candidate.near_forward) {
        return "prefer_prefix";
    }
    if (candidate.used_extra_forward_alt != existing.used_extra_forward_alt) {
        *prefer_candidate = candidate.used_extra_forward_alt;
        return "prefer_extra_forward_alt";
    }
    if (candidate.confidence > existing.confidence + 0.05f) {
        *prefer_candidate = true;
        return "prefer_confidence";
    }
    return "keep_existing";
}

static std::vector<PhysicalBoundaryChain> buildRightForkSearchChains(
    const std::vector<PhysicalBoundaryGuidePoint>& input_points,
    float median_hw,
    float min_forward_ref) {
    if (input_points.empty()) {
        return {};
    }

    std::vector<PhysicalBoundaryGuidePoint> points = input_points;
    std::sort(points.begin(), points.end(), [](const PhysicalBoundaryGuidePoint& a,
                                               const PhysicalBoundaryGuidePoint& b) {
        if (std::abs(a.forward - b.forward) > 1e-3f) {
            return a.forward < b.forward;
        }
        return a.lateral < b.lateral;
    });

    constexpr size_t kRightForkSearchTopKContinuation = 2;
    constexpr size_t kMaxGrowPathsPerRoot = 6;
    constexpr size_t kMaxActiveGrowPathsPerRoot = 8;
    constexpr size_t kMaxRawRightForkSearchChains = 72;
    std::vector<PhysicalBoundaryChain> chains;
    chains.reserve(std::min(points.size() * kRightForkSearchTopKContinuation,
                            kMaxRawRightForkSearchChains));

    for (size_t seed_idx = 0; seed_idx < points.size(); ++seed_idx) {
        std::vector<RightForkSearchGrowState> frontier;
        frontier.push_back({std::vector<ArrowGuide>{points[seed_idx].guide}, seed_idx, false, 0.0f});
        size_t root_chain_count = 0;
        while (!frontier.empty() &&
               chains.size() < kMaxRawRightForkSearchChains &&
               root_chain_count < kMaxGrowPathsPerRoot) {
            RightForkSearchGrowState state = std::move(frontier.back());
            frontier.pop_back();

            std::vector<RightForkSearchGrowOption> next_options;
            next_options.reserve(points.size() - state.last_idx - 1);
            for (size_t i = state.last_idx + 1; i < points.size(); ++i) {
                const PhysicalChainAppendEval eval =
                    evaluatePhysicalChainAppend(state.guides, points[i], median_hw, true);
                if (eval.feasible && std::isfinite(eval.cost)) {
                    next_options.push_back({eval, i});
                }
            }

            if (next_options.empty()) {
                PhysicalBoundaryChain chain =
                    finalizePhysicalBoundaryChain(std::move(state.guides), 0, min_forward_ref);
                if (chain.guides.size() < 2 || chain.span < kPhysicalBoundaryProviderMinSpan) {
                    continue;
                }
                if (chain.max_forward_gap > kPhysicalBoundaryProviderMaxForwardGap) {
                    chain.confidence *= 0.72f;
                    chain.continuity_score *= 0.68f;
                }
                chain.used_extra_forward_alt = state.used_extra_forward_alt;
                chain.grow_accumulated_cost = state.accumulated_cost;
                chain.created_by_right_fork_alt = state.created_by_right_fork_alt;
                chain.right_fork_alt_source_path_ids =
                    state.right_fork_alt_source_path_ids;
                chain.right_fork_alt_source_candidate_id =
                    state.right_fork_alt_source_candidate_id;
                chains.push_back(std::move(chain));
                ++root_chain_count;
                continue;
            }

            std::sort(next_options.begin(), next_options.end(),
                      [](const auto& a, const auto& b) {
                if (std::abs(a.eval.cost - b.eval.cost) > 1e-4f) {
                    return a.eval.cost < b.eval.cost;
                }
                return a.point_idx < b.point_idx;
            });

            const float root_forward = guidePlanForward(state.guides.front());
            const bool branch_sensitive =
                state.guides.size() <= kRightForkSearchBranchSensitiveMaxGuideCount &&
                std::max(0.0f, root_forward - min_forward_ref) <= planMeters(0.36f);
            const size_t keep_count = branch_sensitive
                ? std::min(kRightForkSearchTopKContinuation, next_options.size())
                : 1;
            for (size_t option_idx = 0; option_idx < keep_count; ++option_idx) {
                const size_t next_idx = next_options[option_idx].point_idx;
                RightForkSearchGrowState next_state;
                next_state.guides = state.guides;
                next_state.guides.push_back(points[next_idx].guide);
                next_state.last_idx = next_idx;
                next_state.used_extra_forward_alt = state.used_extra_forward_alt;
                next_state.accumulated_cost =
                    state.accumulated_cost + next_options[option_idx].eval.cost;
                next_state.created_by_right_fork_alt =
                    state.created_by_right_fork_alt;
                next_state.right_fork_alt_depth = state.right_fork_alt_depth;
                next_state.right_fork_alt_source_path_ids =
                    state.right_fork_alt_source_path_ids;
                next_state.right_fork_alt_source_candidate_id =
                    state.right_fork_alt_source_candidate_id;
                frontier.push_back(std::move(next_state));
            }
            if (!branch_sensitive &&
                !state.used_extra_forward_alt &&
                isNearSameLayerSideJump(next_options.front().eval)) {
                int alt_option_idx = -1;
                for (size_t i = 1; i < next_options.size(); ++i) {
                    if (isForwardLikeAlternative(next_options.front().eval,
                                                 next_options[i].eval)) {
                        alt_option_idx = static_cast<int>(i);
                        break;
                    }
                }
                if (alt_option_idx >= 0) {
                    const RightForkSearchGrowOption& alt =
                        next_options[static_cast<size_t>(alt_option_idx)];
                    RightForkSearchGrowState alt_state;
                    alt_state.guides = state.guides;
                    alt_state.guides.push_back(points[alt.point_idx].guide);
                    alt_state.last_idx = alt.point_idx;
                    alt_state.used_extra_forward_alt = true;
                    alt_state.accumulated_cost = state.accumulated_cost + alt.eval.cost;
                    alt_state.created_by_right_fork_alt =
                        state.created_by_right_fork_alt;
                    alt_state.right_fork_alt_depth = state.right_fork_alt_depth;
                    alt_state.right_fork_alt_source_path_ids =
                        state.right_fork_alt_source_path_ids;
                    alt_state.right_fork_alt_source_candidate_id =
                        state.right_fork_alt_source_candidate_id;
                    frontier.push_back(std::move(alt_state));
                }
            }
            if (!branch_sensitive &&
                state.right_fork_alt_depth < 2 &&
                next_options.size() >= 2) {
                const RightForkAltDecision right_fork_alt =
                    chooseRightForkLikeAlternate(
                        state.guides, next_options, points, median_hw);
                const RightForkSearchGrowOption* alt_option = nullptr;
                if (right_fork_alt.created &&
                    right_fork_alt.candidate_option_idx < next_options.size()) {
                    alt_option = &next_options[right_fork_alt.candidate_option_idx];
                }
                if (right_fork_alt.created && alt_option != nullptr) {
                    RightForkSearchGrowState alt_state;
                    alt_state.guides = state.guides;
                    alt_state.guides.push_back(points[alt_option->point_idx].guide);
                    alt_state.last_idx = alt_option->point_idx;
                    alt_state.used_extra_forward_alt = state.used_extra_forward_alt;
                    alt_state.accumulated_cost =
                        state.accumulated_cost + alt_option->eval.cost;
                    alt_state.created_by_right_fork_alt = true;
                    alt_state.right_fork_alt_depth = state.right_fork_alt_depth + 1;
                    alt_state.right_fork_alt_source_path_ids =
                        formatSingleGrowGuideIds(state.guides);
                    alt_state.right_fork_alt_source_candidate_id =
                        points[alt_option->point_idx].guide.tracker_id;
                    frontier.push_back(std::move(alt_state));
                }
            }
            if (frontier.size() > kMaxActiveGrowPathsPerRoot) {
                std::sort(frontier.begin(), frontier.end(),
                          [](const RightForkSearchGrowState& a,
                             const RightForkSearchGrowState& b) {
                    if (a.guides.size() != b.guides.size()) {
                        return a.guides.size() > b.guides.size();
                    }
                    const float a_span = a.guides.empty()
                        ? 0.0f
                        : guidePlanForward(a.guides.back()) - guidePlanForward(a.guides.front());
                    const float b_span = b.guides.empty()
                        ? 0.0f
                        : guidePlanForward(b.guides.back()) - guidePlanForward(b.guides.front());
                    if (std::abs(a_span - b_span) > planMeters(0.03f)) {
                        return a_span > b_span;
                    }
                    if (a.used_extra_forward_alt != b.used_extra_forward_alt) {
                        return a.used_extra_forward_alt;
                    }
                    const float a_root_forward = a.guides.empty()
                        ? std::numeric_limits<float>::infinity()
                        : guidePlanForward(a.guides.front());
                    const float b_root_forward = b.guides.empty()
                        ? std::numeric_limits<float>::infinity()
                        : guidePlanForward(b.guides.front());
                    if (std::abs(a_root_forward - b_root_forward) > planMeters(0.03f)) {
                        return a_root_forward < b_root_forward;
                    }
                    return a.accumulated_cost < b.accumulated_cost;
                });
                frontier.resize(kMaxActiveGrowPathsPerRoot);
                std::reverse(frontier.begin(), frontier.end());
            }
        }
    }

    std::sort(chains.begin(), chains.end(), [](const PhysicalBoundaryChain& a,
                                               const PhysicalBoundaryChain& b) {
        if (a.guides.size() != b.guides.size()) {
            return a.guides.size() > b.guides.size();
        }
        if (std::abs(a.span - b.span) > 1e-3f) {
            return a.span > b.span;
        }
        if (std::abs(a.near_forward - b.near_forward) > planMeters(0.03f)) {
            return a.near_forward < b.near_forward;
        }
        if (a.used_extra_forward_alt != b.used_extra_forward_alt) {
            return a.used_extra_forward_alt;
        }
        return a.confidence > b.confidence;
    });

    std::vector<PhysicalBoundaryChain> unique_chains;
    unique_chains.reserve(chains.size());
    for (auto& chain : chains) {
        int duplicate_index = -1;
        for (size_t i = 0; i < unique_chains.size(); ++i) {
            const float id_ratio = computeGuideIdMatchRatio(chain.guides, unique_chains[i].guides);
            const float loose_ratio = computeGuideMatchRatioLoose(chain.guides,
                                                                  unique_chains[i].guides,
                                                                  10.0f,
                                                                  10.0f);
            if (id_ratio >= 0.80f || loose_ratio >= 0.85f) {
                duplicate_index = static_cast<int>(i);
                break;
            }
        }
        if (duplicate_index < 0) {
            unique_chains.push_back(std::move(chain));
        } else {
            PhysicalBoundaryChain& kept =
                unique_chains[static_cast<size_t>(duplicate_index)];
            bool prefer_new = false;
            choosePreferredSimilarChain(kept, chain, &prefer_new);
            if (prefer_new) {
                kept = std::move(chain);
            }
        }
        if (unique_chains.size() >= kPhysicalBoundaryProviderMaxChainsV2) {
            break;
        }
    }
    return unique_chains;
}

static float computeChainLateralSlope(const PhysicalBoundaryChain& chain) {
    if (chain.guides.size() < 2 || chain.span <= 1e-3f) {
        return 0.0f;
    }
    return (guidePlanLateral(chain.guides.back()) - guidePlanLateral(chain.guides.front())) /
        std::max(chain.span, planMeters(0.02f));
}

static PhysicalBoundaryPrimitive buildPhysicalBoundaryPrimitiveFromChain(
    const PhysicalBoundaryChain& chain,
    float visible_min_forward) {
    PhysicalBoundaryPrimitive primitive;
    primitive.chain = chain;
    primitive.fwd_min = chain.near_forward;
    primitive.fwd_max = chain.far_forward;
    primitive.fwd_span = chain.span;
    primitive.principal_heading_deg = chain.heading_deg;
    primitive.forward_extent_m = chain.span / kWorldPlanScale;
    float lat_min = std::numeric_limits<float>::infinity();
    float lat_max = -std::numeric_limits<float>::infinity();
    for (const auto& guide : chain.guides) {
        lat_min = std::min(lat_min, guidePlanLateral(guide));
        lat_max = std::max(lat_max, guidePlanLateral(guide));
        primitive.guide_ids.push_back(guide.tracker_id);
    }
    if (!std::isfinite(lat_min) || !std::isfinite(lat_max)) {
        lat_min = 0.0f;
        lat_max = 0.0f;
    }
    primitive.lat_min = lat_min;
    primitive.lat_max = lat_max;
    primitive.lat_span = std::max(0.0f, lat_max - lat_min);
    primitive.lateral_extent_m = primitive.lat_span / kWorldPlanScale;
    primitive.root_ok = std::max(0.0f, chain.near_forward - visible_min_forward) <= planMeters(0.24f);
    primitive.near_ok = std::max(0.0f, chain.near_forward - visible_min_forward) <= planMeters(0.34f);
    primitive.near_cov = 0;
    const float near_limit = planMeters(0.34f);
    for (const auto& guide : chain.guides) {
        const float forward = guidePlanForward(guide);
        if (isFiniteFloat(forward) &&
            std::max(0.0f, forward - visible_min_forward) <= near_limit) {
            ++primitive.near_cov;
        }
    }

    const float slope = computeChainLateralSlope(chain);
    const float abs_heading = std::abs(chain.heading_deg);
    const bool near_root_short_branch_like =
        primitive.near_ok &&
        primitive.near_cov >= 2 &&
        chain.guides.size() <= 5 &&
        primitive.forward_extent_m >= 0.12f &&
        primitive.forward_extent_m <= 0.32f &&
        primitive.lateral_extent_m >= 0.12f &&
        (std::abs(slope) >= 0.45f || (abs_heading >= 22.0f && abs_heading <= 78.0f));
    const bool transverse_like =
        !near_root_short_branch_like &&
        primitive.forward_extent_m <= 0.24f &&
        primitive.lateral_extent_m >= std::max(0.28f, primitive.forward_extent_m * 1.35f) &&
        chain.guides.size() >= 3;
    const bool diagonal_like =
        !transverse_like &&
        primitive.forward_extent_m >= 0.10f &&
        primitive.lateral_extent_m >= 0.10f &&
        (std::abs(slope) >= 0.55f || abs_heading >= 28.0f);
    primitive.near_root_short_branch = near_root_short_branch_like;
    primitive.type = transverse_like ? PhysicalBoundaryPrimitiveType::Transverse
                                     : ((diagonal_like || near_root_short_branch_like)
                                            ? PhysicalBoundaryPrimitiveType::Diagonal
                                                      : PhysicalBoundaryPrimitiveType::Forward);
    primitive.quality_score = clampValue(
        0.34f * clampValue(static_cast<float>(chain.guides.size()) / 5.0f, 0.0f, 1.0f) +
            0.22f * clampValue(primitive.forward_extent_m / 0.45f, 0.0f, 1.0f) +
            0.18f * clampValue(primitive.lateral_extent_m / 0.40f, 0.0f, 1.0f) +
            0.14f * (primitive.root_ok ? 1.0f : 0.0f) +
            0.12f * clampValue(chain.confidence, 0.0f, 1.0f),
        0.0f,
        1.0f);
    if (near_root_short_branch_like) {
        primitive.quality_score = clampValue(primitive.quality_score + 0.08f, 0.0f, 1.0f);
    }
    primitive.weak_short_forward = physicalBoundaryPrimitiveIsWeakShortForward(primitive);
    return primitive;
}

static std::vector<PhysicalBoundaryPrimitive> buildTransversePhysicalBoundaryPrimitives(
    const std::vector<PhysicalBoundaryGuidePoint>& points,
    float min_forward_ref,
    float visible_min_forward) {
    std::map<int, std::vector<PhysicalBoundaryGuidePoint>> bucket_points;
    for (const auto& point : points) {
        bucket_points[point.bucket_idx].push_back(point);
    }

    std::vector<PhysicalBoundaryPrimitive> primitives;
    std::unordered_set<int> used_hashes;
    for (auto it = bucket_points.begin(); it != bucket_points.end(); ++it) {
        std::vector<ArrowGuide> guides;
        for (const auto& point : it->second) {
            guides.push_back(point.guide);
        }
        auto next = std::next(it);
        if (next != bucket_points.end() && next->first == it->first + 1) {
            for (const auto& point : next->second) {
                guides.push_back(point.guide);
            }
        }
        if (guides.size() < 3) {
            continue;
        }

        float lat_min = std::numeric_limits<float>::infinity();
        float lat_max = -std::numeric_limits<float>::infinity();
        float fwd_min = std::numeric_limits<float>::infinity();
        float fwd_max = -std::numeric_limits<float>::infinity();
        for (const auto& guide : guides) {
            lat_min = std::min(lat_min, guidePlanLateral(guide));
            lat_max = std::max(lat_max, guidePlanLateral(guide));
            fwd_min = std::min(fwd_min, guidePlanForward(guide));
            fwd_max = std::max(fwd_max, guidePlanForward(guide));
        }
        const float lat_span = std::isfinite(lat_min) && std::isfinite(lat_max) ? (lat_max - lat_min) : 0.0f;
        const float fwd_span = std::isfinite(fwd_min) && std::isfinite(fwd_max) ? (fwd_max - fwd_min) : 0.0f;
        if (lat_span < planMeters(0.30f) || fwd_span > planMeters(0.22f)) {
            continue;
        }

        sortGuidesByDescendingForward(guides);
        PhysicalBoundaryChain chain = finalizePhysicalBoundaryChain(std::move(guides), 0, min_forward_ref);
        PhysicalBoundaryPrimitive primitive =
            buildPhysicalBoundaryPrimitiveFromChain(chain, visible_min_forward);
        if (primitive.near_root_short_branch) {
            primitive.type = PhysicalBoundaryPrimitiveType::Diagonal;
            primitive.quality_score = clampValue(primitive.quality_score + 0.10f, 0.0f, 1.0f);
            primitive.reject_reason = "near_root_short_branch";
        } else {
            primitive.type = PhysicalBoundaryPrimitiveType::Transverse;
            primitive.principal_heading_deg = 90.0f;
            primitive.quality_score = clampValue(primitive.quality_score + 0.18f, 0.0f, 1.0f);
        }

        int hash = 0;
        for (const auto& guide : primitive.chain.guides) {
            hash = hash * 131 + std::max(0, guide.tracker_id + 3);
        }
        if (used_hashes.insert(hash).second) {
            primitives.push_back(std::move(primitive));
        }
    }
    return primitives;
}

static std::vector<PhysicalBoundaryPrimitive> buildPhysicalBoundaryPrimitives(
    const std::vector<PhysicalBoundaryGuidePoint>& points,
    float median_hw,
    float min_forward_ref,
    float visible_min_forward) {
    std::vector<PhysicalBoundaryPrimitive> primitives;
    const std::vector<PhysicalBoundaryChain> chains =
        buildPhysicalBoundaryChainsV2(points, median_hw, min_forward_ref);
    primitives.reserve(chains.size() + 8);
    for (const auto& chain : chains) {
        primitives.push_back(buildPhysicalBoundaryPrimitiveFromChain(chain, visible_min_forward));
    }
    auto transverse =
        buildTransversePhysicalBoundaryPrimitives(points, min_forward_ref, visible_min_forward);
    primitives.insert(primitives.end(), transverse.begin(), transverse.end());
    for (auto& primitive : primitives) {
        primitive.weak_short_forward = physicalBoundaryPrimitiveIsWeakShortForward(primitive);
    }
    for (auto& primitive : primitives) {
        if (primitive.type != PhysicalBoundaryPrimitiveType::Forward) {
            continue;
        }
        for (const auto& other : primitives) {
            if (!physicalBoundaryPrimitiveIsStrongTransverse(other)) {
                continue;
            }
            if (physicalBoundaryPrimitiveSharesGuideId(primitive, other, 1)) {
                primitive.transverse_derived = true;
                break;
            }
        }
    }
    std::sort(primitives.begin(), primitives.end(),
              [](const PhysicalBoundaryPrimitive& a, const PhysicalBoundaryPrimitive& b) {
        if (a.type != b.type) {
            return static_cast<int>(a.type) < static_cast<int>(b.type);
        }
        if (std::abs(a.quality_score - b.quality_score) > 1e-4f) {
            return a.quality_score > b.quality_score;
        }
        return a.chain.span > b.chain.span;
    });
    return primitives;
}

static std::string formatPhysicalBoundaryPrimitiveForDebug(
    const PhysicalBoundaryPrimitive& primitive,
    size_t index) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2)
        << "p" << index
        << "(primitive_type=" << physicalBoundaryPrimitiveTypeName(primitive.type)
        << ",primitive_guide_ids=[";
    for (size_t i = 0; i < primitive.chain.guides.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << primitive.chain.guides[i].tracker_id;
    }
    oss << "]"
        << ",primitive_fwd_range=(" << (primitive.fwd_min / kWorldPlanScale)
        << "," << (primitive.fwd_max / kWorldPlanScale) << ")"
        << ",primitive_lat_range=(" << (primitive.lat_min / kWorldPlanScale)
        << "," << (primitive.lat_max / kWorldPlanScale) << ")"
        << ",primitive_fwd_span=" << primitive.forward_extent_m
        << ",primitive_lat_span=" << primitive.lateral_extent_m
        << ",primitive_heading_deg=" << primitive.principal_heading_deg
        << ",primitive_root_ok=" << (primitive.root_ok ? 1 : 0)
        << ",primitive_near_ok=" << (primitive.near_ok ? 1 : 0)
        << ",primitive_near_cov=" << primitive.near_cov
        << ",primitive_quality=" << primitive.quality_score
        << ",primitive_is_weak_short_forward=" << (primitive.weak_short_forward ? 1 : 0)
        << ",primitive_is_transverse_derived=" << (primitive.transverse_derived ? 1 : 0)
        << ",primitive_is_near_root_short_branch=" << (primitive.near_root_short_branch ? 1 : 0)
        << ",health_reason=" << primitive.reject_reason
        << ")";
    return oss.str();
}

static float averageChainLateralInForwardRange(const PhysicalBoundaryChain& chain,
                                               float range_lo,
                                               float range_hi) {
    float sum = 0.0f;
    int count = 0;
    for (const auto& guide : chain.guides) {
        const float fwd = guidePlanForward(guide);
        if (fwd >= range_lo && fwd <= range_hi) {
            sum += guidePlanLateral(guide);
            ++count;
        }
    }
    return count > 0 ? (sum / static_cast<float>(count)) : chain.median_lateral;
}

static float computePhysicalChainSharedGuideRatio(const PhysicalBoundaryChain& a,
                                                  const PhysicalBoundaryChain& b) {
    std::unordered_set<int> ids_a;
    int id_count_a = 0;
    for (const auto& guide : a.guides) {
        if (guide.tracker_id >= 0) {
            ids_a.insert(guide.tracker_id);
            ++id_count_a;
        }
    }
    int shared_count = 0;
    int total_unique = static_cast<int>(ids_a.size());
    for (const auto& guide : b.guides) {
        if (guide.tracker_id < 0) {
            continue;
        }
        if (ids_a.count(guide.tracker_id) > 0) {
            ++shared_count;
        } else {
            ++total_unique;
        }
    }
    if (total_unique > 0) {
        return static_cast<float>(shared_count) / static_cast<float>(total_unique);
    }
    return computeGuideMatchRatioLoose(a.guides, b.guides, 10.0f, 10.0f);
}

static bool estimatePhysicalPairWidthStats(const PhysicalBoundaryChain& negative_chain,
                                           const PhysicalBoundaryChain& positive_chain,
                                           float min_overlap,
                                           float* out_overlap_forward,
                                           float* out_width_median,
                                           float* out_width_near,
                                           float* out_width_mid,
                                           float* out_width_far,
                                           float* out_left_lat_slope,
                                           float* out_right_lat_slope,
                                           float* out_width_stability,
                                           float* out_far_divergence_penalty,
                                           PhysicalPairWidthStatsDebug* out_debug) {
    if (out_debug != nullptr) {
        *out_debug = PhysicalPairWidthStatsDebug{};
        out_debug->min_overlap_used = min_overlap;
    }
    if (out_overlap_forward != nullptr) {
        *out_overlap_forward = 0.0f;
    }
    if (out_width_median != nullptr) {
        *out_width_median = 0.0f;
    }
    if (out_width_near != nullptr) {
        *out_width_near = 0.0f;
    }
    if (out_width_mid != nullptr) {
        *out_width_mid = 0.0f;
    }
    if (out_width_far != nullptr) {
        *out_width_far = 0.0f;
    }
    if (out_left_lat_slope != nullptr) {
        *out_left_lat_slope = 0.0f;
    }
    if (out_right_lat_slope != nullptr) {
        *out_right_lat_slope = 0.0f;
    }
    if (out_width_stability != nullptr) {
        *out_width_stability = 0.0f;
    }
    if (out_far_divergence_penalty != nullptr) {
        *out_far_divergence_penalty = 0.0f;
    }

    if (negative_chain.guides.size() < 2 || positive_chain.guides.size() < 2) {
        if (out_debug != nullptr) {
            out_debug->reject_reason = "insufficient_guides";
        }
        return false;
    }

    float neg_min = 0.0f;
    float neg_max = 0.0f;
    float pos_min = 0.0f;
    float pos_max = 0.0f;
    computeGuideForwardRange(negative_chain.guides, &neg_min, &neg_max);
    computeGuideForwardRange(positive_chain.guides, &pos_min, &pos_max);
    const float overlap_lo = std::max(neg_min, pos_min);
    const float overlap_hi = std::min(neg_max, pos_max);
    const float overlap_forward = std::max(0.0f, overlap_hi - overlap_lo);
    if (out_overlap_forward != nullptr) {
        *out_overlap_forward = overlap_forward;
    }
    if (out_debug != nullptr) {
        out_debug->overlap_forward = overlap_forward;
    }
    if (overlap_forward < min_overlap) {
        if (out_debug != nullptr) {
            out_debug->reject_reason = "overlap_too_small";
        }
        return false;
    }

    const float near_mid_fraction = 0.60f;
    const float far_start_forward = overlap_lo + near_mid_fraction * overlap_forward;

    const bool short_overlap_sampling = overlap_forward < planMeters(0.20f);
    const int original_sample_count = std::max(
        3,
        std::min(7,
                 static_cast<int>(std::floor(
                     overlap_forward / std::max(planMeters(0.04f),
                                                0.65f * kPhysicalBoundaryProviderBucketWidth))) +
                     2));
    std::vector<float> sample_forwards;
    if (short_overlap_sampling) {
        sample_forwards.reserve(4);
        sample_forwards.push_back(overlap_lo + 0.25f * overlap_forward);
        sample_forwards.push_back(overlap_lo + 0.50f * overlap_forward);
        sample_forwards.push_back(overlap_lo + 0.75f * overlap_forward);
        sample_forwards.push_back(overlap_lo + 0.90f * overlap_forward);
    } else {
        sample_forwards.reserve(static_cast<size_t>(original_sample_count));
        for (int i = 0; i < original_sample_count; ++i) {
            const float t =
                original_sample_count == 1
                    ? 0.0f
                    : (static_cast<float>(i) /
                       static_cast<float>(original_sample_count - 1));
            sample_forwards.push_back(overlap_lo + t * overlap_forward);
        }
    }
    const int sample_count = static_cast<int>(sample_forwards.size());
    std::vector<float> widths;
    widths.reserve(static_cast<size_t>(sample_count));
    std::vector<float> sampled_widths;
    sampled_widths.reserve(static_cast<size_t>(sample_count));
    int far_skipped_count = 0;
    float far_excess_sum = 0.0f;

    for (const float sample_forward : sample_forwards) {
        float negative_lateral = 0.0f;
        float positive_lateral = 0.0f;
        if (!interpolateGuidePlanLateralAtForward(
                negative_chain.guides, sample_forward, negative_lateral) ||
            !interpolateGuidePlanLateralAtForward(
                positive_chain.guides, sample_forward, positive_lateral)) {
            continue;
        }
        const float width = positive_lateral - negative_lateral;
        sampled_widths.push_back(width);
        if (width <= kPhysicalBoundaryProviderMinSeparation) {
            continue;
        }
        const bool is_far = (sample_forward >= far_start_forward);
        if (width > kPhysicalBoundaryProviderMaxWidth) {
            if (is_far) {
                far_skipped_count++;
                far_excess_sum += (width - kPhysicalBoundaryProviderMaxWidth);
                continue;
            } else {
                continue;
            }
        }
        widths.push_back(width);
    }

    if (out_debug != nullptr) {
        out_debug->width_sampling_mode =
            short_overlap_sampling ? "short_inner3" : "original";
        out_debug->sample_count = sample_count;
        out_debug->valid_width_count = static_cast<int>(widths.size());
        out_debug->sampled_forward_values =
            formatPlanMetersListForDebug(sample_forwards);
        out_debug->sampled_width_values =
            formatPlanMetersListForDebug(sampled_widths);
    }

    float far_divergence_penalty = 0.0f;
    if (far_skipped_count > 0) {
        far_divergence_penalty = clampValue(
            0.05f * static_cast<float>(far_skipped_count) +
            0.03f * (far_excess_sum / planMeters(0.20f)),
            0.0f,
            0.20f);
    }
    if (out_far_divergence_penalty != nullptr) {
        *out_far_divergence_penalty = far_divergence_penalty;
    }

    if (widths.size() < 2) {
        if (out_debug != nullptr) {
            out_debug->reject_reason = "valid_width_count_lt_2";
        }
        return false;
    }
    if (out_debug != nullptr) {
        out_debug->reject_reason = "none";
    }

    if (out_width_median != nullptr) {
        *out_width_median = computeMedianFloat(widths);
    }
    if (out_width_near != nullptr) {
        *out_width_near = widths.front();
    }
    const float sample_near_forward = overlap_lo;
    const float sample_mid_forward = overlap_lo + 0.50f * overlap_forward;
    const float sample_far_forward = overlap_hi;
    float neg_near = 0.0f;
    float neg_mid = 0.0f;
    float neg_far = 0.0f;
    float pos_near = 0.0f;
    float pos_mid = 0.0f;
    float pos_far = 0.0f;
    const bool have_near =
        interpolateGuidePlanLateralAtForward(negative_chain.guides, sample_near_forward, neg_near) &&
        interpolateGuidePlanLateralAtForward(positive_chain.guides, sample_near_forward, pos_near);
    const bool have_mid =
        interpolateGuidePlanLateralAtForward(negative_chain.guides, sample_mid_forward, neg_mid) &&
        interpolateGuidePlanLateralAtForward(positive_chain.guides, sample_mid_forward, pos_mid);
    const bool have_far =
        interpolateGuidePlanLateralAtForward(negative_chain.guides, sample_far_forward, neg_far) &&
        interpolateGuidePlanLateralAtForward(positive_chain.guides, sample_far_forward, pos_far);
    if (out_width_mid != nullptr) {
        *out_width_mid = have_mid ? (pos_mid - neg_mid) : computeMedianFloat(widths);
    }
    if (out_width_far != nullptr) {
        *out_width_far = have_far ? (pos_far - neg_far) : widths.back();
    }
    if (overlap_forward > 1e-3f && have_near && have_far) {
        if (out_left_lat_slope != nullptr) {
            *out_left_lat_slope = (neg_far - neg_near) / overlap_forward;
        }
        if (out_right_lat_slope != nullptr) {
            *out_right_lat_slope = (pos_far - pos_near) / overlap_forward;
        }
    }
    if (out_width_stability != nullptr) {
        *out_width_stability = computeChainWidthStability(widths);
    }
    return true;
}

static float scorePhysicalBoundaryPairCandidate(const PhysicalBoundaryPairCandidate& pair) {
    const float min_span = std::min(pair.negative_chain.span, pair.positive_chain.span);
    const float max_span = std::max(pair.negative_chain.span, pair.positive_chain.span);
    const float balance =
        max_span > 1e-3f ? clampValue(min_span / max_span, 0.0f, 1.0f) : 0.0f;
    const float overlap_score = clampValue(
        pair.overlap_forward / planMeters(0.20f),
        0.0f,
        1.0f);
    const float width_stability_score = 1.0f - clampValue(
        pair.width_stability / planMeters(0.18f),
        0.0f,
        1.0f);
    const float heading_score = 1.0f - clampValue(
        pair.heading_consistency / 90.0f,
        0.0f,
        1.0f);
    return clampValue(
        0.22f * pair.negative_chain.confidence +
            0.22f * pair.positive_chain.confidence +
            0.18f * overlap_score +
            0.16f * balance +
            0.12f * width_stability_score +
            0.10f * heading_score,
        0.0f,
        1.0f);
}

static void evaluatePhysicalBoundaryPairSemantics(
    PhysicalBoundaryPairCandidate& pair,
    const PhysicalBoundaryPrimitive& primitive_a,
    const PhysicalBoundaryPrimitive& primitive_b,
    float visible_min_forward);

static void enrichPhysicalBoundaryPairExplanation(
    PhysicalBoundaryPairCandidate& pair,
    const PhysicalBoundaryPrimitive& primitive_a,
    const PhysicalBoundaryPrimitive& primitive_b,
    const std::vector<PhysicalBoundaryPrimitive>& primitives,
    int total_unique_guides) {
    pair.pair_has_transverse =
        primitive_a.type == PhysicalBoundaryPrimitiveType::Transverse ||
        primitive_b.type == PhysicalBoundaryPrimitiveType::Transverse;

    std::unordered_set<int> pair_ids;
    for (const auto& guide : pair.negative_chain.guides) {
        if (guide.tracker_id >= 0) {
            pair_ids.insert(guide.tracker_id);
        }
    }
    for (const auto& guide : pair.positive_chain.guides) {
        if (guide.tracker_id >= 0) {
            pair_ids.insert(guide.tracker_id);
        }
    }
    pair.pair_unique_guide_count = static_cast<int>(pair_ids.size());
    pair.pair_coverage_ratio =
        total_unique_guides > 0
            ? clampValue(static_cast<float>(pair.pair_unique_guide_count) /
                             static_cast<float>(total_unique_guides),
                         0.0f,
                         1.0f)
            : 0.0f;

    int strong_transverse_count = 0;
    float unexplained_transverse_quality = 0.0f;
    for (const auto& primitive : primitives) {
        if (!physicalBoundaryPrimitiveIsStrongTransverse(primitive)) {
            continue;
        }
        ++strong_transverse_count;
        bool explained = false;
        if (pair.pair_has_transverse) {
            explained =
                physicalBoundaryPrimitiveSharesGuideId(primitive_a, primitive, 2) ||
                physicalBoundaryPrimitiveSharesGuideId(primitive_b, primitive, 2);
        }
        if (explained) {
            ++pair.pair_explained_transverse_count;
        } else {
            ++pair.pair_unexplained_transverse_count;
            unexplained_transverse_quality += clampValue(
                primitive.quality_score *
                    clampValue(primitive.lateral_extent_m / 0.60f, 0.0f, 1.0f),
                0.0f,
                1.0f);
        }
    }
    if (strong_transverse_count > 0) {
        pair.pair_unexplained_transverse_risk = clampValue(
            unexplained_transverse_quality / static_cast<float>(strong_transverse_count),
            0.0f,
            1.0f);
    }

    const auto accumulate_forward_penalty = [&](const PhysicalBoundaryPrimitive& primitive) {
        if (primitive.type != PhysicalBoundaryPrimitiveType::Forward) {
            return 0.0f;
        }
        float penalty = 0.0f;
        if (primitive.weak_short_forward) {
            penalty += 0.18f;
        }
        if (primitive.transverse_derived) {
            penalty += 0.14f;
        }
        return penalty;
    };
    pair.pair_fake_forward_penalty = clampValue(
        accumulate_forward_penalty(primitive_a) + accumulate_forward_penalty(primitive_b),
        0.0f,
        0.60f);

    int forward_count = 0;
    float forward_support_sum = 0.0f;
    if (primitive_a.type == PhysicalBoundaryPrimitiveType::Forward) {
        ++forward_count;
        forward_support_sum += physicalBoundaryPrimitiveForwardSupportScore(primitive_a);
    }
    if (primitive_b.type == PhysicalBoundaryPrimitiveType::Forward) {
        ++forward_count;
        forward_support_sum += physicalBoundaryPrimitiveForwardSupportScore(primitive_b);
    }
    pair.pair_forward_side_support_score =
        forward_count > 0 ? (forward_support_sum / static_cast<float>(forward_count)) : 0.5f;

    float semantic_adjust = 0.0f;
    if (pair.semantic == "merge" || pair.semantic == "crossing") {
        semantic_adjust += 0.14f * clampValue(
            static_cast<float>(pair.pair_explained_transverse_count), 0.0f, 2.0f) / 2.0f;
        semantic_adjust += 0.08f * pair.pair_coverage_ratio;
        semantic_adjust += 0.04f * pair.pair_forward_side_support_score;
        semantic_adjust -= 0.04f * pair.pair_fake_forward_penalty;
    } else if (pair.semantic == "opening") {
        if (pair.ft_opening_candidate) {
            semantic_adjust += 0.08f * pair.pair_coverage_ratio;
            semantic_adjust += 0.06f * pair.pair_forward_side_support_score;
            semantic_adjust -= 0.08f * pair.pair_unexplained_transverse_risk;
            semantic_adjust -= 0.06f * pair.pair_fake_forward_penalty;
            semantic_adjust += 0.06f * pair.transverse_opening_branch_score;
        } else {
            semantic_adjust += 0.06f * pair.pair_coverage_ratio;
            semantic_adjust += 0.04f * pair.pair_forward_side_support_score;
            semantic_adjust -= 0.18f * pair.pair_unexplained_transverse_risk;
            semantic_adjust -= 0.16f * pair.pair_fake_forward_penalty;
        }
    } else {
        semantic_adjust += 0.03f * pair.pair_coverage_ratio;
        semantic_adjust -= 0.08f * pair.pair_unexplained_transverse_risk;
        semantic_adjust -= 0.08f * pair.pair_fake_forward_penalty;
    }
    pair.pair_semantic_rank_adjust = clampValue(semantic_adjust, -0.40f, 0.40f);

    std::ostringstream reason;
    reason << std::fixed << std::setprecision(2)
           << "semantic=" << pair.semantic
           << ",cov=" << pair.pair_coverage_ratio
           << ",unexplained_t=" << pair.pair_unexplained_transverse_count
           << ",t_risk=" << pair.pair_unexplained_transverse_risk
           << ",fake_fwd_pen=" << pair.pair_fake_forward_penalty
           << ",fwd_support=" << pair.pair_forward_side_support_score
           << ",merge_band=" << pair.transverse_merge_band_score
           << ",open_branch=" << pair.transverse_opening_branch_score
           << ",t_role=" << pair.transverse_role
           << ",sem_adj=" << pair.pair_semantic_rank_adjust;
    pair.rank_reason = reason.str();
}

static void evaluatePhysicalBoundaryPairSemantics(
    PhysicalBoundaryPairCandidate& pair,
    const PhysicalBoundaryPrimitive& primitive_a,
    const PhysicalBoundaryPrimitive& primitive_b,
    float visible_min_forward) {
    const float overlap_m = pair.overlap_forward / kWorldPlanScale;
    const float near_width_m = pair.width_near / kWorldPlanScale;
    const float mid_width_m = pair.width_mid / kWorldPlanScale;
    const float far_width_m = pair.width_far / kWorldPlanScale;
    const float width_delta_m = far_width_m - near_width_m;
    pair.width_delta = pair.width_far - pair.width_near;
    pair.far_width_ratio =
        near_width_m > 1e-3f ? clampValue(far_width_m / near_width_m, 0.0f, 8.0f) : 0.0f;
    pair.width_slope_m_per_m =
        overlap_m > 1e-3f ? width_delta_m / overlap_m : 0.0f;
    pair.relative_lat_slope = pair.right_lat_slope - pair.left_lat_slope;

    const float min_span_m =
        std::min(pair.negative_chain.span, pair.positive_chain.span) / kWorldPlanScale;
    const float max_span_m =
        std::max(pair.negative_chain.span, pair.positive_chain.span) / kWorldPlanScale;
    const float span_ratio =
        min_span_m > 1e-3f ? max_span_m / min_span_m : std::numeric_limits<float>::infinity();
    const float width_m = pair.width_median / kWorldPlanScale;
    const bool not_sharing_stable = pair.shared_guide_ratio <= 0.20f;
    const bool not_sharing_opening = pair.shared_guide_ratio <= 0.25f;
    const bool has_transverse =
        pair.negative_type == PhysicalBoundaryPrimitiveType::Transverse ||
        pair.positive_type == PhysicalBoundaryPrimitiveType::Transverse;
    const bool has_diagonal =
        pair.negative_type == PhysicalBoundaryPrimitiveType::Diagonal ||
        pair.positive_type == PhysicalBoundaryPrimitiveType::Diagonal;
    pair.pair_type =
        physicalBoundaryPrimitiveTypeTag(pair.negative_type) + "+" +
        physicalBoundaryPrimitiveTypeTag(pair.positive_type);
    pair.closing_like = overlap_m >= 0.20f &&
        near_width_m >= 0.22f &&
        width_delta_m <= -0.10f &&
        pair.width_slope_m_per_m < -0.12f;
    pair.closing_suspect =
        overlap_m < 0.20f &&
        pair.width_slope_m_per_m < -0.50f &&
        pair.relative_lat_slope < -0.40f &&
        width_delta_m < 0.0f;

    const bool strong_width_drop =
        overlap_m >= 0.20f &&
        width_delta_m <= -0.15f &&
        pair.far_width_ratio <= 0.70f &&
        pair.width_slope_m_per_m < -0.20f;
    const bool narrow_far_closing =
        overlap_m >= 0.50f &&
        far_width_m <= 0.08f &&
        pair.far_width_ratio <= 0.35f &&
        pair.closing_like;
    const bool medium_closing_with_convergence =
        overlap_m >= 0.30f &&
        width_delta_m <= -0.12f &&
        pair.far_width_ratio <= 0.72f &&
        pair.relative_lat_slope <= -0.25f &&
        pair.closing_like;
    pair.strong_closing_veto =
        strong_width_drop ||
        narrow_far_closing ||
        medium_closing_with_convergence;

    pair.stable_split_like =
        !has_transverse &&
        span_ratio <= 2.0f &&
        overlap_m >= 0.45f &&
        min_span_m >= 0.45f &&
        not_sharing_stable &&
        !pair.closing_like;

    const bool forward_opening_like =
        !has_transverse &&
        !has_diagonal &&
        not_sharing_opening &&
        overlap_m >= 0.18f &&
        min_span_m >= 0.20f &&
        width_m >= 0.18f &&
        width_m <= 0.65f &&
        far_width_m >= near_width_m - 0.05f &&
        !pair.closing_like &&
        !pair.closing_suspect &&
        !pair.strong_closing_veto;
    const bool diagonal_opening_like =
        !has_transverse &&
        has_diagonal &&
        not_sharing_opening &&
        overlap_m >= 0.14f &&
        min_span_m >= 0.16f &&
        width_m >= 0.14f &&
        width_m <= 0.68f &&
        far_width_m >= near_width_m - 0.03f &&
        pair.relative_lat_slope > 0.0f &&
        !pair.closing_like &&
        !pair.closing_suspect &&
        !pair.strong_closing_veto;

    pair.closing_score = clampValue(
        0.35f * clampValue((near_width_m - far_width_m) / 0.20f, 0.0f, 1.0f) +
            0.30f * clampValue((-pair.width_slope_m_per_m) / 0.30f, 0.0f, 1.0f) +
            0.20f * clampValue((0.75f - pair.far_width_ratio) / 0.35f, 0.0f, 1.0f) +
            0.15f * clampValue(overlap_m / 0.45f, 0.0f, 1.0f),
        0.0f,
        1.0f);
    pair.opening_score = clampValue(
        0.35f * clampValue((width_delta_m + 0.05f) / 0.25f, 0.0f, 1.0f) +
            0.25f * clampValue(overlap_m / 0.45f, 0.0f, 1.0f) +
            0.20f * clampValue(min_span_m / 0.45f, 0.0f, 1.0f) +
            0.20f * (1.0f - clampValue(pair.shared_guide_ratio / 0.25f, 0.0f, 1.0f)),
        0.0f,
        1.0f);
    pair.stable_split_score = clampValue(
        0.30f * clampValue(overlap_m / 0.65f, 0.0f, 1.0f) +
            0.28f * clampValue(min_span_m / 0.65f, 0.0f, 1.0f) +
            0.22f * clampValue(1.0f - (span_ratio - 1.0f) / 1.5f, 0.0f, 1.0f) +
            0.20f * (1.0f - clampValue(pair.shared_guide_ratio / 0.20f, 0.0f, 1.0f)),
        0.0f,
        1.0f);

    pair.semantic = "noise";
    pair.semantic_reason = "unknown_geometry";
    pair.opening_reject_reason = "none";
    pair.opening_like = false;
    pair.transverse_merge_band_score = 0.0f;
    pair.transverse_opening_branch_score = 0.0f;
    pair.transverse_role = "none";
    pair.transverse_role_reason = "none";
    pair.ft_opening_candidate = false;
    pair.ft_opening_block_reason = "none";
    pair.transverse_t_ids.clear();
    pair.transverse_near_forward_m = 0.0f;
    pair.transverse_far_forward_m = 0.0f;
    pair.transverse_lat_min_m = 0.0f;
    pair.transverse_lat_max_m = 0.0f;
    pair.transverse_lat_span_m = 0.0f;
    pair.transverse_guide_count = 0;
    pair.transverse_near_ok = false;
    pair.transverse_near_cov = 0;
    pair.transverse_primitive_debug.clear();

    if (has_transverse) {
        const bool neg_is_t = (pair.negative_type == PhysicalBoundaryPrimitiveType::Transverse);
        const PhysicalBoundaryChain& t_chain =
            neg_is_t ? pair.negative_chain : pair.positive_chain;
        const PhysicalBoundaryPrimitive& t_primitive =
            neg_is_t ? primitive_a : primitive_b;
        const PhysicalBoundaryPrimitive& f_primitive =
            neg_is_t ? primitive_b : primitive_a;

        float t_lat_min = std::numeric_limits<float>::infinity();
        float t_lat_max = -std::numeric_limits<float>::infinity();
        float t_fwd_min = std::numeric_limits<float>::infinity();
        for (const auto& guide : t_chain.guides) {
            const float lat = guidePlanLateral(guide);
            const float fwd = guidePlanForward(guide);
            t_lat_min = std::min(t_lat_min, lat);
            t_lat_max = std::max(t_lat_max, lat);
            t_fwd_min = std::min(t_fwd_min, fwd);
        }
        const float t_lat_span_m =
            (std::isfinite(t_lat_min) && std::isfinite(t_lat_max))
                ? (t_lat_max - t_lat_min) / kWorldPlanScale
                : 0.0f;
        const float t_fwd_min_m =
            std::isfinite(t_fwd_min) ? t_fwd_min / kWorldPlanScale : 0.0f;
        const int t_guide_count = static_cast<int>(t_chain.guides.size());
        const bool t_near_ok =
            std::max(0.0f, t_chain.near_forward - visible_min_forward)
                <= planMeters(0.34f);
        int t_near_cov = 0;
        {
            const float near_limit = planMeters(0.34f);
            for (const auto& guide : t_chain.guides) {
                const float fwd = guidePlanForward(guide);
                if (isFiniteFloat(fwd) &&
                    std::max(0.0f, fwd - visible_min_forward) <= near_limit) {
                    ++t_near_cov;
                }
            }
        }

        const float forward_support_score =
            physicalBoundaryPrimitiveForwardSupportScore(f_primitive);
        const float fake_forward_penalty =
            (f_primitive.weak_short_forward ? 0.18f : 0.0f) +
            (f_primitive.transverse_derived ? 0.14f : 0.0f) +
            (t_primitive.weak_short_forward ? 0.18f : 0.0f) +
            (t_primitive.transverse_derived ? 0.14f : 0.0f);

        float merge_band = 0.0f;
        merge_band += 0.22f * clampValue((t_lat_span_m - 0.40f) / 1.00f, 0.0f, 1.0f);
        merge_band += 0.18f * clampValue((static_cast<float>(t_guide_count) - 3.0f) / 8.0f, 0.0f, 1.0f);
        merge_band += 0.15f * (t_near_ok ? 0.0f : 1.0f) *
            (t_near_cov == 0 ? 1.0f : 0.0f);
        merge_band += 0.15f * clampValue((t_fwd_min_m - 0.35f) / 0.80f, 0.0f, 1.0f);
        merge_band += 0.12f * pair.closing_score;
        merge_band += 0.10f * clampValue(fake_forward_penalty / 0.40f, 0.0f, 1.0f);
        merge_band += 0.08f * (1.0f - forward_support_score);
        merge_band = clampValue(merge_band, 0.0f, 1.0f);

        float opening_branch = 0.0f;
        opening_branch += 0.18f * (1.0f - clampValue((t_lat_span_m - 0.30f) / 0.80f, 0.0f, 1.0f));
        opening_branch += 0.14f * (1.0f - clampValue((static_cast<float>(t_guide_count) - 3.0f) / 7.0f, 0.0f, 1.0f));
        opening_branch += 0.16f * (t_near_ok ? 1.0f : 0.0f) *
            clampValue(static_cast<float>(t_near_cov) / 3.0f, 0.0f, 1.0f);
        opening_branch += 0.14f * pair.opening_score;
        opening_branch += 0.12f * (1.0f - clampValue(fake_forward_penalty / 0.40f, 0.0f, 1.0f));
        opening_branch += 0.10f * forward_support_score;
        opening_branch += 0.08f * clampValue((width_delta_m + 0.10f) / 0.40f, 0.0f, 1.0f);
        opening_branch += 0.08f * clampValue((pair.far_width_ratio - 0.50f) / 3.0f, 0.0f, 1.0f);
        opening_branch = clampValue(opening_branch, 0.0f, 1.0f);

        pair.transverse_merge_band_score = merge_band;
        pair.transverse_opening_branch_score = opening_branch;
        pair.transverse_near_forward_m = t_chain.near_forward / kWorldPlanScale;
        pair.transverse_far_forward_m = t_chain.far_forward / kWorldPlanScale;
        pair.transverse_lat_min_m = t_lat_min / kWorldPlanScale;
        pair.transverse_lat_max_m = t_lat_max / kWorldPlanScale;
        pair.transverse_lat_span_m = t_lat_span_m;
        pair.transverse_guide_count = t_guide_count;
        pair.transverse_near_ok = t_near_ok;
        pair.transverse_near_cov = t_near_cov;

        const bool strong_merge =
            merge_band >= 0.55f && opening_branch < 0.45f;
        const bool opening_support_override =
            !pair.strong_closing_veto &&
            t_near_ok &&
            t_near_cov >= 2 &&
            t_lat_span_m >= 0.30f &&
            t_lat_span_m <= 0.85f &&
            t_guide_count >= 3 &&
            t_guide_count <= 6 &&
            forward_support_score >= 0.75f &&
            fake_forward_penalty <= 0.25f &&
            opening_branch >= merge_band + 0.18f;
        const bool strong_opening =
            opening_branch >= 0.45f &&
            merge_band <= 0.50f &&
            (!pair.closing_suspect || opening_support_override) &&
            clampValue(fake_forward_penalty / 0.40f, 0.0f, 1.0f) < 0.75f;

        if (opening_support_override) {
            pair.closing_suspect = false;
            pair.closing_like = false;
        }

        if (pair.closing_suspect) {
            pair.transverse_role = "far_merge_band";
            pair.transverse_role_reason = "closing_suspect_hard_brake";
            pair.ft_opening_candidate = false;
            pair.ft_opening_block_reason = "closing_suspect";
        } else if (strong_merge) {
            pair.transverse_role = "far_merge_band";
            pair.transverse_role_reason = "merge_band_dominant";
            pair.ft_opening_candidate = false;
            pair.ft_opening_block_reason = "merge_band_dominant";
        } else if (strong_opening) {
            pair.transverse_role = "opening_branch";
            pair.transverse_role_reason = "opening_branch_dominant";
            pair.ft_opening_candidate = true;
            pair.ft_opening_block_reason = "none";
        } else {
            pair.transverse_role = "far_merge_band";
            pair.transverse_role_reason = "merge_band_fallback";
            pair.ft_opening_candidate = false;
            pair.ft_opening_block_reason = "merge_band_fallback";
        }

        if (pair.ft_opening_candidate) {
            pair.semantic = "opening";
            pair.semantic_reason = "transverse_opening_branch";
            pair.opening_like = true;
            pair.stable_split_like = false;
            pair.opening_reject_reason = "none";
        } else if (pair.transverse_role == "far_merge_band") {
            pair.semantic =
                pair.closing_like || pair.closing_suspect ? "merge" : "crossing";
            pair.semantic_reason = pair.closing_like
                ? "forward_transverse_closing_like"
                : "transverse_merge_like";
            pair.opening_reject_reason = "transverse_merge_band";
            pair.opening_like = false;
            pair.stable_split_like = false;
        } else {
            pair.semantic =
                pair.closing_like || pair.closing_suspect ? "merge" : "crossing";
            pair.semantic_reason = pair.closing_like
                ? "forward_transverse_closing_like"
                : "transverse_merge_like";
            pair.opening_reject_reason = "transverse_merge_band";
            pair.opening_like = false;
            pair.stable_split_like = false;
        }
    } else {
        pair.opening_like = forward_opening_like || diagonal_opening_like;

        if (pair.closing_like || pair.strong_closing_veto) {
            pair.semantic = "closing";
            pair.semantic_reason = pair.strong_closing_veto
                ? "strong_closing_veto"
                : "closing_like";
            pair.opening_reject_reason = "closing_geometry";
        } else if (pair.opening_like) {
            pair.semantic = "opening";
            pair.semantic_reason = has_diagonal
                ? "forward_diagonal_opening_like"
                : "forward_forward_opening_like";
        } else if (has_diagonal) {
            pair.semantic = "merge";
            pair.semantic_reason = "forward_diagonal_not_outward_enough";
            pair.opening_reject_reason = "diagonal_direction_not_opening";
        } else if (pair.stable_split_like) {
            pair.semantic = "opening";
            pair.semantic_reason = "stable_split_like";
        }
    }

    (void)mid_width_m;
}

static constexpr bool kBoundaryPlanLateralPositiveIsPhysicalLeft = false;

static float canonicalBoundaryPhysicalLateral(float plan_lateral) {
    return kBoundaryPlanLateralPositiveIsPhysicalLeft ? plan_lateral : -plan_lateral;
}

static BoundaryPairCandidate adaptPhysicalBoundaryPairToBoundaryCandidate(
    const PhysicalBoundaryPairCandidate& pair,
    float visible_min_forward,
    int provider_rank) {
    BoundaryPairCandidate candidate;
    {
        std::ostringstream source;
        source << "physical_provider_v2"
               << ":pair_type=" << pair.pair_type
               << ":pair_semantic=" << pair.semantic
               << ":pair_semantic_reason=" << pair.semantic_reason
               << ":opening_reject_reason=" << pair.opening_reject_reason
               << ":final_pair_source=" << pair.final_pair_source;
        candidate.source = source.str();
    }
    candidate.provider_type = "physical_provider_v2";
    candidate.provider_rank = provider_rank;
    candidate.band_idx = -100 - provider_rank;
    candidate.geometry_source = "semantic_only";

    const bool negative_is_slot_left =
        canonicalBoundaryPhysicalLateral(pair.negative_chain.root_lateral) >=
        canonicalBoundaryPhysicalLateral(pair.positive_chain.root_lateral);
    const PhysicalBoundaryChain& slot_left =
        negative_is_slot_left ? pair.negative_chain : pair.positive_chain;
    const PhysicalBoundaryChain& slot_right =
        negative_is_slot_left ? pair.positive_chain : pair.negative_chain;

    candidate.left_branch = slot_left.guides;
    candidate.right_branch = slot_right.guides;
    candidate.left_shared_prefix_count = 0;
    candidate.right_shared_prefix_count = 0;
    candidate.unique_guide_count =
        static_cast<int>(slot_left.guides.size() + slot_right.guides.size());
    candidate.left_point_count = static_cast<int>(slot_left.guides.size());
    candidate.right_point_count = static_cast<int>(slot_right.guides.size());
    candidate.trunk_avg_step = 0.0f;
    candidate.left_avg_step = averageGuideStepCost(slot_left.guides);
    candidate.right_avg_step = averageGuideStepCost(slot_right.guides);
    candidate.trunk_max_dx = 0.0f;
    candidate.left_max_dx = maxGuideDxJump(slot_left.guides);
    candidate.right_max_dx = maxGuideDxJump(slot_right.guides);
    candidate.center_x = 0.5f * (slot_left.root_lateral + slot_right.root_lateral);
    candidate.spread = pair.width_median;
    candidate.threshold = pair.width_near;
    candidate.excess = std::max(0.0f, pair.width_median - pair.width_near);
    candidate.left_y_span = guideForwardSpan(slot_left.guides);
    candidate.right_y_span = guideForwardSpan(slot_right.guides);
    candidate.left_span_m = candidate.left_y_span / kWorldPlanScale;
    candidate.right_span_m = candidate.right_y_span / kWorldPlanScale;
    candidate.weaker_y_span = std::min(candidate.left_y_span, candidate.right_y_span);
    candidate.stronger_y_span = std::max(candidate.left_y_span, candidate.right_y_span);

    const auto computeBottomReach = [&](const PhysicalBoundaryChain& chain) {
        const float bottom_gap = std::max(0.0f, chain.near_forward - visible_min_forward);
        return clampValue(
            1.0f - bottom_gap / planMeters(0.35f),
            0.0f,
            1.0f);
    };
    candidate.left_bottom_reach_score = computeBottomReach(slot_left);
    candidate.right_bottom_reach_score = computeBottomReach(slot_right);
    candidate.weaker_bottom_reach_score =
        std::min(candidate.left_bottom_reach_score, candidate.right_bottom_reach_score);
    candidate.stronger_bottom_reach_score =
        std::max(candidate.left_bottom_reach_score, candidate.right_bottom_reach_score);

    const auto bottomGapOk = [&](const PhysicalBoundaryChain& chain, float limit_m) {
        const float bottom_gap = std::max(0.0f, chain.near_forward - visible_min_forward);
        return bottom_gap <= planMeters(limit_m);
    };
    candidate.left_root_ok = bottomGapOk(slot_left, 0.24f);
    candidate.right_root_ok = bottomGapOk(slot_right, 0.24f);
    candidate.left_near_root_ok = bottomGapOk(slot_left, 0.34f);
    candidate.right_near_root_ok = bottomGapOk(slot_right, 0.34f);
    candidate.span_balance_ratio =
        candidate.stronger_y_span > 1e-3f
            ? clampValue(candidate.weaker_y_span / candidate.stronger_y_span, 0.0f, 1.0f)
            : 0.0f;
    candidate.provider_overlap_ratio = clampValue(
        pair.overlap_forward / std::max(candidate.stronger_y_span, planMeters(0.08f)),
        0.0f,
        1.0f);
    candidate.weak_far_field_penalty = pair.far_divergence_penalty;
    candidate.common_y_overlap = pair.overlap_forward;
    candidate.provider_overlap_width_m = pair.width_median / kWorldPlanScale;
    candidate.provider_near_width_m = pair.width_near / kWorldPlanScale;
    candidate.provider_overlap_m = pair.overlap_forward / kWorldPlanScale;
    candidate.provider_pair_confidence = pair.pair_confidence;
    candidate.shared_ratio = pair.shared_guide_ratio;
    candidate.far_div_pen = pair.far_divergence_penalty;
    candidate.provider_mid_width_m = pair.width_mid / kWorldPlanScale;
    candidate.provider_far_width_m = pair.width_far / kWorldPlanScale;
    candidate.width_delta_m = candidate.provider_far_width_m - candidate.provider_near_width_m;
    candidate.far_width_ratio = pair.far_width_ratio;
    candidate.width_slope_m_per_m = pair.width_slope_m_per_m;
    candidate.left_lat_slope =
        negative_is_slot_left ? pair.left_lat_slope : pair.right_lat_slope;
    candidate.right_lat_slope =
        negative_is_slot_left ? pair.right_lat_slope : pair.left_lat_slope;
    candidate.relative_lat_slope = candidate.right_lat_slope - candidate.left_lat_slope;
    candidate.closing_like = pair.closing_like;
    candidate.closing_suspect = pair.closing_suspect;
    candidate.strong_closing_veto = pair.strong_closing_veto;
    candidate.left_point_count = static_cast<int>(slot_left.guides.size());
    candidate.right_point_count = static_cast<int>(slot_right.guides.size());

    const bool bilateral_near_ok =
        candidate.left_near_root_ok && candidate.right_near_root_ok;
    candidate.stable_split_like =
        pair.semantic != "crossing" &&
        pair.semantic != "merge" &&
        bilateral_near_ok &&
        candidate.left_root_ok &&
        candidate.right_root_ok &&
        candidate.span_balance_ratio >= 0.50f &&
        candidate.provider_overlap_m >= 0.45f &&
        std::min(candidate.left_span_m, candidate.right_span_m) >= 0.45f &&
        candidate.shared_ratio <= 0.20f &&
        !candidate.closing_like;
    candidate.opening_like =
        pair.semantic == "opening" &&
        candidate.shared_ratio <= 0.25f &&
        candidate.provider_overlap_m >= 0.18f &&
        std::min(candidate.left_span_m, candidate.right_span_m) >= 0.20f &&
        candidate.provider_overlap_width_m >= 0.18f &&
        candidate.provider_overlap_width_m <= 0.65f &&
        (pair.ft_opening_candidate ||
         candidate.provider_far_width_m >= candidate.provider_near_width_m - 0.05f) &&
        !candidate.closing_like &&
        !candidate.closing_suspect &&
        !candidate.strong_closing_veto;
    candidate.stable_split_score = pair.stable_split_score;
    candidate.opening_score = pair.opening_score;
    candidate.closing_score = pair.closing_score;
    candidate.pair_unique_guide_count = pair.pair_unique_guide_count;
    candidate.pair_coverage_ratio = pair.pair_coverage_ratio;
    candidate.pair_explained_transverse_count = pair.pair_explained_transverse_count;
    candidate.pair_unexplained_transverse_count = pair.pair_unexplained_transverse_count;
    candidate.pair_unexplained_transverse_risk = pair.pair_unexplained_transverse_risk;
    candidate.pair_semantic_rank_adjust = pair.pair_semantic_rank_adjust;
    candidate.pair_fake_forward_penalty = pair.pair_fake_forward_penalty;
    candidate.pair_forward_side_support_score = pair.pair_forward_side_support_score;
    candidate.pair_has_transverse = pair.pair_has_transverse;
    candidate.transverse_merge_band_score = pair.transverse_merge_band_score;
    candidate.transverse_opening_branch_score = pair.transverse_opening_branch_score;
    candidate.transverse_role = pair.transverse_role;
    candidate.ft_opening_candidate = pair.ft_opening_candidate;
    candidate.transverse_t_ids = pair.transverse_t_ids;
    candidate.transverse_near_forward_m = pair.transverse_near_forward_m;
    candidate.transverse_far_forward_m = pair.transverse_far_forward_m;
    candidate.transverse_lat_min_m = pair.transverse_lat_min_m;
    candidate.transverse_lat_max_m = pair.transverse_lat_max_m;
    candidate.transverse_lat_span_m = pair.transverse_lat_span_m;
    candidate.transverse_guide_count = pair.transverse_guide_count;
    candidate.transverse_near_ok = pair.transverse_near_ok;
    candidate.transverse_near_cov = pair.transverse_near_cov;
    candidate.transverse_primitive_debug = pair.transverse_primitive_debug;
    candidate.pair_selected_rank_reason = pair.rank_reason;
    return candidate;
}

// [ACTIVE_CHAIN] physical_provider_v2: semantic observation provider. Its
// candidates update BoundaryTrackManager and frame-level semantic arbitration.
static std::vector<BoundaryPairCandidate> buildPhysicalBoundaryProviderCandidates(
    const std::vector<ArrowGuide>& valid_guides,
    const std::vector<ArrowGuide>& center_reference_guides,
    float center_ref_lateral,
    float visible_min_forward,
    float median_hw,
    std::string* provider_summary) {
    if (provider_summary != nullptr) {
        *provider_summary = "";
    }
    if (valid_guides.size() < 4) {
        if (provider_summary != nullptr) {
            *provider_summary = "physical_boundary_provider_v2 chains=0 pairs=0 reject_reason=insufficient_guides";
        }
        return {};
    }

    (void)center_reference_guides;
    (void)center_ref_lateral;

    float min_forward_ref = std::numeric_limits<float>::infinity();
    std::vector<PhysicalBoundaryGuidePoint> points;
    points.reserve(valid_guides.size());
    for (const auto& guide : valid_guides) {
        const cv::Point2f plan = guidePlanPoint(guide);
        if (!isFinitePoint(plan)) {
            continue;
        }
        PhysicalBoundaryGuidePoint point;
        point.guide = guide;
        point.forward = plan.y;
        point.lateral = plan.x;
        points.push_back(point);
        min_forward_ref = std::min(min_forward_ref, point.forward);
    }
    if (points.size() < 4) {
        if (provider_summary != nullptr) {
            *provider_summary = "physical_boundary_provider_v2 chains=0 pairs=0 reject_reason=insufficient_valid_points";
        }
        return {};
    }
    if (!std::isfinite(min_forward_ref)) {
        min_forward_ref = 0.0f;
    }
    for (auto& point : points) {
        const float rel_forward = point.forward - min_forward_ref;
        point.bucket_idx = static_cast<int>(std::floor(
            rel_forward / std::max(1.0f, kPhysicalBoundaryProviderBucketWidth)));
    }

    const std::vector<PhysicalBoundaryPrimitive> primitives =
        buildPhysicalBoundaryPrimitives(points, median_hw, min_forward_ref, visible_min_forward);
    const int total_unique_guides = countUniqueGuideIds(valid_guides);
    int transverse_primitive_count = 0;
    int diagonal_primitive_count = 0;
    for (const auto& primitive : primitives) {
        if (primitive.type == PhysicalBoundaryPrimitiveType::Transverse) {
            ++transverse_primitive_count;
        } else if (primitive.type == PhysicalBoundaryPrimitiveType::Diagonal) {
            ++diagonal_primitive_count;
        }
    }

    std::vector<PhysicalBoundaryPairCandidate> pair_candidates;
    for (size_t i = 0; i < primitives.size(); ++i) {
        for (size_t j = i + 1; j < primitives.size(); ++j) {
            const PhysicalBoundaryPrimitive& primitive_a = primitives[i];
            const PhysicalBoundaryPrimitive& primitive_b = primitives[j];
            if (primitive_a.type == PhysicalBoundaryPrimitiveType::Transverse &&
                primitive_b.type == PhysicalBoundaryPrimitiveType::Transverse) {
                continue;
            }
            const PhysicalBoundaryChain& chain_a = primitive_a.chain;
            const PhysicalBoundaryChain& chain_b = primitive_b.chain;
            const float overlap_lo =
                std::max(chain_a.near_forward, chain_b.near_forward);
            const float overlap_hi =
                std::min(chain_a.far_forward, chain_b.far_forward);
            const float overlap_forward = std::max(0.0f, overlap_hi - overlap_lo);
            const bool has_transverse =
                primitive_a.type == PhysicalBoundaryPrimitiveType::Transverse ||
                primitive_b.type == PhysicalBoundaryPrimitiveType::Transverse;
            const float min_overlap =
                has_transverse ? planMeters(0.03f) : kPhysicalBoundaryProviderMinOverlap;
            if (overlap_forward < min_overlap) {
                continue;
            }

            const float near_mid_hi = overlap_lo + 0.60f * overlap_forward;
            const float lat_a =
                averageChainLateralInForwardRange(chain_a, overlap_lo, near_mid_hi);
            const float lat_b =
                averageChainLateralInForwardRange(chain_b, overlap_lo, near_mid_hi);
            if (std::abs(lat_a - lat_b) < kPhysicalBoundaryProviderMinLateralGap) {
                continue;
            }
            const PhysicalBoundaryChain& lower_chain = lat_a <= lat_b ? chain_a : chain_b;
            const PhysicalBoundaryChain& higher_chain = lat_a <= lat_b ? chain_b : chain_a;
            const PhysicalBoundaryPrimitiveType lower_type =
                lat_a <= lat_b ? primitive_a.type : primitive_b.type;
            const PhysicalBoundaryPrimitiveType higher_type =
                lat_a <= lat_b ? primitive_b.type : primitive_a.type;

            PhysicalBoundaryPairCandidate pair;
            pair.negative_chain = lower_chain;
            pair.positive_chain = higher_chain;
            pair.negative_type = lower_type;
            pair.positive_type = higher_type;
            pair.min_overlap_used = min_overlap;
            float far_divergence = 0.0f;
            PhysicalPairWidthStatsDebug width_debug;
            if (!estimatePhysicalPairWidthStats(lower_chain,
                                                higher_chain,
                                                min_overlap,
                                                &pair.overlap_forward,
                                                &pair.width_median,
                                                &pair.width_near,
                                                &pair.width_mid,
                                                &pair.width_far,
                                                &pair.left_lat_slope,
                                                &pair.right_lat_slope,
                                                &pair.width_stability,
                                                &far_divergence,
                                                &width_debug)) {
                pair.reject_reason = "pair_width_unavailable";
                continue;
            }
            pair.width_sampling_mode = width_debug.width_sampling_mode;
            pair.width_sample_count = width_debug.sample_count;
            pair.valid_width_count = width_debug.valid_width_count;
            pair.sampled_forward_values = width_debug.sampled_forward_values;
            pair.sampled_width_values = width_debug.sampled_width_values;
            pair.width_stats_reject_reason = width_debug.reject_reason;
            pair.far_divergence_penalty = far_divergence;
            pair.heading_consistency = std::abs(normalizeAngleDeltaDeg(
                chain_a.heading_deg - chain_b.heading_deg));
            if (pair.overlap_forward < min_overlap) {
                pair.reject_reason = "pair_overlap_too_small";
                continue;
            }
            if (pair.heading_consistency > (has_transverse ? 110.0f : 95.0f)) {
                pair.reject_reason = "pair_heading_conflict";
                continue;
            }
            pair.shared_guide_ratio =
                computePhysicalChainSharedGuideRatio(lower_chain, higher_chain);
            if (!has_transverse && pair.shared_guide_ratio > 0.45f) {
                pair.reject_reason = "pair_shared_guides";
                continue;
            }
            evaluatePhysicalBoundaryPairSemantics(
                pair, primitive_a, primitive_b, visible_min_forward);
            enrichPhysicalBoundaryPairExplanation(
                pair, primitive_a, primitive_b, primitives, total_unique_guides);
            pair.pair_confidence = scorePhysicalBoundaryPairCandidate(pair);
            pair.pair_confidence = clampValue(
                pair.pair_confidence +
                    0.10f * primitive_a.quality_score +
                    0.10f * primitive_b.quality_score +
                    0.14f * pair.pair_coverage_ratio +
                    0.10f * pair.pair_forward_side_support_score +
                    pair.pair_semantic_rank_adjust -
                    0.12f * pair.pair_unexplained_transverse_risk -
                    0.10f * pair.pair_fake_forward_penalty,
                0.0f,
                1.0f);
            if (pair.far_divergence_penalty > 0.0f) {
                pair.pair_confidence = clampValue(
                    pair.pair_confidence - pair.far_divergence_penalty, 0.0f, 1.0f);
            }
            pair.reject_reason = "none";
            pair_candidates.push_back(std::move(pair));
        }
    }

    std::sort(pair_candidates.begin(), pair_candidates.end(),
              [](const PhysicalBoundaryPairCandidate& a,
                 const PhysicalBoundaryPairCandidate& b) {
        if (std::abs(a.pair_confidence - b.pair_confidence) > 1e-4f) {
            return a.pair_confidence > b.pair_confidence;
        }
        if (std::abs(a.pair_semantic_rank_adjust - b.pair_semantic_rank_adjust) > 1e-4f) {
            return a.pair_semantic_rank_adjust > b.pair_semantic_rank_adjust;
        }
        if (std::abs(a.pair_coverage_ratio - b.pair_coverage_ratio) > 1e-4f) {
            return a.pair_coverage_ratio > b.pair_coverage_ratio;
        }
        if (std::abs(a.overlap_forward - b.overlap_forward) > 1e-3f) {
            return a.overlap_forward > b.overlap_forward;
        }
        return a.width_stability < b.width_stability;
    });

    std::vector<BoundaryPairCandidate> candidates;
    const size_t pair_limit = std::min(kPhysicalBoundaryProviderMaxPairs, pair_candidates.size());
    candidates.reserve(pair_limit);
    for (size_t i = 0; i < pair_limit; ++i) {
        candidates.push_back(adaptPhysicalBoundaryPairToBoundaryCandidate(
            pair_candidates[i], visible_min_forward, static_cast<int>(i)));
    }

    if (provider_summary != nullptr) {
        std::ostringstream summary;
        summary << std::fixed << std::setprecision(2)
                << "physical_boundary_provider_v2"
                << " points=" << points.size()
                << " primitive_count=" << primitives.size()
                << " transverse_primitive_count=" << transverse_primitive_count
                << " diagonal_primitive_count=" << diagonal_primitive_count
                << " pairs=" << candidates.size()
                << " mode=primitive_pair_generation";
        if (!primitives.empty()) {
            summary << " primitive_summary=[";
            const size_t primitive_limit = std::min<size_t>(6, primitives.size());
            for (size_t i = 0; i < primitive_limit; ++i) {
                if (i > 0) {
                    summary << ";";
                }
                summary << formatPhysicalBoundaryPrimitiveForDebug(primitives[i], i);
            }
            summary << "]";
        }
        if (!pair_candidates.empty()) {
            const auto& best_pair = pair_candidates.front();
            summary << " best_width_m=" << (best_pair.width_median / kWorldPlanScale)
                    << " best_near_width_m=" << (best_pair.width_near / kWorldPlanScale)
                    << " best_mid_width_m=" << (best_pair.width_mid / kWorldPlanScale)
                    << " best_far_width_m=" << (best_pair.width_far / kWorldPlanScale)
                    << " width_delta_m=" << (best_pair.width_delta / kWorldPlanScale)
                    << " far_width_ratio=" << best_pair.far_width_ratio
                    << " width_slope=" << best_pair.width_slope_m_per_m
                    << " left_lat_slope=" << best_pair.left_lat_slope
                    << " right_lat_slope=" << best_pair.right_lat_slope
                    << " relative_lat_slope=" << best_pair.relative_lat_slope
                    << " best_overlap_m=" << (best_pair.overlap_forward / kWorldPlanScale)
                    << " best_span_m="
                    << (best_pair.negative_chain.span / kWorldPlanScale) << "/"
                    << (best_pair.positive_chain.span / kWorldPlanScale)
                    << " pair_type=" << best_pair.pair_type
                    << " pair_left_type=" << physicalBoundaryPrimitiveTypeName(best_pair.negative_type)
                    << " pair_right_type=" << physicalBoundaryPrimitiveTypeName(best_pair.positive_type)
                    << " pair_semantic=" << best_pair.semantic
                    << " pair_semantic_reason=" << best_pair.semantic_reason
                    << " opening_reject_reason=" << best_pair.opening_reject_reason
                    << " final_pair_source=" << best_pair.final_pair_source
                    << " best_conf=" << best_pair.pair_confidence
                    << " far_div_pen=" << best_pair.far_divergence_penalty
                    << " shared_ratio=" << best_pair.shared_guide_ratio
                    << " pair_unique_guide_count=" << best_pair.pair_unique_guide_count
                    << " pair_coverage_ratio=" << best_pair.pair_coverage_ratio
                    << " pair_explained_transverse_count=" << best_pair.pair_explained_transverse_count
                    << " pair_unexplained_transverse_count=" << best_pair.pair_unexplained_transverse_count
                    << " pair_unexplained_transverse_risk=" << best_pair.pair_unexplained_transverse_risk
                    << " pair_semantic_rank_adjust=" << best_pair.pair_semantic_rank_adjust
                    << " pair_fake_forward_penalty=" << best_pair.pair_fake_forward_penalty
                    << " pair_forward_side_support_score=" << best_pair.pair_forward_side_support_score
                    << " merge_band_score=" << best_pair.transverse_merge_band_score
                    << " open_branch_score=" << best_pair.transverse_opening_branch_score
                    << " transverse_role=" << best_pair.transverse_role
                    << " ft_opening_candidate=" << (best_pair.ft_opening_candidate ? 1 : 0)
                    << " " << best_pair.transverse_primitive_debug
                    << " pair_selected_rank_reason=" << best_pair.rank_reason
                    << " stable_split_score=" << best_pair.stable_split_score
                    << " opening_score=" << best_pair.opening_score
                    << " closing_score=" << best_pair.closing_score
                    << " stable_split_like=" << (best_pair.stable_split_like ? 1 : 0)
                    << " opening_like=" << (best_pair.opening_like ? 1 : 0)
                    << " closing_like=" << (best_pair.closing_like ? 1 : 0)
                    << " closing_suspect=" << (best_pair.closing_suspect ? 1 : 0)
                    << " strong_closing_veto=" << (best_pair.strong_closing_veto ? 1 : 0)
                    << " reject_reason=" << best_pair.reject_reason;
        } else {
            summary << " reject_reason=no_pair_candidate";
        }
        *provider_summary = summary.str();
    }
    return candidates;
}

struct ForwardGrownVisiblePairCandidate {
    bool valid = false;
    std::vector<ArrowGuide> left_guides;
    std::vector<ArrowGuide> right_guides;
    float left_span = 0.0f;
    float right_span = 0.0f;
    float width_m = 0.0f;
    float overlap_forward = 0.0f;
    float confidence = 0.0f;
    std::string source = "none";
    std::string left_type = "none";
    std::string right_type = "none";
    std::string selected_seed_ids;
    std::string selected_chain_ids;
    int shared_prefix_count = 0;
    int weak_independent_suffix_count = 0;
    float pair_score = 0.0f;
    std::string reject_reason = "none";
    std::string accept_reason = "none";
    std::string relative_assignment = "none";
    int right_fork_search_candidate_count = 0;
    std::string main_forward_source = "none";
    std::string main_forward_ids;
    bool right_fork_branch_search_enabled = false;
    std::string right_fork_anchor_type = "none";
    std::string right_fork_anchor_on_main = "none";
    std::string last_confirmed_right_fork_ids = "none";
    std::string right_fork_candidate_ids;
    std::string current_right_fork_candidate_ids = "none";
    bool suffix_continuation_found = false;
    std::string suffix_continuation_reason = "none";
    bool right_fork_created_by_alt = false;
    bool right_fork_reconnect_main_tail = false;
    bool no_anchor_on_main_seen = false;
    int right_guide_count_when_no_anchor = 0;
    bool candidate_reuses_main_forward_seen = false;
    int right_guide_count_when_reuse_main_forward = 0;
    int right_fork_candidate_reuses_main_forward_reject_count = 0;
    std::string right_fork_candidate_reuses_main_forward_reject_ids = "none";
    std::string right_fork_raw_candidate_details = "none";
    int independent_right_id_count = 0;
    bool independent_right_fork_evidence = false;
    float right_main_overlap = 0.0f;
    float lateral_separation = 0.0f;
    float main_reuse_ratio_max = 0.0f;
    std::string selected_right_fork_model = "none";
    bool detached_candidate_found = false;
    bool detached_candidate_valid = false;
    std::string detached_reject_reason = "none";
    std::string detached_candidate_ids = "none";
    std::string detached_gate_trace = "none";
    float detached_score = 0.0f;
    int detached_independent_id_count = 0;
    float detached_main_overlap = 0.0f;
    float detached_main_reuse_ratio = 0.0f;
    float detached_common_forward_overlap = 0.0f;
    int detached_right_chain_count = 0;
    float detached_right_chain_forward_span = 0.0f;
    float detached_right_chain_lateral_span = 0.0f;
    float detached_right_chain_continuity = 0.0f;
    bool detached_structural_gate_pass = false;
    bool detached_right_side_order_ok = false;
    bool detached_crossing_detected = false;
    bool detached_remerge_detected = false;
    float detached_width_min = 0.0f;
    float detached_width_max = 0.0f;
    float detached_width_mean = 0.0f;
    float detached_width_variation = 0.0f;
    int detached_sample_count = 0;
    int detached_valid_width_count = 0;
    int detached_trend_sample_count = 0;
    float detached_trend_width_near = 0.0f;
    float detached_trend_width_far = 0.0f;
    float detached_trend_width_delta = 0.0f;
    float detached_trend_width_slope = 0.0f;
    float detached_trend_far_width_ratio = 0.0f;
    float detached_trend_tail_min_width = 0.0f;
    std::string final_pair_source = "none";
    std::string semantic_passthrough = "none";
};

static bool guideIdsAreSubsequence(const std::vector<ArrowGuide>& maybe_subsequence,
                                   const std::vector<ArrowGuide>& reference) {
    if (maybe_subsequence.empty()) {
        return true;
    }
    size_t ref_idx = 0;
    for (const auto& guide : maybe_subsequence) {
        if (guide.tracker_id < 0) {
            return false;
        }
        bool found = false;
        while (ref_idx < reference.size()) {
            if (reference[ref_idx].tracker_id == guide.tracker_id) {
                found = true;
                ++ref_idx;
                break;
            }
            ++ref_idx;
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

static std::vector<int> collectGuideTrackerIds(const std::vector<ArrowGuide>& guides) {
    std::vector<int> ids;
    ids.reserve(guides.size());
    for (const auto& guide : guides) {
        if (guide.tracker_id >= 0) {
            ids.push_back(guide.tracker_id);
        }
    }
    return ids;
}

static bool guideIdsAreConfirmedRightForkSuffix(const std::vector<int>& candidate_ids,
                                                const std::vector<int>& confirmed_ids,
                                                std::string* out_reason = nullptr) {
    if (out_reason != nullptr) {
        *out_reason = "none";
    }
    if (confirmed_ids.empty()) {
        if (out_reason != nullptr) {
            *out_reason = "no_last_confirmed_right_fork";
        }
        return false;
    }
    if (candidate_ids.size() < kRightForkConfirmedSuffixMinGuideCount) {
        if (out_reason != nullptr) {
            *out_reason = "candidate_suffix_too_short";
        }
        return false;
    }
    if (candidate_ids.size() >= confirmed_ids.size()) {
        if (out_reason != nullptr) {
            *out_reason = "candidate_not_strict_suffix";
        }
        return false;
    }
    const size_t offset = confirmed_ids.size() - candidate_ids.size();
    for (size_t i = 0; i < candidate_ids.size(); ++i) {
        if (candidate_ids[i] != confirmed_ids[offset + i]) {
            if (out_reason != nullptr) {
                *out_reason = "candidate_ids_not_contiguous_suffix";
            }
            return false;
        }
    }
    if (out_reason != nullptr) {
        *out_reason = "matched_contiguous_confirmed_suffix";
    }
    return true;
}

static std::unordered_set<int> collectSharedVisiblePairGuideIds(
    const std::vector<ArrowGuide>& a,
    const std::vector<ArrowGuide>& b) {
    std::unordered_set<int> a_ids;
    for (const auto& guide : a) {
        if (guide.tracker_id >= 0) {
            a_ids.insert(guide.tracker_id);
        }
    }

    std::unordered_set<int> shared_ids;
    for (const auto& guide : b) {
        if (guide.tracker_id >= 0 && a_ids.count(guide.tracker_id) > 0) {
            shared_ids.insert(guide.tracker_id);
        }
    }
    return shared_ids;
}

static bool computeVisiblePairSeparatedRangeAssignment(
    const PhysicalBoundaryChain& a,
    const PhysicalBoundaryChain& b,
    bool* out_a_is_physical_left,
    float* out_lateral_gap) {
    const std::unordered_set<int> shared_ids =
        collectSharedVisiblePairGuideIds(a.guides, b.guides);

    const float overlap_lo = std::max(a.near_forward, b.near_forward);
    const float overlap_hi = std::min(a.far_forward, b.far_forward);
    if (overlap_hi <= overlap_lo) {
        return false;
    }

    float shared_max_forward = -std::numeric_limits<float>::infinity();
    if (!shared_ids.empty()) {
        auto updateSharedMaxForward = [&](const std::vector<ArrowGuide>& guides) {
            for (const auto& guide : guides) {
                if (guide.tracker_id >= 0 && shared_ids.count(guide.tracker_id) > 0) {
                    const float forward = guidePlanForward(guide);
                    if (isFiniteFloat(forward)) {
                        shared_max_forward = std::max(shared_max_forward, forward);
                    }
                }
            }
        };
        updateSharedMaxForward(a.guides);
        updateSharedMaxForward(b.guides);
    }

    const float assignment_lo =
        std::isfinite(shared_max_forward)
            ? std::max(overlap_lo, shared_max_forward + planMeters(0.06f))
            : overlap_lo;
    if (overlap_hi <= assignment_lo) {
        return false;
    }

    auto lateralRangeInAssignmentWindow = [&](const PhysicalBoundaryChain& chain,
                                              float* out_min,
                                              float* out_max) {
        if (out_min != nullptr) {
            *out_min = 0.0f;
        }
        if (out_max != nullptr) {
            *out_max = 0.0f;
        }
        float lat_min = std::numeric_limits<float>::infinity();
        float lat_max = -std::numeric_limits<float>::infinity();
        int count = 0;
        for (const auto& guide : chain.guides) {
            const float forward = guidePlanForward(guide);
            const float lateral = guidePlanLateral(guide);
            if (!isFiniteFloat(forward) || !isFiniteFloat(lateral)) {
                continue;
            }
            if (forward < assignment_lo || forward > overlap_hi) {
                continue;
            }
            lat_min = std::min(lat_min, lateral);
            lat_max = std::max(lat_max, lateral);
            ++count;
        }
        if (count <= 0 || !std::isfinite(lat_min) || !std::isfinite(lat_max)) {
            return false;
        }
        if (out_min != nullptr) {
            *out_min = lat_min;
        }
        if (out_max != nullptr) {
            *out_max = lat_max;
        }
        return true;
    };

    float a_min = 0.0f;
    float a_max = 0.0f;
    float b_min = 0.0f;
    float b_max = 0.0f;
    if (!lateralRangeInAssignmentWindow(a, &a_min, &a_max) ||
        !lateralRangeInAssignmentWindow(b, &b_min, &b_max)) {
        return false;
    }

    const float separation = planMeters(0.05f);
    if (a_max < b_min - separation) {
        if (out_a_is_physical_left != nullptr) {
            *out_a_is_physical_left = true;
        }
        if (out_lateral_gap != nullptr) {
            *out_lateral_gap = b_min - a_max;
        }
        return true;
    }
    if (b_max < a_min - separation) {
        if (out_a_is_physical_left != nullptr) {
            *out_a_is_physical_left = false;
        }
        if (out_lateral_gap != nullptr) {
            *out_lateral_gap = a_min - b_max;
        }
        return true;
    }
    return false;
}

struct RightForkBranchSearchResult {
    bool found = false;
    PhysicalBoundaryChain chain;
    std::string anchor_type = "none";
    std::string anchor_on_main = "none";
    std::string last_confirmed_right_fork_ids = "none";
    std::string candidate_ids = "none";
    std::string current_candidate_ids = "none";
    bool suffix_continuation_found = false;
    std::string suffix_continuation_reason = "none";
    std::string anchor_candidate_ci_range = "none";
    std::string anchor_match_ci = "none";
    std::string anchor_match_mi = "none";
    std::string no_anchor_candidate_prefix_ids = "none";
    std::string anchor_main_ids = "none";
    bool created_by_alt = false;
    bool reconnect_main_tail = false;
    bool no_anchor_on_main_seen = false;
    int right_guide_count_when_no_anchor = 0;
    bool candidate_reuses_main_forward_seen = false;
    int right_guide_count_when_reuse_main_forward = 0;
    int shared_prefix_count = 0;
    int weak_independent_suffix_count = 0;
    int independent_right_id_count = 0;
    float right_main_overlap = 0.0f;
    float lateral_separation = 0.0f;
    float main_reuse_ratio_max = 0.0f;
    float score = 0.0f;
    float post_split_lateral_trend = 0.0f;
    std::string selected_right_fork_model = "none";
    std::string reject_reason = "no_candidate";
    std::string raw_candidate_details = "none";
    int candidate_reuses_main_forward_reject_count = 0;
    std::string candidate_reuses_main_forward_reject_ids = "none";
    int raw_candidate_count = 0;
    PhysicalBoundaryChain no_anchor_independent_chain;
    std::string no_anchor_independent_candidate_ids = "none";
    float no_anchor_candidate_main_reuse_ratio = 0.0f;
    bool active_no_anchor_independent_candidate_accept = false;
    std::string active_no_anchor_accept_block_reason = "not_evaluated";
    bool independent_right_fork_evidence = false;
    bool active_no_anchor_right_continuity_ok = false;
    std::string active_no_anchor_right_continuity_reason = "not_evaluated";
    int active_no_anchor_last_confirmed_intersection_count = 0;
    std::string active_no_anchor_reject_mainforward_identity = "not_evaluated";
    bool detached_candidate_found = false;
    bool detached_candidate_valid = false;
    std::string detached_reject_reason = "none";
    std::string detached_candidate_ids = "none";
    std::string detached_gate_trace = "none";
    float detached_score = 0.0f;
    int detached_independent_id_count = 0;
    float detached_main_overlap = 0.0f;
    float detached_main_reuse_ratio = 0.0f;
    float detached_common_forward_overlap = 0.0f;
    int detached_right_chain_count = 0;
    float detached_right_chain_forward_span = 0.0f;
    float detached_right_chain_lateral_span = 0.0f;
    float detached_right_chain_continuity = 0.0f;
    bool detached_structural_gate_pass = false;
    bool detached_right_side_order_ok = false;
    bool detached_crossing_detected = false;
    bool detached_remerge_detected = false;
    float detached_width_min = 0.0f;
    float detached_width_max = 0.0f;
    float detached_width_mean = 0.0f;
    float detached_width_variation = 0.0f;
    int detached_sample_count = 0;
    int detached_valid_width_count = 0;
    int detached_trend_sample_count = 0;
    float detached_trend_width_near = 0.0f;
    float detached_trend_width_far = 0.0f;
    float detached_trend_width_delta = 0.0f;
    float detached_trend_width_slope = 0.0f;
    float detached_trend_far_width_ratio = 0.0f;
    float detached_trend_tail_min_width = 0.0f;
};

static int findGuideIndexByTrackerId(const std::vector<ArrowGuide>& guides, int tracker_id) {
    if (tracker_id < 0) {
        return -1;
    }
    for (size_t i = 0; i < guides.size(); ++i) {
        if (guides[i].tracker_id == tracker_id) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

static bool findRightForkAnchorOnMain(const std::vector<ArrowGuide>& main_guides,
                                      const std::vector<ArrowGuide>& candidate,
                                      int* out_main_idx,
                                      int* out_candidate_idx) {
    if (out_main_idx != nullptr) {
        *out_main_idx = -1;
    }
    if (out_candidate_idx != nullptr) {
        *out_candidate_idx = -1;
    }
    if (main_guides.size() < 3 || candidate.size() < 2) {
        return false;
    }

    const int max_main_anchor =
        std::max(0,
                 static_cast<int>(main_guides.size()) -
                     static_cast<int>(kRightForkAnchorMinMainTailGuideCount) - 1);
    int best_main_idx = -1;
    int best_candidate_idx = -1;
    for (size_t ci = 0;
         ci < candidate.size() && ci <= kRightForkAnchorCandidateMaxIndex;
         ++ci) {
        const int mi = findGuideIndexByTrackerId(main_guides, candidate[ci].tracker_id);
        if (mi < 0 || mi > max_main_anchor) {
            continue;
        }
        if (best_main_idx < 0 || mi < best_main_idx ||
            (mi == best_main_idx && static_cast<int>(ci) < best_candidate_idx)) {
            best_main_idx = mi;
            best_candidate_idx = static_cast<int>(ci);
        }
    }
    if (best_main_idx < 0) {
        return false;
    }
    if (out_main_idx != nullptr) {
        *out_main_idx = best_main_idx;
    }
    if (out_candidate_idx != nullptr) {
        *out_candidate_idx = best_candidate_idx;
    }
    return true;
}

struct RightForkReconnectMainTailCheck {
    bool reconnect = false;
    std::string reason = "not_evaluated";
    std::string candidate_ids = "none";
    std::string main_ids = "none";
    std::string shared_prefix_ids = "none";
    int shared_prefix_end_candidate_index = -1;
    int shared_prefix_end_main_index = -1;
    int candidate_departure_index = -1;
    std::string main_tail_ids = "none";
    std::string candidate_after_departure_ids = "none";
    std::string post_departure_main_tail_match_ids = "none";
    int post_departure_main_tail_match_count = 0;
    float post_departure_main_tail_match_ratio = 0.0f;
};

static std::string formatGuideIdsRange(const std::vector<ArrowGuide>& guides,
                                       size_t begin,
                                       size_t end) {
    if (begin >= end || begin >= guides.size()) {
        return "none";
    }
    end = std::min(end, guides.size());
    std::ostringstream oss;
    for (size_t i = begin; i < end; ++i) {
        if (i > begin) {
            oss << ",";
        }
        oss << guides[i].tracker_id;
    }
    return oss.str();
}

static std::string formatTrackerIdsVector(const std::vector<int>& ids) {
    if (ids.empty()) {
        return "none";
    }
    std::ostringstream oss;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << ids[i];
    }
    return oss.str();
}

static RightForkReconnectMainTailCheck evaluateRightForkReconnectsMainTail(
    const std::vector<ArrowGuide>& main_guides,
    const std::vector<ArrowGuide>& candidate,
    int main_anchor_idx,
    int candidate_anchor_idx) {
    RightForkReconnectMainTailCheck check;
    check.candidate_ids = formatSingleGrowGuideIds(candidate);
    check.main_ids = formatSingleGrowGuideIds(main_guides);
    if (main_guides.size() < 3 || candidate.size() < 2) {
        check.reason = "insufficient_guides";
        return check;
    }

    int shared_prefix_count = 0;
    while (static_cast<size_t>(shared_prefix_count) < main_guides.size() &&
           static_cast<size_t>(shared_prefix_count) < candidate.size() &&
           main_guides[static_cast<size_t>(shared_prefix_count)].tracker_id ==
               candidate[static_cast<size_t>(shared_prefix_count)].tracker_id) {
        ++shared_prefix_count;
    }

    if (shared_prefix_count > 0) {
        check.shared_prefix_end_candidate_index = shared_prefix_count - 1;
        check.shared_prefix_end_main_index = shared_prefix_count - 1;
        check.shared_prefix_ids =
            formatGuideIdsRange(candidate, 0, static_cast<size_t>(shared_prefix_count));
    }

    if (candidate_anchor_idx >= 0 && main_anchor_idx < 0 &&
        static_cast<size_t>(candidate_anchor_idx) < candidate.size()) {
        main_anchor_idx = findGuideIndexByTrackerId(
            main_guides,
            candidate[static_cast<size_t>(candidate_anchor_idx)].tracker_id);
    }

    const bool prefix_covers_anchor =
        shared_prefix_count > 0 &&
        (candidate_anchor_idx < 0 ||
         check.shared_prefix_end_candidate_index >= candidate_anchor_idx) &&
        (main_anchor_idx < 0 ||
         check.shared_prefix_end_main_index >= main_anchor_idx);
    if (!prefix_covers_anchor) {
        if (candidate_anchor_idx < 0 || main_anchor_idx < 0) {
            check.reason = "insufficient_anchor";
            return check;
        }
        check.shared_prefix_end_candidate_index = candidate_anchor_idx;
        check.shared_prefix_end_main_index = main_anchor_idx;
        check.shared_prefix_ids = formatGuideIdsRange(
            candidate,
            static_cast<size_t>(candidate_anchor_idx),
            static_cast<size_t>(candidate_anchor_idx + 1));
    }

    check.candidate_departure_index =
        check.shared_prefix_end_candidate_index + 1;
    if (check.candidate_departure_index < 0 ||
        static_cast<size_t>(check.candidate_departure_index) >= candidate.size()) {
        check.reason = "shared_prefix_only";
        return check;
    }

    const int main_tail_begin = check.shared_prefix_end_main_index + 1;
    if (main_tail_begin < 0 ||
        static_cast<size_t>(main_tail_begin) >= main_guides.size()) {
        check.reason = "left_main_no_reconnect";
        check.candidate_after_departure_ids = formatGuideIdsRange(
            candidate,
            static_cast<size_t>(check.candidate_departure_index),
            candidate.size());
        return check;
    }

    check.main_tail_ids = formatGuideIdsRange(
        main_guides,
        static_cast<size_t>(main_tail_begin),
        main_guides.size());
    check.candidate_after_departure_ids = formatGuideIdsRange(
        candidate,
        static_cast<size_t>(check.candidate_departure_index),
        candidate.size());

    std::unordered_set<int> main_tail_id_set;
    for (size_t i = static_cast<size_t>(main_tail_begin);
         i < main_guides.size(); ++i) {
        if (main_guides[i].tracker_id >= 0) {
            main_tail_id_set.insert(main_guides[i].tracker_id);
        }
    }

    bool left_main = false;
    size_t next_main_tail_pos = static_cast<size_t>(main_tail_begin);
    std::vector<int> matched_tail_ids;
    for (size_t ci = static_cast<size_t>(check.candidate_departure_index);
         ci < candidate.size(); ++ci) {
        const int candidate_id = candidate[ci].tracker_id;
        if (candidate_id < 0) {
            continue;
        }
        if (main_tail_id_set.count(candidate_id) == 0) {
            left_main = true;
            continue;
        }
        if (!left_main) {
            continue;
        }
        while (next_main_tail_pos < main_guides.size() &&
               main_guides[next_main_tail_pos].tracker_id != candidate_id) {
            ++next_main_tail_pos;
        }
        if (next_main_tail_pos >= main_guides.size()) {
            continue;
        }
        matched_tail_ids.push_back(candidate_id);
        ++next_main_tail_pos;
    }

    check.post_departure_main_tail_match_count =
        static_cast<int>(matched_tail_ids.size());
    const int main_tail_count =
        static_cast<int>(main_guides.size()) - main_tail_begin;
    check.post_departure_main_tail_match_ratio =
        main_tail_count > 0
            ? clampValue(
                  static_cast<float>(check.post_departure_main_tail_match_count) /
                      static_cast<float>(main_tail_count),
                  0.0f,
                  1.0f)
            : 0.0f;
    check.post_departure_main_tail_match_ids =
        formatTrackerIdsVector(matched_tail_ids);

    if (!left_main) {
        check.reason = "shared_prefix_only";
        return check;
    }
    if (check.post_departure_main_tail_match_count <= 0) {
        check.reason = prefix_covers_anchor
                           ? "shared_prefix_only"
                           : "left_main_no_reconnect";
        return check;
    }

    check.reconnect = true;
    check.reason = "reconnected_main_tail";
    return check;
}

static float computeRightForkPostSplitLateralTrend(
    const std::vector<ArrowGuide>& main_guides,
    const std::vector<ArrowGuide>& candidate,
    int candidate_anchor_idx,
    int* out_sample_count) {
    if (out_sample_count != nullptr) {
        *out_sample_count = 0;
    }
    if (candidate_anchor_idx < -1 ||
        static_cast<size_t>(candidate_anchor_idx + 1) >= candidate.size()) {
        return 0.0f;
    }

    float delta_sum = 0.0f;
    int count = 0;
    for (size_t i = static_cast<size_t>(candidate_anchor_idx + 1);
         i < candidate.size();
         ++i) {
        const float forward = guidePlanForward(candidate[i]);
        const float lateral = guidePlanLateral(candidate[i]);
        float main_lateral = 0.0f;
        if (!isFiniteFloat(forward) || !isFiniteFloat(lateral) ||
            !interpolateGuidePlanLateralAtForward(main_guides, forward, main_lateral)) {
            continue;
        }
        delta_sum += (lateral - main_lateral) / kWorldPlanScale;
        ++count;
    }
    if (out_sample_count != nullptr) {
        *out_sample_count = count;
    }
    return count > 0 ? delta_sum / static_cast<float>(count) : 0.0f;
}

static float computeRightForkAllCandidateLateralSeparation(
    const std::vector<ArrowGuide>& main_guides,
    const std::vector<ArrowGuide>& candidate,
    int* out_sample_count) {
    if (out_sample_count != nullptr) {
        *out_sample_count = 0;
    }

    float delta_sum = 0.0f;
    int count = 0;
    for (const auto& guide : candidate) {
        const float forward = guidePlanForward(guide);
        const float lateral = guidePlanLateral(guide);
        float main_lateral = 0.0f;
        if (!isFiniteFloat(forward) || !isFiniteFloat(lateral) ||
            !interpolateGuidePlanLateralAtForward(main_guides,
                                                  forward,
                                                  main_lateral)) {
            continue;
        }
        delta_sum += (lateral - main_lateral) / kWorldPlanScale;
        ++count;
    }
    if (out_sample_count != nullptr) {
        *out_sample_count = count;
    }
    return count > 0 ? delta_sum / static_cast<float>(count) : 0.0f;
}

struct DetachedRightForkEvaluation {
    bool found = false;
    bool valid = false;
    std::string reject_reason = "not_evaluated";
    std::string candidate_ids = "none";
    std::string gate_trace = "none";
    int independent_id_count = 0;
    float main_overlap = 0.0f;
    float main_reuse_ratio = 0.0f;
    float common_forward_overlap = 0.0f;
    int sample_count = 0;
    int valid_width_count = 0;
    float width_min = 0.0f;
    float width_max = 0.0f;
    float width_mean = 0.0f;
    float width_variation = 0.0f;
    float trend_width_near = 0.0f;
    float trend_width_far = 0.0f;
    float trend_width_delta = 0.0f;
    float trend_width_slope = 0.0f;
    float trend_far_width_ratio = 0.0f;
    float trend_tail_min_width = 0.0f;
    int trend_sample_count = 0;
    bool right_side_order_ok = false;
    bool crossing_detected = false;
    bool remerge_detected = false;
    int right_chain_count = 0;
    float right_chain_forward_span = 0.0f;
    float right_chain_lateral_span = 0.0f;
    float right_chain_continuity = 0.0f;
    bool structural_gate_pass = false;
    float score = 0.0f;
};

static DetachedRightForkEvaluation evaluateDetachedRightForkCandidate(
    const std::vector<ArrowGuide>& main_guides,
    const PhysicalBoundaryChain& candidate,
    float main_reuse_ratio,
    bool main_reuse_by_subsequence) {
    DetachedRightForkEvaluation eval;
    eval.found = !candidate.guides.empty();
    eval.candidate_ids = formatSingleGrowGuideIds(candidate.guides);
    eval.right_chain_count = static_cast<int>(candidate.guides.size());
    eval.right_chain_forward_span = std::max(0.0f, candidate.span / kWorldPlanScale);
    eval.right_chain_continuity = candidate.continuity_score;
    eval.main_reuse_ratio = clampValue(main_reuse_ratio, 0.0f, 1.0f);
    eval.main_overlap = eval.main_reuse_ratio;
    float min_lateral = std::numeric_limits<float>::infinity();
    float max_lateral = -std::numeric_limits<float>::infinity();
    for (const auto& guide : candidate.guides) {
        min_lateral = std::min(min_lateral, guidePlanLateral(guide));
        max_lateral = std::max(max_lateral, guidePlanLateral(guide));
    }
    if (std::isfinite(min_lateral) && std::isfinite(max_lateral)) {
        eval.right_chain_lateral_span =
            std::max(0.0f, (max_lateral - min_lateral) / kWorldPlanScale);
    }
    std::ostringstream gate_trace;
    auto recordGate = [&](const char* gate, bool passed) {
        if (gate_trace.tellp() > 0) gate_trace << ",";
        gate_trace << gate << "=" << (passed ? "pass" : "fail");
    };
    auto fail = [&](const std::string& reason) {
        eval.reject_reason = reason;
        eval.gate_trace = gate_trace.str().empty() ? "none" : gate_trace.str();
        return eval;
    };

    const bool guide_count_ok =
        main_guides.size() >= 3 &&
        candidate.guides.size() >= static_cast<size_t>(kDetachedRightForkMinGuideCount);
    recordGate("guide_count", guide_count_ok);
    if (!guide_count_ok) {
        return fail("detached_insufficient_guides");
    }

    std::unordered_set<int> main_ids;
    for (const auto& guide : main_guides) {
        if (guide.tracker_id >= 0) {
            main_ids.insert(guide.tracker_id);
        }
    }
    for (const auto& guide : candidate.guides) {
        if (guide.tracker_id >= 0 && main_ids.count(guide.tracker_id) == 0) {
            ++eval.independent_id_count;
        }
    }
    const float overlap_lo = std::max(candidate.near_forward, guidePlanForward(main_guides.front()));
    const float overlap_hi = std::min(candidate.far_forward, guidePlanForward(main_guides.back()));
    eval.common_forward_overlap =
        std::max(0.0f, (overlap_hi - overlap_lo) / kWorldPlanScale);
    recordGate("main_subsequence", !main_reuse_by_subsequence);
    if (main_reuse_by_subsequence) {
        return fail("detached_main_subsequence");
    }
    const bool independent_ids_ok =
        eval.independent_id_count >= kDetachedRightForkMinIndependentIds;
    recordGate("independent_ids", independent_ids_ok);
    if (!independent_ids_ok) {
        return fail("detached_independent_id_count_low");
    }
    const bool main_reuse_ok =
        eval.main_reuse_ratio <= kDetachedRightForkMaxMainReuseRatio;
    recordGate("main_reuse", main_reuse_ok);
    if (!main_reuse_ok) {
        return fail("detached_main_reuse_ratio_high");
    }
    const bool main_overlap_ok =
        eval.main_overlap <= kDetachedRightForkMaxRightMainOverlap;
    recordGate("main_overlap", main_overlap_ok);
    if (!main_overlap_ok) {
        return fail("detached_main_overlap_high");
    }
    const bool common_overlap_ok =
        eval.common_forward_overlap >= kDetachedRightForkMinCommonForwardOverlapM;
    recordGate("common_overlap", common_overlap_ok);
    if (!common_overlap_ok) {
        return fail("detached_common_forward_overlap_low");
    }
    const bool forward_span_ok =
        eval.right_chain_forward_span >= kDetachedRightForkMinForwardSpanM;
    recordGate("forward_span", forward_span_ok);
    if (!forward_span_ok) {
        return fail("detached_forward_span_low");
    }
    const bool continuity_ok =
        eval.right_chain_continuity >= kDetachedRightForkMinContinuityScore;
    recordGate("continuity", continuity_ok);
    if (!continuity_ok) {
        return fail("detached_continuity_low");
    }

    std::vector<std::pair<float, float>> widths_by_forward;
    widths_by_forward.reserve(candidate.guides.size());
    for (const auto& guide : candidate.guides) {
        const float forward = guidePlanForward(guide);
        const float lateral = guidePlanLateral(guide);
        if (!isFiniteFloat(forward) || forward < overlap_lo || forward > overlap_hi) {
            continue;
        }
        ++eval.sample_count;
        float main_lateral = 0.0f;
        if (!isFiniteFloat(lateral) ||
            !interpolateGuidePlanLateralAtForward(main_guides, forward, main_lateral)) {
            continue;
        }
        const float width_m = (lateral - main_lateral) / kWorldPlanScale;
        if (!isFiniteFloat(width_m)) {
            continue;
        }
        widths_by_forward.emplace_back(forward, width_m);
        if (width_m <= 0.0f) {
            eval.crossing_detected = true;
        }
    }
    eval.valid_width_count = static_cast<int>(widths_by_forward.size());
    const bool width_samples_ok =
        widths_by_forward.size() >= static_cast<size_t>(kDetachedRightForkMinWidthSamples);
    recordGate("overlap_width_samples", width_samples_ok);
    if (!width_samples_ok) {
        return fail("detached_width_sample_low");
    }
    std::sort(widths_by_forward.begin(), widths_by_forward.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    float width_sum = 0.0f;
    eval.width_min = std::numeric_limits<float>::infinity();
    eval.width_max = -std::numeric_limits<float>::infinity();
    for (const auto& item : widths_by_forward) {
        const float width = item.second;
        eval.width_min = std::min(eval.width_min, width);
        eval.width_max = std::max(eval.width_max, width);
        width_sum += width;
    }
    eval.width_mean = width_sum / static_cast<float>(widths_by_forward.size());
    eval.width_variation = eval.width_max - eval.width_min;
    eval.trend_width_near = widths_by_forward.front().second;
    eval.trend_width_far = widths_by_forward.back().second;
    eval.trend_sample_count = static_cast<int>(widths_by_forward.size());
    eval.trend_width_delta = eval.trend_width_far - eval.trend_width_near;
    const float trend_forward_span =
        std::max(1e-3f, (widths_by_forward.back().first - widths_by_forward.front().first) /
                            kWorldPlanScale);
    eval.trend_width_slope = eval.trend_width_delta / trend_forward_span;
    eval.trend_far_width_ratio =
        eval.trend_width_near > 1e-3f
            ? eval.trend_width_far / eval.trend_width_near
            : 0.0f;
    eval.trend_tail_min_width = eval.trend_width_far;
    if (widths_by_forward.size() >= 2) {
        eval.trend_tail_min_width =
            std::min(widths_by_forward[widths_by_forward.size() - 2].second,
                     widths_by_forward.back().second);
    }
    eval.right_side_order_ok = eval.width_min >= kDetachedRightForkMinWidthM;
    eval.remerge_detected =
        eval.width_mean >= kDetachedRightForkMinMeanWidthM &&
        eval.width_max >= kDetachedRightForkMinMeanWidthM &&
        eval.width_min < kDetachedRightForkMinWidthM;

    recordGate("crossing", !eval.crossing_detected);
    if (eval.crossing_detected) {
        return fail("detached_crossing_detected");
    }
    recordGate("right_side_order", eval.right_side_order_ok);
    if (!eval.right_side_order_ok) {
        return fail("detached_right_side_order_failed");
    }
    const bool mean_width_ok = eval.width_mean >= kDetachedRightForkMinMeanWidthM;
    recordGate("mean_width", mean_width_ok);
    if (!mean_width_ok) {
        return fail("detached_width_mean_low");
    }
    const bool max_width_ok = eval.width_max <= kDetachedRightForkMaxWidthM;
    recordGate("max_width", max_width_ok);
    if (!max_width_ok) {
        return fail("detached_width_max_high");
    }
    const bool width_variation_ok =
        eval.width_variation <= kDetachedRightForkMaxWidthVariationM;
    recordGate("width_variation", width_variation_ok);
    if (!width_variation_ok) {
        return fail("detached_width_variation_high");
    }
    recordGate("remerge", !eval.remerge_detected);
    if (eval.remerge_detected) {
        return fail("detached_remerge_detected");
    }

    eval.structural_gate_pass = true;
    eval.valid = true;
    eval.reject_reason = "none";
    eval.gate_trace = gate_trace.str();
    eval.score = clampValue(
        0.24f * clampValue(eval.common_forward_overlap / 0.45f, 0.0f, 1.0f) +
        0.22f * clampValue(eval.width_mean / 0.45f, 0.0f, 1.0f) +
        0.18f * (1.0f - clampValue(eval.width_variation / 0.35f, 0.0f, 1.0f)) +
        0.16f * clampValue(eval.right_chain_forward_span / 0.45f, 0.0f, 1.0f) +
        0.12f * clampValue(eval.right_chain_continuity, 0.0f, 1.0f) +
        0.08f * (1.0f - clampValue(
            eval.main_reuse_ratio / kDetachedRightForkMaxMainReuseRatio, 0.0f, 1.0f)),
        0.0f,
        1.0f);
    return eval;
}

static RightForkBranchSearchResult searchRightForkBranchFromMain(
    const std::vector<ArrowGuide>& main_guides,
    const std::vector<PhysicalBoundaryGuidePoint>& points,
    float median_hw,
    float min_forward_ref,
    uint32_t frame_id,
    BoundaryTrackManager* boundary_tracks) {
    (void)frame_id;
    RightForkBranchSearchResult best;
    if (main_guides.size() < 3 || points.size() < 4) {
        best.reject_reason = "insufficient_main_or_points";
        return best;
    }

    std::vector<PhysicalBoundaryChain> candidates =
        buildRightForkSearchChains(points, median_hw, min_forward_ref);
    best.raw_candidate_count = static_cast<int>(candidates.size());
    best.anchor_candidate_ci_range =
        "0.." + std::to_string(kRightForkAnchorCandidateMaxIndex);
    best.anchor_main_ids = formatSingleGrowGuideIds(main_guides);
    if (boundary_tracks != nullptr) {
        best.last_confirmed_right_fork_ids =
            boundary_tracks->last_confirmed_right_fork_ids_text;
    }
    if (candidates.empty()) {
        best.reject_reason = "no_right_fork_search_candidates";
        return best;
    }

    std::unordered_set<int> main_id_set;
    for (const auto& guide : main_guides) {
        if (guide.tracker_id >= 0) {
            main_id_set.insert(guide.tracker_id);
        }
    }
    auto updateIndependentRightEvidence =
        [&](const PhysicalBoundaryChain& candidate,
            float main_overlap,
            float lateral_separation,
            const std::string* cand_ids_text = nullptr) {
            int independent_count = 0;
            for (const auto& guide : candidate.guides) {
                if (guide.tracker_id >= 0 &&
                    main_id_set.count(guide.tracker_id) == 0) {
                    ++independent_count;
                }
            }
            if (independent_count > best.independent_right_id_count ||
                (independent_count == best.independent_right_id_count &&
                 lateral_separation > best.lateral_separation)) {
                best.independent_right_id_count = independent_count;
                best.right_main_overlap =
                    clampValue(main_overlap, 0.0f, 1.0f);
                best.lateral_separation =
                    std::max(0.0f, lateral_separation);
                if (cand_ids_text != nullptr) {
                    best.no_anchor_independent_chain = candidate;
                    best.no_anchor_independent_candidate_ids =
                        *cand_ids_text;
                    best.no_anchor_candidate_main_reuse_ratio =
                        main_overlap;
                }
            }
        };

    std::ostringstream raw_candidate_details;
    std::ostringstream reused_main_forward_reject_ids;
    float best_detached_score = -std::numeric_limits<float>::infinity();
    auto copyDetachedEvaluation = [&](const DetachedRightForkEvaluation& eval) {
        best.detached_candidate_found = eval.found;
        best.detached_candidate_valid = eval.valid;
        best.detached_reject_reason = eval.reject_reason;
        best.detached_candidate_ids = eval.candidate_ids;
        best.detached_gate_trace = eval.gate_trace;
        best.detached_score = eval.score;
        best.detached_independent_id_count = eval.independent_id_count;
        best.detached_main_overlap = eval.main_overlap;
        best.detached_main_reuse_ratio = eval.main_reuse_ratio;
        best.detached_common_forward_overlap = eval.common_forward_overlap;
        best.detached_right_chain_count = eval.right_chain_count;
        best.detached_right_chain_forward_span = eval.right_chain_forward_span;
        best.detached_right_chain_lateral_span = eval.right_chain_lateral_span;
        best.detached_right_chain_continuity = eval.right_chain_continuity;
        best.detached_structural_gate_pass = eval.structural_gate_pass;
        best.detached_right_side_order_ok = eval.right_side_order_ok;
        best.detached_crossing_detected = eval.crossing_detected;
        best.detached_remerge_detected = eval.remerge_detected;
        best.detached_width_min = eval.width_min;
        best.detached_width_max = eval.width_max;
        best.detached_width_mean = eval.width_mean;
        best.detached_width_variation = eval.width_variation;
        best.detached_sample_count = eval.sample_count;
        best.detached_valid_width_count = eval.valid_width_count;
        best.detached_trend_sample_count = eval.trend_sample_count;
        best.detached_trend_width_near = eval.trend_width_near;
        best.detached_trend_width_far = eval.trend_width_far;
        best.detached_trend_width_delta = eval.trend_width_delta;
        best.detached_trend_width_slope = eval.trend_width_slope;
        best.detached_trend_far_width_ratio = eval.trend_far_width_ratio;
        best.detached_trend_tail_min_width = eval.trend_tail_min_width;
    };
    auto considerDetachedEvaluation = [&](const DetachedRightForkEvaluation& detached_eval,
                                           const PhysicalBoundaryChain& candidate) {
        const bool is_better =
            !best.detached_candidate_found ||
            (detached_eval.valid && !best.detached_candidate_valid) ||
            (detached_eval.valid == best.detached_candidate_valid &&
             detached_eval.score > best_detached_score);
        if (detached_eval.found && is_better) {
            best_detached_score = detached_eval.score;
            copyDetachedEvaluation(detached_eval);
        }
        if (detached_eval.valid &&
            (!best.found || detached_eval.score > best.score)) {
            best.found = true;
            best.chain = candidate;
            best.anchor_type = "detached";
            best.anchor_on_main = "none";
            best.anchor_match_ci = "none";
            best.anchor_match_mi = "none";
            best.current_candidate_ids = formatSingleGrowGuideIds(candidate.guides);
            best.candidate_ids = best.current_candidate_ids;
            best.suffix_continuation_found = false;
            best.suffix_continuation_reason = "detached_current_frame";
            best.no_anchor_candidate_prefix_ids = formatFirstGuideIdsCompact(
                candidate.guides,
                kRightForkAnchorCandidateMaxIndex + 1);
            best.created_by_alt = candidate.created_by_right_fork_alt;
            best.reconnect_main_tail = false;
            best.shared_prefix_count = 0;
            best.weak_independent_suffix_count =
                static_cast<int>(candidate.guides.size());
            best.score = detached_eval.score;
            best.post_split_lateral_trend = detached_eval.width_mean;
            best.reject_reason = "none";
            best.selected_right_fork_model = "detached";
        }
    };
    for (const auto& candidate : candidates) {
        if (raw_candidate_details.tellp() > 0) {
            raw_candidate_details << ";";
        }
        raw_candidate_details << "ids=[" << formatSingleGrowGuideIds(candidate.guides)
                              << "],n=" << candidate.guides.size();
        if (candidate.guides.size() < 2) {
            raw_candidate_details << ",reject=candidate_too_few_guides";
            if (!best.found) {
                best.reject_reason = "candidate_too_few_guides";
            }
            continue;
        }
        const std::vector<int> candidate_tracker_ids =
            collectGuideTrackerIds(candidate.guides);
        const std::string candidate_ids_text =
            formatSingleGrowGuideIds(candidate.guides);
        const float main_reuse_ratio =
            computeGuideIdMatchRatio(candidate.guides, main_guides);
        best.main_reuse_ratio_max =
            std::max(best.main_reuse_ratio_max, main_reuse_ratio);
        int all_candidate_lateral_sample_count = 0;
        const float all_candidate_lateral_separation =
            std::abs(computeRightForkAllCandidateLateralSeparation(
                main_guides,
                candidate.guides,
                &all_candidate_lateral_sample_count));
        const bool main_reuse_by_subsequence =
            guideIdsAreSubsequence(candidate.guides, main_guides);
        if (main_reuse_ratio >= 0.75f || main_reuse_by_subsequence) {
            int reuse_main_anchor_idx = -1;
            int reuse_candidate_anchor_idx = -1;
            const bool reuse_has_anchor =
                findRightForkAnchorOnMain(main_guides,
                                          candidate.guides,
                                          &reuse_main_anchor_idx,
                                          &reuse_candidate_anchor_idx);
            int reuse_trend_sample_count = 0;
            const float reuse_post_split_lateral_trend =
                reuse_has_anchor
                    ? computeRightForkPostSplitLateralTrend(main_guides,
                                                            candidate.guides,
                                                            reuse_candidate_anchor_idx,
                                                            &reuse_trend_sample_count)
                    : 0.0f;
            const RightForkReconnectMainTailCheck reuse_reconnect_check =
                reuse_has_anchor
                    ? evaluateRightForkReconnectsMainTail(
                          main_guides,
                          candidate.guides,
                          reuse_main_anchor_idx,
                          reuse_candidate_anchor_idx)
                    : RightForkReconnectMainTailCheck();
            const bool reuse_reconnects_main_tail =
                reuse_has_anchor && reuse_reconnect_check.reconnect;
            const bool reuse_has_right_diverging_suffix =
                reuse_has_anchor &&
                reuse_trend_sample_count >= 2 &&
                reuse_post_split_lateral_trend >= 0.07f &&
                !reuse_reconnects_main_tail;
            raw_candidate_details << ",main_reuse_ratio=" << main_reuse_ratio
                                  << ",main_reuse_subseq="
                                  << (main_reuse_by_subsequence ? 1 : 0)
                                  << ",reuse_anchor_ci="
                                  << (reuse_has_anchor
                                          ? std::to_string(reuse_candidate_anchor_idx)
                                          : "none")
                                  << ",reuse_anchor_mi="
                                  << (reuse_has_anchor
                                          ? std::to_string(reuse_main_anchor_idx)
                                          : "none")
                                  << ",reuse_trend_n=" << reuse_trend_sample_count
                                  << ",reuse_trend="
                                  << reuse_post_split_lateral_trend
                                  << ",reuse_reconnect_tail="
                                  << (reuse_reconnects_main_tail ? 1 : 0);
            if (reuse_has_right_diverging_suffix) {
                raw_candidate_details
                    << ",main_reuse_kept=right_diverging_suffix";
            } else {
                if (main_reuse_by_subsequence) {
                    const DetachedRightForkEvaluation detached_eval =
                        evaluateDetachedRightForkCandidate(main_guides,
                                                           candidate,
                                                           main_reuse_ratio,
                                                           true);
                    raw_candidate_details
                        << ",detached_valid=" << (detached_eval.valid ? 1 : 0)
                        << ",detached_reason=" << detached_eval.reject_reason
                        << ",detached_gates=" << detached_eval.gate_trace;
                    considerDetachedEvaluation(detached_eval, candidate);
                }
                raw_candidate_details << ",reject=candidate_reuses_main_forward";
                best.candidate_reuses_main_forward_seen = true;
                best.right_guide_count_when_reuse_main_forward =
                    std::max(best.right_guide_count_when_reuse_main_forward,
                             static_cast<int>(candidate.guides.size()));
                if (best.candidate_reuses_main_forward_reject_count > 0) {
                    reused_main_forward_reject_ids << "|";
                }
                reused_main_forward_reject_ids << formatSingleGrowGuideIds(candidate.guides);
                ++best.candidate_reuses_main_forward_reject_count;
                if (!best.found) {
                    best.reject_reason = "candidate_reuses_main_forward";
                }
                continue;
            }
        }

        int main_anchor_idx = -1;
        int candidate_anchor_idx = -1;
        std::string anchor_type = "none";
        bool has_anchor =
            findRightForkAnchorOnMain(main_guides,
                                      candidate.guides,
                                      &main_anchor_idx,
                                      &candidate_anchor_idx);
        bool suffix_continuation_found = false;
        std::string suffix_continuation_reason = "none";
        const bool suffix_continuation_allowed =
            boundary_tracks != nullptr &&
            boundary_tracks->visible_split_output_active &&
            !boundary_tracks->last_confirmed_right_fork_ids.empty();
        if (has_anchor) {
            anchor_type = "id";
        } else {
            if (!suffix_continuation_allowed) {
                suffix_continuation_reason =
                    boundary_tracks == nullptr
                        ? "no_boundary_manager"
                        : (!boundary_tracks->visible_split_output_active
                               ? "visible_split_not_active"
                               : "no_last_confirmed_right_fork");
            } else {
                suffix_continuation_found =
                    guideIdsAreConfirmedRightForkSuffix(
                        candidate_tracker_ids,
                        boundary_tracks->last_confirmed_right_fork_ids,
                        &suffix_continuation_reason);
                if (suffix_continuation_found) {
                    anchor_type = "suffix_continuation";
                    candidate_anchor_idx = -1;
                    main_anchor_idx = -1;
                    has_anchor = true;
                }
            }
        }
        raw_candidate_details << ",anchor_type=" << anchor_type
                              << ",suffix_continuation_found="
                              << (suffix_continuation_found ? 1 : 0)
                              << ",suffix_continuation_reason="
                              << suffix_continuation_reason;
        if (!has_anchor) {
            const DetachedRightForkEvaluation detached_eval =
                evaluateDetachedRightForkCandidate(main_guides,
                                                   candidate,
                                                   main_reuse_ratio,
                                                   main_reuse_by_subsequence);
            raw_candidate_details << ",detached_valid=" << (detached_eval.valid ? 1 : 0)
                                  << ",detached_reason=" << detached_eval.reject_reason
                                  << ",detached_gates=" << detached_eval.gate_trace;
            considerDetachedEvaluation(detached_eval, candidate);
            if (all_candidate_lateral_sample_count >= 2) {
                updateIndependentRightEvidence(candidate,
                                               main_reuse_ratio,
                                               all_candidate_lateral_separation,
                                               &candidate_ids_text);
            }
            raw_candidate_details << ",anchor_ci=none,anchor_mi=none"
                                  << ",reject=no_anchor_on_main";
            best.no_anchor_on_main_seen = true;
            best.right_guide_count_when_no_anchor =
                std::max(best.right_guide_count_when_no_anchor,
                         static_cast<int>(candidate.guides.size()));
            if (!best.found) {
                best.reject_reason = "no_anchor_on_main";
                best.anchor_type = "none";
                best.current_candidate_ids = candidate_ids_text;
                best.candidate_ids = candidate_ids_text;
                best.suffix_continuation_found = suffix_continuation_found;
                best.suffix_continuation_reason = suffix_continuation_reason;
                best.no_anchor_candidate_prefix_ids =
                    formatFirstGuideIdsCompact(candidate.guides,
                                               kRightForkAnchorCandidateMaxIndex + 1);
            }
            continue;
        }
        if (!best.found) {
            best.anchor_match_ci = anchor_type == "id"
                                       ? std::to_string(candidate_anchor_idx)
                                       : "suffix";
            best.anchor_match_mi = anchor_type == "id"
                                       ? std::to_string(main_anchor_idx)
                                       : "suffix";
        }
        raw_candidate_details << ",anchor_ci="
                              << (anchor_type == "id"
                                      ? std::to_string(candidate_anchor_idx)
                                      : "suffix")
                              << ",anchor_mi="
                              << (anchor_type == "id"
                                      ? std::to_string(main_anchor_idx)
                                      : "suffix");

        int trend_sample_count = 0;
        const float post_split_lateral_trend =
            computeRightForkPostSplitLateralTrend(main_guides,
                                                  candidate.guides,
                                                  candidate_anchor_idx,
                                                  &trend_sample_count);
        raw_candidate_details << ",trend_n=" << trend_sample_count
                              << ",trend=" << post_split_lateral_trend;
        if (trend_sample_count >= 2) {
            updateIndependentRightEvidence(candidate,
                                           main_reuse_ratio,
                                           std::abs(post_split_lateral_trend));
        }
        if (trend_sample_count < 2 ||
            post_split_lateral_trend < kRightForkMinPostSplitLateralTrend) {
            raw_candidate_details << ",reject=not_right_diverging_after_split";
            if (!best.found) {
                best.reject_reason = "not_right_diverging_after_split";
            }
            continue;
        }

        const RightForkReconnectMainTailCheck reconnect_check =
            evaluateRightForkReconnectsMainTail(main_guides,
                                                candidate.guides,
                                                main_anchor_idx,
                                                candidate_anchor_idx);
        const bool reconnect_tail = reconnect_check.reconnect;
        if (reconnect_tail) {
            raw_candidate_details << ",reject=candidate_reconnect_main_tail";
            if (!best.found) {
                best.reject_reason = "candidate_reconnect_main_tail";
                best.reconnect_main_tail = true;
                best.anchor_type = anchor_type;
                best.current_candidate_ids = candidate_ids_text;
                best.candidate_ids = candidate_ids_text;
                best.suffix_continuation_found = suffix_continuation_found;
                best.suffix_continuation_reason = suffix_continuation_reason;
            }
            continue;
        }

        const int independent_suffix =
            anchor_type == "id"
                ? static_cast<int>(candidate.guides.size()) - candidate_anchor_idx - 1
                : static_cast<int>(candidate.guides.size());
        const float score =
            clampValue(0.38f * clampValue(post_split_lateral_trend / 0.24f, 0.0f, 1.0f) +
                           0.26f * clampValue(static_cast<float>(independent_suffix) / 4.0f,
                                               0.0f,
                                               1.0f) +
                           0.22f * clampValue(candidate.span / planMeters(0.45f), 0.0f, 1.0f) +
                           0.14f * clampValue(candidate.confidence, 0.0f, 1.0f),
                        0.0f,
                        1.0f);
        raw_candidate_details << ",score=" << score << ",reject=none";

        if (!best.found || score > best.score) {
            int actual_shared_prefix = 0;
            while (static_cast<size_t>(actual_shared_prefix) < main_guides.size() &&
                   static_cast<size_t>(actual_shared_prefix) < candidate.guides.size() &&
                   main_guides[static_cast<size_t>(actual_shared_prefix)].tracker_id ==
                       candidate.guides[static_cast<size_t>(actual_shared_prefix)].tracker_id) {
                ++actual_shared_prefix;
            }
            best.found = true;
            best.chain = candidate;
            best.anchor_type = anchor_type;
            best.anchor_on_main =
                anchor_type == "id"
                    ? (std::to_string(main_anchor_idx) + ":" +
                       std::to_string(
                           main_guides[static_cast<size_t>(main_anchor_idx)].tracker_id))
                    : "suffix_continuation";
            best.anchor_match_ci = anchor_type == "id"
                                       ? std::to_string(candidate_anchor_idx)
                                       : "suffix";
            best.anchor_match_mi = anchor_type == "id"
                                       ? std::to_string(main_anchor_idx)
                                       : "suffix";
            best.current_candidate_ids = candidate_ids_text;
            best.candidate_ids = candidate_ids_text;
            best.suffix_continuation_found = suffix_continuation_found;
            best.suffix_continuation_reason = suffix_continuation_reason;
            best.created_by_alt = candidate.created_by_right_fork_alt;
            best.reconnect_main_tail = reconnect_tail;
            best.shared_prefix_count = actual_shared_prefix;
            best.weak_independent_suffix_count = independent_suffix;
            best.score = score;
            best.post_split_lateral_trend = post_split_lateral_trend;
            best.reject_reason = "none";
        }
    }

    // Active no-anchor independent candidate acceptance (demo-aligned)
    best.active_no_anchor_independent_candidate_accept = false;
    best.active_no_anchor_accept_block_reason = "not_evaluated";
    const bool active_visible_split_context =
        boundary_tracks != nullptr &&
        (hasSplitRecoverySearchContext(boundary_tracks) ||
         boundary_tracks->visible_split_output_active ||
         boundary_tracks->visible_split_maintain_age > 0);
    const bool independent_right_fork_evidence_for_accept =
        best.independent_right_id_count >= 3 &&
        best.right_main_overlap <= 0.35f &&
        best.lateral_separation >= 0.05f;
    const bool main_reuse_ratio_high_for_accept =
        best.no_anchor_candidate_main_reuse_ratio >= 0.90f &&
        !independent_right_fork_evidence_for_accept;
    const bool lateral_separation_low_for_accept =
        best.lateral_separation < 0.05f;
    const bool right_main_overlap_high_for_accept =
        best.right_main_overlap > 0.45f;
    const bool geometry_not_bad =
        !lateral_separation_low_for_accept &&
        !right_main_overlap_high_for_accept &&
        !main_reuse_ratio_high_for_accept;
    const int consecutive_invalid_for_grace =
        boundary_tracks != nullptr
            ? boundary_tracks->visible_split_provider_gap_frames + 1
            : 999;
    const bool grace_budget_ok = consecutive_invalid_for_grace <= 1;
    bool structural_strong_negative = false;
    std::string structural_strong_negative_reason = "none";
    if (!best.found && best.no_anchor_on_main_seen) {
        const bool no_anchor_candidate_is_not_reuse_main =
            best.no_anchor_candidate_main_reuse_ratio < 0.75f;
        auto markNegative = [&](const std::string& reason) {
            if (!structural_strong_negative) {
                structural_strong_negative = true;
                structural_strong_negative_reason = reason;
            }
        };
        if (!active_visible_split_context) {
            markNegative("no_active_visible_split_context");
        }
        if (!grace_budget_ok) {
            markNegative("consecutive_invalid_visible_pair");
        }
        if (main_reuse_ratio_high_for_accept) {
            markNegative("main_reuse_ratio_high");
        }
        if (lateral_separation_low_for_accept) {
            markNegative("lateral_separation_low");
        }
        if (right_main_overlap_high_for_accept) {
            markNegative("right_main_overlap_high");
        }
        if (best.independent_right_id_count < 3) {
            markNegative("independent_right_id_count_low");
        }
        // Right fork continuity: current no-anchor candidate must be a
        // continuation of the last confirmed right fork, not a fresh chain.
        {
            best.active_no_anchor_right_continuity_ok = false;
            best.active_no_anchor_right_continuity_reason = "not_evaluated";
            best.active_no_anchor_last_confirmed_intersection_count = 0;
            best.active_no_anchor_reject_mainforward_identity =
                "not_evaluated";
            std::unordered_set<int> confirmed_set;
            const bool use_recovery_right_seed =
                hasSplitRecoverySearchContext(boundary_tracks);
            if (use_recovery_right_seed) {
                for (const int id : boundary_tracks->split_recovery_seed_right_ids) {
                    confirmed_set.insert(id);
                }
            }
            const std::string& confirmed_text = best.last_confirmed_right_fork_ids;
            if (!use_recovery_right_seed &&
                confirmed_text != "none" && !confirmed_text.empty()) {
                std::istringstream iss(confirmed_text);
                std::string token;
                while (std::getline(iss, token, ',')) {
                    try {
                        confirmed_set.insert(std::stoi(token));
                    } catch (...) {
                    }
                }
            }
            if (!confirmed_set.empty() &&
                best.no_anchor_independent_chain.guides.size() > 0) {
                int intersection = 0;
                for (const auto& g :
                     best.no_anchor_independent_chain.guides) {
                    if (g.tracker_id >= 0 &&
                        confirmed_set.count(g.tracker_id) > 0) {
                        ++intersection;
                    }
                }
                best.active_no_anchor_last_confirmed_intersection_count =
                    intersection;
                const bool has_suffix =
                    best.suffix_continuation_found;
                const bool has_intersection = intersection >= 2;
                if (has_suffix) {
                    best.active_no_anchor_right_continuity_ok = true;
                    best.active_no_anchor_right_continuity_reason =
                        "suffix_continuation_found";
                } else if (has_intersection) {
                    best.active_no_anchor_right_continuity_ok = true;
                    best.active_no_anchor_right_continuity_reason =
                        "id_intersection_ge_2";
                } else {
                    best.active_no_anchor_right_continuity_ok = false;
                    best.active_no_anchor_right_continuity_reason =
                        "no_intersection_with_last_confirmed_right_fork";
                }
            } else if (confirmed_set.empty()) {
                best.active_no_anchor_right_continuity_ok = false;
                best.active_no_anchor_right_continuity_reason =
                    "no_last_confirmed_right_fork_ids";
            }
            if (!best.active_no_anchor_right_continuity_ok) {
                markNegative(
                    best.active_no_anchor_right_continuity_reason);
            }
        }
        const bool accept_ok =
            active_visible_split_context &&
            no_anchor_candidate_is_not_reuse_main &&
            best.independent_right_id_count >= 3 &&
            best.right_main_overlap <= 0.25f &&
            best.lateral_separation >= 0.20f &&
            best.no_anchor_candidate_main_reuse_ratio < 0.75f &&
            geometry_not_bad &&
            !structural_strong_negative &&
            grace_budget_ok &&
            best.no_anchor_independent_candidate_ids != "none";
        if (accept_ok) {
            best.active_no_anchor_independent_candidate_accept = true;
            best.found = true;
            best.chain = best.no_anchor_independent_chain;
            best.anchor_type = "active_no_anchor_independent_candidate";
            best.anchor_on_main = "active_no_anchor";
            best.anchor_match_ci = "none";
            best.anchor_match_mi = "none";
            best.current_candidate_ids =
                best.no_anchor_independent_candidate_ids;
            best.candidate_ids =
                best.no_anchor_independent_candidate_ids;
            best.selected_right_fork_model = "detached";
            best.suffix_continuation_found = false;
            best.suffix_continuation_reason = "active_no_anchor_accept";
            best.created_by_alt = false;
            best.reconnect_main_tail = false;
            int actual_shared_prefix = 0;
            while (static_cast<size_t>(actual_shared_prefix) < main_guides.size() &&
                   static_cast<size_t>(actual_shared_prefix) <
                       best.chain.guides.size() &&
                   main_guides[static_cast<size_t>(actual_shared_prefix)]
                           .tracker_id ==
                       best.chain.guides[static_cast<size_t>(actual_shared_prefix)]
                           .tracker_id) {
                ++actual_shared_prefix;
            }
            best.shared_prefix_count = actual_shared_prefix;
            best.weak_independent_suffix_count =
                static_cast<int>(best.chain.guides.size()) -
                actual_shared_prefix;
            best.score = 0.0f;
            best.post_split_lateral_trend = best.lateral_separation;
            best.reject_reason = "none";
        } else {
            if (!active_visible_split_context) {
                best.active_no_anchor_accept_block_reason =
                    "no_active_visible_split_context";
            } else if (!no_anchor_candidate_is_not_reuse_main) {
                best.active_no_anchor_accept_block_reason =
                    "candidate_reuses_main_forward";
            } else if (best.independent_right_id_count < 3) {
                best.active_no_anchor_accept_block_reason =
                    "independent_right_id_count_lt_3";
            } else if (best.right_main_overlap > 0.25f) {
                best.active_no_anchor_accept_block_reason =
                    "right_main_overlap_gt_0.25";
            } else if (best.lateral_separation < 0.20f) {
                best.active_no_anchor_accept_block_reason =
                    "lateral_separation_lt_0.20";
            } else if (best.no_anchor_candidate_main_reuse_ratio >= 0.75f) {
                best.active_no_anchor_accept_block_reason =
                    "main_reuse_ratio_max_ge_0.75";
            } else if (!geometry_not_bad) {
                best.active_no_anchor_accept_block_reason =
                    "geometry_not_bad_false";
            } else if (structural_strong_negative) {
                best.active_no_anchor_accept_block_reason =
                    structural_strong_negative_reason;
            } else if (!grace_budget_ok) {
                best.active_no_anchor_accept_block_reason =
                    "grace_budget_exhausted";
            } else if (best.no_anchor_independent_candidate_ids == "none") {
                best.active_no_anchor_accept_block_reason =
                    "no_independent_chain_saved";
            }
        }
    }

    best.raw_candidate_details =
        raw_candidate_details.tellp() > 0 ? raw_candidate_details.str() : "none";
    best.candidate_reuses_main_forward_reject_ids =
        reused_main_forward_reject_ids.tellp() > 0
            ? reused_main_forward_reject_ids.str()
            : "none";
        best.independent_right_fork_evidence =
        best.independent_right_id_count >= 3 &&
        best.right_main_overlap <= 0.35f &&
        best.lateral_separation >= 0.05f;
    if (best.found && best.selected_right_fork_model == "none") {
        best.selected_right_fork_model =
            best.anchor_type == "detached" ? "detached" : "anchored";
    }

    return best;
}

static ForwardGrownVisiblePairCandidate buildForwardGrownVisiblePairCandidate(
    const std::vector<ArrowGuide>& valid_guides,
    const std::vector<ArrowGuide>& main_guides,
    float visible_min_forward,
    float median_hw,
    uint32_t frame_id,
    BoundaryTrackManager* boundary_tracks) {
    ForwardGrownVisiblePairCandidate best;
    best.main_forward_source = "single_anchor_grow";
    best.main_forward_ids = formatSingleGrowGuideIds(main_guides);
    best.right_fork_branch_search_enabled = true;
    best.semantic_passthrough = "physical_provider_v2";
    if (valid_guides.size() < 4) {
        best.source = "forward_grown_visible_pair:insufficient_guides";
        return best;
    }
    if (main_guides.size() < 3) {
        best.source = "forward_grown_visible_pair:insufficient_main_forward";
        best.reject_reason = "insufficient_main_forward";
        return best;
    }

    float min_forward_ref = std::numeric_limits<float>::infinity();
    std::vector<PhysicalBoundaryGuidePoint> points;
    points.reserve(valid_guides.size());
    for (const auto& guide : valid_guides) {
        const cv::Point2f plan = guidePlanPoint(guide);
        if (!isFinitePoint(plan)) {
            continue;
        }
        PhysicalBoundaryGuidePoint point;
        point.guide = guide;
        point.forward = plan.y;
        point.lateral = plan.x;
        points.push_back(point);
        min_forward_ref = std::min(min_forward_ref, point.forward);
    }
    if (points.size() < 4) {
        best.source = "forward_grown_visible_pair:insufficient_valid_points";
        return best;
    }
    if (!std::isfinite(min_forward_ref)) {
        min_forward_ref = 0.0f;
    }

    std::sort(points.begin(), points.end(), [](const PhysicalBoundaryGuidePoint& a,
                                               const PhysicalBoundaryGuidePoint& b) {
        if (std::abs(a.forward - b.forward) > 1e-3f) {
            return a.forward < b.forward;
        }
        return a.lateral < b.lateral;
    });

    PhysicalBoundaryChain main_chain =
        finalizePhysicalBoundaryChain(std::vector<ArrowGuide>(main_guides.begin(),
                                                              main_guides.end()),
                                      0,
                                      min_forward_ref);
    const RightForkBranchSearchResult right_fork =
        searchRightForkBranchFromMain(main_guides,
                                      points,
                                      median_hw,
                                      min_forward_ref,
                                      frame_id,
                                      boundary_tracks);
    best.right_fork_search_candidate_count = right_fork.raw_candidate_count;
    best.selected_seed_ids = "main_forward_anchor:" + best.main_forward_ids;
    best.right_fork_anchor_type = right_fork.anchor_type;
    best.right_fork_anchor_on_main = right_fork.anchor_on_main;
    best.last_confirmed_right_fork_ids =
        right_fork.last_confirmed_right_fork_ids;
    best.right_fork_candidate_ids = right_fork.candidate_ids;
    best.current_right_fork_candidate_ids = right_fork.current_candidate_ids;
    best.suffix_continuation_found = right_fork.suffix_continuation_found;
    best.suffix_continuation_reason = right_fork.suffix_continuation_reason;
    best.right_fork_created_by_alt = right_fork.created_by_alt;
    best.right_fork_reconnect_main_tail = right_fork.reconnect_main_tail;
    best.no_anchor_on_main_seen = right_fork.no_anchor_on_main_seen;
    best.right_guide_count_when_no_anchor =
        right_fork.right_guide_count_when_no_anchor;
    best.candidate_reuses_main_forward_seen =
        right_fork.candidate_reuses_main_forward_seen;
    best.right_guide_count_when_reuse_main_forward =
        right_fork.right_guide_count_when_reuse_main_forward;
    best.right_fork_candidate_reuses_main_forward_reject_count =
        right_fork.candidate_reuses_main_forward_reject_count;
    best.right_fork_candidate_reuses_main_forward_reject_ids =
        right_fork.candidate_reuses_main_forward_reject_ids;
    best.right_fork_raw_candidate_details = right_fork.raw_candidate_details;
    best.independent_right_id_count = right_fork.independent_right_id_count;
    best.independent_right_fork_evidence =
        right_fork.independent_right_fork_evidence;
    best.right_main_overlap = right_fork.right_main_overlap;
    best.lateral_separation = right_fork.lateral_separation;
    best.main_reuse_ratio_max = right_fork.main_reuse_ratio_max;
    best.selected_right_fork_model = right_fork.selected_right_fork_model;
    best.detached_candidate_found = right_fork.detached_candidate_found;
    best.detached_candidate_valid = right_fork.detached_candidate_valid;
    best.detached_reject_reason = right_fork.detached_reject_reason;
    best.detached_candidate_ids = right_fork.detached_candidate_ids;
    best.detached_gate_trace = right_fork.detached_gate_trace;
    best.detached_score = right_fork.detached_score;
    best.detached_independent_id_count = right_fork.detached_independent_id_count;
    best.detached_main_overlap = right_fork.detached_main_overlap;
    best.detached_main_reuse_ratio = right_fork.detached_main_reuse_ratio;
    best.detached_common_forward_overlap = right_fork.detached_common_forward_overlap;
    best.detached_right_chain_count = right_fork.detached_right_chain_count;
    best.detached_right_chain_forward_span = right_fork.detached_right_chain_forward_span;
    best.detached_right_chain_lateral_span = right_fork.detached_right_chain_lateral_span;
    best.detached_right_chain_continuity = right_fork.detached_right_chain_continuity;
    best.detached_structural_gate_pass = right_fork.detached_structural_gate_pass;
    best.detached_right_side_order_ok = right_fork.detached_right_side_order_ok;
    best.detached_crossing_detected = right_fork.detached_crossing_detected;
    best.detached_remerge_detected = right_fork.detached_remerge_detected;
    best.detached_width_min = right_fork.detached_width_min;
    best.detached_width_max = right_fork.detached_width_max;
    best.detached_width_mean = right_fork.detached_width_mean;
    best.detached_width_variation = right_fork.detached_width_variation;
    best.detached_sample_count = right_fork.detached_sample_count;
    best.detached_valid_width_count = right_fork.detached_valid_width_count;
    best.detached_trend_sample_count = right_fork.detached_trend_sample_count;
    best.detached_trend_width_near = right_fork.detached_trend_width_near;
    best.detached_trend_width_far = right_fork.detached_trend_width_far;
    best.detached_trend_width_delta = right_fork.detached_trend_width_delta;
    best.detached_trend_width_slope = right_fork.detached_trend_width_slope;
    best.detached_trend_far_width_ratio = right_fork.detached_trend_far_width_ratio;
    best.detached_trend_tail_min_width = right_fork.detached_trend_tail_min_width;
    if (!right_fork.found) {
        best.source = "forward_grown_visible_pair:no_right_fork_branch";
        best.reject_reason = right_fork.reject_reason;
        best.selected_chain_ids = best.main_forward_ids + "|none";
        return best;
    }

    bool main_is_physical_left = true;
    float selected_lateral_gap = std::abs(right_fork.post_split_lateral_trend) *
                                 kWorldPlanScale;
    computeVisiblePairSeparatedRangeAssignment(main_chain,
                                               right_fork.chain,
                                               &main_is_physical_left,
                                               &selected_lateral_gap);
    const PhysicalBoundaryPrimitive main_primitive =
        buildPhysicalBoundaryPrimitiveFromChain(main_chain, visible_min_forward);
    const PhysicalBoundaryPrimitive right_primitive =
        buildPhysicalBoundaryPrimitiveFromChain(right_fork.chain, visible_min_forward);
    best.valid = true;
    best.left_guides = main_chain.guides;
    best.right_guides = right_fork.chain.guides;
    best.left_span = main_chain.span / kWorldPlanScale;
    best.right_span = right_fork.chain.span / kWorldPlanScale;
    best.width_m = selected_lateral_gap / kWorldPlanScale;
    best.overlap_forward =
        std::max(0.0f,
                 std::min(main_chain.far_forward, right_fork.chain.far_forward) -
                     std::max(main_chain.near_forward, right_fork.chain.near_forward));
    best.confidence = right_fork.score;
    best.left_type = physicalBoundaryPrimitiveTypeName(main_primitive.type);
    best.right_type = physicalBoundaryPrimitiveTypeName(right_primitive.type);
    best.selected_chain_ids = best.main_forward_ids + "|" + right_fork.candidate_ids;
    best.shared_prefix_count = right_fork.shared_prefix_count;
    best.weak_independent_suffix_count = right_fork.weak_independent_suffix_count;
    best.pair_score = right_fork.score;
    best.reject_reason = "none";
    best.accept_reason = "main_forward_plus_right_fork_branch";
    best.relative_assignment =
        main_is_physical_left ? "main_left_right_fork_right"
                              : "main_right_right_fork_left";
    best.final_pair_source = "main_forward_plus_right_fork_branch";
    best.source = "forward_grown_visible_pair:main_forward_plus_right_fork_branch";
    return best;
}

static float computeGuideIdMatchRatio(const std::vector<ArrowGuide>& guides,
                                      const std::vector<ArrowGuide>& reference) {
    if (guides.empty() || reference.empty()) {
        return 0.0f;
    }
    std::unordered_set<int> ref_ids;
    for (const auto& guide : reference) {
        if (guide.tracker_id >= 0) {
            ref_ids.insert(guide.tracker_id);
        }
    }
    if (ref_ids.empty()) {
        return 0.0f;
    }
    int match_count = 0;
    for (const auto& guide : guides) {
        if (guide.tracker_id >= 0 && ref_ids.count(guide.tracker_id) > 0) {
            ++match_count;
        }
    }
    const float denom = static_cast<float>(std::min(guides.size(), reference.size()));
    return denom > 1e-3f
        ? clampValue(static_cast<float>(match_count) / denom, 0.0f, 1.0f)
        : 0.0f;
}







struct BoundaryFinalSplitDecision {
    bool final_split = false;
    bool structural_ok = false;
    float width_m = 0.0f;
    std::string width_source = "none";
    std::string reason = "boundary_gate_rejected";
    std::string reject_reason = "none";
};

static bool candidateSourceContains(const BoundaryPairCandidate& candidate,
                                    const std::string& token) {
    return candidate.source.find(token) != std::string::npos;
}

static bool isStrongFtOpeningBranchCandidate(const BoundaryPairCandidate& candidate) {
    if (candidate.provider_type != "physical_provider_v2") {
        return false;
    }
    if (!candidateSourceContains(candidate, "pair_type=F+T") ||
        !candidateSourceContains(candidate, "pair_semantic=opening")) {
        return false;
    }
    if (candidate.transverse_role != "opening_branch" || !candidate.ft_opening_candidate) {
        return false;
    }
    if (candidate.closing_suspect || candidate.strong_closing_veto || candidate.closing_like) {
        return false;
    }
    if (!(candidate.transverse_near_ok || candidate.transverse_near_cov >= 1)) {
        return false;
    }
    if (candidate.transverse_opening_branch_score <=
        candidate.transverse_merge_band_score + 0.08f) {
        return false;
    }
    if (candidate.pair_forward_side_support_score < 0.75f) {
        return false;
    }
    if (candidate.pair_fake_forward_penalty > 0.25f) {
        return false;
    }
    if (candidate.transverse_lat_span_m > 1.00f) {
        return false;
    }
    if (candidate.transverse_guide_count >= 7 &&
        !candidate.transverse_near_ok &&
        candidate.transverse_near_cov == 0) {
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Phase 5+: demo-aligned multi-candidate frame-level semantic arbitration
// ---------------------------------------------------------------------------

static std::string candidateSourceValue(const BoundaryPairCandidate& candidate,
                                        const std::string& key) {
    const std::string needle = key + "=";
    const size_t begin = candidate.source.find(needle);
    if (begin == std::string::npos) {
        return "none";
    }
    const size_t value_begin = begin + needle.size();
    const size_t end = candidate.source.find(':', value_begin);
    return candidate.source.substr(value_begin,
                                   end == std::string::npos
                                       ? std::string::npos
                                       : end - value_begin);
}

static bool candidateSemanticContains(const BoundaryPairCandidate& candidate,
                                      const std::string& semantic) {
    return candidateSourceContains(candidate, "pair_semantic=" + semantic);
}

static bool candidateHasNearSupport(const BoundaryPairCandidate& candidate) {
    return candidate.left_near_root_ok ||
           candidate.right_near_root_ok ||
           candidate.transverse_near_ok ||
           candidate.transverse_near_cov >= 1;
}

static void appendReasonToken(std::string* reason, const std::string& token) {
    if (reason == nullptr || token.empty()) {
        return;
    }
    if (*reason == "none" || reason->empty()) {
        *reason = token;
    } else {
        *reason += "|" + token;
    }
}

static bool computeGuideRegion(const std::vector<ArrowGuide>& guides,
                               float* fwd_min_m,
                               float* fwd_max_m,
                               float* lat_min_m,
                               float* lat_max_m) {
    if (guides.empty()) {
        return false;
    }
    float fwd_min = std::numeric_limits<float>::infinity();
    float fwd_max = -std::numeric_limits<float>::infinity();
    float lat_min = std::numeric_limits<float>::infinity();
    float lat_max = -std::numeric_limits<float>::infinity();
    for (const auto& guide : guides) {
        const float fwd = guidePlanForward(guide) / kWorldPlanScale;
        const float lat = guidePlanLateral(guide) / kWorldPlanScale;
        if (!isFiniteFloat(fwd) || !isFiniteFloat(lat)) {
            continue;
        }
        fwd_min = std::min(fwd_min, fwd);
        fwd_max = std::max(fwd_max, fwd);
        lat_min = std::min(lat_min, lat);
        lat_max = std::max(lat_max, lat);
    }
    if (!std::isfinite(fwd_min) || !std::isfinite(fwd_max) ||
        !std::isfinite(lat_min) || !std::isfinite(lat_max)) {
        return false;
    }
    if (fwd_min_m != nullptr) *fwd_min_m = fwd_min;
    if (fwd_max_m != nullptr) *fwd_max_m = fwd_max;
    if (lat_min_m != nullptr) *lat_min_m = lat_min;
    if (lat_max_m != nullptr) *lat_max_m = lat_max;
    return true;
}

static bool estimateOpeningBranchRegion(const BoundaryPairCandidate& candidate,
                                        float* fwd_min_m,
                                        float* fwd_max_m,
                                        float* lat_min_m,
                                        float* lat_max_m) {
    const auto& a = candidate.left_branch;
    const auto& b = candidate.right_branch;
    if (a.empty() && b.empty()) {
        return false;
    }
    const auto* branch = &b;
    if (b.empty()) {
        branch = &a;
    } else if (!a.empty()) {
        const bool a_shorter = a.size() < b.size();
        const bool b_shorter = b.size() < a.size();
        if (a_shorter) {
            branch = &a;
        } else if (!b_shorter) {
            float a_lat_min = 0.0f;
            float a_lat_max = 0.0f;
            float b_lat_min = 0.0f;
            float b_lat_max = 0.0f;
            computeGuideRegion(a, nullptr, nullptr, &a_lat_min, &a_lat_max);
            computeGuideRegion(b, nullptr, nullptr, &b_lat_min, &b_lat_max);
            const float a_mag = std::max(std::abs(a_lat_min), std::abs(a_lat_max));
            const float b_mag = std::max(std::abs(b_lat_min), std::abs(b_lat_max));
            branch = a_mag > b_mag ? &a : &b;
        }
    }
    return computeGuideRegion(*branch, fwd_min_m, fwd_max_m, lat_min_m, lat_max_m);
}

static bool rangesOverlapOrNear(float a_min,
                                float a_max,
                                float b_min,
                                float b_max,
                                float slack_m) {
    return !(a_max < b_min - slack_m || b_max < a_min - slack_m);
}

struct FtOpeningBranchSupportInfo {
    bool found = false;
    std::string reason = "none";
    int candidate_rank = -1;
    std::string t_ids;
    std::string t_fwd;
    std::string t_lat;
    float t_lat_sp = 0.0f;
    int t_n = 0;
    bool t_near_ok = false;
    int t_near_cov = 0;
    float open_branch_score = 0.0f;
    float merge_band_score = 0.0f;
};

static FtOpeningBranchSupportInfo
findSupportingFtOpeningBranchForSelectedOpening(
    const BoundaryPairCandidate& selected,
    const std::vector<BoundaryPairCandidate>& candidates) {
    FtOpeningBranchSupportInfo support;
    if (!candidateSourceContains(selected, "pair_semantic=opening")) {
        support.reason = "selected_not_opening";
        return support;
    }

    float selected_fwd_min = 0.0f;
    float selected_fwd_max = 0.0f;
    float selected_lat_min = 0.0f;
    float selected_lat_max = 0.0f;
    if (!estimateOpeningBranchRegion(selected,
                                     &selected_fwd_min,
                                     &selected_fwd_max,
                                     &selected_lat_min,
                                     &selected_lat_max)) {
        support.reason = "selected_region_unavailable";
        return support;
    }

    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto& candidate = candidates[i];
        if (!isStrongFtOpeningBranchCandidate(candidate)) {
            continue;
        }
        if (!rangesOverlapOrNear(selected_fwd_min,
                                 selected_fwd_max,
                                 candidate.transverse_near_forward_m,
                                 candidate.transverse_far_forward_m,
                                 0.18f)) {
            continue;
        }
        if (!rangesOverlapOrNear(selected_lat_min,
                                 selected_lat_max,
                                 candidate.transverse_lat_min_m,
                                 candidate.transverse_lat_max_m,
                                 0.18f)) {
            continue;
        }
        if (std::abs(candidate.provider_overlap_width_m -
                     selected.provider_overlap_width_m) > 0.22f) {
            continue;
        }
        support.found = true;
        support.reason = "same_side_ft_opening_branch_support";
        support.candidate_rank = static_cast<int>(i);
        support.t_ids = candidate.transverse_t_ids;
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2)
                << candidate.transverse_near_forward_m << ","
                << candidate.transverse_far_forward_m;
            support.t_fwd = oss.str();
        }
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2)
                << candidate.transverse_lat_min_m << ","
                << candidate.transverse_lat_max_m;
            support.t_lat = oss.str();
        }
        support.t_lat_sp = candidate.transverse_lat_span_m;
        support.t_n = candidate.transverse_guide_count;
        support.t_near_ok = candidate.transverse_near_ok;
        support.t_near_cov = candidate.transverse_near_cov;
        support.open_branch_score =
            candidate.transverse_opening_branch_score;
        support.merge_band_score = candidate.transverse_merge_band_score;
        return support;
    }

    support.reason = "no_matching_ft_opening_branch_support";
    return support;
}

static bool candidateRegionsCompete(const BoundaryPairCandidate& a,
                                    const BoundaryPairCandidate& b) {
    float af0 = 0.0f, af1 = 0.0f, al0 = 0.0f, al1 = 0.0f;
    float bf0 = 0.0f, bf1 = 0.0f, bl0 = 0.0f, bl1 = 0.0f;
    const bool a_ok = computeGuideRegion(a.left_branch, &af0, &af1, &al0, &al1) ||
                      computeGuideRegion(a.right_branch, &af0, &af1, &al0, &al1);
    if (!a_ok) {
        return false;
    }
    if (!b.left_branch.empty() && !b.right_branch.empty()) {
        float lf0 = 0.0f, lf1 = 0.0f, ll0 = 0.0f, ll1 = 0.0f;
        float rf0 = 0.0f, rf1 = 0.0f, rl0 = 0.0f, rl1 = 0.0f;
        if (computeGuideRegion(b.left_branch, &lf0, &lf1, &ll0, &ll1) &&
            computeGuideRegion(b.right_branch, &rf0, &rf1, &rl0, &rl1)) {
            bf0 = std::min(lf0, rf0);
            bf1 = std::max(lf1, rf1);
            bl0 = std::min(ll0, rl0);
            bl1 = std::max(ll1, rl1);
        } else {
            return false;
        }
    } else if (!computeGuideRegion(b.left_branch.empty() ? b.right_branch : b.left_branch,
                                   &bf0, &bf1, &bl0, &bl1)) {
        return false;
    }
    return rangesOverlapOrNear(af0, af1, bf0, bf1, 0.22f) &&
           rangesOverlapOrNear(al0, al1, bl0, bl1, 0.28f);
}

static std::string makeOpeningRegionKey(const BoundaryPairCandidate& candidate) {
    float fwd_min = 0.0f;
    float fwd_max = 0.0f;
    float lat_min = 0.0f;
    float lat_max = 0.0f;
    if (!estimateOpeningBranchRegion(candidate, &fwd_min, &fwd_max, &lat_min, &lat_max)) {
        return "none";
    }
    const float lat_mid = 0.5f * (lat_min + lat_max);
    const float fwd_mid = 0.5f * (fwd_min + fwd_max);
    const char side = lat_mid < 0.0f ? 'L' : 'R';
    const int lat_bucket = static_cast<int>(std::floor((lat_mid + 2.0f) / 0.22f));
    const int fwd_bucket = static_cast<int>(std::floor(fwd_mid / 0.22f));
    std::ostringstream oss;
    oss << side << ":f" << fwd_bucket << ":l" << lat_bucket;
    return oss.str();
}

static bool openingCandidatesSupportSameRegion(const BoundaryPairCandidate& a,
                                               const BoundaryPairCandidate& b) {
    return candidateRegionsCompete(a, b);
}

static float computeStableSplitEvidence(const BoundaryPairCandidate& candidate) {
    if (!candidate.stable_split_like) {
        return 0.0f;
    }
    const bool width_ok =
        candidate.provider_overlap_width_m >= 0.12f &&
        candidate.provider_overlap_width_m <= 0.60f;
    const bool support_ok =
        candidate.pair_coverage_ratio >= 0.60f &&
        candidate.pair_forward_side_support_score >= 0.55f;
    const bool not_closing =
        !candidate.strong_closing_veto &&
        candidate.closing_score < 0.68f &&
        candidate.width_delta_m > -0.18f;
    if (!(width_ok && support_ok && not_closing)) {
        return 0.0f;
    }
    return clampValue(0.36f * candidate.stable_split_score +
                          0.28f * candidate.pair_coverage_ratio +
                          0.22f * candidate.pair_forward_side_support_score +
                          0.14f * (1.0f - candidate.pair_fake_forward_penalty),
                      0.0f,
                      1.0f);
}

static float computeOpeningEvidence(const BoundaryPairCandidate& candidate,
                                    bool* short_opening) {
    if (short_opening != nullptr) {
        *short_opening = false;
    }
    const bool semantic_opening = candidateSemanticContains(candidate, "opening") ||
                                  candidate.opening_like ||
                                  candidate.transverse_role == "opening_branch";
    if (!semantic_opening) {
        return 0.0f;
    }
    const bool ft_branch = candidate.transverse_role == "opening_branch" &&
                           candidate.ft_opening_candidate;
    const bool fd_opening = candidateSourceContains(candidate, "pair_type=F+D") &&
                            candidateSemanticContains(candidate, "opening");
    const bool near_support =
        candidate.transverse_near_ok ||
        candidate.transverse_near_cov >= 1 ||
        candidate.left_near_root_ok ||
        candidate.right_near_root_ok;
    const bool side_support = candidate.pair_forward_side_support_score >= 0.70f;
    const bool fake_ok = candidate.pair_fake_forward_penalty <= 0.30f;
    const bool opening_beats_merge =
        fd_opening ||
        candidate.transverse_opening_branch_score >
            candidate.transverse_merge_band_score + 0.08f ||
        candidate.opening_score > candidate.closing_score + 0.10f;
    const bool closing_counter_low =
        candidate.closing_score < 0.82f ||
        (candidate.width_delta_m > 0.08f && candidate.far_width_ratio > 0.85f);
    if (!(near_support && side_support && fake_ok && opening_beats_merge &&
          closing_counter_low)) {
        return 0.0f;
    }
    const bool short_branch =
        ft_branch &&
        candidate.transverse_guide_count >= 3 &&
        candidate.transverse_far_forward_m - candidate.transverse_near_forward_m <= 0.24f &&
        candidate.transverse_lat_span_m >= 0.26f;
    if (short_opening != nullptr) {
        *short_opening = short_branch;
    }
    return clampValue(0.30f * candidate.opening_score +
                          0.24f * candidate.transverse_opening_branch_score +
                          0.20f * candidate.pair_forward_side_support_score +
                          0.14f * candidate.pair_coverage_ratio +
                          0.12f * (1.0f - candidate.pair_fake_forward_penalty) +
                          (short_branch ? 0.08f : 0.0f),
                      0.0f,
                      1.0f);
}

static float computeClosingEvidence(const BoundaryPairCandidate& candidate) {
    const bool semantic_closing =
        candidateSemanticContains(candidate, "closing") ||
        candidate.closing_like ||
        candidate.closing_suspect ||
        candidate.strong_closing_veto;
    const bool closing_shape =
        candidate.closing_score >= 0.72f &&
        candidate.pair_coverage_ratio >= 0.68f &&
        (candidate.width_delta_m <= -0.08f ||
         candidate.far_width_ratio <= 0.45f ||
         candidate.provider_far_width_m <= 0.08f) &&
        candidate.pair_forward_side_support_score >= 0.58f;
    if (!(semantic_closing || closing_shape)) {
        return 0.0f;
    }
    return clampValue(0.34f * candidate.closing_score +
                          0.26f * candidate.pair_coverage_ratio +
                          0.18f * candidate.pair_forward_side_support_score +
                          0.12f * clampValue(-candidate.width_delta_m / 0.30f, 0.0f, 1.0f) +
                          0.10f * (candidate.strong_closing_veto ? 1.0f : 0.0f),
                      0.0f,
                      1.0f);
}

static float computeMergeEvidence(const BoundaryPairCandidate& candidate) {
    const bool semantic_merge = candidateSemanticContains(candidate, "merge") ||
                                candidate.transverse_role == "far_merge_band";
    const bool merge_band =
        candidate.transverse_merge_band_score >= 0.52f &&
        candidate.transverse_opening_branch_score <
            candidate.transverse_merge_band_score + 0.04f &&
        candidate.pair_coverage_ratio >= 0.55f &&
        (candidate.pair_fake_forward_penalty >= 0.20f ||
         candidate.pair_unexplained_transverse_risk >= 0.35f ||
         candidate.width_delta_m <= 0.0f);
    if (!(semantic_merge || merge_band)) {
        return 0.0f;
    }
    return clampValue(0.34f * candidate.transverse_merge_band_score +
                          0.22f * candidate.closing_score +
                          0.18f * candidate.pair_coverage_ratio +
                          0.14f * candidate.pair_unexplained_transverse_risk +
                          0.12f * candidate.pair_fake_forward_penalty,
                      0.0f,
                      1.0f);
}

static float computeCrossingRiskEvidence(const BoundaryPairCandidate& candidate) {
    const bool semantic_crossing = candidateSemanticContains(candidate, "crossing");
    const bool crossing_shape =
        candidate.pair_unexplained_transverse_risk >= 0.58f &&
        candidate.pair_coverage_ratio >= 0.55f &&
        candidate.transverse_merge_band_score >= 0.42f &&
        candidate.transverse_opening_branch_score <=
            candidate.transverse_merge_band_score + 0.08f;
    if (!(semantic_crossing || crossing_shape)) {
        return 0.0f;
    }
    return clampValue(0.36f * candidate.pair_unexplained_transverse_risk +
                          0.26f * candidate.transverse_merge_band_score +
                          0.20f * candidate.pair_coverage_ratio +
                          0.18f * candidate.pair_fake_forward_penalty,
                      0.0f,
                      1.0f);
}

static float computeOpeningConsistencyForCandidate(const BoundaryPairCandidate& candidate,
                                                   float opening_evidence,
                                                   std::string* reason) {
    if (opening_evidence <= 0.0f) {
        return 0.0f;
    }
    float score = 0.46f * opening_evidence;
    if (candidate.width_delta_m > 0.0f) {
        score += 0.13f * clampValue(candidate.width_delta_m / 0.40f, 0.0f, 1.0f);
        appendReasonToken(reason, "width_expands");
    }
    if (candidate.far_width_ratio > 1.0f) {
        score += 0.12f * clampValue((candidate.far_width_ratio - 1.0f) / 2.0f, 0.0f, 1.0f);
        appendReasonToken(reason, "far_width_expands");
    }
    if (candidate.opening_score > candidate.closing_score) {
        score += 0.12f * clampValue((candidate.opening_score - candidate.closing_score) / 0.50f,
                                    0.0f,
                                    1.0f);
        appendReasonToken(reason, "opening_beats_closing");
    }
    if (candidate.transverse_opening_branch_score >
        candidate.transverse_merge_band_score + 0.08f) {
        score += 0.11f * clampValue((candidate.transverse_opening_branch_score -
                                     candidate.transverse_merge_band_score) / 0.55f,
                                    0.0f,
                                    1.0f);
        appendReasonToken(reason, "open_branch_beats_merge");
    }
    if (candidate.pair_forward_side_support_score >= 0.70f) {
        score += 0.12f * candidate.pair_forward_side_support_score;
        appendReasonToken(reason, "forward_side_support");
    }
    if (candidate.pair_fake_forward_penalty <= 0.25f) {
        score += 0.07f * (1.0f - clampValue(candidate.pair_fake_forward_penalty / 0.25f,
                                            0.0f,
                                            1.0f));
        appendReasonToken(reason, "fake_penalty_low");
    }
    if (candidateHasNearSupport(candidate)) {
        score += 0.08f;
        appendReasonToken(reason, "near_support");
    }
    if (candidate.pair_coverage_ratio >= 0.65f) {
        score += 0.07f * candidate.pair_coverage_ratio;
        appendReasonToken(reason, "coverage");
    }
    if (candidate.pair_unexplained_transverse_risk >= 0.55f) {
        score -= 0.05f * clampValue((candidate.pair_unexplained_transverse_risk - 0.55f) / 0.45f,
                                    0.0f,
                                    1.0f);
        appendReasonToken(reason, "risk_penalty");
    }
    return clampValue(score, 0.0f, 1.0f);
}

static float computeClosingConsistencyForCandidate(const BoundaryPairCandidate& candidate,
                                                   float closing_evidence,
                                                   float merge_evidence,
                                                   float crossing_evidence,
                                                   std::string* reason) {
    const float base = std::max(closing_evidence, std::max(merge_evidence, crossing_evidence));
    if (base <= 0.0f) {
        return 0.0f;
    }
    float score = 0.48f * base;
    if (candidate.width_delta_m <= 0.0f) {
        score += 0.13f * clampValue((-candidate.width_delta_m) / 0.35f, 0.0f, 1.0f);
        appendReasonToken(reason, "width_not_expanding");
    }
    if (candidate.far_width_ratio <= 0.85f) {
        score += 0.13f * clampValue((0.85f - candidate.far_width_ratio) / 0.85f,
                                    0.0f,
                                    1.0f);
        appendReasonToken(reason, "far_width_low");
    }
    if (candidate.closing_score >= 0.65f) {
        score += 0.13f * candidate.closing_score;
        appendReasonToken(reason, "closing_score_high");
    }
    if (candidate.transverse_merge_band_score >=
        candidate.transverse_opening_branch_score - 0.04f) {
        score += 0.10f * candidate.transverse_merge_band_score;
        appendReasonToken(reason, "merge_band_competes");
    }
    if (candidate.pair_forward_side_support_score < 0.58f) {
        score += 0.08f;
        appendReasonToken(reason, "support_weak");
    }
    if (candidate.pair_fake_forward_penalty >= 0.22f) {
        score += 0.09f * clampValue(candidate.pair_fake_forward_penalty / 0.45f,
                                    0.0f,
                                    1.0f);
        appendReasonToken(reason, "fake_penalty_high");
    }
    if (!candidateHasNearSupport(candidate)) {
        score += 0.06f;
        appendReasonToken(reason, "near_support_weak");
    }
    if (candidate.transverse_role == "far_merge_band" ||
        candidate.transverse_role == "transverse_merge_band") {
        score += 0.08f;
        appendReasonToken(reason, "merge_band_role");
    }
    if (candidate.strong_closing_veto) {
        score += 0.08f;
        appendReasonToken(reason, "strong_closing_veto");
    }
    return clampValue(score, 0.0f, 1.0f);
}

static FrameClosingVetoInfo collectFrameClosingVetoInfo(
    const std::vector<BoundaryPairCandidate>& candidates) {
    FrameClosingVetoInfo info;
    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto& candidate = candidates[i];
        if (candidate.provider_type != "physical_provider_v2") {
            continue;
        }
        if (candidate.strong_closing_veto) {
            info.has_strong_closing_veto = true;
        }
        if (candidate.closing_score > info.best_closing_score) {
            info.best_closing_score = candidate.closing_score;
            info.best_closing_candidate_rank = static_cast<int>(i);
            info.best_closing_pair_type = candidateSourceValue(candidate, "pair_type");
        }
    }
    return info;
}

static FrameSemanticDecision arbitrateFrameSemanticEvidence(
    const std::vector<BoundaryPairCandidate>& candidates,
    const BoundaryPairCandidate* selected_candidate,
    const BoundaryTrackManager& manager) {
    (void)manager;
    FrameSemanticDecision decision;
    int selected_idx = -1;
    if (selected_candidate != nullptr) {
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (&candidates[i] == selected_candidate) {
                selected_idx = static_cast<int>(i);
                break;
            }
        }
    }

    bool best_opening_short = false;
    std::vector<float> opening_consistency_by_idx(candidates.size(), 0.0f);
    std::vector<float> closing_consistency_by_idx(candidates.size(), 0.0f);
    std::vector<std::string> opening_reason_by_idx(candidates.size(), "none");
    std::vector<std::string> closing_reason_by_idx(candidates.size(), "none");
    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto& candidate = candidates[i];
        bool short_opening = false;
        const float opening_score = computeOpeningEvidence(candidate, &short_opening);
        const float closing_score = computeClosingEvidence(candidate);
        const float merge_score = computeMergeEvidence(candidate);
        const float stable_score = computeStableSplitEvidence(candidate);
        const float crossing_score = computeCrossingRiskEvidence(candidate);
        opening_consistency_by_idx[i] =
            computeOpeningConsistencyForCandidate(candidate,
                                                  opening_score,
                                                  &opening_reason_by_idx[i]);
        closing_consistency_by_idx[i] =
            computeClosingConsistencyForCandidate(candidate,
                                                  closing_score,
                                                  merge_score,
                                                  crossing_score,
                                                  &closing_reason_by_idx[i]);

        if (opening_score > decision.best_opening_score) {
            decision.best_opening_score = opening_score;
            decision.best_opening_idx = static_cast<int>(i);
            best_opening_short = short_opening;
        }
        if (closing_score > decision.best_closing_score) {
            decision.best_closing_score = closing_score;
            decision.best_closing_idx = static_cast<int>(i);
        }
        if (merge_score > decision.best_merge_score || crossing_score > decision.best_merge_score) {
            decision.best_merge_score = std::max(merge_score, crossing_score);
            decision.best_merge_idx = static_cast<int>(i);
        }
        if (stable_score > decision.best_stable_score) {
            decision.best_stable_score = stable_score;
            decision.best_stable_idx = static_cast<int>(i);
        }
        if (opening_consistency_by_idx[i] > decision.opening_consistency_score) {
            decision.opening_consistency_score = opening_consistency_by_idx[i];
            decision.opening_consistency_reason = opening_reason_by_idx[i];
        }
        if (closing_consistency_by_idx[i] > decision.closing_consistency_score) {
            decision.closing_consistency_score = closing_consistency_by_idx[i];
            decision.closing_consistency_reason = closing_reason_by_idx[i];
        }
    }

    const BoundaryPairCandidate* best_opening =
        decision.best_opening_idx >= 0
            ? &candidates[static_cast<size_t>(decision.best_opening_idx)]
            : nullptr;
    const BoundaryPairCandidate* best_closing =
        decision.best_closing_idx >= 0
            ? &candidates[static_cast<size_t>(decision.best_closing_idx)]
            : nullptr;
    const BoundaryPairCandidate* best_merge =
        decision.best_merge_idx >= 0
            ? &candidates[static_cast<size_t>(decision.best_merge_idx)]
            : nullptr;

    if (decision.best_opening_idx >= 0) {
        decision.confirmation_region_key =
            makeOpeningRegionKey(candidates[static_cast<size_t>(decision.best_opening_idx)]);
    }
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (static_cast<int>(i) == decision.best_opening_idx ||
            opening_consistency_by_idx[i] < 0.48f ||
            best_opening == nullptr) {
            continue;
        }
        const auto& support = candidates[i];
        const bool fd_ft_mutual_support =
            ((candidateSourceContains(*best_opening, "pair_type=F+D") &&
              candidateSourceContains(support, "pair_type=F+T")) ||
             (candidateSourceContains(*best_opening, "pair_type=F+T") &&
              candidateSourceContains(support, "pair_type=F+D"))) &&
            candidateSemanticContains(support, "opening") &&
            openingCandidatesSupportSameRegion(*best_opening, support);
        if (fd_ft_mutual_support) {
            decision.opening_consistency_score =
                clampValue(decision.opening_consistency_score + 0.14f, 0.0f, 1.0f);
            appendReasonToken(&decision.opening_consistency_reason,
                              "fd_ft_same_region_support");
            break;
        }
    }
    decision.opening_vs_closing_margin =
        decision.opening_consistency_score - decision.closing_consistency_score;

    const bool opening_selected =
        selected_idx >= 0 && selected_idx == decision.best_opening_idx;
    const bool closing_competes =
        best_opening != nullptr && best_closing != nullptr &&
        candidateRegionsCompete(*best_opening, *best_closing);
    const bool merge_competes =
        best_opening != nullptr && best_merge != nullptr &&
        candidateRegionsCompete(*best_opening, *best_merge);
    const bool opening_has_counter_evidence =
        best_opening != nullptr &&
        (best_opening->far_width_ratio <= 0.20f ||
         best_opening->width_delta_m < -0.12f ||
         best_opening->closing_score >= 0.82f);
    const bool frame_opening_strong =
        best_opening != nullptr &&
        decision.opening_consistency_score >= 0.68f &&
        decision.opening_vs_closing_margin >= 0.08f;
    const bool frame_opening_clear =
        best_opening != nullptr &&
        decision.opening_consistency_score >= 0.58f &&
        decision.opening_vs_closing_margin >= 0.03f;
    const bool frame_closing_strong =
        decision.closing_consistency_score >= 0.66f &&
        decision.opening_vs_closing_margin <= -0.08f;

    if (frame_closing_strong &&
        decision.best_merge_score >= 0.64f &&
        (best_opening == nullptr || merge_competes ||
         decision.closing_consistency_score > decision.opening_consistency_score + 0.08f)) {
        decision.semantic_suppressed = true;
        decision.semantic_mode = "merge_or_crossing";
        decision.semantic_reason = "frame_semantic_merge_suppressed";
        decision.suppressing_idx = decision.best_merge_idx;
        decision.selected_semantic_idx = selected_idx;
        decision.competing_semantic_conflict = true;
        decision.competing_conflict_reason = "strong_far_merge_or_crossing";
        return decision;
    }
    if (frame_closing_strong &&
        decision.best_closing_score >= 0.68f &&
        best_opening != nullptr &&
        closing_competes &&
        (decision.closing_consistency_score >= decision.opening_consistency_score + 0.08f ||
         opening_has_counter_evidence)) {
        decision.semantic_suppressed = true;
        decision.semantic_mode = "closing_or_merge";
        decision.semantic_reason = "competing_closing_or_merge_over_opening";
        decision.suppressing_idx = decision.best_closing_idx;
        decision.selected_semantic_idx = decision.best_opening_idx;
        decision.competing_semantic_conflict = true;
        decision.competing_conflict_reason = "same_region_closing_beats_opening";
        return decision;
    }
    if (decision.best_stable_score >= 0.56f &&
        decision.best_closing_score < 0.70f &&
        decision.best_merge_score < 0.62f) {
        decision.semantic_ok = true;
        decision.semantic_mode = "stable_split";
        decision.semantic_reason = "stable_split_confirmed";
        decision.selected_semantic_idx = decision.best_stable_idx;
        return decision;
    }
    if (frame_opening_clear &&
        (decision.best_merge_score < 0.72f || frame_opening_strong) &&
        (decision.best_closing_score < 0.82f || frame_opening_strong)) {
        decision.semantic_pending = true;
        decision.semantic_mode = best_opening_short ? "short_opening" : "opening";
        decision.semantic_reason =
            frame_opening_strong
                ? "frame_opening_consistency_strong"
                : (opening_selected ? "frame_opening_candidate"
                                    : "frame_opening_candidate_nonselected");
        decision.selected_semantic_idx = decision.best_opening_idx;
        return decision;
    }
    if (selected_idx >= 0 &&
        candidates[static_cast<size_t>(selected_idx)].opening_like) {
        decision.semantic_pending = true;
        decision.semantic_mode = "uncertain_opening";
        decision.semantic_reason = "opening_uncertain_pending";
        decision.selected_semantic_idx = selected_idx;
        return decision;
    }
    decision.semantic_reason = "frame_semantic_uncertain";
    decision.selected_semantic_idx = selected_idx;
    return decision;
}

static void applyFrameSemanticDecisionToManager(BoundaryTrackManager* manager,
                                                const FrameSemanticDecision& decision) {
    if (manager == nullptr) {
        return;
    }
    manager->frame_semantic_ok = decision.semantic_ok;
    manager->frame_semantic_pending = decision.semantic_pending;
    manager->frame_semantic_suppressed = decision.semantic_suppressed;
    manager->frame_semantic_mode = decision.semantic_mode;
    manager->frame_semantic_reason = decision.semantic_reason;
    manager->semantic_selected_idx = decision.selected_semantic_idx;
    manager->semantic_suppressing_idx = decision.suppressing_idx;
    manager->best_opening_idx = decision.best_opening_idx;
    manager->best_opening_score = decision.best_opening_score;
    manager->best_closing_idx = decision.best_closing_idx;
    manager->best_closing_score = decision.best_closing_score;
    manager->best_merge_idx = decision.best_merge_idx;
    manager->best_merge_score = decision.best_merge_score;
    manager->best_stable_idx = decision.best_stable_idx;
    manager->best_stable_score = decision.best_stable_score;
    manager->opening_consistency_score = decision.opening_consistency_score;
    manager->closing_consistency_score = decision.closing_consistency_score;
    manager->opening_consistency_reason = decision.opening_consistency_reason;
    manager->closing_consistency_reason = decision.closing_consistency_reason;
    manager->opening_vs_closing_margin = decision.opening_vs_closing_margin;
    manager->confirmation_region_key = decision.confirmation_region_key;
    manager->competing_semantic_conflict = decision.competing_semantic_conflict;
    manager->competing_conflict_reason = decision.competing_conflict_reason;
}

static bool isSplitConfirmationEvidenceMode(const std::string& mode) {
    return mode == "opening_like" ||
           mode == "opening_with_ft_support" ||
           mode == "ft_opening_branch" ||
           mode == "short_opening" ||
           mode == "stable_split_like";
}

static void clearPhysicalSplitConfirmation(BoundaryTrackManager* manager,
                                           const std::string& reject_reason) {
    if (manager == nullptr) {
        return;
    }
    manager->split_confirm_mode = "none";
    manager->split_confirm_count = 0;
    manager->split_confirmed = false;
    manager->split_confirm_reject_reason = reject_reason;
    manager->final_semantic_reason = reject_reason;
    manager->visible_split_output_active = false;
    manager->visible_split_maintain_age = 0;
    manager->visible_split_provider_gap_frames = 0;
}

static bool hasActiveVisibleSplitContext(const BoundaryTrackManager* manager) {
    return manager != nullptr &&
           (manager->visible_split_output_active ||
            manager->visible_split_maintain_age > 0);
}

static bool boundaryCandidateIsUsableOpening(const BoundaryPairCandidate& candidate) {
    if (candidate.provider_type != "physical_provider_v2") {
        return false;
    }
    if (candidate.strong_closing_veto ||
        candidate.closing_like ||
        candidate.closing_suspect) {
        return false;
    }
    const float width_m = candidate.provider_overlap_width_m;
    if (!isFiniteFloat(width_m) || width_m < 0.12f || width_m > 0.75f) {
        return false;
    }
    if (candidate.left_point_count < 2 || candidate.right_point_count < 2) {
        return false;
    }
    return candidate.opening_like || candidate.stable_split_like;
}

// ---------------------------------------------------------------------------
// Phase 5+ A2: boundary candidate health evaluation → pick best, not first
// ---------------------------------------------------------------------------

static constexpr size_t kBoundaryTrackingTopK = 3;
static constexpr float kBoundaryHealthyWidthMinM = 0.10f;
static constexpr float kBoundaryDefaultWidthMaxM = 0.60f;
static constexpr float kBoundaryOpeningWidthMaxM = 0.65f;

struct BoundaryPairHealthEvaluation {
    bool healthy = false;
    bool rejected = true;
    float health_score = -1e9f;
    std::string reject_reason = "not_evaluated";
    float width_m = 0.0f;
    std::string width_source = "none";
    float left_span_m = 0.0f;
    float right_span_m = 0.0f;
    int selected_rank = -1;
    bool normalization_ok = false;
    std::string normalization_reject_reason = "none";
    float root_width_m = 0.0f;
    float normalized_width_m = 0.0f;
    float provider_overlap_width_m = 0.0f;
    float provider_near_width_m = 0.0f;
    float provider_mid_width_m = 0.0f;
    float provider_far_width_m = 0.0f;
    float width_delta_m = 0.0f;
    float far_width_ratio = 0.0f;
    float width_slope_m_per_m = 0.0f;
    float left_lat_slope = 0.0f;
    float right_lat_slope = 0.0f;
    float relative_lat_slope = 0.0f;
    float stable_split_score = 0.0f;
    float opening_score = 0.0f;
    float closing_score = 0.0f;
    bool stable_split_like = false;
    bool opening_like = false;
    bool closing_like = false;
    bool closing_suspect = false;
    bool strong_closing_veto = false;
    float pair_coverage_ratio = 0.0f;
    float pair_unexplained_transverse_risk = 0.0f;
    float pair_semantic_rank_adjust = 0.0f;
    float pair_fake_forward_penalty = 0.0f;
    float pair_forward_side_support_score = 0.0f;
    float transverse_merge_band_score = 0.0f;
    float transverse_opening_branch_score = 0.0f;
    std::string transverse_role = "none";
    bool trigger_split_pair_unavailable = false;
    bool trigger_pair_width_out_of_range = false;
    float width_gate_min_m = 0.0f;
    float width_gate_max_m = 0.0f;
    std::string width_gate_stage = "entry";
    std::string width_gate_fail_side = "none";
};

// B-1: full demo-aligned track association score for a single side.
// Mirrors demo scoreBoundaryObservationAssociation:
//   5-component cost (lateral_residual, heading_residual, forward_gap,
//   near_reach_penalty, overlap_penalty) → converted to [0,1] score.
// When the track has no history (no guides stored), returns neutral 0.5f.
// =========================================================================
// Demo-aligned A2 selector helper functions
// =========================================================================

static BoundarySide inferBoundaryPhysicalSideFromLateral(float plan_lateral) {
    constexpr float kBoundaryMinPhysicalSideSeparation = planMeters(0.04f);
    const float canonical_lateral = canonicalBoundaryPhysicalLateral(plan_lateral);
    if (canonical_lateral > kBoundaryMinPhysicalSideSeparation) {
        return BoundarySide::Left;
    }
    if (canonical_lateral < -kBoundaryMinPhysicalSideSeparation) {
        return BoundarySide::Right;
    }
    return BoundarySide::Unknown;
}

static float computeBoundaryPairObservationConfidence(
    const BoundaryPairCandidate& candidate, BoundarySide slot_side) {
    const bool left_side = slot_side == BoundarySide::Left;
    const int point_count =
        left_side ? candidate.left_point_count : candidate.right_point_count;
    const bool root_ok =
        left_side ? candidate.left_root_ok : candidate.right_root_ok;
    const bool near_root_ok =
        left_side ? candidate.left_near_root_ok : candidate.right_near_root_ok;
    const float span_m =
        left_side ? candidate.left_span_m : candidate.right_span_m;
    const float bottom_reach =
        left_side ? candidate.left_bottom_reach_score
                  : candidate.right_bottom_reach_score;
    const float span_score = clampValue(span_m / 0.18f, 0.0f, 1.0f);
    const float point_score = clampValue(
        static_cast<float>(point_count) / 3.0f,
        0.0f,
        1.0f);
    const float width_health = 1.0f - clampValue(
        std::abs(candidate.provider_overlap_width_m - 0.32f) / 0.32f,
        0.0f,
        1.0f);
    return clampValue(
        0.16f * clampValue(candidate.provider_pair_confidence, 0.0f, 1.0f) +
            0.16f * point_score +
            0.16f * span_score +
            0.14f * clampValue(bottom_reach, 0.0f, 1.0f) +
            0.12f * (root_ok ? 1.0f : 0.0f) +
            0.10f * (near_root_ok ? 1.0f : 0.0f) +
            0.12f * clampValue(candidate.provider_overlap_ratio, 0.0f, 1.0f) +
            0.08f * (1.0f - clampValue(candidate.shared_ratio, 0.0f, 1.0f)) +
            0.06f * width_health,
        0.0f,
        1.0f);
}

static BoundaryObservation buildBoundaryObservationFromGuides(
    const std::vector<ArrowGuide>& fork_guides,
    BoundarySide observation_side,
    const std::string& source,
    float confidence) {
    BoundaryObservation obs;
    if (fork_guides.empty()) {
        return obs;
    }
    obs.valid = true;
    obs.side = observation_side;
    obs.source = source;
    obs.guides = fork_guides;
    obs.confidence = clampValue(confidence, 0.0f, 1.0f);
    obs.root_lateral = worldLateralM(fork_guides.front());
    obs.root_forward = worldForwardM(fork_guides.front());
    obs.near_forward = obs.root_forward;
    obs.far_forward = worldForwardM(fork_guides.back());
    obs.center_lateral = obs.root_lateral;
    if (fork_guides.size() >= 2) {
        const cv::Point2f dir = fork_guides[1].world_pos - fork_guides[0].world_pos;
        const float len = std::sqrt(dir.x * dir.x + dir.y * dir.y);
        if (len > 1e-6f) {
            obs.heading_deg = std::atan2(dir.x, dir.y) * 180.0f / static_cast<float>(CV_PI);
        }
    }
    return obs;
}

static BoundaryObservation buildBoundaryObservationFromBoundaryPairSlot(
    const BoundaryPairCandidate& candidate,
    BoundarySide slot_side,
    BoundarySide observation_side,
    const std::string& source) {
    const bool use_left_slot = slot_side == BoundarySide::Left;
    const std::vector<ArrowGuide>& branch =
        use_left_slot ? candidate.left_branch : candidate.right_branch;
    const int shared_prefix =
        use_left_slot ? candidate.left_shared_prefix_count : candidate.right_shared_prefix_count;
    std::vector<ArrowGuide> fork_guides = guidesAfterSharedPrefix(branch, shared_prefix);
    if (fork_guides.empty()) {
        fork_guides = branch;
    }
    std::ostringstream tagged_source;
    tagged_source << source;
    if (!candidate.provider_type.empty() && source != candidate.provider_type) {
        tagged_source << ":" << candidate.provider_type;
    }
    return buildBoundaryObservationFromGuides(
        fork_guides,
        observation_side,
        tagged_source.str(),
        computeBoundaryPairObservationConfidence(candidate, slot_side));
}

static bool normalizeBoundaryPairPhysicalSides(
    const BoundaryPairCandidate& candidate,
    BoundaryObservation* left_observation,
    BoundaryObservation* right_observation,
    float* width_m,
    std::string* reject_reason,
    BoundarySide* out_physical_left_slot = nullptr,
    BoundarySide* out_physical_right_slot = nullptr) {
    if (left_observation)  *left_observation = BoundaryObservation{};
    if (right_observation) *right_observation = BoundaryObservation{};
    if (width_m)           *width_m = 0.0f;
    if (reject_reason)     *reject_reason = "none";
    if (out_physical_left_slot)  *out_physical_left_slot = BoundarySide::Unknown;
    if (out_physical_right_slot) *out_physical_right_slot = BoundarySide::Unknown;

    const BoundaryObservation slot_left = buildBoundaryObservationFromBoundaryPairSlot(
        candidate, BoundarySide::Left, BoundarySide::Left, "candidate_slot_left");
    const BoundaryObservation slot_right = buildBoundaryObservationFromBoundaryPairSlot(
        candidate, BoundarySide::Right, BoundarySide::Right, "candidate_slot_right");
    if (!slot_left.valid || !slot_right.valid) {
        if (width_m && slot_left.valid && slot_right.valid) {
            *width_m = std::abs(slot_left.root_lateral - slot_right.root_lateral) / kWorldPlanScale;
        }
        if (reject_reason) {
            *reject_reason = !slot_left.valid && !slot_right.valid
                ? "split_pair_invalid"
                : (!slot_left.valid ? "left_branch_invalid" : "right_branch_invalid");
        }
        return false;
    }

    const float left_canonical = canonicalBoundaryPhysicalLateral(slot_left.root_lateral);
    const float right_canonical = canonicalBoundaryPhysicalLateral(slot_right.root_lateral);
    if (width_m) {
        *width_m = std::abs(slot_left.root_lateral - slot_right.root_lateral) / kWorldPlanScale;
    }
    if (!isFiniteFloat(left_canonical) || !isFiniteFloat(right_canonical)) {
        if (reject_reason) *reject_reason = "invalid_lateral";
        return false;
    }
    const bool left_slot_is_physical_left = left_canonical > right_canonical;
    const BoundarySide physical_left_slot =
        left_slot_is_physical_left ? BoundarySide::Left : BoundarySide::Right;
    const BoundarySide physical_right_slot =
        left_slot_is_physical_left ? BoundarySide::Right : BoundarySide::Left;
    if (out_physical_left_slot)  *out_physical_left_slot = physical_left_slot;
    if (out_physical_right_slot) *out_physical_right_slot = physical_right_slot;
    if (left_observation) {
        *left_observation = buildBoundaryObservationFromBoundaryPairSlot(
            candidate, physical_left_slot, BoundarySide::Left,
            candidate.provider_type == "physical_provider_v2" ? "physical_provider_v2" : "selected_boundary_left");
    }
    if (right_observation) {
        *right_observation = buildBoundaryObservationFromBoundaryPairSlot(
            candidate, physical_right_slot, BoundarySide::Right,
            candidate.provider_type == "physical_provider_v2" ? "physical_provider_v2" : "selected_boundary_right");
    }
    return true;
}

static BoundarySelectionResult buildBoundarySelectionResultFromBoundaryPair(
    const BoundaryPairCandidate& candidate, int selected_source_rank) {
    BoundarySelectionResult result;
    result.selected_source_rank = selected_source_rank;
    const BoundaryObservation slot_left = buildBoundaryObservationFromBoundaryPairSlot(
        candidate, BoundarySide::Left, BoundarySide::Left, "candidate_slot_left");
    const BoundaryObservation slot_right = buildBoundaryObservationFromBoundaryPairSlot(
        candidate, BoundarySide::Right, BoundarySide::Right, "candidate_slot_right");
    const bool left_valid = slot_left.valid;
    const bool right_valid = slot_right.valid;
    if (left_valid && right_valid) {
        result.mode = BoundarySelectionMode::SplitPair;
        result.normalized_side_ok = normalizeBoundaryPairPhysicalSides(
            candidate, &result.left_observation, &result.right_observation,
            &result.width_m, &result.reject_reason);
        result.left_active = result.left_observation.valid;
        result.right_active = result.right_observation.valid;
        result.allow_boundary_track_update =
            result.normalized_side_ok && result.left_active && result.right_active;
        if (result.normalized_side_ok) result.reject_reason = "none";
        return result;
    }
    if (left_valid || right_valid) {
        BoundaryObservation single_obs = left_valid ? slot_left : slot_right;
        const BoundarySide physical_side = inferBoundaryPhysicalSideFromLateral(single_obs.root_lateral);
        single_obs.source = physical_side == BoundarySide::Right ? "selected_boundary_right" : "selected_boundary_left";
        single_obs.side = physical_side;
        result.width_m = 0.0f;
        result.normalized_side_ok = false;
        result.allow_boundary_track_update = false;
        result.reject_reason = (physical_side == BoundarySide::Unknown)
            ? "single_branch_side_ambiguous" : "split_pair_incomplete";
        if (physical_side == BoundarySide::Left) {
            result.mode = BoundarySelectionMode::LeftOnly;
            result.left_observation = std::move(single_obs);
            result.left_active = result.left_observation.valid;
        } else if (physical_side == BoundarySide::Right) {
            result.mode = BoundarySelectionMode::RightOnly;
            result.right_observation = std::move(single_obs);
            result.right_active = result.right_observation.valid;
        }
        return result;
    }
    result.mode = BoundarySelectionMode::NoReliableBoundary;
    result.reject_reason = "split_pair_invalid";
    return result;
}

static float boundaryPairBranchSpanMeters(const BoundaryPairCandidate& candidate,
                                           BoundarySide slot_side) {
    return (slot_side == BoundarySide::Left ? candidate.left_y_span : candidate.right_y_span)
           / kWorldPlanScale;
}
static bool boundaryPairBranchNearRootOk(const BoundaryPairCandidate& candidate,
                                          BoundarySide slot_side) {
    return slot_side == BoundarySide::Left ? candidate.left_near_root_ok : candidate.right_near_root_ok;
}
static bool boundaryPairBranchRootOk(const BoundaryPairCandidate& candidate,
                                      BoundarySide slot_side) {
    return slot_side == BoundarySide::Left ? candidate.left_root_ok : candidate.right_root_ok;
}

// Demo-aligned: scoreBoundaryObservationAssociation
static float computeBoundaryTrackLateralAtForward(const BoundaryTrackState& track,
                                                   float target_forward,
                                                   bool* used_interpolation = nullptr) {
    if (used_interpolation) {
        *used_interpolation = false;
    }
    float lateral = track.root_lateral;
    if (!track.guides.empty() &&
        interpolateGuidePlanLateralAtForward(track.guides, target_forward, lateral)) {
        if (used_interpolation) {
            *used_interpolation = true;
        }
        return lateral;
    }
    return track.root_lateral;
}

static BoundaryAssociationScore scoreBoundaryObservationAssociation(
    const BoundaryTrackState& track,
    const BoundaryObservation& observation,
    float visible_min_forward) {
    BoundaryAssociationScore score;
    score.predicted_lateral = track.root_lateral;
    score.predicted_heading = track.predicted_heading;
    if (!observation.valid) { score.reject_reason = "no_observation"; return score; }
    if (observation.side != track.side) { score.reject_reason = "side_mismatch"; return score; }
    if (observation.guides.empty() ||
        !isFiniteFloat(observation.root_lateral) || !isFiniteFloat(observation.root_forward) ||
        observation.far_forward < observation.near_forward) {
        score.reject_reason = "invalid_geometry"; return score;
    }
    bool used_interpolation = false;
    if (!track.guides.empty()) {
        score.predicted_lateral = computeBoundaryTrackLateralAtForward(
            track, observation.root_forward, &used_interpolation);
        score.predicted_heading = isFiniteFloat(track.predicted_heading)
            ? track.predicted_heading : observation.heading_deg;
    } else {
        score.predicted_lateral = observation.root_lateral;
        score.predicted_heading = observation.heading_deg;
    }
    const float id_overlap = computeGuideIdMatchRatio(observation.guides, track.guides);
    const float loose_overlap = computeGuideMatchRatioLoose(observation.guides, track.guides, 16.0f, 16.0f);
    score.overlap = std::max(id_overlap, loose_overlap);
    const float lateral_residual = std::abs(observation.root_lateral - score.predicted_lateral);
    const float heading_residual = std::abs(
        normalizeAngleDeltaDeg(observation.heading_deg - score.predicted_heading));
    const float forward_gap = track.guides.empty()
        ? 0.0f : std::max(0.0f, observation.root_forward - track.near_forward);
    const float near_reach_penalty = std::max(0.0f,
        observation.near_forward - visible_min_forward - planMeters(0.10f));
    const float overlap_penalty = track.guides.empty()
        ? (1.0f - observation.confidence)
        : (1.0f - clampValue(score.overlap, 0.0f, 1.0f));
    score.cost =
        1.40f * (lateral_residual / planMeters(0.18f)) +
        0.95f * (heading_residual / 28.0f) +
        0.75f * (forward_gap / planMeters(0.16f)) +
        0.65f * (near_reach_penalty / planMeters(0.18f)) +
        0.80f * overlap_penalty;
    if (!track.guides.empty() &&
        near_reach_penalty > planMeters(0.18f) &&
        score.overlap < 0.25f) {
        score.reach_gap_warning = true;
    }
    score.reject_reason = "none";
    return score;
}

static void collectBoundaryObservations(
    const BoundarySelectionResult& selection_result,
    BoundaryObservation* left_observation,
    BoundaryObservation* right_observation) {
    if (left_observation)  *left_observation = BoundaryObservation{};
    if (right_observation) *right_observation = BoundaryObservation{};
    if (!selection_result.allow_boundary_track_update) return;
    if (left_observation && selection_result.left_active)
        *left_observation = selection_result.left_observation;
    if (right_observation && selection_result.right_active)
        *right_observation = selection_result.right_observation;
}

// Demo-aligned: updateBoundaryTrackState
static void decayBoundaryTrackState(BoundaryTrackState& track,
                                    const std::string& reject_reason);

static void updateBoundaryTrackState(BoundaryTrackState& track,
                                      const BoundaryObservation& observation,
                                      float visible_min_forward) {
    const BoundaryAssociationScore score =
        scoreBoundaryObservationAssociation(track, observation, visible_min_forward);
    track.selected_obs_cost = score.cost;
    track.obs_source = observation.valid ? observation.source : "none";

    if (score.reject_reason != "none") {
        decayBoundaryTrackState(track, score.reject_reason);
        return;
    }

    const float previous_confidence = track.confidence;
    track.active = true;
    track.confidence = clampValue(
        0.45f * previous_confidence +
            0.40f * observation.confidence +
            0.15f * (1.0f - clampValue(score.cost / 4.0f, 0.0f, 1.0f)),
        0.0f,
        1.0f);
    track.missed_frames = 0;
    track.root_lateral = observation.root_lateral;
    track.root_forward = observation.root_forward;
    track.center_lateral = observation.center_lateral;
    track.near_forward = observation.near_forward;
    track.far_forward = observation.far_forward;
    track.predicted_lateral = observation.root_lateral;
    track.predicted_heading = observation.heading_deg;
    track.guides = observation.guides;
    track.reject_reason = "none";
}

static void decayBoundaryTrackState(BoundaryTrackState& track,
                                    const std::string& reject_reason) {
    if (reject_reason == "no_observation" ||
        reject_reason == "same_side_observation") {
        track.selected_obs_cost = std::numeric_limits<float>::infinity();
    }
    track.reject_reason = reject_reason;
    track.missed_frames += 1;
    track.confidence *= track.active ? 0.82f : 0.72f;
    if (track.missed_frames >= 3 || track.confidence < 0.16f) {
        track.active = false;
    }
    if (track.missed_frames >= 6) {
        track.guides.clear();
        track.confidence = 0.0f;
        track.reject_reason = "stale_track";
    }
}

// Demo-aligned health evaluation with BoundarySelectionResult
static BoundaryPairHealthEvaluation evaluateBoundaryPairHealth(
    const BoundaryPairCandidate& candidate,
    const BoundarySelectionResult& selection_result,
    int selected_rank,
    const BoundaryTrackManager* manager,
    float visible_min_forward) {
    BoundaryPairHealthEvaluation evaluation;
    evaluation.selected_rank = selected_rank;
    evaluation.width_m = selection_result.width_m;
    evaluation.width_source = "selection_result";
    const bool maintain_width_gate =
        manager != nullptr && manager->visible_split_output_active;
    evaluation.width_gate_stage = maintain_width_gate ? "maintain" : "entry";
    evaluation.width_gate_min_m = kBoundaryHealthyWidthMinM;
    evaluation.provider_overlap_width_m = candidate.provider_overlap_width_m;
    evaluation.provider_near_width_m = candidate.provider_near_width_m;
    evaluation.provider_mid_width_m = candidate.provider_mid_width_m;
    evaluation.provider_far_width_m = candidate.provider_far_width_m;
    evaluation.width_delta_m = candidate.width_delta_m;
    evaluation.far_width_ratio = candidate.far_width_ratio;
    evaluation.width_slope_m_per_m = candidate.width_slope_m_per_m;
    evaluation.left_lat_slope = candidate.left_lat_slope;
    evaluation.right_lat_slope = candidate.right_lat_slope;
    evaluation.relative_lat_slope = candidate.relative_lat_slope;
    evaluation.stable_split_score = candidate.stable_split_score;
    evaluation.opening_score = candidate.opening_score;
    evaluation.closing_score = candidate.closing_score;
    evaluation.stable_split_like = candidate.stable_split_like;
    evaluation.opening_like = candidate.opening_like;
    evaluation.closing_like = candidate.closing_like;
    evaluation.closing_suspect = candidate.closing_suspect;
    evaluation.strong_closing_veto = candidate.strong_closing_veto;
    evaluation.pair_coverage_ratio = candidate.pair_coverage_ratio;
    evaluation.pair_unexplained_transverse_risk = candidate.pair_unexplained_transverse_risk;
    evaluation.pair_semantic_rank_adjust = candidate.pair_semantic_rank_adjust;
    evaluation.pair_fake_forward_penalty = candidate.pair_fake_forward_penalty;
    evaluation.pair_forward_side_support_score = candidate.pair_forward_side_support_score;
    evaluation.transverse_merge_band_score = candidate.transverse_merge_band_score;
    evaluation.transverse_opening_branch_score = candidate.transverse_opening_branch_score;
    evaluation.transverse_role = candidate.transverse_role;

    if (selection_result.mode != BoundarySelectionMode::SplitPair ||
        !selection_result.normalized_side_ok ||
        !selection_result.left_observation.valid ||
        !selection_result.right_observation.valid) {
        evaluation.trigger_split_pair_unavailable = true;
        evaluation.reject_reason = selection_result.normalized_side_ok
            ? "split_pair_unavailable" : selection_result.reject_reason;
        return evaluation;
    }

    BoundarySide physical_left_slot = BoundarySide::Unknown;
    BoundarySide physical_right_slot = BoundarySide::Unknown;
    float normalized_width_m = 0.0f;
    std::string normalized_reject_reason = "none";
    if (!normalizeBoundaryPairPhysicalSides(candidate, nullptr, nullptr,
                                             &normalized_width_m, &normalized_reject_reason,
                                             &physical_left_slot, &physical_right_slot)) {
        evaluation.normalization_ok = false;
        evaluation.normalization_reject_reason = normalized_reject_reason;
        evaluation.normalized_width_m = normalized_width_m;
        evaluation.reject_reason = normalized_reject_reason;
        return evaluation;
    }
    evaluation.normalization_ok = true;
    evaluation.root_width_m = normalized_width_m;
    evaluation.normalized_width_m = normalized_width_m;
    evaluation.width_m = normalized_width_m;
    evaluation.width_source = "normalized_root_width";
    const bool use_provider_v2_width =
        candidate.provider_type == "physical_provider_v2" &&
        std::isfinite(evaluation.provider_overlap_width_m) &&
        evaluation.provider_overlap_width_m > 0.0f;
    if (use_provider_v2_width) {
        evaluation.width_m = evaluation.provider_overlap_width_m;
        evaluation.width_source = "provider_overlap_width";
    }
    evaluation.left_span_m = boundaryPairBranchSpanMeters(candidate, physical_left_slot);
    evaluation.right_span_m = boundaryPairBranchSpanMeters(candidate, physical_right_slot);
    const bool left_near_ok = boundaryPairBranchNearRootOk(candidate, physical_left_slot);
    const bool right_near_ok = boundaryPairBranchNearRootOk(candidate, physical_right_slot);
    const bool left_root_ok = boundaryPairBranchRootOk(candidate, physical_left_slot);
    const bool right_root_ok = boundaryPairBranchRootOk(candidate, physical_right_slot);
    const float semantic_width_max = maintain_width_gate
        ? 1.00f : (evaluation.opening_like ? kBoundaryOpeningWidthMaxM : kBoundaryDefaultWidthMaxM);
    evaluation.width_gate_max_m = semantic_width_max;
    if (evaluation.width_m < kBoundaryHealthyWidthMinM || evaluation.width_m > semantic_width_max) {
        evaluation.trigger_pair_width_out_of_range = true;
        evaluation.width_gate_fail_side =
            evaluation.width_m < kBoundaryHealthyWidthMinM ? "too_small" : "too_large";
        evaluation.reject_reason = "pair_width_out_of_range";
        return evaluation;
    }
    if (candidate.provider_type != "physical_provider_v2") {
        evaluation.reject_reason = "non_physical_v2_candidate";
        return evaluation;
    }
    float width_score = 1.0f - clampValue(std::abs(evaluation.width_m - 0.32f) / 0.32f, 0.0f, 1.0f);
    const float coverage_score = 0.25f *
        (static_cast<float>(left_root_ok) + static_cast<float>(right_root_ok) +
         static_cast<float>(left_near_ok) + static_cast<float>(right_near_ok));
    const float rank_score = clampValue(
        1.0f - static_cast<float>(selected_rank) / static_cast<float>(kBoundaryTrackingTopK), 0.0f, 1.0f);
    const float candidate_quality_hint =
        0.40f * clampValue(candidate.provider_pair_confidence, 0.0f, 1.0f) +
        0.35f * clampValue(candidate.provider_overlap_ratio, 0.0f, 1.0f) +
        0.25f * (1.0f - clampValue(candidate.shared_ratio, 0.0f, 1.0f));
    const float explanation_score = clampValue(
        0.40f * evaluation.pair_coverage_ratio +
        0.32f * (1.0f - evaluation.pair_unexplained_transverse_risk) +
        0.28f * evaluation.pair_forward_side_support_score, 0.0f, 1.0f);
    const float semantic_rank_score = clampValue(0.5f + 1.2f * evaluation.pair_semantic_rank_adjust, 0.0f, 1.0f);

    float association_score = 0.5f;
    float association_weight = 0.08f;
    if (manager != nullptr) {
        const BoundaryAssociationScore left_assoc = scoreBoundaryObservationAssociation(
            manager->left, selection_result.left_observation, visible_min_forward);
        const BoundaryAssociationScore right_assoc = scoreBoundaryObservationAssociation(
            manager->right, selection_result.right_observation, visible_min_forward);
        const float left_assoc_score = std::isfinite(left_assoc.cost)
            ? (1.0f - clampValue(left_assoc.cost / 4.0f, 0.0f, 1.0f)) : 0.0f;
        const float right_assoc_score = std::isfinite(right_assoc.cost)
            ? (1.0f - clampValue(right_assoc.cost / 4.0f, 0.0f, 1.0f)) : 0.0f;
        association_score = 0.5f * (left_assoc_score + right_assoc_score);
        association_weight = (manager->left.active || manager->right.active) ? 0.12f : 0.04f;
        if (!manager->left.active && !manager->right.active) association_weight = 0.04f;
    }

    evaluation.health_score =
        0.12f * coverage_score + 0.12f * width_score + 0.10f * rank_score +
        0.12f * candidate_quality_hint + 0.20f * explanation_score +
        0.14f * semantic_rank_score + association_weight * association_score;
    evaluation.health_score -= 0.10f * evaluation.pair_fake_forward_penalty;
    evaluation.health_score -= 0.08f * evaluation.pair_unexplained_transverse_risk;
    evaluation.health_score = clampValue(evaluation.health_score, 0.0f, 1.0f);
    evaluation.reject_reason = "none";
    evaluation.healthy = true;
    evaluation.rejected = false;
    return evaluation;
}

// Demo-aligned: selectBoundaryObservationForTracking
static BoundarySelectionResult selectBoundaryObservationForTracking(
    const std::vector<BoundaryPairCandidate>& split_candidates,
    const BoundaryTrackManager* manager,
    float visible_min_forward) {
    BoundarySelectionResult best_result;
    BoundaryPairHealthEvaluation best_health;
    BoundaryPairHealthEvaluation best_rejected_health;
    bool have_best = false;
    bool have_rejected = false;
    const size_t topk = std::min(kBoundaryTrackingTopK, split_candidates.size());
    for (size_t i = 0; i < topk; ++i) {
        BoundarySelectionResult candidate_result =
            buildBoundarySelectionResultFromBoundaryPair(split_candidates[i], static_cast<int>(i));
        candidate_result.allow_boundary_track_update = false;
        const BoundaryPairHealthEvaluation health = evaluateBoundaryPairHealth(
            split_candidates[i], candidate_result, static_cast<int>(i), manager, visible_min_forward);
        candidate_result.debug_health_score = health.health_score;
        if (health.healthy) {
            candidate_result.mode = BoundarySelectionMode::SplitPair;
            candidate_result.reject_reason = "none";
            candidate_result.allow_boundary_track_update = true;
            candidate_result.width_m = health.width_m;
            candidate_result.width_source = health.width_source;
            candidate_result.debug_health_score = health.health_score;
            const bool take_candidate =
                !have_best ||
                health.health_score > best_health.health_score + 1e-4f ||
                (std::abs(health.health_score - best_health.health_score) <= 1e-4f &&
                 candidate_result.selected_source_rank < best_result.selected_source_rank);
            if (take_candidate) {
                best_result = std::move(candidate_result);
                best_health = health;
                have_best = true;
            }
        } else if (!have_rejected ||
                   static_cast<int>(i) < best_rejected_health.selected_rank) {
            best_rejected_health = health;
            have_rejected = true;
        }
    }
    if (have_best) {
        best_result.debug_health_score = best_health.health_score;
        return best_result;
    }
    BoundarySelectionResult rejected_result;
    rejected_result.mode = BoundarySelectionMode::NoReliableBoundary;
    rejected_result.allow_boundary_track_update = false;
    rejected_result.normalized_side_ok = false;
    rejected_result.reject_reason =
        have_rejected ? best_rejected_health.reject_reason : "no_split_candidate";
    rejected_result.selected_source_rank =
        have_rejected ? best_rejected_health.selected_rank : -1;
    rejected_result.debug_health_score =
        have_rejected ? best_rejected_health.health_score : -1e9f;
    return rejected_result;
}

// Demo-aligned updateBoundaryTracks — uses selectBoundaryObservationForTracking
// and populates BoundaryTrackState left/right.
static void updateBoundaryTracks(BoundaryTrackManager* manager,
                                 const std::vector<BoundaryPairCandidate>& candidates,
                                 float visible_min_forward,
                                 BranchDebugInfo*) {
    if (manager == nullptr) return;

    // Reset all per-frame semantic fields (demo-aligned)
    manager->selected_pair_source = "none";
    manager->selected_pair_rank = -1;
    manager->selected_pair_width_m = 0.0f;
    manager->provider_final_width_m = 0.0f;
    manager->final_pair_width_m = 0.0f;
    manager->final_width_source = "none";
    manager->selector_pair_width_out_of_range = false;
    manager->selector_split_pair_unavailable = false;
    manager->selector_non_physical_v2_candidate = false;
    manager->selector_selected_non_physical_v2 = false;
    manager->selected_candidate_provider_type = "none";
    manager->selector_reject_reason = "none";
    manager->selector_selected_pair_width_m = 0.0f;
    manager->selector_selected_width_source = "none";
    manager->selector_best_rejected_rank = -1;
    manager->selector_best_rejected_score = -1e9f;
    manager->selector_best_rejected_reason = "none";
    manager->provider_semantic_not_opening_seen = false;
    manager->provider_semantic_used_as_weak_negative = false;
    manager->preserve_blocked_reason = "none";
    manager->selected_closing_like = false;
    manager->selected_closing_suspect = false;
    manager->selected_strong_closing_veto = false;
    manager->selected_opening_like = false;
    manager->selected_stable_split_like = false;
    manager->final_semantic_reason = "none";
    manager->frame_semantic_ok = false;
    manager->frame_semantic_pending = false;
    manager->frame_semantic_suppressed = false;
    manager->frame_semantic_mode = "none";
    manager->frame_semantic_reason = "none";
    manager->has_ft_opening_branch_support = false;
    manager->ft_support_reason = "none";
    manager->ft_support_t_ids.clear();
    manager->ft_support_t_fwd.clear();
    manager->ft_support_t_lat.clear();
    manager->ft_support_t_lat_sp = 0.0f;
    manager->ft_support_t_n = 0;
    manager->ft_support_t_near_ok = false;
    manager->ft_support_t_near_cov = 0;
    manager->ft_support_open_branch_score = 0.0f;
    manager->ft_support_merge_band_score = 0.0f;
    manager->selected_opening_score = 0.0f;
    manager->selected_open_branch_score = 0.0f;
    manager->selected_merge_band_score = 0.0f;
    manager->selected_transverse_merge_band_score = 0.0f;
    manager->selected_transverse_opening_branch_score = 0.0f;
    manager->selected_transverse_role = "none";
    manager->selected_pair_coverage_ratio = 0.0f;
    manager->selected_pair_unexplained_transverse_risk = 0.0f;
    manager->selected_pair_semantic_rank_adjust = 0.0f;
    manager->selected_pair_fake_forward_penalty = 0.0f;
    manager->selected_pair_forward_side_support_score = 0.0f;
    manager->frame_has_strong_closing_veto = false;
    manager->frame_best_closing_score = 0.0f;
    manager->opening_blocked_by_frame_closing_veto = false;
    manager->opening_block_reason = "none";
    manager->opening_vs_closing_margin = 0.0f;
    manager->selected_risk_downgraded_by_frame_opening = false;
    manager->semantic_selected_idx = -1;
    manager->semantic_suppressing_idx = -1;
    manager->best_opening_idx = -1;
    manager->best_opening_score = 0.0f;
    manager->best_closing_idx = -1;
    manager->best_closing_score = 0.0f;
    manager->best_merge_idx = -1;
    manager->best_merge_score = 0.0f;
    manager->best_stable_idx = -1;
    manager->best_stable_score = 0.0f;
    manager->opening_consistency_score = 0.0f;
    manager->closing_consistency_score = 0.0f;
    manager->opening_consistency_reason = "none";
    manager->closing_consistency_reason = "none";
    manager->confirmation_region_key = "none";
    manager->opening_hypothesis_id = "none";
    manager->confirmation_continued_by_region = false;
    manager->competing_semantic_conflict = false;
    manager->competing_conflict_reason = "none";
    manager->frame_best_closing_candidate_rank = -1;
    manager->frame_best_closing_pair_type = "none";
    manager->selected_right_fork_model = "none";
    manager->detached_candidate_valid = false;
    manager->detached_candidate_ids = "none";
    manager->detached_gate_trace = "none";
    manager->detached_score = 0.0f;
    manager->detached_structural_gate_pass = false;
    manager->detached_right_side_order_ok = false;
    manager->detached_crossing_detected = false;
    manager->detached_remerge_detected = false;
    manager->detached_independent_id_count = 0;
    manager->detached_main_overlap = 0.0f;
    manager->detached_main_reuse_ratio = 0.0f;
    manager->detached_common_forward_overlap = 0.0f;
    manager->detached_right_chain_count = 0;
    manager->detached_right_chain_forward_span = 0.0f;
    manager->detached_right_chain_lateral_span = 0.0f;
    manager->detached_lateral_separation = 0.0f;
    manager->detached_right_chain_continuity = 0.0f;
    manager->detached_width_min = 0.0f;
    manager->detached_width_max = 0.0f;
    manager->detached_width_mean = 0.0f;
    manager->detached_width_variation = 0.0f;
    manager->detached_sample_count = 0;
    manager->detached_valid_width_count = 0;
    manager->detached_trend_sample_count = 0;
    manager->detached_trend_width_near = 0.0f;
    manager->detached_trend_width_far = 0.0f;
    manager->detached_trend_width_delta = 0.0f;
    manager->detached_trend_width_slope = 0.0f;
    manager->detached_trend_far_width_ratio = 0.0f;
    manager->detached_trend_tail_min_width = 0.0f;
    manager->detached_fresh_entry_gate_evaluated = false;
    manager->detached_fresh_entry_gate_allowed = false;
    manager->detached_fresh_entry_gate_reason = "not_evaluated";
    manager->detached_fresh_entry_basic_valid = false;
    manager->detached_fresh_entry_structural_support = false;
    manager->detached_fresh_entry_specific_closing = false;
    manager->detached_fresh_entry_structural_reason = "not_evaluated";
    manager->detached_fresh_entry_closing_reason = "not_evaluated";
    manager->detached_fresh_entry_support_score = 0;
    manager->detached_fresh_entry_same_region_closing_count = 0;
    manager->left_active = false;
    manager->right_active = false;
    manager->left_obs_source = "none";
    manager->right_obs_source = "none";

    // Scan all candidates for frame-level closing veto
    for (const auto& candidate : candidates) {
        if (candidate.strong_closing_veto) manager->frame_has_strong_closing_veto = true;
        if (candidate.closing_score > manager->frame_best_closing_score)
            manager->frame_best_closing_score = candidate.closing_score;
    }

    // A2: select best candidate via demo-aligned selector
    const BoundarySelectionResult selection_result =
        selectBoundaryObservationForTracking(candidates, manager, visible_min_forward);
    manager->selector_reject_reason = selection_result.reject_reason;
    manager->selector_best_rejected_rank = selection_result.allow_boundary_track_update
        ? -1
        : selection_result.selected_source_rank;
    manager->selector_best_rejected_score = selection_result.allow_boundary_track_update
        ? -1e9f
        : selection_result.debug_health_score;
    manager->selector_best_rejected_reason = selection_result.allow_boundary_track_update
        ? "none"
        : selection_result.reject_reason;

    BoundaryObservation left_observation;
    BoundaryObservation right_observation;
    collectBoundaryObservations(selection_result, &left_observation, &right_observation);

    const BoundaryPairCandidate* selected_candidate = nullptr;
    if (selection_result.allow_boundary_track_update &&
        selection_result.selected_source_rank >= 0 &&
        selection_result.selected_source_rank < static_cast<int>(candidates.size())) {
        selected_candidate = &candidates[static_cast<size_t>(selection_result.selected_source_rank)];
        manager->selected_pair_source = selected_candidate->provider_type;
        manager->selected_pair_rank = selection_result.selected_source_rank;
        manager->selected_pair_width_m = selection_result.width_m;
        manager->selected_candidate_provider_type = selected_candidate->provider_type;
        manager->selector_selected_pair_width_m = selection_result.width_m;
        manager->selector_selected_width_source = selection_result.width_source;
        manager->selected_stable_split_like = selected_candidate->stable_split_like;
        manager->selected_opening_like = selected_candidate->opening_like;
        manager->selected_closing_like = selected_candidate->closing_like;
        manager->selected_closing_suspect = selected_candidate->closing_suspect;
        manager->selected_strong_closing_veto = selected_candidate->strong_closing_veto;
        manager->selected_pair_coverage_ratio = selected_candidate->pair_coverage_ratio;
        manager->selected_pair_unexplained_transverse_risk =
            selected_candidate->pair_unexplained_transverse_risk;
        manager->selected_pair_semantic_rank_adjust = selected_candidate->pair_semantic_rank_adjust;
        manager->selected_pair_fake_forward_penalty = selected_candidate->pair_fake_forward_penalty;
        manager->selected_pair_forward_side_support_score =
            selected_candidate->pair_forward_side_support_score;
        manager->selected_transverse_merge_band_score = selected_candidate->transverse_merge_band_score;
        manager->selected_transverse_opening_branch_score = selected_candidate->transverse_opening_branch_score;
        manager->selected_transverse_role = selected_candidate->transverse_role;
        const bool semantic_merge =
            selected_candidate->source.find("pair_semantic=merge") != std::string::npos;
        const bool semantic_crossing =
            selected_candidate->source.find("pair_semantic=crossing") != std::string::npos;
        const bool is_transverse_opening =
            selected_candidate->transverse_role == "opening_branch";
        manager->final_semantic_reason =
            selected_candidate->strong_closing_veto ? "physical_v2_closing_pair"
            : (selected_candidate->closing_like ? "physical_v2_closing_like"
               : (selected_candidate->closing_suspect ? "physical_v2_closing_suspect"
                  : (semantic_merge ? "primitive_merge_like"
                     : (semantic_crossing ? "primitive_crossing_like"
                        : (selected_candidate->stable_split_like ? "stable_split_confirmed"
                           : (is_transverse_opening ? "transverse_opening_candidate"
                              : (selected_candidate->opening_like ? "opening_pending_confirmation"
                                 : "unknown_pair_not_confirmed")))))));
        if (selected_candidate->provider_type == "physical_provider_v2") {
            manager->provider_final_width_m = selection_result.width_m;
            manager->final_pair_width_m = selection_result.width_m;
            manager->final_width_source = selection_result.width_source.empty()
                ? "physical_provider_v2" : selection_result.width_source;
        }
        FtOpeningBranchSupportInfo ft_support;
        if (isStrongFtOpeningBranchCandidate(*selected_candidate)) {
            ft_support.found = true;
            ft_support.reason = "selected_ft_opening_branch";
            ft_support.candidate_rank = selection_result.selected_source_rank;
            ft_support.t_ids = selected_candidate->transverse_t_ids;
            {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(2)
                    << selected_candidate->transverse_near_forward_m << ","
                    << selected_candidate->transverse_far_forward_m;
                ft_support.t_fwd = oss.str();
            }
            {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(2)
                    << selected_candidate->transverse_lat_min_m << ","
                    << selected_candidate->transverse_lat_max_m;
                ft_support.t_lat = oss.str();
            }
            ft_support.t_lat_sp = selected_candidate->transverse_lat_span_m;
            ft_support.t_n = selected_candidate->transverse_guide_count;
            ft_support.t_near_ok = selected_candidate->transverse_near_ok;
            ft_support.t_near_cov = selected_candidate->transverse_near_cov;
            ft_support.open_branch_score =
                selected_candidate->transverse_opening_branch_score;
            ft_support.merge_band_score =
                selected_candidate->transverse_merge_band_score;
        } else if (selected_candidate->opening_like) {
            ft_support = findSupportingFtOpeningBranchForSelectedOpening(
                *selected_candidate,
                candidates);
        } else {
            ft_support.reason = "selected_not_opening_hypothesis";
        }
        manager->has_ft_opening_branch_support = ft_support.found;
        manager->ft_support_reason = ft_support.reason;
        manager->ft_support_t_ids = ft_support.t_ids;
        manager->ft_support_t_fwd = ft_support.t_fwd;
        manager->ft_support_t_lat = ft_support.t_lat;
        manager->ft_support_t_lat_sp = ft_support.t_lat_sp;
        manager->ft_support_t_n = ft_support.t_n;
        manager->ft_support_t_near_ok = ft_support.t_near_ok;
        manager->ft_support_t_near_cov = ft_support.t_near_cov;
        manager->ft_support_open_branch_score = ft_support.open_branch_score;
        manager->ft_support_merge_band_score = ft_support.merge_band_score;
    } else {
        // Demo-aligned: no formal selector gate flags set from rejected reason.
        // Only record debug info (best rejected rank / score / reason).
        if (selection_result.selected_source_rank >= 0 &&
            selection_result.selected_source_rank < static_cast<int>(candidates.size())) {
            const BoundaryPairCandidate& rejected =
                candidates[static_cast<size_t>(selection_result.selected_source_rank)];
            manager->selected_candidate_provider_type = rejected.provider_type;
            manager->selector_selected_pair_width_m =
                selection_result.width_m;
            manager->selector_selected_width_source =
                selection_result.width_source;
        }
    }

    // A-1: populate per-frame left/right active + obs_source from selection
    manager->left_active = selection_result.left_active;
    manager->right_active = selection_result.right_active;
    manager->left_obs_source = selection_result.left_active
        ? selection_result.left_observation.source : "none";
    manager->right_obs_source = selection_result.right_active
        ? selection_result.right_observation.source : "none";

    // Update demo-aligned BoundaryTrackState per side
    updateBoundaryTrackState(manager->left, left_observation, visible_min_forward);
    updateBoundaryTrackState(manager->right, right_observation, visible_min_forward);

    // Sync to B-1 flat fields for backward compatibility
    manager->left_track_active = manager->left.active;
    manager->right_track_active = manager->right.active;
    manager->left_track_confidence = manager->left.confidence;
    manager->right_track_confidence = manager->right.confidence;
    manager->left_track_missed_frames = manager->left.missed_frames;
    manager->right_track_missed_frames = manager->right.missed_frames;
    manager->left_track_root_lateral_m = manager->left.root_lateral;
    manager->right_track_root_lateral_m = manager->right.root_lateral;
    manager->left_track_root_forward_m = manager->left.root_forward;
    manager->right_track_root_forward_m = manager->right.root_forward;
    manager->left_track_near_forward_m = manager->left.near_forward;
    manager->right_track_near_forward_m = manager->right.near_forward;
    manager->left_track_predicted_lateral_m = manager->left.predicted_lateral;
    manager->right_track_predicted_lateral_m = manager->right.predicted_lateral;
    manager->left_track_predicted_heading_deg = manager->left.predicted_heading;
    manager->right_track_predicted_heading_deg = manager->right.predicted_heading;
    manager->left_track_guides = manager->left.guides;
    manager->right_track_guides = manager->right.guides;

}

struct DetachedFreshEntryGateResult {
    bool evaluated = false;
    bool allowed = true;
    std::string reason = "not_evaluated";
    bool basic_valid = false;
    bool structural_support = false;
    bool specific_closing = false;
    std::string structural_reason = "not_evaluated";
    std::string closing_reason = "not_evaluated";
    int support_score = 0;
    int same_region_closing_count = 0;
};

static int countBoundaryCandidateSharedIds(
    const BoundaryPairCandidate& candidate,
    const std::unordered_set<int>& detached_ids) {
    std::unordered_set<int> matched_ids;
    auto collect = [&](const std::vector<ArrowGuide>& guides) {
        for (const auto& guide : guides) {
            if (guide.tracker_id >= 0 && detached_ids.count(guide.tracker_id) > 0) {
                matched_ids.insert(guide.tracker_id);
            }
        }
    };
    collect(candidate.left_branch);
    collect(candidate.right_branch);
    return static_cast<int>(matched_ids.size());
}

static DetachedFreshEntryGateResult evaluateDetachedFreshEntryGate(
    const BoundaryTrackManager& manager,
    bool opening_hypothesis_active,
    const std::vector<BoundaryPairCandidate>& split_candidates) {
    DetachedFreshEntryGateResult result;
    const bool fresh_detached_entry =
        opening_hypothesis_active &&
        !manager.split_confirmed &&
        !manager.visible_split_output_active &&
        manager.visible_split_maintain_age <= 0 &&
        manager.selected_right_fork_model == "detached";
    if (!fresh_detached_entry) {
        result.evaluated = false;
        result.allowed = true;
        result.reason = "not_fresh_detached_entry";
        return result;
    }

    result.evaluated = true;
    result.basic_valid =
        manager.detached_candidate_valid &&
        manager.detached_structural_gate_pass &&
        manager.detached_right_side_order_ok &&
        !manager.detached_crossing_detected;
    if (!result.basic_valid) {
        result.allowed = false;
        result.reason = "detached_basic_invalid";
        result.structural_reason = "basic_invalid";
        return result;
    }

    int independence_score = 0;
    if (manager.detached_independent_id_count >= kDetachedRightForkMinIndependentIds) {
        ++independence_score;
    }
    if (manager.detached_main_overlap <= kDetachedRightForkMaxRightMainOverlap) {
        ++independence_score;
    }
    if (manager.detached_main_reuse_ratio <= kDetachedRightForkMaxMainReuseRatio) {
        ++independence_score;
    }
    int continuity_score = 0;
    if (manager.detached_right_chain_count >= kDetachedRightForkMinGuideCount) {
        ++continuity_score;
    }
    if (manager.detached_right_chain_forward_span >= kDetachedRightForkMinForwardSpanM) {
        ++continuity_score;
    }
    if (manager.detached_right_chain_continuity >= kDetachedRightForkMinContinuityScore) {
        ++continuity_score;
    }
    int separation_score = 0;
    if (manager.detached_lateral_separation >= kDetachedRightForkMinWidthM) {
        ++separation_score;
    }
    if (manager.detached_width_mean >= kDetachedRightForkMinMeanWidthM) {
        ++separation_score;
    }
    if (manager.detached_width_min >= kDetachedRightForkMinWidthM) {
        ++separation_score;
    }
    const bool common_overlap_support =
        manager.detached_common_forward_overlap >=
        kDetachedRightForkMinCommonForwardOverlapM;
    result.support_score = independence_score + continuity_score +
                           separation_score + (common_overlap_support ? 1 : 0);
    result.structural_support =
        independence_score >= 2 && continuity_score >= 2 &&
        separation_score >= 2 && result.support_score >= 7;
    {
        std::ostringstream reason;
        reason << "ind=" << independence_score
               << ",cont=" << continuity_score
               << ",sep=" << separation_score
               << ",common=" << (common_overlap_support ? 1 : 0)
               << ",score=" << result.support_score;
        result.structural_reason = reason.str();
    }
    if (!result.structural_support) {
        result.allowed = false;
        result.reason = "detached_structural_support_weak";
        return result;
    }

    if (manager.detached_remerge_detected) {
        result.specific_closing = true;
        result.closing_reason = "detached_remerge_detected";
    }
    if (!result.specific_closing &&
        manager.detached_trend_sample_count >= kDetachedRightForkMinWidthSamples) {
        const bool tail_near_main =
            manager.detached_trend_tail_min_width < kDetachedRightForkMinWidthM ||
            manager.detached_trend_width_far < kDetachedRightForkMinWidthM;
        const bool width_contracting =
            manager.detached_trend_width_delta <= -kDetachedRightForkMinMeanWidthM &&
            manager.detached_trend_far_width_ratio <= 0.70f;
        const bool slope_contracting =
            manager.detached_trend_width_slope <= -0.35f &&
            manager.detached_trend_far_width_ratio <= 0.80f;
        if (tail_near_main) {
            result.specific_closing = true;
            result.closing_reason = "detached_tail_near_main";
        } else if (width_contracting || slope_contracting) {
            result.specific_closing = true;
            result.closing_reason = width_contracting
                ? "detached_width_contracting"
                : "detached_width_slope_contracting";
        }
    }

    if (!result.specific_closing) {
        const std::vector<int> detached_id_vector =
            parseTrackerIdsText(manager.detached_candidate_ids);
        const std::unordered_set<int> detached_ids(detached_id_vector.begin(),
                                                   detached_id_vector.end());
        const BoundaryPairCandidate* selected_candidate = nullptr;
        if (manager.selected_pair_rank >= 0 &&
            manager.selected_pair_rank < static_cast<int>(split_candidates.size())) {
            selected_candidate =
                &split_candidates[static_cast<size_t>(manager.selected_pair_rank)];
        }
        for (const auto& candidate : split_candidates) {
            if (candidate.provider_type != "physical_provider_v2" ||
                countBoundaryCandidateSharedIds(candidate, detached_ids) < 2) {
                continue;
            }
            const bool same_selected_opening =
                selected_candidate != nullptr && &candidate == selected_candidate &&
                (candidateSemanticContains(candidate, "opening") ||
                 candidate.opening_like ||
                 candidate.transverse_role == "opening_branch");
            const bool semantic_closing =
                candidateSemanticContains(candidate, "closing") ||
                candidate.closing_like || candidate.closing_suspect ||
                candidate.strong_closing_veto;
            const bool semantic_merge =
                candidateSemanticContains(candidate, "merge") ||
                candidate.transverse_role == "far_merge_band";
            const bool closing_shape =
                candidate.closing_score >= 0.82f &&
                candidate.width_delta_m <= -0.12f &&
                candidate.far_width_ratio <= 0.70f;
            if (!same_selected_opening &&
                (closing_shape ||
                 (semantic_closing && candidate.closing_score >= 0.78f) ||
                 (semantic_merge && closing_shape))) {
                ++result.same_region_closing_count;
            }
        }
        if (result.same_region_closing_count > 0) {
            result.specific_closing = true;
            result.closing_reason = "detached_same_region_closing_conflict";
        }
    }

    if (result.specific_closing) {
        result.allowed = false;
        result.reason = "detached_self_closing_trend";
        return result;
    }
    result.allowed = true;
    result.reason = "detached_fresh_entry_allowed";
    result.closing_reason = "none";
    return result;
}

struct StableDoubleBoundaryRecoveryEvaluation {
    bool evidence = false;
    bool context_active = false;
    bool confirmed_this_frame = false;
    std::string reason = "not_evaluated";
};

static StableDoubleBoundaryRecoveryEvaluation evaluateStableDoubleBoundaryRecovery(
    uint32_t frame_id,
    BoundaryTrackManager* manager,
    const std::vector<ArrowGuide>& main_guides,
    const ForwardGrownVisiblePairCandidate& visible_pair,
    const BoundaryPairCandidate* selected_candidate,
    bool visible_split_output_active_before_frame,
    bool split_confirmed_before_frame) {
    (void)frame_id;
    (void)split_confirmed_before_frame;
    StableDoubleBoundaryRecoveryEvaluation result;
    auto fail = [&](const std::string& reason) {
        result.reason = reason;
        clearStableDoubleBoundaryRecoveryContext(manager, reason);
        return result;
    };

    if (manager == nullptr) {
        return fail("no_boundary_manager");
    }
    if (visible_split_output_active_before_frame) {
        return fail("split_context_already_active");
    }
    if (manager->previous_accepted_main_ids.empty()) {
        return fail("no_historical_main_identity");
    }
    if (main_guides.size() < static_cast<size_t>(kDetachedRightForkMinGuideCount)) {
        return fail("main_chain_too_short");
    }
    int main_common_ids = 0;
    float main_match_ratio = 0.0f;
    if (!guideChainMatchesIdentityIds(main_guides,
                                      manager->previous_accepted_main_ids,
                                      &main_common_ids,
                                      &main_match_ratio)) {
        return fail("historical_main_not_matched");
    }
    if (guideForwardSpan(main_guides) / kWorldPlanScale <
        kSplitMainIdentityMinForwardSpanM) {
        return fail("historical_main_span_low");
    }

    const bool no_anchor_recovery_shape =
        visible_pair.no_anchor_on_main_seen ||
        visible_pair.reject_reason == "no_anchor_on_main" ||
        visible_pair.detached_candidate_found;
    if (!no_anchor_recovery_shape) {
        return fail("no_no_anchor_recovery_shape");
    }
    if (!visible_pair.detached_candidate_found) {
        return fail("no_detached_candidate");
    }
    if (visible_pair.detached_independent_id_count <
        kDetachedRightForkMinIndependentIds) {
        return fail("detached_independent_id_count_low");
    }
    if (visible_pair.detached_main_overlap >
        kDetachedRightForkMaxRightMainOverlap) {
        return fail("detached_main_overlap_high");
    }
    if (visible_pair.detached_main_reuse_ratio >
        kDetachedRightForkMaxMainReuseRatio) {
        return fail("detached_main_reuse_ratio_high");
    }
    if (visible_pair.detached_common_forward_overlap <
        kStableDoubleBoundaryRecoveryMinCommonForwardOverlapM) {
        return fail("detached_common_forward_overlap_low");
    }
    if (visible_pair.detached_right_chain_count <
        kDetachedRightForkMinGuideCount) {
        return fail("detached_chain_too_short");
    }
    if (visible_pair.detached_right_chain_forward_span <
        kDetachedRightForkMinForwardSpanM) {
        return fail("detached_forward_span_low");
    }
    if (visible_pair.detached_right_chain_continuity <
        kDetachedRightForkMinContinuityScore) {
        return fail("detached_continuity_low");
    }
    if (visible_pair.lateral_separation <
        kStableDoubleBoundaryRecoveryMinLateralSeparationM) {
        return fail("lateral_separation_low");
    }
    if (visible_pair.detached_crossing_detected) {
        return fail("detached_crossing_detected");
    }
    if (visible_pair.detached_remerge_detected) {
        return fail("detached_remerge_detected");
    }
    if (visible_pair.detached_width_max >
        kStableDoubleBoundaryRecoveryMaxWidthM) {
        return fail("detached_width_too_large_for_recovery");
    }
    if (selected_candidate == nullptr ||
        selected_candidate->provider_type != "physical_provider_v2") {
        return fail("no_physical_provider_pair");
    }
    if (selected_candidate->strong_closing_veto ||
        selected_candidate->closing_like ||
        selected_candidate->closing_suspect ||
        selected_candidate->closing_score >= 0.85f ||
        selected_candidate->pair_unexplained_transverse_risk >= 0.58f) {
        return fail("selected_pair_closing_or_crossing_risk");
    }

    const std::vector<int> main_seed_ids = collectGuideTrackerIds(main_guides);
    const std::vector<int> right_seed_ids =
        collectIndependentRecoveryRightSeedIds(
            visible_pair.detached_candidate_ids,
            visible_pair.current_right_fork_candidate_ids,
            main_seed_ids);
    if (right_seed_ids.size() <
        static_cast<size_t>(kDetachedRightForkMinIndependentIds)) {
        return fail("right_seed_ids_low");
    }

    result.evidence = true;
    result.reason = "stable_double_boundary_recovery";
    const int previous_count = manager->stable_double_boundary_recovery_count;
    manager->stable_double_boundary_recovery_count = std::min(
        previous_count + 1,
        kStableDoubleBoundaryRecoveryRequiredFrames);
    manager->stable_double_boundary_recovery_reject_reason = "none";
    if (manager->stable_double_boundary_recovery_count >=
        kStableDoubleBoundaryRecoveryRequiredFrames) {
        result.confirmed_this_frame =
            previous_count < kStableDoubleBoundaryRecoveryRequiredFrames;
        manager->split_recovery_seed_main_ids = main_seed_ids;
        manager->split_recovery_seed_right_ids = right_seed_ids;
        manager->split_recovery_context_active = true;
        manager->split_recovery_context_age = std::min(
            manager->split_recovery_context_age + 1,
            kStableDoubleBoundaryRecoveryMaxAgeFrames);
        manager->split_recovery_context_reason = result.reason;
    } else {
        manager->split_recovery_context_active = false;
        manager->split_recovery_context_age = 0;
        manager->split_recovery_seed_main_ids.clear();
        manager->split_recovery_seed_right_ids.clear();
        manager->split_recovery_context_reason =
            "stable_double_boundary_recovery_pending";
    }
    result.context_active = hasSplitRecoverySearchContext(manager);
    return result;
}

// Phase 5+: Full demo-aligned computeBoundaryFinalSplitDecision.
// Calls arbitrateFrameSemanticEvidence for multi-candidate frame-level
// semantic arbitration, then makes the final split / reject decision.
// Includes opening_explanation_risky (A3) and short_opening + frame
// closing veto (A4).
static BoundaryFinalSplitDecision computeBoundaryFinalSplitDecision(
    BoundaryTrackManager* manager,
    const std::vector<BoundaryPairCandidate>& split_candidates) {
    BoundaryFinalSplitDecision decision;
    if (manager == nullptr) {
        decision.reject_reason = "no_boundary_manager";
        return decision;
    }

    decision.width_m = manager->final_pair_width_m;
    decision.width_source = manager->final_width_source;

    const BoundaryPairCandidate* selected_candidate = nullptr;
    if (manager->selected_pair_rank >= 0 &&
        manager->selected_pair_rank < static_cast<int>(split_candidates.size())) {
        selected_candidate =
            &split_candidates[static_cast<size_t>(manager->selected_pair_rank)];
    }

    // ---- Phase 5+: Full multi-candidate frame-level semantic arbitration ----
    const bool strong_ft_opening_selected =
        selected_candidate != nullptr &&
        isStrongFtOpeningBranchCandidate(*selected_candidate);

    const FrameClosingVetoInfo frame_closing_veto =
        collectFrameClosingVetoInfo(split_candidates);
    manager->frame_best_closing_score = frame_closing_veto.best_closing_score;
    manager->frame_best_closing_candidate_rank =
        frame_closing_veto.best_closing_candidate_rank;
    manager->frame_best_closing_pair_type =
        frame_closing_veto.best_closing_pair_type;
    manager->frame_has_strong_closing_veto =
        frame_closing_veto.has_strong_closing_veto;

    if (selected_candidate != nullptr) {
        manager->selected_opening_score = selected_candidate->opening_score;
        manager->selected_open_branch_score =
            selected_candidate->transverse_opening_branch_score;
        manager->selected_merge_band_score =
            selected_candidate->transverse_merge_band_score;
    }

    const FrameSemanticDecision frame_decision =
        arbitrateFrameSemanticEvidence(split_candidates, selected_candidate, *manager);
    applyFrameSemanticDecisionToManager(manager, frame_decision);

    // ---- opening_dominates (consistency-score-based, aligned with demo) ----
    const bool frame_opening_dominates =
        frame_decision.semantic_pending &&
        (frame_decision.semantic_mode == "opening" ||
         frame_decision.semantic_mode == "short_opening" ||
         frame_decision.semantic_mode == "uncertain_opening") &&
        frame_decision.opening_consistency_score >= 0.68f &&
        frame_decision.opening_vs_closing_margin >= 0.08f;

    const bool frame_has_usable_split_evidence =
        (frame_decision.semantic_pending &&
         (frame_decision.semantic_mode == "opening" ||
          frame_decision.semantic_mode == "short_opening" ||
          frame_decision.semantic_mode == "uncertain_opening")) ||
        (frame_decision.semantic_ok &&
         frame_decision.semantic_mode == "stable_split");
    const bool active_visible_split_context =
        hasActiveVisibleSplitContext(manager);

    // ---- Hard gates: source, semantic suppression, closing veto ----
    // Demo uses BoundaryTrackState active/obs_source here, not per-frame
    // selected-observation active flags.
    if (!manager->left.active) {
        decision.reject_reason = "left_inactive";
        clearPhysicalSplitConfirmation(manager, decision.reject_reason);
        return decision;
    }
    if (!manager->right.active) {
        decision.reject_reason = "right_inactive";
        clearPhysicalSplitConfirmation(manager, decision.reject_reason);
        return decision;
    }
    if (manager->selected_pair_source != "physical_provider_v2") {
        decision.reject_reason = "non_physical_v2_pair";
        decision.reason = decision.reject_reason;
        if (!active_visible_split_context) {
            clearPhysicalSplitConfirmation(manager, decision.reject_reason);
        }
        return decision;
    }
    if (manager->left.obs_source != "physical_provider_v2") {
        decision.reject_reason = "left_non_physical_v2_obs";
        clearPhysicalSplitConfirmation(manager, decision.reject_reason);
        return decision;
    }
    if (manager->right.obs_source != "physical_provider_v2") {
        decision.reject_reason = "right_non_physical_v2_obs";
        clearPhysicalSplitConfirmation(manager, decision.reject_reason);
        return decision;
    }
    if (manager->final_width_source != "provider_overlap_width" &&
        manager->final_width_source != "physical_provider_v2") {
        decision.reject_reason = "non_physical_v2_width";
        clearPhysicalSplitConfirmation(manager, decision.reject_reason);
        return decision;
    }

    if (frame_decision.semantic_suppressed) {
        decision.reject_reason = frame_decision.semantic_reason;
        decision.reason = frame_decision.semantic_reason;
        manager->final_semantic_reason = decision.reason;
        if (!active_visible_split_context) {
            clearPhysicalSplitConfirmation(manager, decision.reject_reason);
        }
        return decision;
    }

    if (manager->selected_strong_closing_veto && !frame_opening_dominates) {
        decision.reject_reason = "physical_v2_closing_pair";
        decision.reason = "physical_v2_closing_pair";
        manager->final_semantic_reason = decision.reason;
        if (!active_visible_split_context) {
            clearPhysicalSplitConfirmation(manager, decision.reject_reason);
        }
        return decision;
    }
    if (manager->selected_closing_like && !frame_opening_dominates) {
        decision.reject_reason = "physical_v2_closing_like";
        decision.reason = "physical_v2_closing_like";
        manager->final_semantic_reason = decision.reason;
        if (!active_visible_split_context) {
            clearPhysicalSplitConfirmation(manager, decision.reject_reason);
        }
        return decision;
    }
    if (manager->selected_closing_suspect && !strong_ft_opening_selected &&
        !frame_has_usable_split_evidence && !frame_opening_dominates) {
        decision.reject_reason = "physical_v2_closing_suspect";
        decision.reason = "physical_v2_closing_suspect";
        manager->final_semantic_reason = decision.reason;
        if (!active_visible_split_context) {
            clearPhysicalSplitConfirmation(manager, decision.reject_reason);
        }
        return decision;
    }

    // ---- Width gate ----
    if (!isFiniteFloat(decision.width_m)) {
        decision.reject_reason = "width_invalid";
        clearPhysicalSplitConfirmation(manager, decision.reject_reason);
        return decision;
    }
    if (decision.width_m < 0.12f) {
        decision.reject_reason = "width_too_small";
        clearPhysicalSplitConfirmation(manager, decision.reject_reason);
        return decision;
    }
    if (decision.width_m > 0.75f) {
        decision.reject_reason = "width_too_large";
        clearPhysicalSplitConfirmation(manager, decision.reject_reason);
        return decision;
    }

    // ---- Split confirmation: stable_split ----
    manager->split_confirm_reject_reason = "none";
    const bool already_confirmed_split = manager->split_confirmed;
    const std::string previous_confirm_mode = manager->split_confirm_mode;

    if (frame_decision.semantic_ok &&
        frame_decision.semantic_mode == "stable_split") {
        manager->split_confirm_mode = "stable_split_like";
        if (already_confirmed_split) {
            manager->split_confirm_count = std::max(manager->split_confirm_count, 2);
            manager->split_confirmed = true;
            decision.final_split = true;
            decision.reason = "stable_split_confirmed";
            manager->final_semantic_reason = decision.reason;
            return decision;
        }
        manager->split_confirm_count =
            isSplitConfirmationEvidenceMode(previous_confirm_mode)
                ? std::max(manager->split_confirm_count + 1, 1)
                : 1;
        manager->split_confirmed = false;
        decision.reject_reason = "stable_split_pending_confirmation";
        decision.reason = "stable_split_pending_confirmation";
        manager->split_confirm_reject_reason = decision.reject_reason;
        manager->final_semantic_reason = decision.reason;
        return decision;
    }

    // ---- Split confirmation: opening hypothesis ----
    const bool opening_hypothesis_active =
        frame_decision.semantic_pending &&
        (frame_decision.semantic_mode == "opening" ||
         frame_decision.semantic_mode == "short_opening" ||
         frame_decision.semantic_mode == "uncertain_opening");

    if (opening_hypothesis_active) {
        // A3: opening_explanation_risky check (demo-aligned)
        const bool opening_explanation_risky =
            manager->selected_pair_unexplained_transverse_risk >= 0.55f ||
            manager->selected_pair_fake_forward_penalty >= 0.35f ||
            manager->selected_pair_semantic_rank_adjust <= -0.10f;
        const bool can_downgrade_opening_risk =
            (manager->selected_opening_like && manager->has_ft_opening_branch_support) ||
            frame_opening_dominates;
        if (opening_explanation_risky && !can_downgrade_opening_risk &&
            !(already_confirmed_split && frame_has_usable_split_evidence)) {
            decision.reject_reason = "opening_explanation_risk";
            decision.reason = "opening_explanation_risk";
            manager->final_semantic_reason = decision.reason;
            if (!active_visible_split_context) {
                clearPhysicalSplitConfirmation(manager, decision.reject_reason);
            }
            return decision;
        }
        if (opening_explanation_risky &&
            (can_downgrade_opening_risk ||
             (already_confirmed_split && frame_has_usable_split_evidence))) {
            manager->split_confirm_reject_reason =
                already_confirmed_split && !can_downgrade_opening_risk
                    ? "opening_explanation_risk_held_by_confirmed_opening"
                    : (frame_opening_dominates
                    ? "opening_explanation_risk_downgraded_by_frame_opening"
                    : "opening_explanation_risk_downgraded_by_ft_support");
            manager->selected_risk_downgraded_by_frame_opening =
                frame_opening_dominates;
        }

        // Cross-frame opening hypothesis region tracking
        const bool region_continued =
            frame_decision.confirmation_region_key != "none" &&
            frame_decision.confirmation_region_key == manager->previous_opening_region_key;
        manager->confirmation_continued_by_region = region_continued;
        if (frame_decision.confirmation_region_key != "none") {
            if (region_continued) {
                manager->opening_hypothesis_id = manager->previous_opening_hypothesis_id;
            } else {
                std::ostringstream hypothesis_id;
                hypothesis_id << "opening_" << manager->next_opening_hypothesis_seq++;
                manager->opening_hypothesis_id = hypothesis_id.str();
            }
            manager->previous_opening_region_key = frame_decision.confirmation_region_key;
            manager->previous_opening_hypothesis_id = manager->opening_hypothesis_id;
        }

        const std::string confirm_mode =
            frame_decision.semantic_mode == "short_opening"
                ? "short_opening"
                : (strong_ft_opening_selected
                       ? (manager->selected_opening_like ? "opening_with_ft_support"
                                                         : "ft_opening_branch")
                       : (manager->has_ft_opening_branch_support
                              ? "opening_with_ft_support"
                              : "opening_like"));
        const std::string pending_reason =
            confirm_mode == "short_opening"
                ? "short_opening_pending_confirmation"
                : (confirm_mode == "ft_opening_branch"
                ? "ft_opening_branch_pending_confirmation"
                : (confirm_mode == "opening_with_ft_support"
                       ? "opening_with_ft_support_pending_confirmation"
                       : "opening_pending_confirmation"));
        const std::string confirmed_reason =
            confirm_mode == "short_opening"
                ? "short_opening_confirmed"
                : (confirm_mode == "ft_opening_branch"
                ? "ft_opening_branch_confirmed"
                : (confirm_mode == "opening_with_ft_support"
                       ? "opening_with_ft_support_confirmed"
                       : "opening_confirmed"));

        const DetachedFreshEntryGateResult detached_entry_gate =
            evaluateDetachedFreshEntryGate(*manager,
                                           opening_hypothesis_active,
                                           split_candidates);
        manager->detached_fresh_entry_gate_evaluated =
            detached_entry_gate.evaluated;
        manager->detached_fresh_entry_gate_allowed =
            detached_entry_gate.allowed;
        manager->detached_fresh_entry_gate_reason =
            detached_entry_gate.reason;
        manager->detached_fresh_entry_basic_valid = detached_entry_gate.basic_valid;
        manager->detached_fresh_entry_structural_support =
            detached_entry_gate.structural_support;
        manager->detached_fresh_entry_specific_closing =
            detached_entry_gate.specific_closing;
        manager->detached_fresh_entry_structural_reason =
            detached_entry_gate.structural_reason;
        manager->detached_fresh_entry_closing_reason =
            detached_entry_gate.closing_reason;
        manager->detached_fresh_entry_support_score =
            detached_entry_gate.support_score;
        manager->detached_fresh_entry_same_region_closing_count =
            detached_entry_gate.same_region_closing_count;
        if (detached_entry_gate.evaluated && !detached_entry_gate.allowed) {
            decision.reject_reason = detached_entry_gate.reason;
            decision.reason = detached_entry_gate.reason;
            manager->final_semantic_reason = decision.reason;
            if (!active_visible_split_context) {
                clearPhysicalSplitConfirmation(manager, decision.reject_reason);
            }
            return decision;
        }

        if (already_confirmed_split) {
            manager->split_confirm_mode = confirm_mode;
            manager->split_confirm_count = std::max(manager->split_confirm_count, 2);
            manager->split_confirmed = true;
            decision.final_split = true;
            decision.reason = confirmed_reason;
            manager->final_semantic_reason = decision.reason;
            return decision;
        }

        const int next_confirm_count =
            (isSplitConfirmationEvidenceMode(previous_confirm_mode) || region_continued)
                ? manager->split_confirm_count + 1
                : 1;

        // A4: frame closing veto for fresh F+T short_opening entries
        const bool fresh_entry =
            !manager->visible_split_output_active && !already_confirmed_split;
        const bool selected_ft_opening =
            selected_candidate != nullptr &&
            candidateSourceContains(*selected_candidate, "pair_type=F+T") &&
            candidateSourceContains(*selected_candidate, "pair_semantic=opening");
        const bool frame_closing_veto_active =
            fresh_entry &&
            selected_ft_opening &&
            confirm_mode == "short_opening" &&
            next_confirm_count >= 2 &&
            (frame_closing_veto.has_strong_closing_veto ||
             frame_closing_veto.best_closing_score >= 0.90f ||
             frame_closing_veto.best_closing_score >
                 manager->selected_opening_score + 0.25f);
        if (frame_closing_veto_active) {
            manager->split_confirm_mode = confirm_mode;
            manager->split_confirm_count = 1;
            manager->split_confirmed = false;
            manager->opening_blocked_by_frame_closing_veto = true;
            manager->opening_block_reason =
                frame_closing_veto.has_strong_closing_veto
                    ? "frame_strong_closing_veto"
                    : (frame_closing_veto.best_closing_score >= 0.90f
                           ? "frame_closing_score_high"
                           : "frame_closing_beats_selected_opening");
            decision.reject_reason = "short_opening_blocked_by_frame_closing_veto";
            decision.reason = "short_opening_blocked_by_frame_closing_veto";
            manager->split_confirm_reject_reason = decision.reject_reason;
            manager->final_semantic_reason = decision.reason;
            return decision;
        }

        manager->split_confirm_count = next_confirm_count;
        manager->split_confirm_mode = confirm_mode;
        manager->split_confirmed = manager->split_confirm_count >= 2;
        if (manager->split_confirmed) {
            decision.final_split = true;
            decision.reason = confirmed_reason;
            manager->final_semantic_reason = decision.reason;
            return decision;
        }
        decision.reject_reason = pending_reason;
        decision.reason = pending_reason;
        manager->final_semantic_reason = decision.reason;
        return decision;
    }

    // Fallback: physical_provider_v2 but unknown pair type
    if (manager->selected_pair_source == "physical_provider_v2") {
        decision.reject_reason = "unknown_pair_not_confirmed";
        decision.reason = "unknown_pair_not_confirmed";
        manager->final_semantic_reason = decision.reason;
        if (!active_visible_split_context) {
            clearPhysicalSplitConfirmation(manager, decision.reject_reason);
        }
        return decision;
    }

    return decision;
}

static std::vector<ArrowBranch> splitArrowGuidesIntoBranchesTwoStep(
    const std::vector<ArrowGuide>& guides,
    uint32_t frame_id,
    int,
    int,
    BranchDebugInfo* debug_info,
    BoundaryTrackManager* boundary_tracks,
    LineLossTraceFields* line_loss) {
    if (debug_info) {
        *debug_info = BranchDebugInfo{};
        debug_info->status = "single";
    }

    std::vector<ArrowGuide> main_guides =
        selectSingleMainGuidesAnchorGrow(
            guides, frame_id, boundary_tracks, 1.0f, line_loss, debug_info);
    if (main_guides.empty()) {
        return {};
    }

    float median_hw = 8.0f;
    std::vector<float> hws;
    hws.reserve(main_guides.size());
    for (const auto& guide : main_guides) {
        hws.push_back(guidePlanHalfWidth(guide));
    }
    if (!hws.empty()) {
        const size_t mid = hws.size() / 2;
        std::nth_element(hws.begin(),
                         hws.begin() + static_cast<std::ptrdiff_t>(mid),
                         hws.end());
        median_hw = hws[mid];
    }

    PolyFit2 main_fit;
    finalizeSingleBranchGuides(main_guides, median_hw, 0.0f, &main_fit);

    if (main_guides.empty()) {
        return {};
    }

    float visible_min_forward =
        main_guides.empty() ? 0.0f : guidePlanForward(main_guides.front());
    for (const auto& guide : guides) {
        const float forward = guidePlanForward(guide);
        if (isFiniteFloat(forward)) {
            visible_min_forward = std::min(visible_min_forward, forward);
        }
    }

    float center_ref_lateral = 0.0f;
    if (main_fit.valid && isFiniteFloat(visible_min_forward)) {
        center_ref_lateral = main_fit.eval(visible_min_forward);
    } else if (!main_guides.empty()) {
        center_ref_lateral = guidePlanLateral(main_guides.front());
    }

    std::string provider_summary;
    const std::vector<BoundaryPairCandidate> split_candidates =
        buildPhysicalBoundaryProviderCandidates(guides,
                                                main_guides,
                                                center_ref_lateral,
                                                visible_min_forward,
                                                median_hw,
                                                &provider_summary);

    ForwardGrownVisiblePairCandidate visible_pair =
        buildForwardGrownVisiblePairCandidate(
            guides,
            main_guides,
            visible_min_forward,
            median_hw,
            frame_id,
            boundary_tracks);
    const bool was_recently_split_before_decision =
        boundary_tracks != nullptr && boundary_tracks->visible_split_output_active;
    const int split_maintain_age_before_decision =
        boundary_tracks != nullptr ? boundary_tracks->visible_split_maintain_age : 0;
    const bool split_confirmed_before_frame =
        boundary_tracks != nullptr && boundary_tracks->split_confirmed;
    const int split_confirm_count_before_frame =
        boundary_tracks != nullptr ? boundary_tracks->split_confirm_count : 0;
    const std::string split_confirm_mode_before_frame =
        boundary_tracks != nullptr ? boundary_tracks->split_confirm_mode : "none";
    bool preserve_visible_split_on_single = false;
    int preserved_visible_split_maintain_age = 0;
    bool preserve_recovery_context_on_single = false;

    if (debug_info) {
        debug_info->assigned_points = static_cast<int>(main_guides.size());
        debug_info->rejected_points =
            std::max(0, static_cast<int>(guides.size()) -
                            static_cast<int>(main_guides.size()));
        debug_info->main_forward_span = guideForwardSpan(main_guides);
        debug_info->main_max_dx_jump = maxGuideDxJump(main_guides);
        debug_info->output_branches = main_guides.empty() ? 0 : 1;
        debug_info->boundary_final_split = false;
        debug_info->boundary_final_split_reason = "single_main_anchor_grow";
        std::ostringstream visible_summary;
        visible_summary << "visible_pair_source=" << visible_pair.source
                        << " valid=" << (visible_pair.valid ? 1 : 0)
                        << " provider_candidates=" << split_candidates.size()
                        << " right_fork_candidates="
                        << visible_pair.right_fork_search_candidate_count
                        << " right_fork_anchor="
                        << visible_pair.right_fork_anchor_on_main
                        << " selected_right_fork_model="
                        << visible_pair.selected_right_fork_model
                        << " detached_valid="
                        << (visible_pair.detached_candidate_valid ? 1 : 0)
                        << " detached_reason="
                        << visible_pair.detached_reject_reason
                        << " right_fork_ids="
                        << visible_pair.right_fork_candidate_ids
                        << " reject_reason=" << visible_pair.reject_reason;
        if (!provider_summary.empty()) {
            visible_summary << " " << provider_summary;
        }
        debug_info->physical_boundary_provider_summary = visible_summary.str();
    }
    // ---- demo-aligned split decision / maintain / preserve-state-only ----
    // Save full state before this frame's evaluation
    const bool active_split_state_before_frame =
        boundary_tracks != nullptr &&
        (boundary_tracks->split_confirmed ||
         boundary_tracks->visible_split_output_active ||
         boundary_tracks->visible_split_maintain_age > 0 ||
         hasSplitRecoverySearchContext(boundary_tracks) ||
         !boundary_tracks->last_confirmed_right_fork_ids.empty());
    const std::string split_confirm_reject_reason_before_frame =
        boundary_tracks != nullptr ? boundary_tracks->split_confirm_reject_reason : "none";
    const bool visible_split_output_active_before_frame =
        boundary_tracks != nullptr && boundary_tracks->visible_split_output_active;
    const int visible_split_maintain_age_before_frame =
        boundary_tracks != nullptr ? boundary_tracks->visible_split_maintain_age : 0;
    const bool active_visible_split_context =
        visible_split_output_active_before_frame ||
        visible_split_maintain_age_before_frame > 0;
    const int visible_split_provider_gap_frames_before_frame =
        boundary_tracks != nullptr ? boundary_tracks->visible_split_provider_gap_frames : 0;

    updateBoundaryTracks(
        boundary_tracks,
        split_candidates,
        visible_min_forward,
        debug_info);
    if (boundary_tracks != nullptr) {
        boundary_tracks->selected_right_fork_model = visible_pair.selected_right_fork_model;
        boundary_tracks->detached_candidate_valid = visible_pair.detached_candidate_valid;
        boundary_tracks->detached_candidate_ids = visible_pair.detached_candidate_ids;
        boundary_tracks->detached_gate_trace = visible_pair.detached_gate_trace;
        boundary_tracks->detached_score = visible_pair.detached_score;
        boundary_tracks->detached_structural_gate_pass =
            visible_pair.detached_structural_gate_pass;
        boundary_tracks->detached_right_side_order_ok =
            visible_pair.detached_right_side_order_ok;
        boundary_tracks->detached_crossing_detected =
            visible_pair.detached_crossing_detected;
        boundary_tracks->detached_remerge_detected =
            visible_pair.detached_remerge_detected;
        boundary_tracks->detached_independent_id_count =
            visible_pair.detached_independent_id_count;
        boundary_tracks->detached_main_overlap = visible_pair.detached_main_overlap;
        boundary_tracks->detached_main_reuse_ratio =
            visible_pair.detached_main_reuse_ratio;
        boundary_tracks->detached_common_forward_overlap =
            visible_pair.detached_common_forward_overlap;
        boundary_tracks->detached_right_chain_count =
            visible_pair.detached_right_chain_count;
        boundary_tracks->detached_right_chain_forward_span =
            visible_pair.detached_right_chain_forward_span;
        boundary_tracks->detached_right_chain_lateral_span =
            visible_pair.detached_right_chain_lateral_span;
        boundary_tracks->detached_lateral_separation =
            visible_pair.lateral_separation;
        boundary_tracks->detached_right_chain_continuity =
            visible_pair.detached_right_chain_continuity;
        boundary_tracks->detached_width_min = visible_pair.detached_width_min;
        boundary_tracks->detached_width_max = visible_pair.detached_width_max;
        boundary_tracks->detached_width_mean = visible_pair.detached_width_mean;
        boundary_tracks->detached_width_variation =
            visible_pair.detached_width_variation;
        boundary_tracks->detached_sample_count = visible_pair.detached_sample_count;
        boundary_tracks->detached_valid_width_count =
            visible_pair.detached_valid_width_count;
        boundary_tracks->detached_trend_sample_count =
            visible_pair.detached_trend_sample_count;
        boundary_tracks->detached_trend_width_near =
            visible_pair.detached_trend_width_near;
        boundary_tracks->detached_trend_width_far =
            visible_pair.detached_trend_width_far;
        boundary_tracks->detached_trend_width_delta =
            visible_pair.detached_trend_width_delta;
        boundary_tracks->detached_trend_width_slope =
            visible_pair.detached_trend_width_slope;
        boundary_tracks->detached_trend_far_width_ratio =
            visible_pair.detached_trend_far_width_ratio;
        boundary_tracks->detached_trend_tail_min_width =
            visible_pair.detached_trend_tail_min_width;
    }
    const BoundaryPairCandidate* recovery_selected_candidate = nullptr;
    if (boundary_tracks != nullptr &&
        boundary_tracks->selected_pair_rank >= 0 &&
        boundary_tracks->selected_pair_rank <
            static_cast<int>(split_candidates.size())) {
        recovery_selected_candidate =
            &split_candidates[static_cast<size_t>(boundary_tracks->selected_pair_rank)];
    }
    const StableDoubleBoundaryRecoveryEvaluation recovery_evaluation =
        evaluateStableDoubleBoundaryRecovery(
            frame_id,
            boundary_tracks,
            main_guides,
            visible_pair,
            recovery_selected_candidate,
            visible_split_output_active_before_frame,
            split_confirmed_before_frame);
    preserve_recovery_context_on_single =
        recovery_evaluation.evidence ||
        hasSplitRecoverySearchContext(boundary_tracks);
    if (debug_info != nullptr && boundary_tracks != nullptr) {
        debug_info->stable_double_boundary_recovery_evidence =
            recovery_evaluation.evidence;
        debug_info->stable_double_boundary_recovery_confirmed_this_frame =
            recovery_evaluation.confirmed_this_frame;
        debug_info->split_recovery_context_active =
            boundary_tracks->split_recovery_context_active;
        debug_info->split_recovery_context_age =
            boundary_tracks->split_recovery_context_age;
        debug_info->stable_double_boundary_recovery_count =
            boundary_tracks->stable_double_boundary_recovery_count;
        debug_info->split_recovery_context_reason =
            boundary_tracks->split_recovery_context_reason;
        debug_info->stable_double_boundary_recovery_reject_reason =
            boundary_tracks->stable_double_boundary_recovery_reject_reason;
        debug_info->split_recovery_seed_main_ids = formatTrackerIdsCompact(
            boundary_tracks->split_recovery_seed_main_ids);
        debug_info->split_recovery_seed_right_ids = formatTrackerIdsCompact(
            boundary_tracks->split_recovery_seed_right_ids);
    }
    BoundaryFinalSplitDecision final_split_decision =
        computeBoundaryFinalSplitDecision(boundary_tracks, split_candidates);

    const BoundaryPairCandidate* selected_candidate = nullptr;
    if (boundary_tracks != nullptr &&
        boundary_tracks->selected_pair_rank >= 0 &&
        boundary_tracks->selected_pair_rank < static_cast<int>(split_candidates.size())) {
        selected_candidate =
            &split_candidates[static_cast<size_t>(boundary_tracks->selected_pair_rank)];
    }
    const auto extractSourceToken = [](const std::string& source,
                                       const std::string& key) {
        const std::string needle = key + "=";
        const size_t begin = source.find(needle);
        if (begin == std::string::npos) {
            return std::string("none");
        }
        const size_t value_begin = begin + needle.size();
        const size_t end = source.find(':', value_begin);
        return source.substr(value_begin,
                             end == std::string::npos ? std::string::npos
                                                      : end - value_begin);
    };
    const std::string selected_semantic_pair_type =
        selected_candidate != nullptr
            ? extractSourceToken(selected_candidate->source, "pair_type")
            : "none";
    const std::string selected_semantic_pair_semantic =
        selected_candidate != nullptr
            ? extractSourceToken(selected_candidate->source, "pair_semantic")
            : "none";
    const std::string selected_provider_semantic_candidate_source =
        selected_candidate != nullptr ? selected_candidate->source : "none";
    const std::string selected_provider_semantic_reason =
        selected_candidate != nullptr
            ? extractSourceToken(selected_candidate->source, "pair_semantic_reason")
            : "none";

    // ---- visible veto reason (demo-aligned, uses frame-level arbitration result) ----
    const bool semantic_usable_opening_evidence =
        (boundary_tracks != nullptr) &&
        ((boundary_tracks->frame_semantic_pending &&
          (boundary_tracks->frame_semantic_mode == "opening" ||
           boundary_tracks->frame_semantic_mode == "short_opening" ||
           boundary_tracks->frame_semantic_mode == "uncertain_opening")) ||
         (boundary_tracks->frame_semantic_ok &&
          boundary_tracks->frame_semantic_mode == "stable_split"));
    std::string visible_veto_reason = "none";
    if (boundary_tracks != nullptr) {
        if (boundary_tracks->frame_semantic_suppressed) {
            visible_veto_reason = boundary_tracks->frame_semantic_reason;
        } else if (boundary_tracks->selected_strong_closing_veto) {
            visible_veto_reason = "selected_strong_closing_veto";
        } else if (boundary_tracks->selected_closing_like) {
            visible_veto_reason = "selected_closing_like";
        } else if (boundary_tracks->selected_closing_suspect &&
                   !semantic_usable_opening_evidence) {
            visible_veto_reason = "selected_closing_suspect";
        }
    }
    if (visible_veto_reason != "none") {
        if (!active_visible_split_context) {
            clearPhysicalSplitConfirmation(boundary_tracks, visible_veto_reason);
        }
        if (boundary_tracks != nullptr) {
            boundary_tracks->final_semantic_reason = visible_veto_reason;
        }
        final_split_decision.final_split = false;
        final_split_decision.reason = visible_veto_reason;
        final_split_decision.reject_reason = visible_veto_reason;
    }

    // ---- visible split gate base pass ----
    // A-2: demo-aligned guard conditions (defense-in-depth beyond visible_veto_reason).
    // Even though visible_veto_reason handles closing/suspect via
    // final_split_decision.final_split, these gate-level checks protect against
    // edge cases where preserve-state-only restores stale confirmation state
    // while frame-level semantic suppression or closing evidence should still
    // block visible split output.
    const bool visible_split_gate_base_pass =
        final_split_decision.final_split &&
        boundary_tracks != nullptr &&
        boundary_tracks->split_confirmed &&
        visible_pair.valid &&
        boundary_tracks->selected_pair_source == "physical_provider_v2" &&
        !boundary_tracks->frame_semantic_suppressed &&
        !boundary_tracks->selected_strong_closing_veto &&
        !boundary_tracks->selected_closing_like &&
        (!boundary_tracks->selected_closing_suspect ||
         semantic_usable_opening_evidence);

    // ---- weak branch / maturity ----
    const bool semantic_provider_available =
        selected_candidate != nullptr &&
        boundary_tracks != nullptr &&
        boundary_tracks->selected_pair_rank >= 0;
    const bool provider_selected_pair_available = semantic_provider_available;
    const bool selector_pair_width_out_of_range =
        boundary_tracks != nullptr &&
        boundary_tracks->selector_pair_width_out_of_range;
    const bool selector_split_pair_unavailable =
        boundary_tracks != nullptr &&
        boundary_tracks->selector_split_pair_unavailable;
    const bool selector_non_physical_v2_candidate =
        boundary_tracks != nullptr &&
        boundary_tracks->selector_non_physical_v2_candidate;
    const bool weak_branch_maintain_grace_available =
        was_recently_split_before_decision &&
        visible_pair.valid;
    const int normal_branch_min_required =
        weak_branch_maintain_grace_available ? 2 : 3;
    const int weak_branch_count =
        visible_pair.valid
            ? std::min(static_cast<int>(visible_pair.left_guides.size()),
                       static_cast<int>(visible_pair.right_guides.size()))
            : 0;
    bool maturity_count_pass =
        weak_branch_count >= normal_branch_min_required;
    bool visible_split_allowed = visible_split_gate_base_pass;
    if (visible_split_gate_base_pass && !maturity_count_pass) {
        visible_split_allowed = false;
    }
    const std::string maturity_count_fail_reason =
        maturity_count_pass
            ? "none"
            : ("normal_branch_count_lt_" +
               std::to_string(normal_branch_min_required));
    std::string visible_gate_fail_reason = "none";
    if (!visible_split_allowed && visible_gate_fail_reason == "none") {
        if (!final_split_decision.final_split) {
            visible_gate_fail_reason =
                !final_split_decision.reject_reason.empty() &&
                final_split_decision.reject_reason != "none"
                    ? final_split_decision.reject_reason
                    : final_split_decision.reason;
        } else if (boundary_tracks != nullptr && !boundary_tracks->split_confirmed) {
            visible_gate_fail_reason = "split_not_confirmed";
        } else if (!visible_pair.valid) {
            visible_gate_fail_reason = visible_pair.source;
        } else if (!maturity_count_pass) {
            visible_gate_fail_reason = maturity_count_fail_reason;
        } else if (boundary_tracks != nullptr &&
                   boundary_tracks->selected_pair_source != "physical_provider_v2") {
            visible_gate_fail_reason = "non_physical_selected_pair";
        } else if (boundary_tracks != nullptr &&
                   boundary_tracks->frame_semantic_suppressed) {
            visible_gate_fail_reason = "frame_semantic_suppressed";
        } else if (visible_veto_reason != "none") {
            visible_gate_fail_reason = visible_veto_reason;
        } else {
            visible_gate_fail_reason = "visible_gate_rejected";
        }
    }

    // ---- original_final_split (before maintain override) ----
    const bool original_final_split = visible_split_allowed;
    const std::string original_block_reason =
        original_final_split ? "none" : visible_gate_fail_reason;

    // ---- structural hard gate for maintain ----
    const bool provider_width_finite =
        boundary_tracks != nullptr &&
        isFiniteFloat(boundary_tracks->final_pair_width_m) &&
        boundary_tracks->final_pair_width_m > 0.0f;
    std::string split_maintain_structural_fail_reason = "none";
    if (!visible_pair.valid) {
        split_maintain_structural_fail_reason =
            visible_pair.reject_reason != "none" ? visible_pair.reject_reason
                                                 : "visible_pair_invalid";
    } else if (visible_pair.reject_reason != "none") {
        split_maintain_structural_fail_reason = visible_pair.reject_reason;
    } else if (!maturity_count_pass) {
        split_maintain_structural_fail_reason = maturity_count_fail_reason;
    } else if (selector_pair_width_out_of_range) {
        split_maintain_structural_fail_reason = "pair_width_out_of_range";
    } else if (!was_recently_split_before_decision && !semantic_provider_available) {
        split_maintain_structural_fail_reason = "semantic_provider_unavailable";
    } else if (!was_recently_split_before_decision && selector_split_pair_unavailable) {
        split_maintain_structural_fail_reason = "split_pair_unavailable";
    } else if (!was_recently_split_before_decision &&
               boundary_tracks != nullptr &&
               (boundary_tracks->selected_pair_source != "physical_provider_v2" ||
                selected_candidate == nullptr ||
                selected_candidate->provider_type != "physical_provider_v2")) {
        split_maintain_structural_fail_reason = "non_physical_v2_candidate";
    } else if (!was_recently_split_before_decision &&
               selector_non_physical_v2_candidate) {
        split_maintain_structural_fail_reason = "non_physical_v2_candidate";
    } else if (!provider_width_finite &&
               (!was_recently_split_before_decision ||
                provider_selected_pair_available)) {
        split_maintain_structural_fail_reason = "provider_width_invalid";
    }
    const bool split_maintain_structural_ok =
        split_maintain_structural_fail_reason == "none";
    const bool structural_hard_gate_pass = split_maintain_structural_ok;
    const std::string structural_hard_gate_fail_reason =
        structural_hard_gate_pass ? "none" : split_maintain_structural_fail_reason;

    // ---- split entry / maintain classification ----
    const bool split_entry_by_semantic_confirm =
        !was_recently_split_before_decision && original_final_split;
    const bool split_maintain_by_structure =
        was_recently_split_before_decision && structural_hard_gate_pass;
    const bool semantic_exit_veto_ignored =
        split_maintain_by_structure && !original_final_split;

    // ---- transient failure type check for preserve-state-only ----
    const bool no_anchor_on_main_seen =
        visible_pair.no_anchor_on_main_seen ||
        visible_pair.reject_reason == "no_anchor_on_main";
    const bool candidate_reuses_main_forward_seen =
        visible_pair.candidate_reuses_main_forward_seen ||
        visible_pair.reject_reason == "candidate_reuses_main_forward";
    const bool split_confirmed_only_context =
        split_confirmed_before_frame && !active_visible_split_context;
    const bool transient_failure_type_ok =
        !visible_pair.valid &&
        (no_anchor_on_main_seen ||
         visible_pair.reject_reason == "no_anchor_on_main" ||
         (candidate_reuses_main_forward_seen && no_anchor_on_main_seen) ||
         visible_pair.suffix_continuation_found ||
         visible_pair.suffix_continuation_reason == "no_last_confirmed_right_fork");
    const bool independent_right_fork_evidence =
        visible_pair.independent_right_id_count >= 3 &&
        visible_pair.right_main_overlap <= 0.35f &&
        visible_pair.lateral_separation >= 0.05f;
    const int consecutive_invalid_visible_pair_count =
        visible_pair.valid ? 0 : (visible_split_provider_gap_frames_before_frame + 1);
    const bool grace_budget_ok =
        visible_pair.valid || consecutive_invalid_visible_pair_count <= 1;
    const bool main_reuse_ratio_high =
        visible_pair.main_reuse_ratio_max >= 0.90f &&
        !independent_right_fork_evidence;
    const bool lateral_separation_low =
        visible_pair.lateral_separation < 0.05f;
    const bool right_main_overlap_high =
        visible_pair.right_main_overlap > 0.45f;
    const bool independent_right_ids_low =
        visible_pair.independent_right_id_count < 3;

    // Demo-aligned: provider_semantic_not_opening_seen / weak negative
    const bool provider_semantic_not_opening_seen =
        selected_semantic_pair_semantic != "opening";
    const bool provider_semantic_used_as_weak_negative =
        provider_semantic_not_opening_seen &&
        (main_reuse_ratio_high ||
         lateral_separation_low ||
         right_main_overlap_high ||
         independent_right_ids_low);

    // ---- structural_strong_negative evaluation ----
    bool structural_strong_negative = false;
    std::string structural_strong_negative_reason = "none";
    auto markStructuralNegative = [&](const std::string& reason) {
        if (!structural_strong_negative) {
            structural_strong_negative = true;
            structural_strong_negative_reason = reason;
        }
    };
    if (!active_visible_split_context) {
        markStructuralNegative(split_confirmed_only_context
                                   ? "split_confirmed_only_context"
                                   : "no_active_visible_split_context");
    }
    if (!grace_budget_ok) {
        markStructuralNegative("consecutive_invalid_visible_pair");
    }
    if (main_reuse_ratio_high) {
        markStructuralNegative("main_reuse_ratio_high");
    }
    if (lateral_separation_low) {
        markStructuralNegative("lateral_separation_low");
    }
    if (right_main_overlap_high) {
        markStructuralNegative("right_main_overlap_high");
    }
    if (independent_right_ids_low) {
        markStructuralNegative("independent_right_id_count_low");
    }
    if (boundary_tracks != nullptr &&
        boundary_tracks->selector_pair_width_out_of_range) {
        markStructuralNegative("pair_width_out_of_range");
    }
    const bool geometry_not_bad =
        !lateral_separation_low &&
        !right_main_overlap_high &&
        !main_reuse_ratio_high &&
        (boundary_tracks == nullptr ||
         !boundary_tracks->selector_pair_width_out_of_range);

    // ---- preserve_state_without_forced_output ----
    const bool preserve_state_without_forced_output =
        active_visible_split_context &&
        transient_failure_type_ok &&
        independent_right_fork_evidence &&
        geometry_not_bad &&
        !structural_strong_negative &&
        grace_budget_ok;
    const bool active_no_anchor_preserve_state_only =
        preserve_state_without_forced_output && no_anchor_on_main_seen;
    const bool active_reuse_main_forward_preserve_state_only =
        preserve_state_without_forced_output &&
        !no_anchor_on_main_seen &&
        candidate_reuses_main_forward_seen;
    std::string preserve_blocked_reason = "none";
    if (!preserve_state_without_forced_output &&
        (no_anchor_on_main_seen || candidate_reuses_main_forward_seen)) {
        if (!active_visible_split_context) {
            preserve_blocked_reason =
                split_confirmed_only_context
                    ? "split_confirmed_only_context"
                    : "no_active_visible_split_context";
        } else if (!transient_failure_type_ok) {
            preserve_blocked_reason = "transient_failure_type_not_ok";
        } else if (!independent_right_fork_evidence) {
            preserve_blocked_reason = "independent_right_fork_evidence_missing";
        } else if (!geometry_not_bad) {
            preserve_blocked_reason = structural_strong_negative_reason;
        } else if (structural_strong_negative) {
            preserve_blocked_reason = structural_strong_negative_reason;
        } else if (!grace_budget_ok) {
            preserve_blocked_reason = "grace_budget_exhausted";
        }
    }
    if (boundary_tracks != nullptr) {
        boundary_tracks->provider_semantic_not_opening_seen =
            provider_semantic_not_opening_seen;
        boundary_tracks->provider_semantic_used_as_weak_negative =
            provider_semantic_used_as_weak_negative;
        boundary_tracks->preserve_blocked_reason = preserve_blocked_reason;
    }

    // ---- visible_split_allowed after maintain override ----
    bool split_maintain_used = split_maintain_by_structure && !original_final_split;
    std::string split_maintain_reason =
        split_maintain_used ? "maintain_split_by_structure" : "none";

    visible_split_allowed = was_recently_split_before_decision
        ? structural_hard_gate_pass
        : original_final_split;
    visible_gate_fail_reason = visible_split_allowed
        ? "none"
        : (was_recently_split_before_decision
               ? structural_hard_gate_fail_reason
               : original_block_reason);

    // ---- provider gap grace ----
    const bool provider_grace_candidate =
        was_recently_split_before_decision &&
        visible_pair.valid &&
        !visible_split_allowed &&
        structural_hard_gate_fail_reason == "semantic_provider_unavailable";
    bool provider_grace_used = false;

    // ---- Apply preserve / grace decisions to boundary_tracks state ----
    bool preserved_visible_split_output_active = false;
    if (boundary_tracks != nullptr) {
        if (structural_hard_gate_pass) {
            boundary_tracks->visible_split_provider_gap_frames = 0;
        } else if (preserve_state_without_forced_output) {
            boundary_tracks->split_confirm_mode = split_confirm_mode_before_frame;
            boundary_tracks->split_confirm_count = split_confirm_count_before_frame;
            boundary_tracks->split_confirmed = split_confirmed_before_frame;
            boundary_tracks->split_confirm_reject_reason =
                split_confirm_reject_reason_before_frame;
            boundary_tracks->visible_split_output_active =
                visible_split_output_active_before_frame;
            boundary_tracks->visible_split_maintain_age =
                visible_split_maintain_age_before_frame;
            boundary_tracks->visible_split_provider_gap_frames =
                consecutive_invalid_visible_pair_count;
            preserve_visible_split_on_single = true;
            preserved_visible_split_output_active =
                visible_split_output_active_before_frame;
            preserved_visible_split_maintain_age =
                visible_split_maintain_age_before_frame;
            split_maintain_reason =
                active_no_anchor_preserve_state_only
                    ? "no_anchor_on_main_preserve_state_only"
                    : "candidate_reuses_main_forward_preserve_state_only";
        } else if (provider_grace_candidate) {
            boundary_tracks->visible_split_provider_gap_frames += 1;
            provider_grace_used =
                boundary_tracks->visible_split_provider_gap_frames <= 1;
            if (provider_grace_used) {
                preserve_visible_split_on_single = true;
                preserved_visible_split_output_active = true;
                preserved_visible_split_maintain_age =
                    split_maintain_age_before_decision;
                boundary_tracks->visible_split_output_active = true;
                boundary_tracks->visible_split_maintain_age =
                    split_maintain_age_before_decision;
            } else {
                boundary_tracks->visible_split_provider_gap_frames = 0;
            }
        } else if (!visible_pair.valid ||
                   structural_hard_gate_fail_reason != "semantic_provider_unavailable") {
            boundary_tracks->visible_split_provider_gap_frames = 0;
        }
        if (was_recently_split_before_decision &&
            structural_hard_gate_pass &&
            original_final_split) {
            split_maintain_reason = "structure_confirmed_existing_split";
        }
    }

    // ---- fallback with preserve-state ----
    auto fallbackSingle = [&]() {
        if (boundary_tracks != nullptr) {
            boundary_tracks->previous_accepted_main_ids =
                trackerIdsFromGuidesForIdentity(main_guides);
            if (preserve_visible_split_on_single) {
                boundary_tracks->visible_split_output_active =
                    preserved_visible_split_output_active;
                boundary_tracks->visible_split_maintain_age =
                    preserved_visible_split_maintain_age;
            } else if (preserve_recovery_context_on_single) {
                boundary_tracks->visible_split_output_active = false;
                boundary_tracks->visible_split_maintain_age = 0;
            } else {
                boundary_tracks->visible_split_output_active = false;
                boundary_tracks->visible_split_maintain_age = 0;
                boundary_tracks->visible_split_provider_gap_frames = 0;
                boundary_tracks->previous_output_main_ids.clear();
                boundary_tracks->previous_output_right_ids.clear();
                boundary_tracks->pending_main_challenger_ids.clear();
                boundary_tracks->pending_main_challenger_confirm_count = 0;
                boundary_tracks->main_identity_reverse_lock_remaining = 0;
                boundary_tracks->last_switched_from_main_ids.clear();
            }
        }
        if (debug_info != nullptr) {
            debug_info->output_branches = 1;
        }
        return std::vector<ArrowBranch>{
            buildSingleMainBranch(std::move(main_guides), main_fit)};
    };

    // ---- debug_info population for split decision ----
    if (debug_info) {
        debug_info->boundary_final_split = visible_split_allowed;
        debug_info->boundary_final_split_reason =
            visible_split_allowed
                ? (was_recently_split_before_decision
                       ? (split_maintain_reason != "none"
                              ? split_maintain_reason
                              : "maintain_split_by_structure")
                       : final_split_decision.reason)
                : (was_recently_split_before_decision
                       ? structural_hard_gate_fail_reason
                       : final_split_decision.reason);
        debug_info->boundary_reject_reason =
            visible_split_allowed
                ? "none"
                : (was_recently_split_before_decision
                       ? structural_hard_gate_fail_reason
                       : final_split_decision.reject_reason);
        debug_info->selected_semantic_pair_type = selected_semantic_pair_type;
        debug_info->selected_semantic_pair_semantic = selected_semantic_pair_semantic;
        debug_info->provider_semantic_candidate_source =
            selected_provider_semantic_candidate_source;
        debug_info->provider_semantic_reason = selected_provider_semantic_reason;
        debug_info->usable_opening_evidence = semantic_usable_opening_evidence;
        debug_info->split_maintain_used = split_maintain_used;
        debug_info->split_maintain_reason = split_maintain_reason;
        debug_info->split_maintain_age = split_maintain_age_before_decision;
        debug_info->original_final_split = original_final_split;
        debug_info->original_block_reason = original_block_reason;
        debug_info->final_split_after_maintain = visible_split_allowed;
        debug_info->was_recently_split_before_decision =
            was_recently_split_before_decision;
        debug_info->visible_split_output_active =
            boundary_tracks != nullptr ? boundary_tracks->visible_split_output_active : false;
        debug_info->visible_split_maintain_age =
            boundary_tracks != nullptr ? boundary_tracks->visible_split_maintain_age : 0;
        debug_info->split_confirm_mode =
            boundary_tracks != nullptr ? boundary_tracks->split_confirm_mode : "none";
        debug_info->split_confirm_count =
            boundary_tracks != nullptr ? boundary_tracks->split_confirm_count : 0;
        debug_info->split_confirmed =
            boundary_tracks != nullptr && boundary_tracks->split_confirmed;
        debug_info->split_confirm_reject_reason =
            boundary_tracks != nullptr ? boundary_tracks->split_confirm_reject_reason : "none";
        debug_info->final_semantic_reason =
            boundary_tracks != nullptr ? boundary_tracks->final_semantic_reason : "none";
        debug_info->has_ft_opening_branch_support =
            boundary_tracks != nullptr &&
            boundary_tracks->has_ft_opening_branch_support;
        debug_info->ft_support_reason =
            boundary_tracks != nullptr ? boundary_tracks->ft_support_reason : "none";
        debug_info->ft_support_t_ids =
            boundary_tracks != nullptr ? boundary_tracks->ft_support_t_ids : "";
        debug_info->ft_support_t_fwd =
            boundary_tracks != nullptr ? boundary_tracks->ft_support_t_fwd : "";
        debug_info->ft_support_t_lat =
            boundary_tracks != nullptr ? boundary_tracks->ft_support_t_lat : "";
        debug_info->ft_support_t_lat_sp =
            boundary_tracks != nullptr ? boundary_tracks->ft_support_t_lat_sp : 0.0f;
        debug_info->ft_support_t_n =
            boundary_tracks != nullptr ? boundary_tracks->ft_support_t_n : 0;
        debug_info->ft_support_t_near_ok =
            boundary_tracks != nullptr && boundary_tracks->ft_support_t_near_ok;
        debug_info->ft_support_t_near_cov =
            boundary_tracks != nullptr ? boundary_tracks->ft_support_t_near_cov : 0;
        debug_info->ft_support_open_branch_score =
            boundary_tracks != nullptr
                ? boundary_tracks->ft_support_open_branch_score
                : 0.0f;
        debug_info->ft_support_merge_band_score =
            boundary_tracks != nullptr
                ? boundary_tracks->ft_support_merge_band_score
                : 0.0f;
        debug_info->no_anchor_on_main_seen = no_anchor_on_main_seen;
        debug_info->no_anchor_on_main_preserve_state_only =
            active_no_anchor_preserve_state_only;
        debug_info->candidate_reuses_main_forward_seen =
            candidate_reuses_main_forward_seen;
        debug_info->candidate_reuses_main_forward_preserve_state_only =
            active_reuse_main_forward_preserve_state_only;
        debug_info->active_visible_split_context = active_visible_split_context;
        debug_info->split_confirmed_only_context = split_confirmed_only_context;
        debug_info->transient_failure_type_ok = transient_failure_type_ok;
        debug_info->independent_right_fork_evidence =
            independent_right_fork_evidence;
        debug_info->selected_right_fork_model =
            visible_pair.selected_right_fork_model;
        debug_info->detached_candidate_found =
            visible_pair.detached_candidate_found;
        debug_info->detached_candidate_valid =
            visible_pair.detached_candidate_valid;
        debug_info->detached_reject_reason =
            visible_pair.detached_reject_reason;
        debug_info->detached_candidate_ids = visible_pair.detached_candidate_ids;
        debug_info->detached_gate_trace = visible_pair.detached_gate_trace;
        debug_info->detached_score = visible_pair.detached_score;
        debug_info->detached_independent_id_count =
            visible_pair.detached_independent_id_count;
        debug_info->detached_main_overlap =
            visible_pair.detached_main_overlap;
        debug_info->detached_main_reuse_ratio =
            visible_pair.detached_main_reuse_ratio;
        debug_info->detached_common_forward_overlap =
            visible_pair.detached_common_forward_overlap;
        debug_info->detached_right_chain_count =
            visible_pair.detached_right_chain_count;
        debug_info->detached_right_chain_forward_span =
            visible_pair.detached_right_chain_forward_span;
        debug_info->detached_right_chain_lateral_span =
            visible_pair.detached_right_chain_lateral_span;
        debug_info->detached_lateral_separation =
            visible_pair.lateral_separation;
        debug_info->detached_right_chain_continuity =
            visible_pair.detached_right_chain_continuity;
        debug_info->detached_structural_gate_pass =
            visible_pair.detached_structural_gate_pass;
        debug_info->detached_right_side_order_ok =
            visible_pair.detached_right_side_order_ok;
        debug_info->detached_crossing_detected =
            visible_pair.detached_crossing_detected;
        debug_info->detached_remerge_detected =
            visible_pair.detached_remerge_detected;
        debug_info->detached_width_min = visible_pair.detached_width_min;
        debug_info->detached_width_max = visible_pair.detached_width_max;
        debug_info->detached_width_mean = visible_pair.detached_width_mean;
        debug_info->detached_width_variation = visible_pair.detached_width_variation;
        debug_info->detached_sample_count = visible_pair.detached_sample_count;
        debug_info->detached_valid_width_count =
            visible_pair.detached_valid_width_count;
        debug_info->detached_trend_sample_count =
            visible_pair.detached_trend_sample_count;
        debug_info->detached_trend_width_near =
            visible_pair.detached_trend_width_near;
        debug_info->detached_trend_width_far =
            visible_pair.detached_trend_width_far;
        debug_info->detached_trend_width_delta =
            visible_pair.detached_trend_width_delta;
        debug_info->detached_trend_width_slope =
            visible_pair.detached_trend_width_slope;
        debug_info->detached_trend_far_width_ratio =
            visible_pair.detached_trend_far_width_ratio;
        debug_info->detached_trend_tail_min_width =
            visible_pair.detached_trend_tail_min_width;
        debug_info->detached_fresh_entry_gate_evaluated =
            boundary_tracks != nullptr && boundary_tracks->detached_fresh_entry_gate_evaluated;
        debug_info->detached_fresh_entry_gate_allowed =
            boundary_tracks != nullptr && boundary_tracks->detached_fresh_entry_gate_allowed;
        debug_info->detached_fresh_entry_gate_reason =
            boundary_tracks != nullptr
                ? boundary_tracks->detached_fresh_entry_gate_reason
                : "not_evaluated";
        debug_info->detached_fresh_entry_basic_valid =
            boundary_tracks != nullptr && boundary_tracks->detached_fresh_entry_basic_valid;
        debug_info->detached_fresh_entry_structural_support =
            boundary_tracks != nullptr && boundary_tracks->detached_fresh_entry_structural_support;
        debug_info->detached_fresh_entry_specific_closing =
            boundary_tracks != nullptr && boundary_tracks->detached_fresh_entry_specific_closing;
        debug_info->detached_fresh_entry_structural_reason =
            boundary_tracks != nullptr
                ? boundary_tracks->detached_fresh_entry_structural_reason
                : "not_evaluated";
        debug_info->detached_fresh_entry_closing_reason =
            boundary_tracks != nullptr
                ? boundary_tracks->detached_fresh_entry_closing_reason
                : "not_evaluated";
        debug_info->detached_fresh_entry_support_score =
            boundary_tracks != nullptr
                ? boundary_tracks->detached_fresh_entry_support_score
                : 0;
        debug_info->detached_fresh_entry_same_region_closing_count =
            boundary_tracks != nullptr
                ? boundary_tracks->detached_fresh_entry_same_region_closing_count
                : 0;
        debug_info->geometry_not_bad = geometry_not_bad;
        debug_info->structural_strong_negative = structural_strong_negative;
        debug_info->structural_strong_negative_reason =
            structural_strong_negative_reason;
        debug_info->provider_semantic_not_opening_seen =
            provider_semantic_not_opening_seen;
        debug_info->provider_semantic_used_as_weak_negative =
            provider_semantic_used_as_weak_negative;
        debug_info->preserve_state_only = preserve_state_without_forced_output;
        debug_info->preserve_blocked_reason = preserve_blocked_reason;
        debug_info->grace_budget_ok = grace_budget_ok;
        debug_info->consecutive_invalid_visible_pair_count =
            consecutive_invalid_visible_pair_count;
        debug_info->structural_hard_gate_pass = structural_hard_gate_pass;
        debug_info->structural_hard_gate_fail_reason =
            structural_hard_gate_fail_reason;
        debug_info->split_maintain_structural_ok = split_maintain_structural_ok;
        debug_info->split_maintain_structural_fail_reason =
            split_maintain_structural_fail_reason;
        debug_info->split_entry_by_semantic_confirm = split_entry_by_semantic_confirm;
        debug_info->split_maintain_by_structure = split_maintain_by_structure;
        debug_info->semantic_exit_veto_ignored = semantic_exit_veto_ignored;
        debug_info->visible_split_provider_gap_frames =
            boundary_tracks != nullptr ? boundary_tracks->visible_split_provider_gap_frames : 0;
        debug_info->provider_grace_candidate = provider_grace_candidate;
        debug_info->provider_grace_used = provider_grace_used;
        debug_info->visible_gate_pass = visible_split_allowed;
        debug_info->visible_gate_fail_reason =
            visible_split_allowed ? "none" : visible_gate_fail_reason;
        debug_info->maturity_count_pass = maturity_count_pass;
        debug_info->last_confirmed_right_fork_ids =
            visible_pair.last_confirmed_right_fork_ids;
        debug_info->current_right_fork_candidate_ids =
            visible_pair.current_right_fork_candidate_ids;
        debug_info->suffix_continuation_found =
            visible_pair.suffix_continuation_found;
        debug_info->suffix_continuation_reason =
            visible_pair.suffix_continuation_reason;
    }

    // ---- output split branches ----
    if (!visible_split_allowed) {
        const bool fork_context_like =
            visible_pair.valid &&
            visible_pair.accept_reason == "main_forward_plus_right_fork_branch";
        if (fork_context_like) {
            std::vector<ArrowGuide> rescored_main =
                selectSingleMainGuidesAnchorGrow(
                    guides, frame_id, boundary_tracks,
                    kForkContextTailBonusScale,
                    line_loss,
                    debug_info);
            if (!rescored_main.empty()) {
                main_fit = PolyFit2{};
                finalizeSingleBranchGuides(rescored_main, median_hw,
                                           0.0f, &main_fit);
                main_guides = std::move(rescored_main);
                if (debug_info) {
                    debug_info->main_forward_span =
                        guideForwardSpan(main_guides);
                    debug_info->main_max_dx_jump =
                        maxGuideDxJump(main_guides);
                }
            }
        }
        return fallbackSingle();
    }

    ArrowBranch main_branch =
        buildBoundaryVisibleBranch(visible_pair.left_guides, "main", 0);
    ArrowBranch right_branch =
        buildBoundaryVisibleBranch(visible_pair.right_guides,
                                   "right",
                                   visible_pair.shared_prefix_count);
    if (main_branch.guides.empty() || right_branch.guides.empty()) {
        if (debug_info) {
            debug_info->boundary_final_split = false;
            debug_info->boundary_reject_reason = "selected_branch_empty";
        }
        return fallbackSingle();
    }
    if (boundary_tracks != nullptr) {
        boundary_tracks->visible_split_output_active = true;
        boundary_tracks->visible_split_maintain_age =
            was_recently_split_before_decision ? split_maintain_age_before_decision + 1 : 0;
        boundary_tracks->visible_split_provider_gap_frames = 0;
        boundary_tracks->previous_output_main_ids =
            trackerIdsFromGuidesForIdentity(main_branch.guides);
        boundary_tracks->previous_output_right_ids =
            trackerIdsFromGuidesForIdentity(right_branch.guides);
        boundary_tracks->previous_accepted_main_ids =
            boundary_tracks->previous_output_main_ids;
        if (visible_pair.right_fork_anchor_type == "id" ||
            visible_pair.selected_right_fork_model == "detached") {
            std::vector<int> right_fork_ids =
                collectGuideTrackerIds(visible_pair.right_guides);
            if (!right_fork_ids.empty()) {
                boundary_tracks->last_confirmed_right_fork_ids =
                    right_fork_ids;
                boundary_tracks->last_confirmed_right_fork_ids_text =
                    formatTrackerIdsCompact(
                        boundary_tracks->last_confirmed_right_fork_ids);
                boundary_tracks->last_confirmed_right_fork_frame = frame_id;
            }
        }
    }

    if (debug_info) {
        debug_info->status = "split";
        debug_info->output_branches = 2;
        debug_info->boundary_final_split = true;
        debug_info->boundary_final_split_reason = split_maintain_reason != "none"
            ? split_maintain_reason
            : final_split_decision.reason;
        debug_info->final_output_preserve_labels = true;
        debug_info->final_output_anchor_sort_applied = false;
    }

    std::vector<ArrowBranch> split_branches;
    split_branches.reserve(2);
    split_branches.push_back(std::move(main_branch));
    split_branches.push_back(std::move(right_branch));
    return split_branches;
}

static std::vector<CenterlineSample> buildCenterlineSamplesFromPolyImpl(const PolyFit2& fit,
                                                                        float y_max,
                                                                        float y_min,
                                                                        int frame_w,
                                                                        int frame_h) {
    std::vector<CenterlineSample> centerline;
    if (!fit.valid || frame_w <= 1 || frame_h <= 1) return centerline;

    const float plan_forward_min = std::min(y_min, y_max);
    const float plan_forward_max = std::max(y_min, y_max);
    if (plan_forward_max - plan_forward_min < 2.0f) return centerline;

    const int y_start = static_cast<int>(std::floor(plan_forward_min));
    const int y_end = static_cast<int>(std::ceil(plan_forward_max));
    const int y_step = 8;

    centerline.reserve(static_cast<size_t>((y_end - y_start) / y_step + 2));
    for (int y = y_start; y <= y_end; y += y_step) {
        const float x = fit.eval(static_cast<float>(y));
        if (!std::isfinite(x)) continue;
        bool proj_ok = false;
        const cv::Point2f px = planPointToPixel(cv::Point2f(x, static_cast<float>(y)), &proj_ok);
        if (!proj_ok) continue;
        CenterlineSample s;
        s.y = clampValue(px.y, 0.0f, static_cast<float>(frame_h - 1));
        s.x = clampValue(px.x, 0.0f, static_cast<float>(frame_w - 1));
        centerline.push_back(s);
    }
    return centerline;
}

static int detectSingleBranchPreviewTailStart(const std::vector<ArrowGuide>& guides,
                                              float median_hw) {
    if (guides.size() < 5) {
        return -1;
    }

    const float max_tail_dy = std::max(planMeters(0.04f), median_hw * 0.28f);
    const float min_tail_dx = std::max(planMeters(0.06f), median_hw * 0.35f);
    float tail_sign = 0.0f;
    int edge_count = 0;
    int start_idx = static_cast<int>(guides.size() - 1);

    for (int i = static_cast<int>(guides.size()) - 1; i >= 1; --i) {
        const float dx = guidePlanLateral(guides[static_cast<size_t>(i)]) -
                         guidePlanLateral(guides[static_cast<size_t>(i - 1)]);
        const float dy = guidePlanForward(guides[static_cast<size_t>(i)]) -
                         guidePlanForward(guides[static_cast<size_t>(i - 1)]);
        if (dy < -planMeters(0.003f) || dy > max_tail_dy || std::abs(dx) < min_tail_dx) {
            break;
        }
        const float sign = dx >= 0.0f ? 1.0f : -1.0f;
        if (tail_sign == 0.0f) {
            tail_sign = sign;
        } else if (tail_sign * sign < 0.0f) {
            break;
        }
        start_idx = i - 1;
        ++edge_count;
    }

    const int tail_point_count = static_cast<int>(guides.size()) - start_idx;
    if (edge_count < 2 || tail_point_count < 3 || start_idx < 1) {
        return -1;
    }

    const float tail_forward_span =
        guidePlanForward(guides.back()) -
        guidePlanForward(guides[static_cast<size_t>(start_idx)]);
    const float tail_x_span =
        std::abs(guidePlanLateral(guides.back()) -
                 guidePlanLateral(guides[static_cast<size_t>(start_idx)]));
    if (tail_forward_span > std::max(planMeters(0.06f), median_hw * 0.45f) ||
        tail_x_span < std::max(planMeters(0.17f), median_hw * 1.00f)) {
        return -1;
    }

    const float entry_dx =
        guidePlanLateral(guides[static_cast<size_t>(start_idx)]) -
        guidePlanLateral(guides[static_cast<size_t>(start_idx - 1)]);
    if (tail_sign * entry_dx <= 0.0f ||
        std::abs(entry_dx) < std::max(planMeters(0.09f), median_hw * 0.55f)) {
        return -1;
    }

    if (twoStepBackEndpointPreviewSupport(guides, median_hw) < 0.25f) {
        return -1;
    }
    return start_idx;
}

static std::vector<CenterlineSample> buildCenterlineSamplesFromPolyTailBlend(
    const std::vector<ArrowGuide>& guides,
    int tail_start_idx,
    int frame_w,
    int frame_h) {
    std::vector<CenterlineSample> centerline;
    if (guides.size() < 4 ||
        tail_start_idx <= 0 ||
        tail_start_idx >= static_cast<int>(guides.size()) - 1) {
        return centerline;
    }

    std::vector<ArrowGuide> trunk_guides(
        guides.begin(),
        guides.begin() + static_cast<std::ptrdiff_t>(tail_start_idx + 1));
    std::vector<ArrowGuide> tail_guides(
        guides.begin() + static_cast<std::ptrdiff_t>(tail_start_idx),
        guides.end());
    if (trunk_guides.size() < 3 || tail_guides.size() < 2) {
        return centerline;
    }

    const PolyFit2 trunk_fit = refitPoly2OnAllGuides(trunk_guides);
    if (!trunk_fit.valid) {
        return centerline;
    }

    centerline = buildCenterlineSamplesFromPolyImpl(
        trunk_fit,
        guidePlanForward(trunk_guides.front()),
        guidePlanForward(trunk_guides.back()),
        frame_w,
        frame_h);

    auto appendIfDistinct = [&](float x, float y) {
        if (!std::isfinite(x) || !std::isfinite(y)) {
            return;
        }
        if (!centerline.empty()) {
            const auto& last = centerline.back();
            if (std::abs(last.y - y) <= 0.5f &&
                std::abs(last.x - x) <= 0.5f) {
                return;
            }
            if (y > last.y + 0.5f) {
                return;
            }
        }
        CenterlineSample s;
        s.x = clampValue(x, 0.0f, static_cast<float>(frame_w - 1));
        s.y = clampValue(y, 0.0f, static_cast<float>(frame_h - 1));
        centerline.push_back(s);
    };

    const auto& anchor = tail_guides.front();
    appendIfDistinct(anchor.pos.x, anchor.pos.y);
    for (size_t i = 1; i < tail_guides.size(); ++i) {
        appendIfDistinct(tail_guides[i].pos.x, tail_guides[i].pos.y);
    }
    return centerline;
}

static void applyBranchAppearance(ArrowBranch& branch, const std::string& label) {
    branch.label = label;
    if (label == "main") {
        branch.color = cv::Scalar(0, 0, 255);
    } else if (label == "right") {
        branch.color = cv::Scalar(0, 165, 255);
    } else {
        branch.color = cv::Scalar(255, 255, 255);
    }
}

// Final branch post-processing preserves main/right labels and records that
// anchor_x sorting is not applied.
static void finalizeVisibleBranches(std::vector<ArrowBranch>& branches,
                                    BranchDebugInfo* debug_info = nullptr) {
    branches.erase(std::remove_if(branches.begin(), branches.end(), [](const ArrowBranch& b) {
        return b.guides.empty();
    }), branches.end());

    for (auto& branch : branches) {
        branch.anchor_x = computeBranchAnchorXAfterSharedPrefix(branch.guides, branch.shared_prefix_count);
    }
    if (debug_info) {
        debug_info->finalize_input_branch_count = static_cast<int>(branches.size());
        debug_info->final_output_anchor_sort_applied = false;
    }

    if (branches.size() <= 1) {
        if (!branches.empty()) {
            applyBranchAppearance(branches.front(), "main");
        }
        if (debug_info) {
            if (branches.empty()) {
                debug_info->status = "error";
            } else if (debug_info->status.empty() || debug_info->status == "single") {
                debug_info->status = "single";
            }
            debug_info->output_branches = static_cast<int>(branches.size());
        }
        return;
    }

    for (auto& branch : branches) {
        applyBranchAppearance(branch, branch.label);
    }
    if (debug_info) {
        if (debug_info->status.empty()) {
            debug_info->status = "split";
        }
        debug_info->output_branches = static_cast<int>(branches.size());
        debug_info->final_output_preserve_labels = true;
    }
}


}  // namespace

bool normalizeVector(const cv::Point2f& v, cv::Point2f& out_unit, float* out_length) {
    return normalizeVectorImpl(v, out_unit, out_length);
}

cv::Point2f worldToPixelMeters(const cv::Point2f& world, bool* valid) {
    return worldToPixel(world, valid);
}

float pointToVehicleBodyDistanceMeters(const cv::Point2f& world,
                                       float vehicle_front_m,
                                       float vehicle_rear_m,
                                       float vehicle_width_m) {
    if (!std::isfinite(world.x) || !std::isfinite(world.y)) {
        return std::numeric_limits<float>::infinity();
    }
    const float half_width = 0.5f * std::max(0.01f, vehicle_width_m);
    const float front = std::max(0.0f, vehicle_front_m);
    const float rear = std::max(0.0f, vehicle_rear_m);
    const float dx = std::max(std::abs(world.x) - half_width, 0.0f);
    float dy = 0.0f;
    if (world.y < -rear) {
        dy = -rear - world.y;
    } else if (world.y > front) {
        dy = world.y - front;
    }
    return std::sqrt(dx * dx + dy * dy);
}

bool estimateTailDirection(const std::vector<ArrowGuide>& guides,
                           cv::Point2f& out_unit,
                           float* out_length,
                           size_t max_lookback) {
    return estimateTailDirectionImpl(guides, out_unit, out_length, max_lookback);
}

float angleBetweenUnitVectorsDeg(const cv::Point2f& a, const cv::Point2f& b) {
    return angleBetweenUnitVectorsDegImpl(a, b);
}

float forwardScoreFromUnitDirection(const cv::Point2f& unit_dir) {
    return forwardScoreFromUnitDirectionImpl(unit_dir);
}

std::vector<CenterlineSample> buildCenterlineSamplesFromPoly(const PolyFit2& fit,
                                                             float y_max,
                                                             float y_min,
                                                             int frame_w,
                                                             int frame_h) {
    return buildCenterlineSamplesFromPolyImpl(fit, y_max, y_min, frame_w, frame_h);
}


float averageGuideHalfWidth(const std::vector<ArrowGuide>& guides) {
    if (guides.empty()) return 0.0f;
    float sum = 0.0f;
    for (const auto& guide : guides) sum += guide.half_width;
    return sum / static_cast<float>(guides.size());
}

float turnTightness(const std::vector<cv::Point2f>& pts) {
    if (pts.size() < 3) return 0.0f;
    const cv::Point2f a = pts.front();
    const cv::Point2f b = pts[pts.size() / 2];
    const cv::Point2f c = pts.back();
    const cv::Point2f ab = b - a;
    const cv::Point2f bc = c - b;
    const float lab = std::hypot(ab.x, ab.y);
    const float lbc = std::hypot(bc.x, bc.y);
    if (lab < 1e-3f || lbc < 1e-3f) return 0.0f;
    const float cosang = std::clamp((ab.x * bc.x + ab.y * bc.y) / (lab * lbc), -1.0f, 1.0f);
    return std::clamp(static_cast<float>(std::acos(cosang) / (CV_PI / 6.0f)), 0.0f, 1.0f);
}

std::vector<cv::Point2f> smoothPath(const std::vector<cv::Point2f>& path, int base_window) {
    if (path.size() < 5) return path;
    int window = std::min(std::max(3, base_window), static_cast<int>(path.size()) / 3);
    if (window % 2 == 0) ++window;
    const int half = window / 2;
    std::vector<cv::Point2f> result;
    result.reserve(path.size());
    for (size_t i = 0; i < path.size(); ++i) {
        if (i < static_cast<size_t>(half) || i + half >= path.size()) {
            result.push_back(path[i]);
            continue;
        }
        cv::Point2f sum(0.0f, 0.0f);
        for (int j = -half; j <= half; ++j) sum += path[i + j];
        result.push_back(sum * (1.0f / static_cast<float>(window)));
    }
    return result;
}

std::vector<cv::Point2f> dedupPath(const std::vector<cv::Point2f>& path) {
    std::vector<cv::Point2f> output;
    output.reserve(path.size());
    for (const auto& point : path) {
        if (!isFinitePoint(point)) continue;
        if (!output.empty()) {
            const float dx = point.x - output.back().x;
            const float dy = point.y - output.back().y;
            if (dx * dx + dy * dy < 4.0f) continue;
        }
        output.push_back(point);
    }
    return output;
}

BranchPlanningResult planVisibleBranchesForFrame(
    uint32_t frame_id,
    const ArrowGuideFrameData& arrow_frame,
    int arrow_detection_count,
    int frame_w,
    int frame_h,
    PathPlannerRuntimeState& runtime_state) {
    BranchPlanningResult result;
    BranchDebugInfo* debug_info = &result.debug;
    result.line_loss.line_arrow_object_count = arrow_detection_count;
    result.line_loss.line_raw_guide_count =
        static_cast<int>(arrow_frame.guides_before_branching);
    result.line_loss.line_valid_world_guide_count =
        static_cast<int>(std::count_if(
            arrow_frame.guides.begin(),
            arrow_frame.guides.end(),
            [](const ArrowGuide& guide) {
                return hasValidWorldGuide(guide);
            }));
    result.visible_branches = splitArrowGuidesIntoBranchesTwoStep(
        arrow_frame.guides,
        frame_id,
        frame_w,
        frame_h,
        debug_info,
        &runtime_state.boundary_tracks,
        &result.line_loss);
    finalizeVisibleBranches(result.visible_branches, debug_info);
    result.line_loss.line_visible_branch_count =
        static_cast<int>(result.visible_branches.size());
    result.road_corridors = buildRoadCorridorSetForBranches(
        result.visible_branches, frame_w, frame_h, 0.15f);
    result.debug.road_corridor_count = static_cast<int>(result.road_corridors.corridors.size());
    result.debug.road_corridor_valid = result.road_corridors.valid;
    result.debug.road_corridor_width_mean_m = result.road_corridors.width_mean_m;
    result.debug.road_corridor_invalid_reason = result.road_corridors.invalid_reason;
    (void)frame_id;
    return result;
}

BranchCenterlineResult buildBranchCenterlineResult(const ArrowBranch& branch,
                                                   int frame_w,
                                                   int frame_h) {
    BranchCenterlineResult result;
    result.centerline_guides = guidesForBranchCenterline(branch, &result.debug);
    const bool selected_guides_match_branch =
        result.centerline_guides.size() == branch.guides.size() &&
        std::equal(result.centerline_guides.begin(),
                   result.centerline_guides.end(),
                   branch.guides.begin(),
                   [](const ArrowGuide& a, const ArrowGuide& b) {
                       return std::abs(a.pos.x - b.pos.x) < 1e-3f &&
                              std::abs(a.pos.y - b.pos.y) < 1e-3f;
                   });

    bool guides_truncated_by_reversal = false;
    {
        constexpr float kWorldForwardReverseThreshold = 0.018f;
        constexpr float kWorldForwardReverseWarmupThreshold = 0.020f;
        for (size_t i = 2; i < result.centerline_guides.size(); ++i) {
            if (!result.centerline_guides[i - 2].has_world_pos ||
                !result.centerline_guides[i - 1].has_world_pos ||
                !result.centerline_guides[i].has_world_pos) {
                continue;
            }
            const float prev_delta = worldForwardM(result.centerline_guides[i - 1]) -
                                     worldForwardM(result.centerline_guides[i - 2]);
            const float curr_delta = worldForwardM(result.centerline_guides[i]) -
                                     worldForwardM(result.centerline_guides[i - 1]);
            if (prev_delta > kWorldForwardReverseWarmupThreshold &&
                curr_delta < -kWorldForwardReverseThreshold) {
                result.centerline_guides.resize(i);
                guides_truncated_by_reversal = true;
                break;
            }
        }
    }

    PolyFit2 selected_fit;
    float selected_poly_y_min = 0.0f;
    float selected_poly_y_max = 0.0f;
    if (branch.has_poly_fit && selected_guides_match_branch) {
        selected_fit.a = branch.poly_a;
        selected_fit.b = branch.poly_b;
        selected_fit.c = branch.poly_c;
        selected_fit.valid = true;
        selected_poly_y_min = branch.poly_y_min;
        selected_poly_y_max = branch.poly_y_max;
    } else if (result.centerline_guides.size() >= 3) {
        selected_fit = refitPoly2OnAllGuides(result.centerline_guides);
        if (selected_fit.valid) {
            selected_poly_y_min = guidePlanForward(result.centerline_guides.front());
            selected_poly_y_max = guidePlanForward(result.centerline_guides.back());
        }
    }

    if (selected_fit.valid && !guides_truncated_by_reversal) {
        const float avg_hw = averageGuideHalfWidthPlan(result.centerline_guides);
        // [REVIEW_NEEDED] Tail-blend centerline generation is active and
        // affects centerline/road_corridor, even though it resembles old tail
        // extension logic. Confirm with output diffs before any cleanup.
        const int preview_tail_start =
            detectSingleBranchPreviewTailStart(result.centerline_guides, avg_hw);
        if (preview_tail_start >= 0) {
            result.centerline = buildCenterlineSamplesFromPolyTailBlend(
                result.centerline_guides, preview_tail_start, frame_w, frame_h);
            if (!result.centerline.empty()) {
                result.fit_mode = "poly_tail_blend";
            }
        }

        if (result.centerline.empty()) {
            result.centerline = buildCenterlineSamplesFromPolyImpl(
                selected_fit, selected_poly_y_max, selected_poly_y_min, frame_w, frame_h);
            result.fit_mode = "poly";
        }

        result.debug.input_guides = static_cast<int>(result.centerline_guides.size());
        result.debug.valid_guides = static_cast<int>(result.centerline_guides.size());
        result.debug.guide_y_span =
            std::max(0.0f, selected_poly_y_max - selected_poly_y_min);
        result.debug.raw_samples = static_cast<int>(result.centerline.size());
        result.debug.filtered_samples = static_cast<int>(result.centerline.size());
        result.debug.sample_step_y = 8;
        if (!result.centerline.empty()) {
            result.debug.sample_y_max = result.centerline.front().y;
            result.debug.sample_y_min = result.centerline.back().y;
        } else if (!result.centerline_guides.empty()) {
            result.debug.sample_y_max = selected_poly_y_max;
            result.debug.sample_y_min = selected_poly_y_min;
        }
        return result;
    }

    // No valid poly fit 鈥?force-refit from guides and use poly sampling.
    if (result.centerline_guides.size() >= 3) {
        const PolyFit2 refit = refitPoly2OnAllGuides(result.centerline_guides);
        if (refit.valid) {
            result.centerline = buildCenterlineSamplesFromPolyImpl(
                refit,
                guidePlanForward(result.centerline_guides.front()),
                guidePlanForward(result.centerline_guides.back()),
                frame_w,
                frame_h);
            result.fit_mode = "poly";
        }
    }
    if (!result.centerline.empty()) {
        result.debug.input_guides = static_cast<int>(result.centerline_guides.size());
        result.debug.valid_guides = static_cast<int>(result.centerline_guides.size());
        result.debug.raw_samples = static_cast<int>(result.centerline.size());
        result.debug.filtered_samples = static_cast<int>(result.centerline.size());
    }
    return result;
}


namespace {
static float worldForwardM(const cv::Point2f& world) {
    return world.y;
}

static std::vector<cv::Point2f> centerlineSamplesToWorldPoints(
    const std::vector<CenterlineSample>& centerline,
    std::string* invalid_reason = nullptr) {
    if (invalid_reason != nullptr) {
        *invalid_reason = "none";
    }

    std::vector<cv::Point2f> world_points;
    world_points.reserve(centerline.size());
    for (const auto& sample : centerline) {
        if (!isFiniteFloat(sample.x) || !isFiniteFloat(sample.y)) {
            if (invalid_reason != nullptr) {
                *invalid_reason = "non_finite_pixel_centerline";
            }
            continue;
        }
        bool valid = false;
        const cv::Point2f world =
            pixelToWorldMeters(cv::Point2f(sample.x, sample.y), &valid);
        if (!valid || !isFinitePoint(world)) {
            if (invalid_reason != nullptr) {
                *invalid_reason = "pixel_to_world_failed";
            }
            continue;
        }
        world_points.push_back(world);
    }

    if (world_points.size() < 3 && invalid_reason != nullptr &&
        *invalid_reason == "none") {
        *invalid_reason = "centerline_too_short";
    }
    return world_points;
}

static void updateRoadCorridorSetWidthStats(RoadCorridorSet& set) {
    float width_sum = 0.0f;
    int width_count = 0;
    set.width_min_m = std::numeric_limits<float>::infinity();
    set.width_max_m = 0.0f;
    set.width_mean_m = 0.0f;

    for (const auto& corridor : set.corridors) {
        if (!corridor.valid) {
            continue;
        }
        for (const auto& point : corridor.points) {
            const cv::Point2f delta = point.left_world - point.right_world;
            const float width =
                std::sqrt(delta.x * delta.x + delta.y * delta.y);
            if (!isFiniteFloat(width)) {
                continue;
            }
            set.width_min_m = std::min(set.width_min_m, width);
            set.width_max_m = std::max(set.width_max_m, width);
            width_sum += width;
            ++width_count;
        }
    }

    if (width_count > 0) {
        set.width_mean_m = width_sum / static_cast<float>(width_count);
    } else {
        set.width_min_m = 0.0f;
        set.width_max_m = 0.0f;
    }
}

static RoadCorridor buildRoadCorridorFromWorldCenterline(
    const std::vector<cv::Point2f>& center_world,
    RoadCorridorRole role,
    int branch_id,
    float half_width_m = 0.15f) {
    RoadCorridor corridor;
    corridor.role = role;
    corridor.branch_id = branch_id;
    corridor.half_width_m = half_width_m;

    if (center_world.size() < 3) {
        corridor.invalid_reason = "centerline_too_short";
        return corridor;
    }

    constexpr float kMinAdjacentDistanceM = 0.005f;
    constexpr float kForwardReverseSlackM = 0.020f;
    for (size_t i = 0; i < center_world.size(); ++i) {
        if (!isFinitePoint(center_world[i])) {
            corridor.invalid_reason = "non_finite_world_centerline";
            return corridor;
        }
        if (i == 0) {
            continue;
        }
        const cv::Point2f delta = center_world[i] - center_world[i - 1];
        const float dist = std::sqrt(delta.x * delta.x + delta.y * delta.y);
        if (!isFiniteFloat(dist) || dist < kMinAdjacentDistanceM) {
            corridor.invalid_reason = "adjacent_points_too_close";
            return corridor;
        }
        const float dfwd =
            worldForwardM(center_world[i]) - worldForwardM(center_world[i - 1]);
        if (dfwd < -kForwardReverseSlackM) {
            corridor.invalid_reason = "forward_reversal";
            return corridor;
        }
    }

    corridor.points.reserve(center_world.size());
    cv::Point2f prev_normal(0.0f, 0.0f);
    bool has_prev_normal = false;
    for (size_t i = 0; i < center_world.size(); ++i) {
        cv::Point2f tangent_src;
        if (i == 0) {
            tangent_src = center_world[1] - center_world[0];
        } else if (i + 1 == center_world.size()) {
            tangent_src = center_world[i] - center_world[i - 1];
        } else {
            tangent_src = center_world[i + 1] - center_world[i - 1];
        }

        cv::Point2f tangent;
        if (!normalizeVector(tangent_src, tangent)) {
            corridor.points.clear();
            corridor.invalid_reason = "tangent_normalize_failed";
            return corridor;
        }

        cv::Point2f normal(-tangent.y, tangent.x);
        if (has_prev_normal && dotPoint(prev_normal, normal) < 0.0f) {
            normal.x = -normal.x;
            normal.y = -normal.y;
        }
        prev_normal = normal;
        has_prev_normal = true;

        RoadCorridorPoint point;
        point.center_world = center_world[i];
        point.tangent = tangent;
        point.normal = normal;
        point.half_width_m = half_width_m;
        point.left_world = point.center_world + point.normal * half_width_m;
        point.right_world = point.center_world - point.normal * half_width_m;
        corridor.points.push_back(point);
    }

    corridor.valid = true;
    corridor.invalid_reason = "none";
    return corridor;
}

// [ACTIVE_CHAIN] Builds road_corridor output from final branch centerlines.
static RoadCorridorSet buildRoadCorridorSetForBranches(
    const std::vector<ArrowBranch>& final_branches,
    int frame_w,
    int frame_h,
    float half_width_m) {
    RoadCorridorSet set;
    if (final_branches.size() == 1) {
        const auto& branch = final_branches.front();
        // Accept "main", "left", or "right" as valid single-branch labels.
        // "right" can appear as the sole output in edge cases (e.g., preserve-state-only).
        if (branch.label != "main" && branch.label != "left" && branch.label != "right") {
            set.invalid_reason = "unexpected_single_branch_label";
            return set;
        }
        const BranchCenterlineResult centerline_result =
            buildBranchCenterlineResult(branch, frame_w, frame_h);
        std::string reason = "none";
        const std::vector<cv::Point2f> center_world =
            centerlineSamplesToWorldPoints(centerline_result.centerline, &reason);
        RoadCorridor corridor;
        if (reason != "none") {
            corridor.role = RoadCorridorRole::LeftBranch;
            corridor.branch_id = 0;
            corridor.invalid_reason = reason;
        } else {
            corridor = buildRoadCorridorFromWorldCenterline(
                center_world, RoadCorridorRole::LeftBranch, 0, half_width_m);
        }
        set.corridors.push_back(std::move(corridor));
    } else if (final_branches.size() == 2) {
        int main_idx = -1;
        int legacy_left_idx = -1;
        int right_idx = -1;
        for (size_t i = 0; i < final_branches.size(); ++i) {
            if (final_branches[i].label == "main") {
                main_idx = static_cast<int>(i);
            } else if (final_branches[i].label == "left") {
                legacy_left_idx = static_cast<int>(i);
            } else if (final_branches[i].label == "right") {
                right_idx = static_cast<int>(i);
            }
        }
        if (main_idx < 0) {
            main_idx = legacy_left_idx;
        }
        if (main_idx < 0 || right_idx < 0) {
            set.invalid_reason = "unexpected_split_branch_labels";
            return set;
        }

        const std::array<int, 2> branch_indices = {main_idx, right_idx};
        const std::array<RoadCorridorRole, 2> roles = {
            RoadCorridorRole::LeftBranch,
            RoadCorridorRole::RightBranch};
        for (size_t out_idx = 0; out_idx < branch_indices.size(); ++out_idx) {
            const int branch_idx = branch_indices[out_idx];
            const BranchCenterlineResult centerline_result =
                buildBranchCenterlineResult(
                    final_branches[static_cast<size_t>(branch_idx)], frame_w, frame_h);
            std::string reason = "none";
            const std::vector<cv::Point2f> center_world =
                centerlineSamplesToWorldPoints(centerline_result.centerline, &reason);
            RoadCorridor corridor;
            if (reason != "none") {
                corridor.role = roles[out_idx];
                corridor.branch_id = branch_idx;
                corridor.invalid_reason = reason;
            } else {
                corridor = buildRoadCorridorFromWorldCenterline(
                    center_world, roles[out_idx], branch_idx, half_width_m);
            }
            set.corridors.push_back(std::move(corridor));
        }
    } else {
        set.invalid_reason = "unexpected_branch_count";
        return set;
    }

    set.valid = !set.corridors.empty() &&
                std::any_of(set.corridors.begin(), set.corridors.end(),
                            [](const RoadCorridor& corridor) {
                                return corridor.valid;
                            });
    if (!set.valid) {
        set.invalid_reason = "corridor_invalid";
        for (const auto& corridor : set.corridors) {
            if (!corridor.valid && corridor.invalid_reason != "none") {
                set.invalid_reason = corridor.invalid_reason;
                break;
            }
        }
    } else {
        set.invalid_reason = "none";
    }
    updateRoadCorridorSetWidthStats(set);
    return set;
}


}  // namespace

std::vector<PlanningRoutePoint> buildPlanningRouteFromRoadCorridors(
    const RoadCorridorSet& corridor_set,
    int frame_w,
    int frame_h,
    RoadCorridorRole preferred_role,
    bool allow_role_fallback) {
    std::vector<PlanningRoutePoint> route;
    if (!corridor_set.valid || corridor_set.corridors.empty()) return route;

    const RoadCorridor* selected = nullptr;
    for (const auto& corridor : corridor_set.corridors) {
        if (corridor.valid && corridor.role == preferred_role) {
            selected = &corridor;
            break;
        }
    }
    if (selected == nullptr && allow_role_fallback) {
        for (const auto& corridor : corridor_set.corridors) {
            if (corridor.valid) {
                selected = &corridor;
                break;
            }
        }
    }
    if (selected == nullptr || selected->points.size() < 2) return route;

    route.reserve(selected->points.size());
    for (const auto& cp : selected->points) {
        bool center_ok = false;
        bool left_ok = false;
        bool right_ok = false;
        const cv::Point2f center_px = worldToPixel(cp.center_world, &center_ok);
        const cv::Point2f left_px = worldToPixel(cp.left_world, &left_ok);
        const cv::Point2f right_px = worldToPixel(cp.right_world, &right_ok);
        if (!center_ok || !left_ok || !right_ok ||
            !isFinitePoint(center_px) || !isFinitePoint(left_px) || !isFinitePoint(right_px)) {
            continue;
        }
        PlanningRoutePoint point;
        point.pos = cv::Point2f(
            clampValue(center_px.x, 0.0f, static_cast<float>(frame_w - 1)),
            clampValue(center_px.y, 0.0f, static_cast<float>(frame_h - 1)));
        point.left = std::min(left_px.x, right_px.x);
        point.right = std::max(left_px.x, right_px.x);
        point.half_width = 0.5f * std::abs(right_px.x - left_px.x);
        point.has_bounds = point.right > point.left;
        point.width_clipped = false;
        point.width_limited = false;
        route.push_back(point);
    }
    return route;
}

cv::Point2f pixelToWorldMeters(const cv::Point2f& pixel, bool* valid);

std::vector<PlanningObject> collectRawPersonObjects(
    const std::vector<PredictResult>& detections,
    const cv::Size& frame_size,
    float max_forward_m) {
    std::vector<PlanningObject> objects;
    if (frame_size.width <= 0 || frame_size.height <= 0) {
        return objects;
    }

    const cv::Rect frame_rect(0, 0, frame_size.width, frame_size.height);
    int raw_id = 0;
    for (const auto& result : detections) {
        if (result.type != TYPE_PEOPLE ||
            result.width <= 0 || result.height <= 0) {
            continue;
        }

        const cv::Rect clipped_box =
            cv::Rect(result.x, result.y, result.width, result.height) & frame_rect;
        if (clipped_box.width <= 0 || clipped_box.height <= 0) {
            continue;
        }

        PlanningObject out;
        out.id = raw_id++;
        out.type = ObjectType::PERSON;
        out.pos = cv::Point2f(
            static_cast<float>(clipped_box.x) + 0.5f * static_cast<float>(clipped_box.width),
            static_cast<float>(clipped_box.y) + 0.5f * static_cast<float>(clipped_box.height));
        out.bbox_w = static_cast<float>(clipped_box.width);
        out.bbox_h = static_cast<float>(clipped_box.height);
        out.foot_pos = cv::Point2f(
            out.pos.x,
            static_cast<float>(clipped_box.y + clipped_box.height));
        out.has_foot_pos = isFinitePoint(out.foot_pos);
        if (!out.has_foot_pos) {
            continue;
        }

        bool world_valid = false;
        out.world_pos_m = pixelToWorldMeters(out.foot_pos, &world_valid);
        out.has_world_pos_m = world_valid && isFinitePoint(out.world_pos_m);
        if (!out.has_world_pos_m || !isFinitePoint(out.world_pos_m) ||
            out.world_pos_m.y < 0.0f || out.world_pos_m.y > max_forward_m) {
            continue;
        }

        out.distance_to_self_m = static_cast<float>(cv::norm(out.world_pos_m));
        out.from_raw_detection = true;
        out.observed_this_frame = true;
        out.current_frame_visible = true;
        objects.push_back(out);
    }
    return objects;
}

std::vector<PlanningObject> collectPlanningObjects(
    ObjectTracker& tracker,
    int frame_idx,
    float gold_position_correction_scale) {
    std::vector<PlanningObject> objects;
    const ObjectType types[] = {ObjectType::CAR, ObjectType::GOLD};
    for (ObjectType type : types) {
        auto tracked = tracker.getObjectsByType(type);
        for (const auto* obj : tracked) {
            if (obj == nullptr || !isFinitePoint(obj->pos)) continue;
            if (type == ObjectType::GOLD && obj->disappear_count > 0) continue;
            PlanningObject out;
            out.id = obj->id;
            out.type = obj->type;
            out.pos = obj->pos;
            out.velocity = obj->velocity;
            out.bbox_w = obj->bbox_w;
            out.bbox_h = obj->bbox_h;
            out.from_raw_detection = false;
            out.observed_this_frame = obj->observed_this_frame;
            out.current_frame_visible = out.observed_this_frame;
            const cv::Point2f projection_pixel =
                type == ObjectType::GOLD
                    ? obj->pos
                    : cv::Point2f(
                          obj->pos.x,
                          obj->pos.y + 0.5f * std::max(1.0f, obj->bbox_h));
            out.foot_pos = projection_pixel;
            out.has_foot_pos = isFinitePoint(projection_pixel);
            if (out.has_foot_pos) {
                bool world_valid = false;
                const cv::Point2f apparent_world_m =
                    pixelToWorldMeters(projection_pixel, &world_valid);
                out.world_pos_m =
                    type == ObjectType::GOLD
                        ? correctGoldWorldPosition(
                              apparent_world_m,
                              gold_position_correction_scale)
                        : apparent_world_m;
                out.has_world_pos_m =
                    world_valid && isFinitePoint(out.world_pos_m);
                if (out.has_world_pos_m) {
                    out.distance_to_self_m =
                        static_cast<float>(cv::norm(out.world_pos_m));
                }
            }
            objects.push_back(out);
        }
    }
    return objects;
}

cv::Point2f pixelToWorldMeters(const cv::Point2f& pixel, bool* valid) {
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
    if (!std::isfinite(pixel.x) || !std::isfinite(pixel.y)) return cv::Point2f(0.0f, 0.0f);
    const double denom = h20 * static_cast<double>(pixel.x) + h21 * static_cast<double>(pixel.y) + h22;
    if (!std::isfinite(denom) || std::abs(denom) < 1e-9) return cv::Point2f(0.0f, 0.0f);
    const double wx = (h00 * static_cast<double>(pixel.x) + h01 * static_cast<double>(pixel.y) + h02) / denom;
    const double wy = (h10 * static_cast<double>(pixel.x) + h11 * static_cast<double>(pixel.y) + h12) / denom;
    if (!std::isfinite(wx) || !std::isfinite(wy)) return cv::Point2f(0.0f, 0.0f);
    if (valid) *valid = true;
    return cv::Point2f(static_cast<float>(wx), static_cast<float>(wy));
}

cv::Point2f correctGoldWorldPosition(
    const cv::Point2f& apparent_world_m,
    float correction_scale) {
    if (!isFinitePoint(apparent_world_m) || !std::isfinite(correction_scale)) {
        return cv::Point2f(0.0f, 0.0f);
    }
    const float bounded_scale = clampValue(correction_scale, 0.0f, 1.0f);
    return apparent_world_m * bounded_scale;
}

bool isPathUsable(const std::vector<cv::Point2f>& path, int width, int height) {
    if (path.size() < 2 || width <= 0 || height <= 0) {
        return false;
    }

    const float margin = 200.0f;
    for (const auto& point : path) {
        if (!isFinitePoint(point)) {
            return false;
        }
        if (point.x < -450.0f || point.x > width + 450.0f || point.y < -margin || point.y > height + margin) {
            return false;
        }
    }
    return true;
}

void drawPath(cv::Mat& frame,
              const std::vector<cv::Point2f>& path,
              const cv::Scalar& color,
              int thickness) {
    if (path.size() < 2) {
        return;
    }
    for (size_t i = 1; i < path.size(); ++i) {
        cv::line(frame, path[i - 1], path[i], color, thickness, cv::LINE_AA);
    }
}

// ============================================================================
// Pedestrian stop-state machine reset (shared with Car.cpp)
// ============================================================================
void enterGoldIdleAt(GoldBiasRuntime& gold, float platform_m) {
    platform_m = isFiniteFloat(platform_m) ? platform_m : 0.0f;
    gold.stable_base_m = platform_m;
    gold.target_m = platform_m;
    gold.applied_m = platform_m;
    gold.transition_state = GoldTransitionState::IDLE;
    gold.pending_bias_m = 0.0f;
    gold.pending_confirm_count = 0;
    gold.no_executable_gold_frames = 0;
    gold.target_reached_frames = 0;
    assert(gold.stable_base_m == gold.target_m &&
           gold.target_m == gold.applied_m);
}

void resetGoldBiasRuntime(PathPlanningRuntime& runtime) {
    enterGoldIdleAt(runtime.gold, 0.0f);
}

void resetPedestrianStopRuntime(PathPlanningRuntime& runtime,
                                const std::string& reason) {
    resetPedestrianBehavior(runtime.pedestrian_behavior);
    runtime.pedestrian_behavior.reason = reason;
    runtime.pedestrian_launch_source = PedestrianLaunchSource::NONE;
    runtime.pedestrian_danger_corridor_servo_initialized = false;
    runtime.pedestrian_danger_corridor_servo_locked = false;
    runtime.pedestrian_danger_corridor_servo_filtered = 0.0f;
}

// ============================================================================
// Local Path Planning
// ============================================================================
namespace {

constexpr float kDefaultRouteHalfWidth = 55.0f;
constexpr float kCorridorMargin = 8.0f;
constexpr float kObstacleSideLockThresholdM = 0.06f;
constexpr float kVehicleCurveSampleSpacingM = 0.30f;
constexpr float kVehicleCurveMetricThresholdM = 0.03f;
constexpr float kVehicleCurveEgoOppositeThresholdM = 0.08f;

void setPlannerSpeed(PathPlanningResult& result,
                     float speed_mps,
                     const std::string& source,
                     const std::string& risk = "normal") {
    result.target_speed_mps = std::max(0.0f, speed_mps);
    result.debug.target_speed_mps = result.target_speed_mps;
    result.debug.planner_speed_source = source;
    result.debug.speed_reason = source;
    result.debug.speed_limit_reason = source == "cruise" ? "" : source;
    result.debug.speed_risk_level = risk;
}

const char* pedestrianLaunchSourceName(PedestrianLaunchSource source) {
    switch (source) {
        case PedestrianLaunchSource::STOPPED_WAIT:
            return "launch_from_stop";
        case PedestrianLaunchSource::DRIVING_SLOW:
            return "launch_from_slow";
        case PedestrianLaunchSource::NONE:
            return "none";
    }
    return "none";
}

float pedestrianLaunchSpeedMps(const PathPlanningInput& input,
                               PedestrianLaunchSource source) {
    if (source == PedestrianLaunchSource::STOPPED_WAIT) {
        return std::max(0.0f, input.pedestrian_stopped_launch_speed_mps);
    }
    return std::max(0.0f, input.pedestrian_slow_launch_speed_mps);
}

void syncPedestrianStopDebug(const PathPlanningRuntime& runtime,
                             PathPlanningResult& result) {
    const PedestrianStopState state = runtime.pedestrian_behavior.state;
    result.pedestrian_stop_state = state;
    result.pedestrian_stop_state_active =
        state != PedestrianStopState::DRIVING;
    result.pedestrian_launch_committed =
        state == PedestrianStopState::LAUNCH_COMMITTED;
    result.startup_launch_protect_active =
        result.pedestrian_launch_committed;
    result.pedestrian_emergency_stop =
        state == PedestrianStopState::STOPPED_WAIT;
    result.debug.pedestrian_stop_state = state;
    result.debug.pedestrian_stop_state_name =
        pedestrianStopStateName(state);
    result.debug.launch_remaining =
        runtime.pedestrian_behavior.launch_remaining;
    result.debug.pedestrian_launch_source =
        pedestrianLaunchSourceName(runtime.pedestrian_launch_source);
    result.debug.startup_launch_protect_remaining =
        runtime.pedestrian_behavior.launch_remaining;
    result.debug.launch_protect_speed_applied =
        state == PedestrianStopState::LAUNCH_COMMITTED;
    result.debug.ped_danger_servo_locked =
        runtime.pedestrian_danger_corridor_servo_locked;
    result.debug.ped_stop_car_side_metric_m =
        runtime.pedestrian_behavior.stop_car_side_metric_m;
    result.debug.ped_stop_locked_car_side = pedestrianStopCarSideName(
        runtime.pedestrian_behavior.locked_stop_car_side);
    result.debug.ped_release_required_side =
        pedestrianReleaseRequiredSideName(
            runtime.pedestrian_behavior.release_required_side);
}

bool interpolateRouteAtY(const std::vector<PlanningRoutePoint>& route,
                         float y,
                         PlanningRoutePoint& out);

const char* objectTypeName(ObjectType type) {
    switch (type) {
        case ObjectType::PERSON: return "person";
        case ObjectType::CAR: return "car";
        case ObjectType::ARROW: return "arrow";
        default: return "unknown";
    }
}

float localPlannerDt(const PathPlanningInput& input,
                     const PathPlanningRuntime& runtime) {
    if (input.timestamp_valid && runtime.has_last_timestamp) {
        const double dt = input.timestamp_sec - runtime.last_timestamp_sec;
        if (std::isfinite(dt) && dt >= 0.005 && dt <= 0.25) {
            return static_cast<float>(dt);
        }
    }
    return 1.0f / 25.0f;
}

float pointToVehicleBodyDistance(const cv::Point2f& world,
                                 float vehicle_front_m,
                                 float vehicle_rear_m,
                                 float vehicle_width_m) {
    return pointToVehicleBodyDistanceMeters(world,
                                            vehicle_front_m,
                                            vehicle_rear_m,
                                            vehicle_width_m);
}

float pointToVehicleBodyDistance(const cv::Point2f& world,
                                 const PathPlanningInput& input) {
    return pointToVehicleBodyDistance(world,
                                      input.vehicle_front_m,
                                      input.vehicle_rear_m,
                                      input.vehicle_width_m);
}

bool routeFrameAtPixelY(const std::vector<PlanningRoutePoint>& route,
                        float y,
                        PlanningRoutePoint& route_point,
                        cv::Point2f& center_world,
                        cv::Point2f& normal_world) {
    if (!interpolateRouteAtY(route, y, route_point)) {
        return false;
    }
    bool center_valid = false;
    center_world = pixelToWorldMeters(route_point.pos, &center_valid);
    if (!center_valid || !isFinitePoint(center_world)) {
        return false;
    }

    PlanningRoutePoint prev_point;
    PlanningRoutePoint next_point;
    cv::Point2f prev_world;
    cv::Point2f next_world;
    bool prev_valid = false;
    bool next_valid = false;
    if (interpolateRouteAtY(route, y - 18.0f, prev_point)) {
        prev_world = pixelToWorldMeters(prev_point.pos, &prev_valid);
    }
    if (interpolateRouteAtY(route, y + 18.0f, next_point)) {
        next_world = pixelToWorldMeters(next_point.pos, &next_valid);
    }
    if (prev_valid && next_valid && isFinitePoint(prev_world) && isFinitePoint(next_world)) {
        const cv::Point2f tangent = next_world - prev_world;
        const float len = static_cast<float>(cv::norm(tangent));
        if (len > 1e-4f) {
            normal_world = cv::Point2f(-tangent.y / len, tangent.x / len);
            if (normal_world.x < 0.0f) {
                normal_world *= -1.0f;
            }
            return true;
        }
    }
    normal_world = cv::Point2f(1.0f, 0.0f);
    return true;
}

bool routeFrameNearestWorld(const std::vector<PlanningRoutePoint>& route,
                            const cv::Point2f& world,
                            PlanningRoutePoint& route_point,
                            cv::Point2f& center_world,
                            cv::Point2f& normal_world) {
    if (route.size() < 2 || !isFinitePoint(world)) {
        return false;
    }

    float best_dist2 = std::numeric_limits<float>::infinity();
    bool found = false;
    for (size_t i = 0; i + 1 < route.size(); ++i) {
        bool a_valid = false;
        bool b_valid = false;
        const cv::Point2f a_world = pixelToWorldMeters(route[i].pos, &a_valid);
        const cv::Point2f b_world = pixelToWorldMeters(route[i + 1].pos, &b_valid);
        if (!a_valid || !b_valid || !isFinitePoint(a_world) || !isFinitePoint(b_world)) {
            continue;
        }
        const cv::Point2f seg = b_world - a_world;
        const float seg_len2 = dotPoint(seg, seg);
        if (!isFiniteFloat(seg_len2) || seg_len2 < 1e-8f) {
            continue;
        }
        const float t = clampValue(dotPoint(world - a_world, seg) / seg_len2, 0.0f, 1.0f);
        const cv::Point2f candidate = a_world + seg * t;
        const cv::Point2f diff = world - candidate;
        const float dist2 = dotPoint(diff, diff);
        if (isFiniteFloat(dist2) && dist2 < best_dist2) {
            best_dist2 = dist2;
            center_world = candidate;
            const float seg_len = std::sqrt(seg_len2);
            normal_world = cv::Point2f(-seg.y / seg_len, seg.x / seg_len);
            if (normal_world.x < 0.0f) {
                normal_world *= -1.0f;
            }
            route_point = route[i];
            route_point.pos = route[i].pos + (route[i + 1].pos - route[i].pos) * t;
            route_point.half_width =
                route[i].half_width + (route[i + 1].half_width - route[i].half_width) * t;
            found = true;
        }
    }
    return found;
}

float obstacleLateralFromRoute(const PlanningObject& obj,
                               const std::vector<PlanningRoutePoint>& route,
                               cv::Point2f* out_normal = nullptr,
                               PlanningRoutePoint* out_route_point = nullptr) {
    PlanningRoutePoint rp;
    cv::Point2f center_world;
    cv::Point2f normal_world;
    if (!obj.has_world_pos_m ||
        !routeFrameNearestWorld(route, obj.world_pos_m, rp, center_world, normal_world)) {
        if (out_normal) *out_normal = cv::Point2f(1.0f, 0.0f);
        if (out_route_point) *out_route_point = rp;
        return 0.0f;
    }
    if (out_normal) *out_normal = normal_world;
    if (out_route_point) *out_route_point = rp;
    return dotPoint(obj.world_pos_m - center_world, normal_world);
}

cv::Point2f shiftPointByRouteOffset(const cv::Point2f& pixel,
                                    const std::vector<PlanningRoutePoint>& route,
                                    float offset_m,
                                    bool& valid) {
    valid = false;
    if (!isFinitePoint(pixel) || std::abs(offset_m) < 1e-5f) {
        valid = isFinitePoint(pixel);
        return pixel;
    }
    PlanningRoutePoint rp;
    cv::Point2f center_world;
    cv::Point2f normal_world;
    if (!routeFrameAtPixelY(route, pixel.y, rp, center_world, normal_world)) {
        return pixel;
    }
    bool world_valid = false;
    const cv::Point2f world = pixelToWorldMeters(pixel, &world_valid);
    if (!world_valid || !isFinitePoint(world)) {
        return pixel;
    }

    // Keep the route sample on its original image row. Reusing only the x
    // coordinate from a full world-offset projection changes the world point
    // when that projection also changes y, which amplifies offsets on curves.
    auto offsetErrorAtX = [&](float x, float& error) {
        bool candidate_valid = false;
        const cv::Point2f candidate_world =
            pixelToWorldMeters(cv::Point2f(x, pixel.y), &candidate_valid);
        if (!candidate_valid || !isFinitePoint(candidate_world)) {
            return false;
        }
        error = dotPoint(candidate_world - world, normal_world) - offset_m;
        return isFiniteFloat(error);
    };

    constexpr float kDirectionProbePx = 1.0f;
    float left_probe_error = 0.0f;
    float right_probe_error = 0.0f;
    if (!offsetErrorAtX(pixel.x - kDirectionProbePx, left_probe_error) ||
        !offsetErrorAtX(pixel.x + kDirectionProbePx, right_probe_error)) {
        return pixel;
    }
    const float search_direction =
        std::abs(left_probe_error) < std::abs(right_probe_error) ? -1.0f : 1.0f;

    const float base_error = -offset_m;
    float far_x = pixel.x;
    float far_error = base_error;
    bool bracketed = false;
    float search_step_px = 8.0f;
    constexpr int kMaxBracketExpansions = 10;
    for (int i = 0; i < kMaxBracketExpansions; ++i) {
        far_x = pixel.x + search_direction * search_step_px;
        if (!offsetErrorAtX(far_x, far_error)) {
            return pixel;
        }
        if ((base_error <= 0.0f && far_error >= 0.0f) ||
            (base_error >= 0.0f && far_error <= 0.0f)) {
            bracketed = true;
            break;
        }
        search_step_px *= 2.0f;
    }
    if (!bracketed) {
        return pixel;
    }

    float lower_x = pixel.x;
    float lower_error = base_error;
    float upper_x = far_x;
    constexpr int kBisectionIterations = 32;
    constexpr float kOffsetToleranceM = 1e-5f;
    for (int i = 0; i < kBisectionIterations; ++i) {
        const float mid_x = 0.5f * (lower_x + upper_x);
        float mid_error = 0.0f;
        if (!offsetErrorAtX(mid_x, mid_error)) {
            return pixel;
        }
        if (std::abs(mid_error) <= kOffsetToleranceM) {
            valid = true;
            return cv::Point2f(mid_x, pixel.y);
        }
        if ((lower_error <= 0.0f && mid_error >= 0.0f) ||
            (lower_error >= 0.0f && mid_error <= 0.0f)) {
            upper_x = mid_x;
        } else {
            lower_x = mid_x;
            lower_error = mid_error;
        }
    }

    valid = true;
    return cv::Point2f(0.5f * (lower_x + upper_x), pixel.y);
}

ObstacleAvoidanceTrack* findObstacleTrack(std::vector<ObstacleAvoidanceTrack>& tracks,
                                          ObjectType type,
                                          int id) {
    for (auto& track : tracks) {
        if (track.type == type && track.id == id) {
            return &track;
        }
    }
    return nullptr;
}

const ObstacleAvoidanceTrack* findObstacleTrack(const std::vector<ObstacleAvoidanceTrack>& tracks,
                                                ObjectType type,
                                                int id) {
    for (const auto& track : tracks) {
        if (track.type == type && track.id == id) {
            return &track;
        }
    }
    return nullptr;
}

void updateObstacleTracks(const PathPlanningInput& input,
                          const std::vector<PlanningRoutePoint>& route,
                          PathPlanningRuntime& runtime) {
    runtime.obstacle_tracks.erase(
        std::remove_if(runtime.obstacle_tracks.begin(),
                       runtime.obstacle_tracks.end(),
                       [](const ObstacleAvoidanceTrack& track) {
                           return track.type == ObjectType::PERSON;
                       }),
        runtime.obstacle_tracks.end());
    for (auto& track : runtime.obstacle_tracks) {
        track.updated = false;
    }

    const int confirm_frames = std::max(1, input.obstacle_confirm_frames);
    const int vehicle_confirm_frames = 2;
    const int exit_lost_frames = std::max(0, input.obstacle_exit_lost_frames);

    for (const auto& obj : input.objects) {
        if (obj.type != ObjectType::CAR || !obj.has_world_pos_m ||
            !isFinitePoint(obj.world_pos_m)) {
            continue;
        }
        ObstacleAvoidanceTrack* track =
            findObstacleTrack(runtime.obstacle_tracks, obj.type, obj.id);
        const bool maintain_active_vehicle =
            !obj.observed_this_frame && track != nullptr && track->confirmed &&
            runtime.avoid_scenario == AvoidScenario::VehicleOnly &&
            (runtime.obstacle_state == ObstacleAvoidanceState::Prepare ||
             runtime.obstacle_state == ObstacleAvoidanceState::Active) &&
            runtime.obstacle_primary_track_id == track->id;
        if (!obj.observed_this_frame) {
            if (maintain_active_vehicle) {
                track->updated = true;
                track->lost_frames = 0;
                track->observed_this_frame = false;
                track->world_pos_m = track->last_observed_world_pos_m;
            }
            continue;
        }
        if (track == nullptr) {
            runtime.obstacle_tracks.push_back(ObstacleAvoidanceTrack{});
            track = &runtime.obstacle_tracks.back();
            track->id = obj.id;
            track->type = obj.type;
        }

        const bool was_continuous = track->lost_frames == 0 && track->seen_streak > 0;
        track->seen_streak = was_continuous ? (track->seen_streak + 1) : 1;
        track->lost_frames = 0;
        const int type_confirm_frames =
            obj.type == ObjectType::CAR ? vehicle_confirm_frames : confirm_frames;
        track->confirmed =
            track->confirmed || track->seen_streak >= type_confirm_frames;
        track->updated = true;
        track->observed_this_frame = true;
        track->pixel_foot = obj.has_foot_pos ? obj.foot_pos : obj.pos;
        track->world_pos_m = obj.world_pos_m;
        track->point_distance_to_vehicle_m =
            pointToVehicleBodyDistance(obj.world_pos_m, input);
        track->distance_to_vehicle_m = track->point_distance_to_vehicle_m;
        track->lateral_from_route_m = obstacleLateralFromRoute(obj, route);
        PlanningRoutePoint rp;
        if (interpolateRouteAtY(route, track->pixel_foot.y, rp)) {
            track->center_delta_px = rp.pos.x - track->pixel_foot.x;
        } else {
            track->center_delta_px = 0.0f;
        }
        track->last_world_pos_m = obj.world_pos_m;
        track->has_last_world_pos_m = true;
        track->last_observed_world_pos_m = obj.world_pos_m;
        track->has_last_observed_world_pos_m = true;
        track->last_timestamp_sec = input.timestamp_valid ? input.timestamp_sec : 0.0;
    }

    for (auto& track : runtime.obstacle_tracks) {
        if (track.updated) continue;
        track.lost_frames++;
        track.seen_streak = 0;
        track.observed_this_frame = false;
        if (track.lost_frames > exit_lost_frames) {
            track.confirmed = false;
        }
    }

    runtime.obstacle_tracks.erase(
        std::remove_if(runtime.obstacle_tracks.begin(),
                       runtime.obstacle_tracks.end(),
                       [exit_lost_frames](const ObstacleAvoidanceTrack& track) {
                           return track.lost_frames > exit_lost_frames;
                       }),
        runtime.obstacle_tracks.end());
}

bool trackIsConfirmedVisible(const ObstacleAvoidanceTrack& track) {
    return track.confirmed && track.updated && track.lost_frames == 0;
}

ObstacleAvoidanceTrack* findVisibleTrack(std::vector<ObstacleAvoidanceTrack>& tracks,
                                         ObjectType type,
                                         int id) {
    for (auto& track : tracks) {
        if (track.id == id && track.type == type && trackIsConfirmedVisible(track)) {
            return &track;
        }
    }
    return nullptr;
}

const ObstacleAvoidanceTrack* findVisibleTrack(const std::vector<ObstacleAvoidanceTrack>& tracks,
                                               ObjectType type,
                                               int id) {
    for (const auto& track : tracks) {
        if (track.id == id && track.type == type && trackIsConfirmedVisible(track)) {
            return &track;
        }
    }
    return nullptr;
}

std::vector<ObstacleAvoidanceTrack*> visibleTracksOfType(
    std::vector<ObstacleAvoidanceTrack>& tracks,
    ObjectType type) {
    std::vector<ObstacleAvoidanceTrack*> out;
    for (auto& track : tracks) {
        if (track.type == type && trackIsConfirmedVisible(track) &&
            isFiniteFloat(track.distance_to_vehicle_m)) {
            out.push_back(&track);
        }
    }
    return out;
}

int avoidSideAwayFromLateral(float lateral_m) {
    return lateral_m >= 0.0f ? -1 : 1;
}

int normalizeAvoidSide(int side) {
    if (side < 0) return -1;
    if (side > 0) return 1;
    return 0;
}

const PlanningObject* findPlanningObject(const std::vector<PlanningObject>& objects,
                                         ObjectType type,
                                         int id) {
    for (const auto& object : objects) {
        if (object.type == type && object.id == id) {
            return &object;
        }
    }
    return nullptr;
}

bool vehicleRequiredOffsetFromBboxEdge(
    const PlanningObject& vehicle,
    const std::vector<PlanningRoutePoint>& route,
    int avoid_side,
    float edge_clearance_m,
    float& out_required_offset_m) {
    const int normalized_side = normalizeAvoidSide(avoid_side);
    if (normalized_side == 0 || !vehicle.observed_this_frame ||
        !vehicle.current_frame_visible || !vehicle.has_foot_pos ||
        !isFinitePoint(vehicle.foot_pos) || !isFiniteFloat(vehicle.bbox_w) ||
        vehicle.bbox_w <= 0.0f) {
        return false;
    }

    cv::Point2f edge_pixel = vehicle.foot_pos;
    edge_pixel.x += normalized_side > 0
                        ? 0.5f * vehicle.bbox_w
                        : -0.5f * vehicle.bbox_w;
    bool edge_world_valid = false;
    const cv::Point2f edge_world =
        pixelToWorldMeters(edge_pixel, &edge_world_valid);
    if (!edge_world_valid || !isFinitePoint(edge_world)) {
        return false;
    }

    PlanningRoutePoint route_point;
    cv::Point2f route_center_world;
    cv::Point2f route_normal_world;
    if (!routeFrameNearestWorld(route,
                                edge_world,
                                route_point,
                                route_center_world,
                                route_normal_world)) {
        return false;
    }

    const float signed_edge_lateral =
        dotPoint(edge_world - route_center_world, route_normal_world);
    const float edge_along_avoid_side =
        static_cast<float>(normalized_side) * signed_edge_lateral;
    if (!isFiniteFloat(edge_along_avoid_side)) {
        return false;
    }
    out_required_offset_m =
        std::max(0.0f, edge_along_avoid_side) + std::max(0.0f, edge_clearance_m);
    return isFiniteFloat(out_required_offset_m);
}

int sideFromObstacleOffset(float offset_m) {
    // buildOffsetPath applies world + normal * offset_m, while pedestrian
    // targets are side * obstacle_avoid_offset_m, so the signs map directly.
    return offset_m < 0.0f ? -1 : (offset_m > 0.0f ? 1 : 0);
}

int chooseVehicleAvoidSide(float lateral_m, int previous_avoid_side) {
    static constexpr float kReusePreviousSideLateralEpsilonM = 0.08f;
    static constexpr int kDefaultVehicleAvoidSide = 1;
    const int previous_side = normalizeAvoidSide(previous_avoid_side);
    const int fallback_side =
        previous_side != 0 ? previous_side : kDefaultVehicleAvoidSide;
    if (isFiniteFloat(lateral_m) &&
        std::abs(lateral_m) >= kReusePreviousSideLateralEpsilonM) {
        return avoidSideAwayFromLateral(lateral_m);
    }
    if (isFiniteFloat(lateral_m) && lateral_m != 0.0f &&
        lateral_m * static_cast<float>(fallback_side) >= 0.0f) {
        return avoidSideAwayFromLateral(lateral_m);
    }
    return fallback_side;
}

bool routeLateralAtForward(const std::vector<PlanningRoutePoint>& route,
                           float target_forward_m,
                           float& lateral_m) {
    for (size_t i = 0; i + 1 < route.size(); ++i) {
        bool a_valid = false;
        bool b_valid = false;
        const cv::Point2f a_world = pixelToWorldMeters(route[i].pos, &a_valid);
        const cv::Point2f b_world = pixelToWorldMeters(route[i + 1].pos, &b_valid);
        if (!a_valid || !b_valid ||
            !isFinitePoint(a_world) || !isFinitePoint(b_world)) {
            continue;
        }
        const float forward_delta_m = b_world.y - a_world.y;
        if (std::abs(forward_delta_m) < 1e-4f ||
            target_forward_m < std::min(a_world.y, b_world.y) ||
            target_forward_m > std::max(a_world.y, b_world.y)) {
            continue;
        }
        const float t =
            (target_forward_m - a_world.y) / forward_delta_m;
        lateral_m = a_world.x + (b_world.x - a_world.x) * t;
        return isFiniteFloat(lateral_m);
    }
    return false;
}

bool nearestRouteWorldPoint(const std::vector<PlanningRoutePoint>& route,
                            float& forward_m,
                            float& lateral_m) {
    forward_m = std::numeric_limits<float>::infinity();
    lateral_m = 0.0f;
    bool found = false;
    for (const auto& route_point : route) {
        bool world_valid = false;
        const cv::Point2f world =
            pixelToWorldMeters(route_point.pos, &world_valid);
        if (!world_valid || !isFinitePoint(world) || world.y < 0.0f ||
            world.y >= forward_m) {
            continue;
        }
        forward_m = world.y;
        lateral_m = world.x;
        found = true;
    }
    return found;
}

float vehicleCurveMetricFromRoute(
    const std::vector<PlanningRoutePoint>& route,
    bool& valid) {
    valid = false;
    float near_lateral_m = 0.0f;
    float mid_lateral_m = 0.0f;
    float far_lateral_m = 0.0f;
    float near_forward_m = 0.0f;
    if (!nearestRouteWorldPoint(route,
                                near_forward_m,
                                near_lateral_m) ||
        !routeLateralAtForward(route,
                               near_forward_m + kVehicleCurveSampleSpacingM,
                               mid_lateral_m) ||
        !routeLateralAtForward(route,
                               near_forward_m +
                                   2.0f * kVehicleCurveSampleSpacingM,
                               far_lateral_m)) {
        return 0.0f;
    }
    valid = true;
    return far_lateral_m - 2.0f * mid_lateral_m + near_lateral_m;
}

std::vector<PlanningRoutePoint> sanitizedRoute(const std::vector<PlanningRoutePoint>& route) {
    std::vector<PlanningRoutePoint> out;
    out.reserve(route.size());
    for (auto p : route) {
        if (!isFinitePoint(p.pos)) continue;
        if (!isFiniteFloat(p.half_width) || p.half_width <= 1.0f)
            p.half_width = kDefaultRouteHalfWidth;
        p.half_width = clampValue(p.half_width, 30.0f, 80.0f);
        out.push_back(p);
    }
    std::sort(out.begin(), out.end(), [](const PlanningRoutePoint& a, const PlanningRoutePoint& b) {
        return a.pos.y > b.pos.y;
    });
    out.erase(std::unique(out.begin(), out.end(), [](const PlanningRoutePoint& a, const PlanningRoutePoint& b) {
        return std::abs(a.pos.y - b.pos.y) < 2.0f && std::abs(a.pos.x - b.pos.x) < 2.0f;
    }), out.end());
    return out;
}

bool nearRouteLateralFromRoute(const std::vector<PlanningRoutePoint>& route,
                               float& lateral_m,
                               int& sample_count) {
    lateral_m = 0.0f;
    sample_count = 0;
    if (route.size() < 2) return false;

    std::vector<float> valid_x;
    valid_x.reserve(9);
    for (int station_index = 0; station_index <= 8; ++station_index) {
        const float station_y_m =
            0.20f + 0.025f * static_cast<float>(station_index);
        for (size_t i = 0; i + 1 < route.size(); ++i) {
            bool a_valid = false;
            bool b_valid = false;
            const cv::Point2f a = pixelToWorldMeters(route[i].pos, &a_valid);
            const cv::Point2f b = pixelToWorldMeters(route[i + 1].pos, &b_valid);
            if (!a_valid || !b_valid || !isFinitePoint(a) || !isFinitePoint(b)) {
                continue;
            }
            const float dy = b.y - a.y;
            if (std::abs(dy) < 1e-6f ||
                station_y_m < std::min(a.y, b.y) ||
                station_y_m > std::max(a.y, b.y)) {
                continue;
            }
            const float t = (station_y_m - a.y) / dy;
            if (!isFiniteFloat(t) || t < 0.0f || t > 1.0f) {
                continue;
            }
            const float x = a.x + t * (b.x - a.x);
            if (isFiniteFloat(x)) {
                valid_x.push_back(x);
                break;
            }
        }
    }

    sample_count = static_cast<int>(valid_x.size());
    if (valid_x.empty()) return false;
    std::sort(valid_x.begin(), valid_x.end());
    const size_t middle = valid_x.size() / 2;
    lateral_m = valid_x.size() % 2 == 0
        ? 0.5f * (valid_x[middle - 1] + valid_x[middle])
        : valid_x[middle];
    return isFiniteFloat(lateral_m);
}

bool interpolateRouteAtY(const std::vector<PlanningRoutePoint>& route,
                         float y, PlanningRoutePoint& out) {
    if (route.size() < 2 || !isFiniteFloat(y)) return false;
    for (size_t i = 0; i + 1 < route.size(); ++i) {
        const auto& a = route[i];
        const auto& b = route[i + 1];
        const float dy = b.pos.y - a.pos.y;
        if (std::abs(dy) < 1e-4f) continue;
        if ((y <= a.pos.y && y >= b.pos.y) || (y >= a.pos.y && y <= b.pos.y)) {
            const float t = (y - a.pos.y) / dy;
            out.pos.x = a.pos.x + t * (b.pos.x - a.pos.x);
            out.pos.y = y;
            out.half_width = a.half_width + t * (b.half_width - a.half_width);
            out.half_width = clampValue(out.half_width, 30.0f, 80.0f);
            return isFinitePoint(out.pos);
        }
    }
    const auto& endpoint = (std::abs(y - route.front().pos.y) < std::abs(y - route.back().pos.y))
                               ? route.front() : route.back();
    out = endpoint;
    out.pos.y = y;
    return true;
}

float corridorMinX(const PlanningRoutePoint& route_point) {
    if (route_point.has_bounds && isFiniteFloat(route_point.left))
        return route_point.left + kCorridorMargin;
    return route_point.pos.x - route_point.half_width + kCorridorMargin;
}

float corridorMaxX(const PlanningRoutePoint& route_point) {
    if (route_point.has_bounds && isFiniteFloat(route_point.right))
        return route_point.right - kCorridorMargin;
    return route_point.pos.x + route_point.half_width - kCorridorMargin;
}

std::vector<PlanningObject> objectsOfType(const std::vector<PlanningObject>& objects, ObjectType type) {
    std::vector<PlanningObject> out;
    for (const auto& obj : objects)
        if (obj.type == type && isFinitePoint(obj.pos)) out.push_back(obj);
    return out;
}

const char* modeName(LocalPlannerMode mode) {
    switch (mode) {
        case LocalPlannerMode::Cruise: return "cruise";
        case LocalPlannerMode::AvoidPrepare: return "avoid_prepare";
        case LocalPlannerMode::AvoidActive: return "avoid_active";
        case LocalPlannerMode::AvoidRecover: return "avoid_recover";
        case LocalPlannerMode::Disabled:
        default: return "disabled";
    }
}

std::vector<cv::Point2f> buildOffsetPath(const std::vector<PlanningRoutePoint>& route,
                                         float offset_m,
                                         float zero_epsilon_m,
                                         int offset_extension_side = 0,
                                         float offset_extension_limit_m = 0.0f) {
    std::vector<cv::Point2f> path;
    path.reserve(route.size());
    const int normalized_extension_side = normalizeAvoidSide(offset_extension_side);
    const float normalized_extension_limit_m =
        std::max(0.0f, offset_extension_limit_m);
    for (const auto& rp : route) {
        float min_x = corridorMinX(rp);
        float max_x = corridorMaxX(rp);
        cv::Point2f path_point(clampValue(rp.pos.x, min_x, max_x), rp.pos.y);
        if (normalized_extension_side != 0 && normalized_extension_limit_m > 0.0f) {
            bool max_offset_valid = false;
            const cv::Point2f max_offset_point = shiftPointByRouteOffset(
                path_point,
                route,
                static_cast<float>(normalized_extension_side) * normalized_extension_limit_m,
                max_offset_valid);
            if (max_offset_valid && isFinitePoint(max_offset_point)) {
                if (max_offset_point.x < path_point.x) {
                    min_x = std::min(min_x, max_offset_point.x);
                } else if (max_offset_point.x > path_point.x) {
                    max_x = std::max(max_x, max_offset_point.x);
                }
            }
        }
        if (std::abs(offset_m) > zero_epsilon_m) {
            bool shifted_valid = false;
            const cv::Point2f shifted =
                shiftPointByRouteOffset(path_point, route, offset_m, shifted_valid);
            if (shifted_valid && isFinitePoint(shifted)) {
                path_point.x = clampValue(shifted.x, min_x, max_x);
                path_point.y = rp.pos.y;
            }
        }
        path.push_back(path_point);
    }
    return path;
}

float computeRealizedVehicleOffset(const std::vector<cv::Point2f>& path,
                                   const std::vector<PlanningRoutePoint>& route,
                                   int avoid_side) {
    const int normalized_side = normalizeAvoidSide(avoid_side);
    if (normalized_side == 0 || path.empty() || route.size() < 2) {
        return 0.0f;
    }

    float max_directed_offset_m = 0.0f;
    bool found_valid_point = false;
    for (const auto& pixel : path) {
        bool world_valid = false;
        const cv::Point2f world = pixelToWorldMeters(pixel, &world_valid);
        if (!world_valid || !isFinitePoint(world)) {
            continue;
        }

        PlanningRoutePoint route_point;
        cv::Point2f route_center_world;
        cv::Point2f route_normal_world;
        if (!routeFrameNearestWorld(route,
                                    world,
                                    route_point,
                                    route_center_world,
                                    route_normal_world)) {
            continue;
        }
        const float signed_lateral =
            dotPoint(world - route_center_world, route_normal_world);
        const float directed_lateral =
            static_cast<float>(normalized_side) * signed_lateral;
        if (!isFiniteFloat(directed_lateral)) {
            continue;
        }
        max_directed_offset_m =
            std::max(max_directed_offset_m, std::max(0.0f, directed_lateral));
        found_valid_point = true;
    }

    return found_valid_point
               ? static_cast<float>(normalized_side) * max_directed_offset_m
               : 0.0f;
}

struct PathProjection {
    bool valid = false;
    int segment_index = -1;
    float projection_t = 0.0f;
    float raw_projection_t = 0.0f;
    float along_path_s = 0.0f;
    float lateral_offset_m = 0.0f;
    float distance_to_path_m = std::numeric_limits<float>::infinity();
    cv::Point2f projection_world = cv::Point2f(0.0f, 0.0f);
    cv::Point2f normal_world = cv::Point2f(1.0f, 0.0f);
    cv::Point2f segment_start_world = cv::Point2f(0.0f, 0.0f);
    cv::Point2f segment_end_world = cv::Point2f(0.0f, 0.0f);
    bool is_terminal_segment = false;
};

bool pathToWorldSamples(const std::vector<cv::Point2f>& path,
                        std::vector<cv::Point2f>& world_points,
                        std::vector<float>& cumulative_s) {
    world_points.clear();
    cumulative_s.clear();
    if (path.size() < 2) return false;

    world_points.reserve(path.size());
    cumulative_s.reserve(path.size());
    for (const auto& point : path) {
        bool valid = false;
        const cv::Point2f world = pixelToWorldMeters(point, &valid);
        if (!valid || !isFinitePoint(world)) return false;
        world_points.push_back(world);
        if (cumulative_s.empty()) {
            cumulative_s.push_back(0.0f);
        } else {
            cumulative_s.push_back(
                cumulative_s.back() +
                static_cast<float>(cv::norm(world_points.back() -
                                            world_points[world_points.size() - 2])));
        }
    }
    return world_points.size() >= 2;
}

cv::Point2f orientedPathNormal(const cv::Point2f& tangent_world) {
    const float len = static_cast<float>(cv::norm(tangent_world));
    if (!isFiniteFloat(len) || len < 1e-5f) {
        return cv::Point2f(1.0f, 0.0f);
    }
    cv::Point2f normal(-tangent_world.y / len, tangent_world.x / len);
    if (normal.x < 0.0f) {
        normal *= -1.0f;
    }
    return normal;
}

PathProjection projectWorldPointToPath(const cv::Point2f& world,
                                       const std::vector<cv::Point2f>& world_points,
                                       const std::vector<float>& cumulative_s) {
    PathProjection out;
    if (!isFinitePoint(world) || world_points.size() < 2 ||
        world_points.size() != cumulative_s.size()) {
        return out;
    }

    float best_dist2 = std::numeric_limits<float>::infinity();
    for (size_t i = 0; i + 1 < world_points.size(); ++i) {
        const cv::Point2f a = world_points[i];
        const cv::Point2f b = world_points[i + 1];
        const cv::Point2f ab = b - a;
        const float ab2 = dotPoint(ab, ab);
        if (!isFiniteFloat(ab2) || ab2 < 1e-8f) continue;

        const float raw_t = dotPoint(world - a, ab) / ab2;
        const float t = clampValue(raw_t, 0.0f, 1.0f);
        const cv::Point2f projection = a + ab * t;
        const cv::Point2f delta = world - projection;
        const float dist2 = dotPoint(delta, delta);
        if (!isFiniteFloat(dist2) || dist2 >= best_dist2) continue;

        const float seg_len = std::sqrt(ab2);
        const cv::Point2f normal = orientedPathNormal(ab);
        out.valid = true;
        out.segment_index = static_cast<int>(i);
        out.projection_t = t;
        out.raw_projection_t = raw_t;
        out.projection_world = projection;
        out.normal_world = normal;
        out.segment_start_world = a;
        out.segment_end_world = b;
        out.is_terminal_segment = i + 2 == world_points.size();
        out.along_path_s = cumulative_s[i] + seg_len * t;
        out.lateral_offset_m = dotPoint(delta, normal);
        out.distance_to_path_m = std::sqrt(dist2);
        best_dist2 = dist2;
    }
    return out;
}

constexpr int kGoldBiasConfirmFrames = 2;
constexpr int kGoldTargetSettleFrames = 2;
constexpr float kGoldBiasEpsilonM = 1e-4f;

struct ProjectedGoldForBias {
    cv::Point2f world_pos_m = cv::Point2f(0.0f, 0.0f);
    float safety_lateral_m = 0.0f;
    float compensated_lateral_m = 0.0f;
    float along_s_m = 0.0f;
};

struct GoldBiasCandidate {
    float bias_m = 0.0f;
    float action_delta_m = 0.0f;
    int near_capture_count = 0;
    int capture_count = 0;
    float capture_along_s_sum = 0.0f;
    float capture_min_along_s_m = std::numeric_limits<float>::infinity();
};

struct GoldBiasEvaluation {
    float best_bias_m = 0.0f;
    int best_near_capture_count = 0;
    int best_capture_count = 0;
    float best_capture_along_s_sum = 0.0f;
    float best_capture_min_along_s_m =
        std::numeric_limits<float>::infinity();
    int visible_count = 0;
    int projectable_count = 0;
    std::string no_capture_diag;
    std::string reason = "no_gold";
};

bool logicalSafetyPathWorldSamples(
    const std::vector<PlanningRoutePoint>& route,
    float safety_offset_m,
    std::vector<cv::Point2f>& world_points,
    std::vector<float>& cumulative_s) {
    std::vector<cv::Point2f> safety_path;
    safety_path.reserve(route.size());
    for (const auto& route_point : route) {
        bool shifted_valid = false;
        const cv::Point2f shifted = shiftPointByRouteOffset(
            route_point.pos, route, safety_offset_m, shifted_valid);
        if (!shifted_valid || !isFinitePoint(shifted)) {
            return false;
        }
        safety_path.push_back(shifted);
    }
    return pathToWorldSamples(safety_path, world_points, cumulative_s);
}

GoldBiasEvaluation evaluateGoldBiasCandidates(
    const std::vector<PlanningRoutePoint>& route,
    const PathPlanningInput& cfg,
    float safety_offset_m,
    const GoldBiasRuntime& gold) {
    GoldBiasEvaluation evaluation;
    std::vector<cv::Point2f> safety_world_points;
    std::vector<float> safety_cumulative_s;
    if (!logicalSafetyPathWorldSamples(route,
                                       safety_offset_m,
                                       safety_world_points,
                                       safety_cumulative_s)) {
        evaluation.reason = "safety_path_invalid";
        return evaluation;
    }

    std::vector<ProjectedGoldForBias> projected_golds;
    for (const auto& gold_object : objectsOfType(cfg.objects, ObjectType::GOLD)) {
        if (!gold_object.observed_this_frame ||
            !gold_object.current_frame_visible ||
            !gold_object.has_world_pos_m ||
            !isFinitePoint(gold_object.world_pos_m) ||
            gold_object.world_pos_m.y <= 0.0f) {
            continue;
        }
        evaluation.visible_count++;
        const PathProjection projection = projectWorldPointToPath(
            gold_object.world_pos_m,
            safety_world_points,
            safety_cumulative_s);
        if (!projection.valid || projection.raw_projection_t < -1e-3f ||
            projection.raw_projection_t > 1.001f) {
            continue;
        }

        const float lateral_m = projection.lateral_offset_m;
        float compensated_lateral_m = lateral_m;
        if (std::abs(lateral_m) <= cfg.gold_capture_tolerance_m) {
            compensated_lateral_m = 0.0f;
        } else {
            compensated_lateral_m +=
                lateral_m > 0.0f ? cfg.gold_capture_compensation_m
                                 : -cfg.gold_capture_compensation_m;
        }
        projected_golds.push_back(
            ProjectedGoldForBias{gold_object.world_pos_m,
                                 lateral_m,
                                 compensated_lateral_m,
                                 projection.along_path_s});
        evaluation.projectable_count++;
    }

    std::vector<GoldBiasCandidate> candidates;
    candidates.push_back(GoldBiasCandidate{});
    const int bias_step_count = static_cast<int>(
        std::floor(cfg.gold_max_bias_m / cfg.gold_bias_step_m + 1e-5f));
    for (int step_index = 1; step_index <= bias_step_count; ++step_index) {
        const float magnitude =
            std::min(cfg.gold_max_bias_m,
                     cfg.gold_bias_step_m * static_cast<float>(step_index));
        candidates.push_back(GoldBiasCandidate{-magnitude});
        candidates.push_back(GoldBiasCandidate{magnitude});
    }

    const float action_reference_m =
        gold.transition_state == GoldTransitionState::RECENTER
            ? gold.applied_m
            : gold.stable_base_m;
    int executable_candidate_count = 0;
    for (auto& candidate : candidates) {
        candidate.action_delta_m =
            std::abs(candidate.bias_m - action_reference_m);
        const bool exceeds_new_action_limit =
            candidate.action_delta_m >
            cfg.gold_max_new_delta_m + kGoldBiasEpsilonM;
        const bool opposes_safety_offset =
            std::abs(safety_offset_m) > cfg.obstacle_offset_zero_epsilon_m &&
            candidate.bias_m * safety_offset_m < 0.0f;
        const int stable_side = sideFromObstacleOffset(gold.stable_base_m);
        const int candidate_side = sideFromObstacleOffset(candidate.bias_m);
        const bool crosses_recenter_side =
            gold.transition_state == GoldTransitionState::RECENTER &&
            std::abs(gold.applied_m) > cfg.gold_center_release_m &&
            stable_side != 0 && candidate_side != 0 &&
            candidate_side != stable_side;
        if (exceeds_new_action_limit || opposes_safety_offset ||
            crosses_recenter_side) {
            candidate.capture_count = -1;
            continue;
        }
        executable_candidate_count++;
        for (const auto& projected_gold : projected_golds) {
            const bool static_capture =
                std::abs(candidate.bias_m -
                         projected_gold.compensated_lateral_m) <=
                cfg.gold_capture_tolerance_m;
            if (static_capture) {
                candidate.capture_count++;
                candidate.capture_along_s_sum += projected_gold.along_s_m;
                candidate.capture_min_along_s_m = std::min(
                    candidate.capture_min_along_s_m,
                    projected_gold.along_s_m);
                if (projected_gold.along_s_m <= cfg.gold_near_priority_distance_m) {
                    candidate.near_capture_count++;
                }
            }
        }
    }

    const auto better_candidate = [&](const GoldBiasCandidate& lhs,
                                      const GoldBiasCandidate& rhs) {
        const bool lhs_valid = lhs.capture_count >= 0;
        const bool rhs_valid = rhs.capture_count >= 0;
        if (lhs_valid != rhs_valid) {
            return lhs_valid;
        }
        if (lhs.near_capture_count != rhs.near_capture_count) {
            return lhs.near_capture_count > rhs.near_capture_count;
        }
        if (lhs.capture_count != rhs.capture_count) {
            return lhs.capture_count > rhs.capture_count;
        }
        if (std::abs(lhs.capture_along_s_sum - rhs.capture_along_s_sum) >
            kGoldBiasEpsilonM) {
            return lhs.capture_along_s_sum < rhs.capture_along_s_sum;
        }
        if (std::abs(lhs.action_delta_m - rhs.action_delta_m) >
            kGoldBiasEpsilonM) {
            return lhs.action_delta_m < rhs.action_delta_m;
        }
        if (std::abs(std::abs(lhs.bias_m) - std::abs(rhs.bias_m)) >
            kGoldBiasEpsilonM) {
            return std::abs(lhs.bias_m) < std::abs(rhs.bias_m);
        }
        return lhs.bias_m < rhs.bias_m;
    };
    const auto best_it = std::min_element(
        candidates.begin(),
        candidates.end(),
        [&](const GoldBiasCandidate& lhs, const GoldBiasCandidate& rhs) {
            return better_candidate(lhs, rhs);
        });
    if (best_it != candidates.end() && best_it->capture_count >= 0) {
        evaluation.best_bias_m = best_it->bias_m;
        evaluation.best_near_capture_count = best_it->near_capture_count;
        evaluation.best_capture_count = best_it->capture_count;
        evaluation.best_capture_along_s_sum = best_it->capture_along_s_sum;
        evaluation.best_capture_min_along_s_m =
            best_it->capture_min_along_s_m;
    }
    evaluation.reason = evaluation.visible_count == 0
                            ? "no_gold"
                            : (evaluation.projectable_count == 0
                                   ? "no_projectable_gold"
                                   : (executable_candidate_count == 0
                                          ? "no_executable_gold_candidate"
                                          : (evaluation.best_capture_count == 0
                                          ? "no_capture_benefit"
                                          : "bias_selected")));
    if (evaluation.projectable_count > 0 &&
        evaluation.best_capture_count == 0) {
        struct CandidatePathDiagnostic {
            float bias_m = 0.0f;
            std::vector<cv::Point2f> world_points;
            std::vector<float> cumulative_s;
        };
        std::vector<CandidatePathDiagnostic> candidate_paths;
        candidate_paths.reserve(candidates.size());
        for (const auto& candidate : candidates) {
            CandidatePathDiagnostic path;
            path.bias_m = candidate.bias_m;
            if (logicalSafetyPathWorldSamples(
                    route,
                    safety_offset_m + candidate.bias_m,
                    path.world_points,
                    path.cumulative_s)) {
                candidate_paths.push_back(std::move(path));
            }
        }

        std::ostringstream diag;
        diag << std::fixed << std::setprecision(3);
        for (size_t gold_index = 0;
             gold_index < projected_golds.size();
             ++gold_index) {
            const auto& projected_gold = projected_golds[gold_index];
            float nearest_bias_m = 0.0f;
            float nearest_distance_m = std::numeric_limits<float>::infinity();
            bool nearest_valid = false;
            for (const auto& candidate_path : candidate_paths) {
                const PathProjection candidate_projection =
                    projectWorldPointToPath(
                        projected_gold.world_pos_m,
                        candidate_path.world_points,
                        candidate_path.cumulative_s);
                if (!candidate_projection.valid ||
                    !isFiniteFloat(candidate_projection.distance_to_path_m) ||
                    candidate_projection.distance_to_path_m >=
                        nearest_distance_m) {
                    continue;
                }
                nearest_valid = true;
                nearest_bias_m = candidate_path.bias_m;
                nearest_distance_m = candidate_projection.distance_to_path_m;
            }
            if (gold_index > 0) diag << ";";
            diag << "g" << gold_index
                 << "|s=" << projected_gold.along_s_m
                 << "|lat=" << projected_gold.safety_lateral_m
                 << "|bias=";
            if (nearest_valid) {
                diag << nearest_bias_m
                     << "|min_d=" << nearest_distance_m;
            } else {
                diag << "nan|min_d=nan";
            }
        }
        evaluation.no_capture_diag = diag.str();
    }
    return evaluation;
}

bool goldBiasLateStartRejected(const GoldBiasEvaluation& evaluation,
                               const PathPlanningInput& cfg,
                               const GoldBiasRuntime& gold) {
    const bool continues_pending_bias =
        gold.pending_confirm_count > 0 &&
        std::abs(evaluation.best_bias_m - gold.pending_bias_m) <=
            kGoldBiasEpsilonM;
    const float action_reference_m =
        gold.transition_state == GoldTransitionState::RECENTER
            ? gold.applied_m
            : gold.stable_base_m;
    const float late_start_delta_m =
        std::abs(evaluation.best_bias_m - action_reference_m);
    return evaluation.best_capture_count > 0 &&
           !continues_pending_bias &&
           evaluation.best_capture_min_along_s_m <=
               cfg.gold_late_start_forward_m &&
           late_start_delta_m > cfg.gold_late_start_delta_m;
}

void clearGoldPending(GoldBiasRuntime& gold) {
    gold.pending_bias_m = 0.0f;
    gold.pending_confirm_count = 0;
}

void confirmGoldBias(float best_bias_m, GoldBiasRuntime& gold) {
    if (gold.pending_confirm_count > 0 &&
        std::abs(best_bias_m - gold.pending_bias_m) <=
            kGoldBiasEpsilonM) {
        gold.pending_confirm_count++;
    } else {
        gold.pending_bias_m = best_bias_m;
        gold.pending_confirm_count = 1;
    }
    if (gold.pending_confirm_count >= kGoldBiasConfirmFrames) {
        gold.target_m = best_bias_m;
        gold.transition_state = GoldTransitionState::HARD;
        gold.target_reached_frames = 0;
        gold.no_executable_gold_frames = 0;
        clearGoldPending(gold);
    }
}

void advanceGoldTransition(float dt,
                           const PathPlanningInput& cfg,
                           float commanded_target_m,
                           GoldBiasRuntime& gold) {
    if (gold.transition_state != GoldTransitionState::HARD &&
        gold.transition_state != GoldTransitionState::RECENTER) {
        return;
    }
    const float max_step = std::max(0.0f, cfg.gold_bias_rate_mps) * dt;
    const float delta = commanded_target_m - gold.applied_m;
    if (std::abs(delta) <= max_step) {
        gold.applied_m = commanded_target_m;
    } else {
        gold.applied_m += delta > 0.0f ? max_step : -max_step;
    }

    if (std::abs(gold.applied_m - gold.target_m) <= kGoldBiasEpsilonM) {
        gold.target_reached_frames++;
    } else {
        gold.target_reached_frames = 0;
    }
    if (gold.target_reached_frames >= kGoldTargetSettleFrames) {
        const float completed_target_m = gold.target_m;
        if (gold.transition_state == GoldTransitionState::HARD) {
            enterGoldIdleAt(gold, completed_target_m);
        } else {
            enterGoldIdleAt(gold, 0.0f);
        }
    }
}

const char* goldTransitionStateName(GoldTransitionState state) {
    switch (state) {
        case GoldTransitionState::HARD: return "HARD";
        case GoldTransitionState::RECENTER: return "RECENTER";
        case GoldTransitionState::IDLE:
        default: return "IDLE";
    }
}

void syncGoldDebug(const GoldBiasRuntime& gold,
                   float effective_gold_bias_m,
                   PathPlanningDebug& debug) {
    if (gold.transition_state == GoldTransitionState::IDLE) {
        assert(std::abs(gold.stable_base_m - gold.target_m) <=
                   kGoldBiasEpsilonM &&
               std::abs(gold.target_m - gold.applied_m) <=
                   kGoldBiasEpsilonM);
    }
    debug.gold_stable_base_m = gold.stable_base_m;
    debug.gold_transition_state_name =
        goldTransitionStateName(gold.transition_state);
    debug.gold_target_bias_m = gold.target_m;
    debug.gold_applied_bias_m = gold.applied_m;
    debug.gold_effective_bias_m = effective_gold_bias_m;
    debug.gold_pending_bias_m = gold.pending_bias_m;
    debug.gold_pending_confirm_count = gold.pending_confirm_count;
    debug.gold_no_executable_frames = gold.no_executable_gold_frames;
    debug.gold_target_reached_frames = gold.target_reached_frames;
}

float updateGoldBiasSelector(
    const std::vector<PlanningRoutePoint>& route,
    const PathPlanningInput& cfg,
    float safety_offset_m,
    float dt,
    const PathPlanningRuntime& runtime,
    GoldBiasRuntime& gold,
    PathPlanningResult& result) {
    PathPlanningDebug& debug = result.debug;
    const bool pedestrian_observation_starts_slow =
        cfg.pedestrian_slow_enable &&
        !runtime.pedestrian_behavior.slow_dismissed &&
        std::any_of(
            cfg.raw_person_objects.begin(),
            cfg.raw_person_objects.end(),
            [&](const PlanningObject& person) {
                return person.type == ObjectType::PERSON &&
                       person.from_raw_detection &&
                       person.observed_this_frame &&
                       person.has_world_pos_m &&
                       isFinitePoint(person.world_pos_m) &&
                       std::hypot(person.world_pos_m.x,
                                  person.world_pos_m.y) <=
                           std::max(0.0f,
                                    cfg.pedestrian_slow_distance_m);
            });
    const bool pedestrian_behavior_freeze =
        runtime.pedestrian_behavior.state != PedestrianStopState::DRIVING ||
        runtime.pedestrian_behavior.slow_active ||
        pedestrian_observation_starts_slow;
    const bool vehicle_safety_freeze =
        runtime.avoid_scenario == AvoidScenario::VehicleOnly &&
        (runtime.obstacle_state == ObstacleAvoidanceState::Active ||
         runtime.obstacle_state == ObstacleAvoidanceState::Recover);

    if (pedestrian_behavior_freeze) {
        debug.gold_selection_frozen = true;
        debug.gold_bias_reason = "pedestrian_behavior_frozen";
        clearGoldPending(gold);
    } else if (vehicle_safety_freeze) {
        // Vehicle Safety owns the lateral handoff. Preserve the confirmed
        // Gold transition exactly, discard only unconfirmed intent, and
        // resume from the same planner-owned state after Safety exits.
        debug.gold_selection_frozen = true;
        debug.gold_bias_reason = "vehicle_safety_frozen";
        clearGoldPending(gold);
    } else if (gold.transition_state == GoldTransitionState::HARD) {
        debug.gold_bias_reason = "hard_transition";
        clearGoldPending(gold);
        advanceGoldTransition(dt, cfg, gold.target_m, gold);
    } else {
        const GoldBiasEvaluation evaluation = evaluateGoldBiasCandidates(
            route, cfg, safety_offset_m, gold);
        debug.gold_bias_reason = evaluation.reason;
        debug.gold_visible_count = evaluation.visible_count;
        debug.gold_projectable_count = evaluation.projectable_count;
        debug.gold_best_near_capture_count =
            evaluation.best_near_capture_count;
        debug.gold_best_capture_count = evaluation.best_capture_count;
        debug.gold_best_capture_along_s_sum =
            evaluation.best_capture_along_s_sum;
        debug.gold_no_capture_diag = evaluation.no_capture_diag;
        debug.gold_best_bias_m = evaluation.best_bias_m;
        const float action_reference_m =
            gold.transition_state == GoldTransitionState::RECENTER
                ? gold.applied_m
                : gold.stable_base_m;
        debug.gold_late_start_delta_m =
            std::abs(evaluation.best_bias_m - action_reference_m);
        if (evaluation.best_capture_count > 0) {
            debug.gold_late_start_s_m =
                evaluation.best_capture_min_along_s_m;
        }

        const bool late_start_rejected =
            goldBiasLateStartRejected(evaluation, cfg, gold);
        debug.gold_late_start_rejected = late_start_rejected;
        const bool has_executable_capture =
            evaluation.best_capture_count > 0 && !late_start_rejected;
        if (has_executable_capture) {
            gold.no_executable_gold_frames = 0;
            if (gold.transition_state == GoldTransitionState::IDLE &&
                std::abs(evaluation.best_bias_m - gold.stable_base_m) <=
                    kGoldBiasEpsilonM) {
                clearGoldPending(gold);
                debug.gold_bias_reason = "stable_base_captures_gold";
            } else {
                confirmGoldBias(evaluation.best_bias_m, gold);
            }
        } else {
            clearGoldPending(gold);
            if (late_start_rejected) {
                debug.gold_bias_reason = "late_start_rejected";
            }
            if (gold.transition_state == GoldTransitionState::IDLE) {
                gold.no_executable_gold_frames++;
                if (gold.no_executable_gold_frames >= cfg.gold_recenter_hold_frames &&
                    std::abs(gold.stable_base_m) > kGoldBiasEpsilonM) {
                    gold.target_m = 0.0f;
                    gold.transition_state = GoldTransitionState::RECENTER;
                    gold.target_reached_frames = 0;
                    gold.no_executable_gold_frames = 0;
                    debug.gold_bias_reason = "recenter_started";
                }
            }
        }

        if (gold.transition_state == GoldTransitionState::RECENTER) {
            advanceGoldTransition(dt, cfg, gold.target_m, gold);
        }
    }

    float effective_gold_bias_m = gold.applied_m;
    if (std::abs(safety_offset_m) > cfg.obstacle_offset_zero_epsilon_m &&
        safety_offset_m * effective_gold_bias_m < 0.0f) {
        effective_gold_bias_m = 0.0f;
    }
    syncGoldDebug(gold, effective_gold_bias_m, debug);
    return effective_gold_bias_m;
}

float cross2d(const cv::Point2f& a, const cv::Point2f& b) {
    return a.x * b.y - a.y * b.x;
}

bool pointOnSegmentInclusive(const cv::Point2f& point,
                             const cv::Point2f& a,
                             const cv::Point2f& b) {
    static constexpr float kSegmentEpsilon = 1e-5f;
    if (std::abs(cross2d(b - a, point - a)) > kSegmentEpsilon) {
        return false;
    }
    return point.x >= std::min(a.x, b.x) - kSegmentEpsilon &&
           point.x <= std::max(a.x, b.x) + kSegmentEpsilon &&
           point.y >= std::min(a.y, b.y) - kSegmentEpsilon &&
           point.y <= std::max(a.y, b.y) + kSegmentEpsilon;
}

bool segmentsIntersectInclusive(const cv::Point2f& a0,
                                const cv::Point2f& a1,
                                const cv::Point2f& b0,
                                const cv::Point2f& b1) {
    static constexpr float kSegmentEpsilon = 1e-5f;
    const float c0 = cross2d(a1 - a0, b0 - a0);
    const float c1 = cross2d(a1 - a0, b1 - a0);
    const float c2 = cross2d(b1 - b0, a0 - b0);
    const float c3 = cross2d(b1 - b0, a1 - b0);
    if (((c0 > kSegmentEpsilon && c1 < -kSegmentEpsilon) ||
         (c0 < -kSegmentEpsilon && c1 > kSegmentEpsilon)) &&
        ((c2 > kSegmentEpsilon && c3 < -kSegmentEpsilon) ||
         (c2 < -kSegmentEpsilon && c3 > kSegmentEpsilon))) {
        return true;
    }
    return (std::abs(c0) <= kSegmentEpsilon &&
            pointOnSegmentInclusive(b0, a0, a1)) ||
           (std::abs(c1) <= kSegmentEpsilon &&
            pointOnSegmentInclusive(b1, a0, a1)) ||
           (std::abs(c2) <= kSegmentEpsilon &&
            pointOnSegmentInclusive(a0, b0, b1)) ||
           (std::abs(c3) <= kSegmentEpsilon &&
            pointOnSegmentInclusive(a1, b0, b1));
}

bool buildPedestrianNormalStrip(
    const std::vector<PlanningRoutePoint>& route,
    int debug_id,
    const cv::Point2f& pedestrian_world,
    float half_width_along_road_m,
    PedestrianNormalStripGeometry& strip) {
    strip = PedestrianNormalStripGeometry{};
    strip.track_id = debug_id;
    strip.pedestrian_world = pedestrian_world;
    strip.half_width_along_road_m =
        std::max(0.0f, half_width_along_road_m);
    strip.half_length_across_road_m = 0.40f;
    if (route.size() < 2 || !isFinitePoint(pedestrian_world) ||
        strip.half_width_along_road_m <= 0.0f) {
        return false;
    }

    float best_distance_sq = std::numeric_limits<float>::infinity();
    for (size_t i = 0; i + 1 < route.size(); ++i) {
        bool a_valid = false;
        bool b_valid = false;
        const cv::Point2f a = pixelToWorldMeters(route[i].pos, &a_valid);
        const cv::Point2f b = pixelToWorldMeters(route[i + 1].pos, &b_valid);
        if (!a_valid || !b_valid || !isFinitePoint(a) || !isFinitePoint(b)) {
            continue;
        }
        const cv::Point2f segment = b - a;
        const float segment_length_sq = dotPoint(segment, segment);
        if (!isFiniteFloat(segment_length_sq) || segment_length_sq <= 1e-8f) {
            continue;
        }
        const float raw_t = dotPoint(pedestrian_world - a, segment) /
                            segment_length_sq;
        const float clamped_t = clampValue(raw_t, 0.0f, 1.0f);
        const cv::Point2f closest = a + segment * clamped_t;
        const float distance_sq = dotPoint(
            pedestrian_world - closest,
            pedestrian_world - closest);
        if (!isFiniteFloat(distance_sq) || distance_sq >= best_distance_sq) {
            continue;
        }

        const float segment_length = std::sqrt(segment_length_sq);
        strip.center_world = a + segment * raw_t;
        strip.tangent_unit = segment * (1.0f / segment_length);
        strip.normal_unit = cv::Point2f(-strip.tangent_unit.y,
                                       strip.tangent_unit.x);
        if (strip.normal_unit.x < 0.0f) {
            strip.normal_unit *= -1.0f;
        }
        best_distance_sq = distance_sq;
        strip.valid = true;
    }

    if (!strip.valid) {
        return false;
    }
    const cv::Point2f forward_center =
        strip.center_world +
        strip.tangent_unit * strip.half_width_along_road_m;
    const cv::Point2f across =
        strip.normal_unit * strip.half_length_across_road_m;
    strip.forward_edge_left_world = forward_center - across;
    strip.forward_edge_right_world = forward_center + across;
    return true;
}

bool pedestrianStripForwardEdgeIntersectsCorridor(
    const PedestrianNormalStripGeometry& strip,
    const PedestrianDangerCorridorGeometry& corridor) {
    if (!strip.valid || !corridor.valid ||
        corridor.polygon_world.size() < 3) {
        return false;
    }
    if (cv::pointPolygonTest(corridor.polygon_world,
                             strip.forward_edge_left_world,
                             false) >= 0.0 ||
        cv::pointPolygonTest(corridor.polygon_world,
                             strip.forward_edge_right_world,
                             false) >= 0.0) {
        return true;
    }
    for (size_t i = 0; i < corridor.polygon_world.size(); ++i) {
        const cv::Point2f& a = corridor.polygon_world[i];
        const cv::Point2f& b =
            corridor.polygon_world[(i + 1) % corridor.polygon_world.size()];
        if (segmentsIntersectInclusive(strip.forward_edge_left_world,
                                       strip.forward_edge_right_world,
                                       a,
                                       b)) {
            return true;
        }
    }
    return false;
}

void appendUniqueTrackId(std::vector<int>& ids, int id) {
    if (id < 0 || std::find(ids.begin(), ids.end(), id) != ids.end()) {
        return;
    }
    ids.push_back(id);
}

}  // namespace

PedestrianDangerCorridorGeometry buildPedestrianDangerCorridor(
    float servo_output,
    float near_width_m,
    float far_width_m,
    float length_m,
    float sample_step_m,
    float servo_straight_deadband,
    float servo_radius_max_abs) {
    PedestrianDangerCorridorGeometry geometry;
    if (!pedestrianDangerCorridorValid(
            near_width_m, far_width_m, length_m, sample_step_m)) {
        near_width_m = kPedestrianDangerDefaultNearWidthM;
        far_width_m = kPedestrianDangerDefaultFarWidthM;
        length_m = kPedestrianDangerDefaultLengthM;
        sample_step_m = kPedestrianDangerDefaultSampleStepM;
    }

    geometry.servo_output = std::isfinite(servo_output) ? servo_output : 0.0f;
    geometry.near_width_m = near_width_m;
    geometry.far_width_m = far_width_m;
    geometry.length_m = length_m;
    geometry.sample_step_m = sample_step_m;

    const float deadband = std::max(0.0f, servo_straight_deadband);
    const float max_servo_abs =
        std::max(deadband, std::max(0.0f, servo_radius_max_abs));
    const float servo_abs = std::abs(geometry.servo_output);
    geometry.servo_abs_used = std::min(servo_abs, max_servo_abs);
    geometry.turn_sign = geometry.servo_output > 0.0f
                             ? 1
                             : (geometry.servo_output < 0.0f ? -1 : 0);
    geometry.straight = geometry.servo_abs_used < deadband ||
                        geometry.turn_sign == 0;

    if (!geometry.straight) {
        // Fixed reference speed v=1.0 m/s. The fitted speed multiplier is
        // therefore exactly 1 and actual low-speed commands do not distort
        // the servo-to-radius relationship.
        static constexpr float kRadiusDenom0 = -0.047328472f;
        static constexpr float kRadiusDenom1 = 0.019933312f;
        static constexpr float kRadiusDenom2 = -1.396452957e-5f;
        const float denominator =
            kRadiusDenom0 + kRadiusDenom1 * geometry.servo_abs_used +
            kRadiusDenom2 * geometry.servo_abs_used * geometry.servo_abs_used;
        if (std::isfinite(denominator) && denominator > 1e-5f) {
            geometry.turn_radius_m = 1.0f / denominator;
        } else {
            geometry.straight = true;
            geometry.turn_sign = 0;
            geometry.turn_radius_m = 0.0f;
        }
    }

    const int sample_count = std::max(
        2,
        static_cast<int>(std::ceil(length_m / sample_step_m)) + 1);
    geometry.centerline_world.reserve(sample_count);
    geometry.left_boundary_world.reserve(sample_count);
    geometry.right_boundary_world.reserve(sample_count);

    for (int i = 0; i < sample_count; ++i) {
        const float ratio = static_cast<float>(i) /
                            static_cast<float>(sample_count - 1);
        const float along_s_m = length_m * ratio;
        cv::Point2f center(0.0f, along_s_m);
        cv::Point2f normal(1.0f, 0.0f);
        if (!geometry.straight && geometry.turn_radius_m > 0.0f) {
            const float theta = along_s_m / geometry.turn_radius_m;
            const float turn_sign = static_cast<float>(geometry.turn_sign);
            center.x = turn_sign * geometry.turn_radius_m *
                       (1.0f - std::cos(theta));
            center.y = geometry.turn_radius_m * std::sin(theta);
            normal.x = std::cos(theta);
            normal.y = -turn_sign * std::sin(theta);
        }

        const float width_m = pedestrianDangerCorridorAllowedWidthM(
            along_s_m, near_width_m, far_width_m, length_m);
        const float half_width_m = 0.5f * width_m;
        geometry.centerline_world.push_back(center);
        geometry.left_boundary_world.push_back(center - normal * half_width_m);
        geometry.right_boundary_world.push_back(center + normal * half_width_m);
    }

    geometry.polygon_world.reserve(
        geometry.left_boundary_world.size() +
        geometry.right_boundary_world.size());
    for (const auto& point : geometry.left_boundary_world) {
        geometry.polygon_world.push_back(point);
    }
    for (auto it = geometry.right_boundary_world.rbegin();
         it != geometry.right_boundary_world.rend(); ++it) {
        geometry.polygon_world.push_back(*it);
    }
    geometry.valid = geometry.centerline_world.size() >= 2 &&
                     geometry.polygon_world.size() >= 4;
    return geometry;
}

PedestrianDangerHitResult evaluatePedestrianDangerCorridor(
    const PedestrianDangerCorridorGeometry& corridor,
    const cv::Point2f& person_world_m) {
    PedestrianDangerHitResult hit;
    if (!isFinitePoint(person_world_m)) {
        return hit;
    }

    if (corridor.valid && corridor.polygon_world.size() >= 3) {
        const double signed_polygon_distance = cv::pointPolygonTest(
            corridor.polygon_world, person_world_m, true);
        hit.boundary_clearance_m =
            static_cast<float>(-signed_polygon_distance);
        hit.corridor_hit = signed_polygon_distance >= 0.0;

        float best_distance_sq = std::numeric_limits<float>::infinity();
        float cumulative_s_m = 0.0f;
        for (size_t i = 0; i + 1 < corridor.centerline_world.size(); ++i) {
            const cv::Point2f a = corridor.centerline_world[i];
            const cv::Point2f b = corridor.centerline_world[i + 1];
            const cv::Point2f ab = b - a;
            const float segment_length_sq = dotPoint(ab, ab);
            if (!std::isfinite(segment_length_sq) ||
                segment_length_sq <= 1e-8f) {
                continue;
            }
            const float segment_length_m = std::sqrt(segment_length_sq);
            const float projection_t = clampValue(
                dotPoint(person_world_m - a, ab) / segment_length_sq,
                0.0f,
                1.0f);
            const cv::Point2f closest = a + ab * projection_t;
            const cv::Point2f delta = person_world_m - closest;
            const float distance_sq = dotPoint(delta, delta);
            if (distance_sq < best_distance_sq) {
                best_distance_sq = distance_sq;
                hit.closest_world = closest;
                hit.along_s_m = cumulative_s_m +
                                projection_t * segment_length_m;
                const cv::Point2f right_normal(
                    ab.y / segment_length_m,
                    -ab.x / segment_length_m);
                hit.signed_lateral_m = dotPoint(delta, right_normal);
            }
            cumulative_s_m += segment_length_m;
        }
        if (std::isfinite(best_distance_sq)) {
            hit.lateral_distance_m = std::sqrt(best_distance_sq);
            hit.allowed_width_m = pedestrianDangerCorridorAllowedWidthM(
                hit.along_s_m,
                corridor.near_width_m,
                corridor.far_width_m,
                corridor.length_m);
        }
    }

    hit.hit = hit.corridor_hit;
    return hit;
}

PathPlanningResult planLocalPath(const PathPlanningInput& input,
                                 PathPlanningRuntime& runtime) {
    PathPlanningResult result;
    result.debug.obstacle_state = runtime.obstacle_state;
    result.debug.avoid_scenario = runtime.avoid_scenario;
    result.debug.obstacle_offset_current_m = runtime.obstacle_offset_current_m;
    result.debug.obstacle_offset_target_m = runtime.obstacle_offset_target_m;
    result.debug.final_applied_path_offset_m = 0.0f;
    syncGoldDebug(runtime.gold, runtime.gold.applied_m, result.debug);
    result.debug.ego_speed_mps = input.ego_speed_mps;
    result.debug.ego_center_lateral_m = runtime.ego_center_lateral_m;

    PathPlanningInput cfg = input;
    cfg.obstacle_confirm_frames = std::max(1, cfg.obstacle_confirm_frames);
    cfg.obstacle_exit_lost_frames = std::max(0, cfg.obstacle_exit_lost_frames);
    cfg.vehicle_recover_hold_frames = std::max(0, cfg.vehicle_recover_hold_frames);
    cfg.vehicle_recover_hold_max_lifetime_frames =
        std::max(0, cfg.vehicle_recover_hold_max_lifetime_frames);
    cfg.obstacle_seen_hold_frames = std::max(0, cfg.obstacle_seen_hold_frames);
    cfg.obstacle_prepare_distance_m =
        std::max(0.05f, cfg.obstacle_prepare_distance_m);
    cfg.obstacle_activate_distance_m =
        std::max(0.05f, cfg.obstacle_activate_distance_m);
    cfg.vehicle_front_m = std::max(0.0f, cfg.vehicle_front_m);
    cfg.vehicle_rear_m = std::max(0.0f, cfg.vehicle_rear_m);
    cfg.vehicle_length_m =
        std::max(0.05f, std::max(cfg.vehicle_length_m,
                                 cfg.vehicle_front_m + cfg.vehicle_rear_m));
    cfg.vehicle_width_m = std::max(0.05f, cfg.vehicle_width_m);
    cfg.obstacle_avoid_offset_m =
        std::max(0.0f, std::abs(cfg.obstacle_avoid_offset_m));
    cfg.vehicle_avoidance_edge_clearance_m =
        std::max(0.0f, cfg.vehicle_avoidance_edge_clearance_m);
    cfg.vehicle_avoidance_max_offset_m =
        std::max(cfg.obstacle_avoid_offset_m,
                 std::max(0.0f, cfg.vehicle_avoidance_max_offset_m));
    cfg.vehicle_curve_avoidance_extra_offset_m =
        std::max(0.0f, cfg.vehicle_curve_avoidance_extra_offset_m);
    cfg.vehicle_avoidance_opposite_steer_threshold_deg =
        std::max(0.0f, cfg.vehicle_avoidance_opposite_steer_threshold_deg);
    cfg.vehicle_avoidance_opposite_steer_hold_frames =
        std::max(0, cfg.vehicle_avoidance_opposite_steer_hold_frames);
    cfg.obstacle_offset_rate_mps =
        std::max(0.01f, cfg.obstacle_offset_rate_mps);
    cfg.obstacle_offset_zero_epsilon_m =
        std::max(0.0f, cfg.obstacle_offset_zero_epsilon_m);
    if (!pedestrianDangerCorridorValid(
            cfg.pedestrian_danger_corridor_near_width_m,
            cfg.pedestrian_danger_corridor_far_width_m,
            cfg.pedestrian_danger_corridor_length_m,
            cfg.pedestrian_danger_corridor_sample_step_m)) {
        cfg.pedestrian_danger_corridor_near_width_m =
            kPedestrianDangerDefaultNearWidthM;
        cfg.pedestrian_danger_corridor_far_width_m =
            kPedestrianDangerDefaultFarWidthM;
        cfg.pedestrian_danger_corridor_length_m =
            kPedestrianDangerDefaultLengthM;
        cfg.pedestrian_danger_corridor_sample_step_m =
            kPedestrianDangerDefaultSampleStepM;
    }
    cfg.pedestrian_servo_straight_deadband =
        std::max(0.0f, cfg.pedestrian_servo_straight_deadband);
    cfg.pedestrian_servo_radius_max_abs =
        std::max(cfg.pedestrian_servo_straight_deadband,
                 cfg.pedestrian_servo_radius_max_abs);
    cfg.pedestrian_slow_distance_m =
        std::max(0.0f, cfg.pedestrian_slow_distance_m);
    cfg.pedestrian_slow_speed_mps =
        std::max(0.0f, cfg.pedestrian_slow_speed_mps);
    cfg.pedestrian_normal_strip_half_width_m =
        std::max(0.0f, cfg.pedestrian_normal_strip_half_width_m);
    cfg.pedestrian_stopped_launch_speed_mps =
        std::max(0.0f, cfg.pedestrian_stopped_launch_speed_mps);
    cfg.pedestrian_slow_launch_speed_mps =
        std::max(0.0f, cfg.pedestrian_slow_launch_speed_mps);
    cfg.gold_bias_step_m = std::max(0.001f, cfg.gold_bias_step_m);
    cfg.gold_max_bias_m = std::max(0.0f, cfg.gold_max_bias_m);
    cfg.gold_capture_compensation_m =
        std::max(0.0f, cfg.gold_capture_compensation_m);
    cfg.gold_capture_tolerance_m =
        std::max(0.0f, cfg.gold_capture_tolerance_m);
    cfg.gold_bias_rate_mps = std::max(0.0f, cfg.gold_bias_rate_mps);
    cfg.gold_max_new_delta_m = std::max(0.0f, cfg.gold_max_new_delta_m);
    cfg.gold_recenter_hold_frames =
        std::max(1, cfg.gold_recenter_hold_frames);
    cfg.gold_near_priority_distance_m =
        std::max(0.0f, cfg.gold_near_priority_distance_m);
    cfg.gold_late_start_forward_m =
        std::max(0.0f, cfg.gold_late_start_forward_m);
    cfg.gold_late_start_delta_m =
        std::max(0.0f, cfg.gold_late_start_delta_m);
    cfg.gold_center_release_m =
        std::max(0.0f, cfg.gold_center_release_m);
    cfg.path_smooth_window = std::max(1, cfg.path_smooth_window);
    cfg.avoid_lookahead_distance_m = std::max(0.05f, cfg.avoid_lookahead_distance_m);

    auto updateRuntimeClock = [&]() {
        if (input.timestamp_valid && std::isfinite(input.timestamp_sec)) {
            runtime.last_timestamp_sec = input.timestamp_sec;
            runtime.has_last_timestamp = true;
        }
    };

    auto clearVehicleCurveAvoidanceBoost = [&]() {
        runtime.vehicle_curve_avoid_boost_episode_active = false;
        runtime.vehicle_curve_avoid_boost_active_last_frame = false;
        runtime.vehicle_curve_avoid_boost_track_id = -1;
        runtime.vehicle_curve_avoid_boost_side = 0;
        runtime.vehicle_curve_avoid_hold_remaining = 0;
    };

    auto clearObstacleRuntime = [&]() {
        runtime.obstacle_state = ObstacleAvoidanceState::Cruise;
        runtime.avoid_scenario = AvoidScenario::None;
        runtime.obstacle_offset_target_m = 0.0f;
        runtime.obstacle_offset_current_m = 0.0f;
        runtime.vehicle_episode_max_required_offset_m = 0.0f;
        runtime.obstacle_tracks.clear();
        runtime.obstacle_seen_hold_frames = 0;
        resetPedestrianStopRuntime(runtime, "runtime_reset");
        runtime.obstacle_primary_track_id = -1;
        runtime.obstacle_primary_track_type = ObjectType::UNKNOWN;
        runtime.vehicle_avoid_side_locked = false;
        runtime.vehicle_locked_avoid_side = 0;
        runtime.vehicle_recover_hold_frames_remaining = 0;
        runtime.vehicle_recover_hold_started = false;
        runtime.vehicle_recover_hold_age_frames = 0;
        clearVehicleCurveAvoidanceBoost();
    };

    if (!input.obstacle_runtime_enabled) {
        clearObstacleRuntime();
        result.debug.obstacle_state = runtime.obstacle_state;
        result.debug.obstacle_offset_current_m = runtime.obstacle_offset_current_m;
        result.debug.obstacle_offset_target_m = runtime.obstacle_offset_target_m;
    }

    result.debug.ped_danger_corridor_near_width_m =
        cfg.pedestrian_danger_corridor_near_width_m;
    result.debug.vehicle_recover_hold_age_frames =
        runtime.vehicle_recover_hold_age_frames;
    result.debug.vehicle_recover_hold_max_lifetime_frames =
        cfg.vehicle_recover_hold_max_lifetime_frames;
    result.debug.ped_danger_corridor_far_width_m =
        cfg.pedestrian_danger_corridor_far_width_m;
    result.debug.ped_danger_corridor_length_m =
        cfg.pedestrian_danger_corridor_length_m;
    result.debug.ped_danger_corridor_sample_step_m =
        cfg.pedestrian_danger_corridor_sample_step_m;
    result.debug.raw_person_count = static_cast<int>(input.raw_person_objects.size());
    result.debug.slow_candidate = false;
    result.debug.emergency_candidate = false;
    result.debug.active_block_reasons = "none";
    syncPedestrianStopDebug(runtime, result);

    const std::vector<PlanningRoutePoint> route = sanitizedRoute(input.route);
    result.debug.near_route_lateral_valid = nearRouteLateralFromRoute(
        route,
        result.debug.near_route_lateral_m,
        result.debug.near_route_lateral_sample_count);
    PlanningRoutePoint ego_route_point;
    cv::Point2f route_center_world;
    cv::Point2f route_normal_world;
    const cv::Point2f ego_world(0.0f, 0.0f);
    const bool ego_center_lateral_valid = routeFrameNearestWorld(
        route,
        ego_world,
        ego_route_point,
        route_center_world,
        route_normal_world);
    const float ego_center_lateral_m = ego_center_lateral_valid
        ? dotPoint(ego_world - route_center_world, route_normal_world)
        : runtime.ego_center_lateral_m;
    if (ego_center_lateral_valid && isFiniteFloat(ego_center_lateral_m)) {
        runtime.ego_center_lateral_m = ego_center_lateral_m;
        runtime.ego_center_lateral_valid = true;
    } else {
        runtime.ego_center_lateral_valid = false;
    }
    result.debug.ego_center_lateral_m = runtime.ego_center_lateral_m;
    auto refreshPathUnavailableObstacleTracks = [&]() {
        if (cfg.obstacle_avoid_enable && cfg.obstacle_runtime_enabled) {
            updateObstacleTracks(cfg, route, runtime);
        } else {
            clearObstacleRuntime();
        }
    };

    const bool route_selected = input.visible_branch_count == 1 || input.external_route_decision_valid;
    if (!route_selected) {
        refreshPathUnavailableObstacleTracks();
        clearGoldPending(runtime.gold);
        syncGoldDebug(runtime.gold, runtime.gold.applied_m, result.debug);
        result.debug.gold_bias_reason = "path_unavailable";
        result.debug.startup_launch_release_event = false;
        result.debug.ped_emergency_suppressed_by_launch = false;
        result.debug.final_applied_path_offset_m = 0.0f;
        result.debug.reason = "multi_branch_wait_route_decision";
        result.debug.path_usable = false;
        result.debug.blocked_by_path =
            runtime.pedestrian_behavior.state == PedestrianStopState::STOPPED_WAIT;
        result.debug.launch_paused_by_path =
            runtime.pedestrian_behavior.state == PedestrianStopState::LAUNCH_COMMITTED;
        // [ALIGN-DEMO] route_hold removed — do not reuse prev_path as waypoints.
        // Demo does not have route_hold; return empty path so the caller
        // falls back to centerline candidate or stops.
        result.debug.mode = LocalPlannerMode::Disabled;
        result.debug.route_hold_active = false;
        result.waypoints.clear();
        result.planning_enabled = false;
        result.target_speed_mps = 0.0f;
        result.debug.target_speed_mps = result.target_speed_mps;
        result.debug.speed_risk_level =
            runtime.pedestrian_behavior.state == PedestrianStopState::STOPPED_WAIT ? "stop" : "normal";
        result.debug.speed_limit_reason = "path_unavailable";
        result.debug.speed_reason = "path_unavailable";
        result.debug.planner_speed_source = "path_unavailable";
        result.debug.path_points = 0;
        syncPedestrianStopDebug(runtime, result);
        updateRuntimeClock();
        return result;
    }

    if (route.size() < 2 || input.frame_size.width <= 0 || input.frame_size.height <= 0) {
        refreshPathUnavailableObstacleTracks();
        clearGoldPending(runtime.gold);
        syncGoldDebug(runtime.gold, runtime.gold.applied_m, result.debug);
        result.debug.gold_bias_reason = "path_unavailable";
        result.debug.startup_launch_release_event = false;
        result.debug.ped_emergency_suppressed_by_launch = false;
        result.debug.final_applied_path_offset_m = 0.0f;
        result.debug.reason = "route_unavailable";
        result.debug.path_usable = false;
        result.debug.blocked_by_path =
            runtime.pedestrian_behavior.state == PedestrianStopState::STOPPED_WAIT;
        result.debug.launch_paused_by_path =
            runtime.pedestrian_behavior.state == PedestrianStopState::LAUNCH_COMMITTED;
        // [ALIGN-DEMO] route_hold removed — do not reuse prev_path as waypoints.
        result.debug.mode = LocalPlannerMode::Disabled;
        result.debug.route_hold_active = false;
        result.waypoints.clear();
        result.planning_enabled = false;
        result.target_speed_mps = 0.0f;
        result.debug.target_speed_mps = result.target_speed_mps;
        result.debug.speed_risk_level =
            runtime.pedestrian_behavior.state == PedestrianStopState::STOPPED_WAIT ? "stop" : "normal";
        result.debug.speed_limit_reason = "path_unavailable";
        result.debug.speed_reason = "path_unavailable";
        result.debug.planner_speed_source = "path_unavailable";
        result.debug.path_points = 0;
        syncPedestrianStopDebug(runtime, result);
        updateRuntimeClock();
        return result;
    }

    const float cruise_speed =
        std::max(0.0f, cfg.pedestrian_cruise_speed_mps);

    result.planning_enabled = true;
    result.target_speed_mps = cruise_speed;
    result.debug.target_speed_mps = cruise_speed;
    result.debug.speed_reason = "cruise";
    result.debug.planner_speed_source = "cruise";
    result.debug.speed_risk_level = "normal";
    result.debug.speed_limit_reason.clear();

    const float dt = localPlannerDt(cfg, runtime);
    if (cfg.obstacle_avoid_enable && cfg.obstacle_runtime_enabled) {
        updateObstacleTracks(cfg, route, runtime);
    } else {
        clearObstacleRuntime();
    }

    auto cars = visibleTracksOfType(runtime.obstacle_tracks, ObjectType::CAR);
    const int visible_obstacle_count = static_cast<int>(cars.size());
    if (!cars.empty()) {
        runtime.obstacle_seen_hold_frames = cfg.obstacle_seen_hold_frames;
    } else if (runtime.obstacle_seen_hold_frames > 0) {
        --runtime.obstacle_seen_hold_frames;
    }
    const bool obstacle_seen_recent = runtime.obstacle_seen_hold_frames > 0;

    auto nearestTrack = [](const std::vector<ObstacleAvoidanceTrack*>& tracks) {
        ObstacleAvoidanceTrack* best = nullptr;
        float best_distance_m = std::numeric_limits<float>::infinity();
        for (auto* track : tracks) {
            if (track != nullptr &&
                track->distance_to_vehicle_m < best_distance_m) {
                best = track;
                best_distance_m = track->distance_to_vehicle_m;
            }
        }
        return best;
    };

    ObstacleAvoidanceTrack* nearest_car = nearestTrack(cars);
    ObstacleAvoidanceTrack* primary_track = nullptr;
    bool shift_state = false;
    int selected_avoid_side =
        runtime.obstacle_avoid_side == 0 ? 1 : runtime.obstacle_avoid_side;
    const int forced_avoid_side =
        cfg.force_left_obstacle_avoidance ? -1 : 0;
    const bool vehicle_episode_was_active =
        runtime.avoid_scenario == AvoidScenario::VehicleOnly &&
        runtime.obstacle_state != ObstacleAvoidanceState::Cruise;
    const float previous_obstacle_offset_target_m =
        runtime.obstacle_offset_target_m;
    AvoidScenario scenario = AvoidScenario::None;
    bool vehicle_recover_hold_active = false;

    if (nearest_car != nullptr &&
        nearest_car->distance_to_vehicle_m <=
            cfg.obstacle_prepare_distance_m) {
        scenario = AvoidScenario::VehicleOnly;
        primary_track = nearest_car;
        shift_state = true;
        runtime.vehicle_recover_hold_frames_remaining = 0;
        runtime.vehicle_recover_hold_started = false;
        runtime.vehicle_recover_hold_age_frames = 0;
        if (!vehicle_episode_was_active) {
            runtime.vehicle_episode_max_required_offset_m = 0.0f;
        }

        const bool vehicle_locked_side_toward_obstacle =
            runtime.vehicle_avoid_side_locked &&
            runtime.vehicle_locked_avoid_side != 0 &&
            nearest_car->lateral_from_route_m *
                    static_cast<float>(runtime.vehicle_locked_avoid_side) >
                0.0f;
        const bool entering_vehicle_context =
            runtime.avoid_scenario != AvoidScenario::VehicleOnly ||
            runtime.obstacle_state == ObstacleAvoidanceState::Cruise ||
            runtime.obstacle_state == ObstacleAvoidanceState::Recover ||
            !runtime.vehicle_avoid_side_locked ||
            runtime.vehicle_locked_avoid_side == 0 ||
            vehicle_locked_side_toward_obstacle;
        if (forced_avoid_side != 0) {
            runtime.vehicle_locked_avoid_side = forced_avoid_side;
            runtime.vehicle_avoid_side_locked = true;
        } else if (entering_vehicle_context &&
                   nearest_car->observed_this_frame) {
            runtime.vehicle_locked_avoid_side =
                chooseVehicleAvoidSide(nearest_car->lateral_from_route_m,
                                       runtime.obstacle_avoid_side);
            runtime.vehicle_avoid_side_locked = true;
        }
        selected_avoid_side = runtime.vehicle_locked_avoid_side;
    }

    if (!cfg.obstacle_avoid_enable || !cfg.obstacle_runtime_enabled) {
        scenario = AvoidScenario::None;
        shift_state = false;
        runtime.vehicle_avoid_side_locked = false;
        runtime.vehicle_locked_avoid_side = 0;
        runtime.vehicle_recover_hold_frames_remaining = 0;
        runtime.vehicle_recover_hold_started = false;
        runtime.vehicle_recover_hold_age_frames = 0;
    }

    float desired_obstacle_offset_m = 0.0f;
    if (shift_state) {
        if (forced_avoid_side != 0) {
            runtime.vehicle_locked_avoid_side = forced_avoid_side;
            runtime.vehicle_avoid_side_locked = true;
        } else if (!runtime.vehicle_avoid_side_locked ||
                   runtime.vehicle_locked_avoid_side == 0) {
            runtime.vehicle_locked_avoid_side =
                normalizeAvoidSide(selected_avoid_side);
            if (runtime.vehicle_locked_avoid_side == 0) {
                runtime.vehicle_locked_avoid_side = 1;
            }
            runtime.vehicle_avoid_side_locked = true;
        }
        runtime.obstacle_avoid_side = runtime.vehicle_locked_avoid_side;

        const PlanningObject* primary_vehicle =
            findPlanningObject(cfg.objects, ObjectType::CAR,
                               primary_track->id);
        float required_offset_m = 0.0f;
        if (primary_vehicle != nullptr &&
            vehicleRequiredOffsetFromBboxEdge(
                *primary_vehicle,
                route,
                runtime.obstacle_avoid_side,
                cfg.vehicle_avoidance_edge_clearance_m,
                required_offset_m)) {
            runtime.vehicle_episode_max_required_offset_m =
                std::max(runtime.vehicle_episode_max_required_offset_m,
                         required_offset_m);
        }
        const float vehicle_target_offset_m = std::min(
            std::max(cfg.obstacle_avoid_offset_m,
                     runtime.vehicle_episode_max_required_offset_m),
            cfg.vehicle_avoidance_max_offset_m);
        desired_obstacle_offset_m =
            static_cast<float>(runtime.obstacle_avoid_side) *
            vehicle_target_offset_m;

        runtime.obstacle_state =
            primary_track->distance_to_vehicle_m <=
                    cfg.obstacle_activate_distance_m
                ? ObstacleAvoidanceState::Active
                : ObstacleAvoidanceState::Prepare;
        runtime.avoid_scenario = AvoidScenario::VehicleOnly;
        runtime.obstacle_primary_track_id = primary_track->id;
        runtime.obstacle_primary_track_type = ObjectType::CAR;
    } else {
        const bool recovering_vehicle_context =
            runtime.avoid_scenario == AvoidScenario::VehicleOnly &&
            runtime.obstacle_state != ObstacleAvoidanceState::Cruise &&
            runtime.vehicle_avoid_side_locked &&
            runtime.vehicle_locked_avoid_side != 0 &&
            std::abs(runtime.obstacle_offset_current_m) >
                cfg.obstacle_offset_zero_epsilon_m;
        runtime.avoid_scenario =
            recovering_vehicle_context
                ? AvoidScenario::VehicleOnly
                : AvoidScenario::None;
        runtime.obstacle_primary_track_id = -1;
        runtime.obstacle_primary_track_type = ObjectType::UNKNOWN;
        runtime.obstacle_avoid_side =
            recovering_vehicle_context
                ? runtime.vehicle_locked_avoid_side
                : 0;

        if (recovering_vehicle_context &&
            nearest_car == nullptr &&
            runtime.obstacle_state != ObstacleAvoidanceState::Recover &&
            !runtime.vehicle_recover_hold_started &&
            cfg.vehicle_recover_hold_frames > 0) {
            runtime.vehicle_recover_hold_frames_remaining =
                cfg.vehicle_recover_hold_frames;
            runtime.vehicle_recover_hold_started = true;
            runtime.vehicle_recover_hold_age_frames = 0;
        }
        const bool recover_hold_refresh_within_lifetime =
            runtime.vehicle_recover_hold_age_frames <
            cfg.vehicle_recover_hold_max_lifetime_frames;
        const float steering_along_locked_side =
            static_cast<float>(runtime.vehicle_locked_avoid_side) *
            input.steering_angle_deg;
        if (recovering_vehicle_context &&
            runtime.vehicle_recover_hold_started &&
            recover_hold_refresh_within_lifetime &&
            input.steering_angle_valid &&
            isFiniteFloat(input.steering_angle_deg) &&
            steering_along_locked_side <
                -cfg.vehicle_avoidance_opposite_steer_threshold_deg) {
            runtime.vehicle_recover_hold_frames_remaining =
                std::max(
                    runtime.vehicle_recover_hold_frames_remaining,
                    cfg.vehicle_avoidance_opposite_steer_hold_frames);
        }
        if (recovering_vehicle_context &&
            runtime.vehicle_recover_hold_frames_remaining > 0) {
            vehicle_recover_hold_active = true;
            runtime.obstacle_offset_target_m =
                runtime.obstacle_offset_current_m;
            --runtime.vehicle_recover_hold_frames_remaining;
            runtime.vehicle_recover_hold_age_frames++;
        } else if (!recovering_vehicle_context) {
            runtime.vehicle_avoid_side_locked = false;
            runtime.vehicle_locked_avoid_side = 0;
            runtime.vehicle_recover_hold_frames_remaining = 0;
            runtime.vehicle_recover_hold_started = false;
            runtime.vehicle_recover_hold_age_frames = 0;
        }
    }

    const int vehicle_avoid_side =
        normalizeAvoidSide(runtime.vehicle_locked_avoid_side);
    const int current_vehicle_track_id = runtime.obstacle_primary_track_id;
    const bool current_vehicle_episode_valid =
        shift_state &&
        scenario == AvoidScenario::VehicleOnly &&
        runtime.avoid_scenario == AvoidScenario::VehicleOnly &&
        runtime.vehicle_avoid_side_locked &&
        vehicle_avoid_side != 0 &&
        current_vehicle_track_id >= 0;
    const bool vehicle_curve_recover_continuation =
        vehicle_recover_hold_active &&
        runtime.vehicle_curve_avoid_boost_episode_active &&
        runtime.vehicle_curve_avoid_boost_active_last_frame &&
        runtime.vehicle_curve_avoid_boost_side == vehicle_avoid_side;
    if (runtime.vehicle_curve_avoid_boost_episode_active &&
        !vehicle_curve_recover_continuation &&
        (!current_vehicle_episode_valid ||
         runtime.vehicle_curve_avoid_boost_track_id != current_vehicle_track_id ||
         runtime.vehicle_curve_avoid_boost_side != vehicle_avoid_side)) {
        clearVehicleCurveAvoidanceBoost();
    }
    const bool vehicle_curve_context_active =
        current_vehicle_episode_valid &&
        runtime.obstacle_state == ObstacleAvoidanceState::Active &&
        runtime.vehicle_locked_avoid_side != 0;
    bool vehicle_curve_metric_valid = false;
    float vehicle_curve_metric = 0.0f;
    int vehicle_curve_direction = 0;
    if (vehicle_curve_context_active) {
        vehicle_curve_metric =
            vehicleCurveMetricFromRoute(route, vehicle_curve_metric_valid);
        if (vehicle_curve_metric_valid) {
            if (vehicle_curve_metric >= kVehicleCurveMetricThresholdM) {
                vehicle_curve_direction = 1;
            } else if (vehicle_curve_metric <= -kVehicleCurveMetricThresholdM) {
                vehicle_curve_direction = -1;
            }
        }
    }
    const float ego_lateral_along_avoid_side =
        static_cast<float>(vehicle_avoid_side) * runtime.ego_center_lateral_m;
    const bool vehicle_ego_clearly_opposite =
        runtime.ego_center_lateral_valid &&
        ego_lateral_along_avoid_side <=
            -kVehicleCurveEgoOppositeThresholdM;
    const bool vehicle_curve_initial_trigger =
        vehicle_curve_context_active &&
        vehicle_curve_metric_valid &&
        vehicle_curve_direction == -vehicle_avoid_side;
    if (vehicle_curve_initial_trigger) {
        runtime.vehicle_curve_avoid_boost_episode_active = true;
        runtime.vehicle_curve_avoid_boost_track_id = current_vehicle_track_id;
        runtime.vehicle_curve_avoid_boost_side = vehicle_avoid_side;
    }
    const bool vehicle_curve_boost_owner_matches =
        runtime.vehicle_curve_avoid_boost_episode_active &&
        current_vehicle_episode_valid &&
        runtime.vehicle_curve_avoid_boost_track_id == current_vehicle_track_id &&
        runtime.vehicle_curve_avoid_boost_side == vehicle_avoid_side;
    const float steering_along_avoid_side_deg =
        static_cast<float>(vehicle_avoid_side) * input.steering_angle_deg;
    const bool vehicle_opposite_steer_active =
        !vehicle_curve_initial_trigger &&
        vehicle_curve_context_active &&
        vehicle_curve_boost_owner_matches &&
        input.steering_angle_valid &&
        isFiniteFloat(input.steering_angle_deg) &&
        steering_along_avoid_side_deg <
            -cfg.vehicle_avoidance_opposite_steer_threshold_deg;
    if (vehicle_opposite_steer_active) {
        runtime.vehicle_curve_avoid_hold_remaining =
            cfg.vehicle_avoidance_opposite_steer_hold_frames;
    }
    const bool vehicle_curve_avoid_boost_active =
        vehicle_curve_recover_continuation ||
        (vehicle_curve_context_active &&
         vehicle_curve_boost_owner_matches &&
         (vehicle_curve_initial_trigger ||
          runtime.vehicle_curve_avoid_hold_remaining > 0));
    if (!vehicle_curve_initial_trigger &&
        !vehicle_opposite_steer_active &&
        vehicle_curve_boost_owner_matches &&
        runtime.vehicle_curve_avoid_hold_remaining > 0) {
        runtime.vehicle_curve_avoid_hold_remaining--;
    }
    if (vehicle_curve_avoid_boost_active &&
        !vehicle_curve_recover_continuation) {
        const float boosted_offset_m = std::min(
            std::abs(desired_obstacle_offset_m) +
                cfg.vehicle_curve_avoidance_extra_offset_m,
            cfg.vehicle_avoidance_max_offset_m);
        desired_obstacle_offset_m =
            static_cast<float>(vehicle_avoid_side) * boosted_offset_m;
    }
    result.debug.vehicle_curve_metric = vehicle_curve_metric;
    result.debug.vehicle_curve_direction = vehicle_curve_direction;
    result.debug.vehicle_curve_avoid_boost_active =
        vehicle_curve_avoid_boost_active;
    runtime.vehicle_curve_avoid_boost_active_last_frame =
        vehicle_curve_avoid_boost_active;
    result.debug.vehicle_opposite_steer_active =
        vehicle_opposite_steer_active;
    result.debug.vehicle_curve_avoid_hold_remaining =
        runtime.vehicle_curve_avoid_hold_remaining;

    if (vehicle_recover_hold_active) {
        runtime.obstacle_offset_target_m =
            vehicle_curve_recover_continuation
                ? previous_obstacle_offset_target_m
                : runtime.obstacle_offset_current_m;
    } else {
        runtime.obstacle_offset_target_m = desired_obstacle_offset_m;
    }
    const float max_offset_step = cfg.obstacle_offset_rate_mps * dt;
    const float offset_delta =
        runtime.obstacle_offset_target_m - runtime.obstacle_offset_current_m;
    if (std::abs(offset_delta) <= max_offset_step) {
        runtime.obstacle_offset_current_m = runtime.obstacle_offset_target_m;
    } else {
        runtime.obstacle_offset_current_m +=
            offset_delta > 0.0f ? max_offset_step : -max_offset_step;
    }
    if (std::abs(runtime.obstacle_offset_current_m) <= cfg.obstacle_offset_zero_epsilon_m &&
        std::abs(runtime.obstacle_offset_target_m) <= cfg.obstacle_offset_zero_epsilon_m) {
        runtime.obstacle_offset_current_m = 0.0f;
    }
    if (!shift_state && !vehicle_recover_hold_active) {
        if (std::abs(runtime.obstacle_offset_current_m) >
            cfg.obstacle_offset_zero_epsilon_m) {
            runtime.obstacle_state = ObstacleAvoidanceState::Recover;
            if (runtime.avoid_scenario == AvoidScenario::VehicleOnly &&
                runtime.vehicle_avoid_side_locked &&
                runtime.vehicle_locked_avoid_side != 0) {
                runtime.obstacle_avoid_side = runtime.vehicle_locked_avoid_side;
            }
        } else {
            runtime.obstacle_state = ObstacleAvoidanceState::Cruise;
            runtime.avoid_scenario = AvoidScenario::None;
            runtime.obstacle_avoid_side = 0;
            runtime.vehicle_avoid_side_locked = false;
            runtime.vehicle_locked_avoid_side = 0;
            runtime.vehicle_recover_hold_frames_remaining = 0;
            runtime.vehicle_recover_hold_started = false;
            runtime.vehicle_recover_hold_age_frames = 0;
        }
    }
    const bool vehicle_episode_is_active =
        runtime.avoid_scenario == AvoidScenario::VehicleOnly &&
        runtime.obstacle_state != ObstacleAvoidanceState::Cruise;
    if (!vehicle_episode_is_active) {
        runtime.vehicle_episode_max_required_offset_m = 0.0f;
    }
    result.obstacle_avoidance_engaged =
        obstacle_seen_recent ||
        runtime.avoid_scenario != AvoidScenario::None ||
        std::abs(runtime.obstacle_offset_target_m) > cfg.obstacle_offset_zero_epsilon_m ||
        std::abs(runtime.obstacle_offset_current_m) > cfg.obstacle_offset_zero_epsilon_m;

    result.debug.obstacle_state = runtime.obstacle_state;
    result.debug.avoid_scenario = runtime.avoid_scenario;
    result.debug.obstacle_count = visible_obstacle_count;
    result.debug.obstacle_active_count = shift_state ? 1 : 0;
    result.debug.obstacle_avoid_side = runtime.obstacle_avoid_side;
    result.debug.vehicle_avoid_side_locked = runtime.vehicle_avoid_side_locked;
    result.debug.vehicle_locked_avoid_side = runtime.vehicle_locked_avoid_side;
    result.debug.vehicle_recover_hold_active = vehicle_recover_hold_active;
    result.debug.vehicle_recover_hold_remaining =
        runtime.vehicle_recover_hold_frames_remaining;
    result.debug.vehicle_recover_hold_age_frames =
        runtime.vehicle_recover_hold_age_frames;
    result.debug.vehicle_recover_hold_max_lifetime_frames =
        cfg.vehicle_recover_hold_max_lifetime_frames;
    result.debug.obstacle_offset_current_m = runtime.obstacle_offset_current_m;
    result.debug.obstacle_offset_target_m = runtime.obstacle_offset_target_m;
    if (primary_track != nullptr) {
        result.debug.obstacle_primary_id = primary_track->id;
        result.debug.obstacle_primary_type = objectTypeName(primary_track->type);
        result.debug.obstacle_target_pixel_x = primary_track->pixel_foot.x;
        result.debug.obstacle_target_pixel_y = primary_track->pixel_foot.y;
        result.debug.obstacle_distance_m = primary_track->distance_to_vehicle_m;
        result.debug.obstacle_point_distance_m =
            primary_track->point_distance_to_vehicle_m;
        result.debug.obstacle_lateral_m = primary_track->lateral_from_route_m;
    }

    const bool obstacle_prepare =
        runtime.obstacle_state == ObstacleAvoidanceState::Prepare;
    const bool obstacle_active =
        runtime.obstacle_state == ObstacleAvoidanceState::Active;
    const bool obstacle_recover =
        runtime.obstacle_state == ObstacleAvoidanceState::Recover;

    LocalPlannerMode mode = LocalPlannerMode::Cruise;
    if (obstacle_active) {
        mode = LocalPlannerMode::AvoidActive;
    } else if (obstacle_prepare) {
        mode = LocalPlannerMode::AvoidPrepare;
    } else if (obstacle_recover) {
        mode = LocalPlannerMode::AvoidRecover;
    }
    result.debug.mode = mode;
    result.debug.reason = modeName(mode);
    result.speed_factor = 1.0f;
    if (obstacle_prepare || obstacle_active) {
        result.target_speed_mps = cruise_speed;
        if (runtime.avoid_scenario == AvoidScenario::VehicleOnly) {
            result.debug.speed_risk_level = "normal";
            result.debug.speed_limit_reason.clear();
            result.debug.speed_reason = "cruise";
        } else {
            result.debug.speed_risk_level = "caution";
            result.debug.speed_limit_reason = "obstacle_offset";
            result.debug.speed_reason = result.debug.speed_limit_reason;
        }
    }

    const bool vehicle_corridor_extension_active =
        runtime.avoid_scenario == AvoidScenario::VehicleOnly &&
        runtime.obstacle_state != ObstacleAvoidanceState::Cruise &&
        normalizeAvoidSide(runtime.obstacle_avoid_side) != 0;
    const float safety_offset_m = runtime.obstacle_offset_current_m;
    const bool vehicle_gold_freeze_active =
        runtime.avoid_scenario == AvoidScenario::VehicleOnly &&
        (runtime.obstacle_state == ObstacleAvoidanceState::Active ||
         runtime.obstacle_state == ObstacleAvoidanceState::Recover);
    if (vehicle_gold_freeze_active) {
        // Pending intent is not part of a confirmed transition and must not
        // survive a Vehicle Safety takeover, even if this local path is later
        // rejected and the staged Gold state is not committed.
        clearGoldPending(runtime.gold);
    }
    GoldBiasRuntime staged_gold = runtime.gold;
    const float effective_gold_bias_m = updateGoldBiasSelector(
        route,
        cfg,
        safety_offset_m,
        dt,
        runtime,
        staged_gold,
        result);
    const float final_offset_m = safety_offset_m + effective_gold_bias_m;

    int path_extension_side = 0;
    float path_extension_limit_m = 0.0f;
    if (vehicle_corridor_extension_active) {
        path_extension_side = runtime.obstacle_avoid_side;
        path_extension_limit_m = cfg.vehicle_avoidance_max_offset_m;
    }
    if (std::abs(effective_gold_bias_m) > kGoldBiasEpsilonM &&
        std::abs(final_offset_m) > cfg.obstacle_offset_zero_epsilon_m) {
        path_extension_side = sideFromObstacleOffset(final_offset_m);
        path_extension_limit_m =
            std::max(path_extension_limit_m, std::abs(final_offset_m));
    }
    std::vector<cv::Point2f> path = buildOffsetPath(
        route,
        final_offset_m,
        cfg.obstacle_offset_zero_epsilon_m,
        path_extension_side,
        path_extension_limit_m);

    if (cfg.path_smooth_window > 1) {
        path = smoothPath(path, cfg.path_smooth_window);
    }

    const bool final_path_usable = isPathUsable(
        path, input.frame_size.width, input.frame_size.height);
    result.debug.path_usable = final_path_usable;
    if (final_path_usable) {
        runtime.gold = staged_gold;
    } else {
        float committed_effective_gold_m = runtime.gold.applied_m;
        if (std::abs(safety_offset_m) > cfg.obstacle_offset_zero_epsilon_m &&
            safety_offset_m * committed_effective_gold_m < 0.0f) {
            committed_effective_gold_m = 0.0f;
        }
        syncGoldDebug(runtime.gold, committed_effective_gold_m, result.debug);
        result.debug.gold_bias_reason = "path_unusable_gold_not_committed";
    }

    result.debug.realized_offset_m =
        vehicle_corridor_extension_active
            ? computeRealizedVehicleOffset(path,
                                           route,
                                           runtime.obstacle_avoid_side)
            : 0.0f;

    result.debug.final_applied_path_offset_m = final_offset_m;

    if (runtime.pedestrian_behavior.state == PedestrianStopState::STOPPED_WAIT) {
        setPlannerSpeed(result, 0.0f, "pedestrian_stopped_wait", "stop");
        result.obstacle_avoidance_engaged = true;
    } else if (runtime.pedestrian_behavior.state == PedestrianStopState::LAUNCH_COMMITTED) {
        setPlannerSpeed(
            result,
            pedestrianLaunchSpeedMps(cfg, runtime.pedestrian_launch_source),
            pedestrianLaunchSourceName(runtime.pedestrian_launch_source),
            "launch");
        result.debug.launch_protect_speed_applied = true;
    } else if (result.debug.planner_speed_source.empty()) {
        result.debug.planner_speed_source = "cruise";
    }

    syncPedestrianStopDebug(runtime, result);
    if (!path.empty()) {
        result.debug.debug_base_path_x = route.front().pos.x;
        result.debug.debug_final_path_x = path.front().x;
        result.debug.debug_path_y = path.front().y;
    }
    result.debug.cluster_active = false;
    result.debug.avoid_lookahead_active = shift_state;
    result.debug.recommended_lookahead_m =
        result.debug.avoid_lookahead_active ? cfg.avoid_lookahead_distance_m : 0.0f;

    result.waypoints = std::move(path);
    result.debug.path_points = static_cast<int>(result.waypoints.size());
    result.debug.target_speed_mps = result.target_speed_mps;
    updateRuntimeClock();
    return result;
}

void finalizePedestrianBehavior(
    const PathPlanningInput& input,
    float current_servo_output,
    bool current_servo_valid,
    const std::string& current_servo_source,
    float last_applied_servo_output,
    bool last_applied_servo_valid,
    PathPlanningRuntime& runtime,
    PathPlanningResult& result)
{
    //含多次fallback的初始化舵机转角
    const bool use_current_servo =
        current_servo_valid && std::isfinite(current_servo_output);
    const bool use_last_servo =
        !use_current_servo && last_applied_servo_valid &&
        std::isfinite(last_applied_servo_output);
    const float raw_servo_for_corridor =
        use_current_servo
            ? current_servo_output
            : (use_last_servo ? last_applied_servo_output : 0.0f);

    if (runtime.pedestrian_behavior.state == PedestrianStopState::DRIVING) {
        runtime.pedestrian_danger_corridor_servo_locked = false;
    }
    if (!runtime.pedestrian_danger_corridor_servo_initialized) {
        runtime.pedestrian_danger_corridor_servo_filtered =
            raw_servo_for_corridor;
        runtime.pedestrian_danger_corridor_servo_initialized = true;
    } else if (!runtime.pedestrian_danger_corridor_servo_locked) {
        const float alpha = std::clamp(
            input.pedestrian_danger_corridor_servo_smooth_alpha, 0.0f, 1.0f);
        float delta =
            alpha * (raw_servo_for_corridor -
                     runtime.pedestrian_danger_corridor_servo_filtered);
        const float max_step = std::max(
            0.0f, input.pedestrian_danger_corridor_servo_max_step);
        if (max_step > 0.0f) {
            delta = std::clamp(delta, -max_step, max_step);
        }
        runtime.pedestrian_danger_corridor_servo_filtered += delta;
    }
    //低通加限幅滤波后传入构造危险区
    result.debug.pedestrian_danger_corridor =
        buildPedestrianDangerCorridor(
            runtime.pedestrian_danger_corridor_servo_filtered,
            input.pedestrian_danger_corridor_near_width_m,
            input.pedestrian_danger_corridor_far_width_m,
            input.pedestrian_danger_corridor_length_m,
            input.pedestrian_danger_corridor_sample_step_m,
            input.pedestrian_servo_straight_deadband,
            input.pedestrian_servo_radius_max_abs);
    const auto& corridor = result.debug.pedestrian_danger_corridor;
    result.debug.ped_danger_servo_raw = raw_servo_for_corridor;
    result.debug.ped_danger_servo_output = corridor.servo_output;
    result.debug.ped_danger_servo_abs_used = corridor.servo_abs_used;
    result.debug.ped_danger_servo_valid = use_current_servo || use_last_servo;
    result.debug.ped_danger_servo_locked =
        runtime.pedestrian_danger_corridor_servo_locked;
    result.debug.ped_danger_servo_source =
        use_current_servo
            ? current_servo_source
            : (use_last_servo ? "last_applied" : "straight_fallback");
    result.debug.ped_danger_corridor_near_width_m = corridor.near_width_m;
    result.debug.ped_danger_corridor_far_width_m = corridor.far_width_m;
    result.debug.ped_danger_corridor_length_m = corridor.length_m;
    result.debug.ped_danger_corridor_sample_step_m = corridor.sample_step_m;
    result.debug.ped_danger_turn_radius_m = corridor.turn_radius_m;
    result.debug.ped_danger_turn_sign = corridor.turn_sign;
    result.debug.ped_danger_straight = corridor.straight;
    result.debug.pedestrian_normal_strips.clear();

    const std::vector<PlanningRoutePoint> route = sanitizedRoute(input.route);
    PedestrianFrameEvidence frame;
    frame.near_route_lateral_m = result.debug.near_route_lateral_m;
    frame.near_route_lateral_valid = result.debug.near_route_lateral_valid;
    const PlanningObject* nearest_person = nullptr;
    PedestrianDangerHitResult nearest_hit;
    float nearest_distance_m = std::numeric_limits<float>::infinity();
    bool any_hard_hit = false;
    bool any_conflict = false;

    for (const auto& person : input.raw_person_objects) {
        if (person.type != ObjectType::PERSON ||
            !person.from_raw_detection ||
            !person.observed_this_frame ||
            !person.has_world_pos_m ||
            !isFinitePoint(person.world_pos_m)) {
            continue;
        }

        const PedestrianDangerHitResult hit =
            evaluatePedestrianDangerCorridor(corridor, person.world_pos_m);
        const float distance_m =
            std::hypot(person.world_pos_m.x, person.world_pos_m.y);

        PedestrianNormalStripGeometry strip;
        const bool strip_valid =
            buildPedestrianNormalStrip(
                route,
                person.id,
                person.world_pos_m,
                input.pedestrian_normal_strip_half_width_m,
                strip);
        const bool potential_conflict =
            strip_valid &&
            pedestrianStripForwardEdgeIntersectsCorridor(strip, corridor);
        if (strip_valid) {
            strip.pedestrian_inside_corridor = hit.hit;
            strip.forward_edge_intersects_corridor = potential_conflict;
            result.debug.pedestrian_normal_strips.push_back(strip);
        }

        PedestrianObservationEvidence evidence;
        evidence.x_m = person.world_pos_m.x;
        evidence.y_m = person.world_pos_m.y;
        evidence.distance_to_vehicle_m = distance_m;
        evidence.corridor_boundary_clearance_m =
            hit.boundary_clearance_m;
        evidence.within_slow_distance =
            input.pedestrian_slow_enable &&
            distance_m <= std::max(0.0f, input.pedestrian_slow_distance_m);
        evidence.potential_conflict = potential_conflict;
        const float half_allowed_width_m = 0.5f * hit.allowed_width_m;
        evidence.left_outside =
            std::isfinite(hit.signed_lateral_m) &&
            hit.allowed_width_m > 0.0f &&
            hit.signed_lateral_m < -half_allowed_width_m;
        evidence.right_outside =
            std::isfinite(hit.signed_lateral_m) &&
            hit.allowed_width_m > 0.0f &&
            hit.signed_lateral_m > half_allowed_width_m;
        frame.people.push_back(evidence);

        any_hard_hit = any_hard_hit || hit.hit;
        any_conflict = any_conflict || potential_conflict;
        if (nearest_person == nullptr || distance_m < nearest_distance_m) {
            nearest_person = &person;
            nearest_distance_m = distance_m;
            nearest_hit = hit;
        }
    }

    const PedestrianStopState state_before_update =
        runtime.pedestrian_behavior.state;
    const PedestrianBehaviorDecision decision =
        updatePedestrianBehavior(
            frame, input.pedestrian_behavior_config,
            runtime.pedestrian_behavior);
    if (decision.entered_launch) {
        runtime.pedestrian_launch_source =
            state_before_update == PedestrianStopState::STOPPED_WAIT
                ? PedestrianLaunchSource::STOPPED_WAIT
                : PedestrianLaunchSource::DRIVING_SLOW;
    }
    if (decision.entered_stop) {
        // The release geometry is frozen immediately on STOP entry. It does not
        // depend on encoder standstill or subsequent path/servo changes.
        runtime.pedestrian_danger_corridor_servo_locked = true;
    }


    result.debug.slow_candidate = runtime.pedestrian_behavior.slow_active;
    result.debug.pedestrian_observed_this_frame = !frame.people.empty();
    result.debug.emergency_candidate = any_hard_hit || any_conflict;
    result.debug.ped_emergency_hit = any_hard_hit;
    result.debug.ped_emergency_hit_corridor = any_hard_hit;
    result.debug.pedestrian_normal_strip_intersection = any_conflict;
    result.debug.pedestrian_route_motion_approaching =
        decision.motion == PedestrianRouteMotion::APPROACHING;
    result.debug.pedestrian_route_motion_away =
        decision.motion == PedestrianRouteMotion::MOVING_AWAY;
    result.debug.pedestrian_route_motion_unknown =
        decision.motion == PedestrianRouteMotion::UNKNOWN;
    result.debug.pedestrian_motion_name =
        pedestrianRouteMotionName(decision.motion);
    result.debug.pedestrian_motion_reason = decision.reason;
    const PedestrianMotionRuntime* debug_motion =
        &runtime.pedestrian_behavior.motion_runtime;
    if (runtime.pedestrian_behavior.state ==
        PedestrianStopState::STOPPED_WAIT) {
        debug_motion = &runtime.pedestrian_behavior.stop_motion_runtime;
    }
    result.debug.pedestrian_motion_clearance_net_m =
        debug_motion->clearance_history_m.size() >= 2
            ? debug_motion->clearance_history_m.back() -
                  debug_motion->clearance_history_m.front()
            : 0.0f;
    result.debug.pedestrian_boundary_clearance_m =
        !debug_motion->clearance_history_m.empty()
            ? debug_motion->clearance_history_m.back()
            : (nearest_person != nullptr &&
                       std::isfinite(nearest_hit.boundary_clearance_m)
                   ? nearest_hit.boundary_clearance_m
                   : 0.0f);
    result.debug.pedestrian_seen_inside =
        runtime.pedestrian_behavior.seen_inside;
    result.debug.pedestrian_clear_streak =
        runtime.pedestrian_behavior.clear_streak;
    result.debug.pedestrian_stop_lost_streak =
        runtime.pedestrian_behavior.ped_stop_lost_streak;
    result.debug.pedestrian_exit_evidence =
        runtime.pedestrian_behavior.exit_evidence;
    result.debug.pedestrian_exit_lost_grace =
        runtime.pedestrian_behavior.exit_lost_grace;
    result.debug.pedestrian_slow_observation_count =
        runtime.pedestrian_behavior.slow_observation_count;
    result.debug.pedestrian_slow_lost_count =
        runtime.pedestrian_behavior.slow_lost_count;
    result.debug.pedestrian_slow_away_streak =
        runtime.pedestrian_behavior.slow_away_streak;
    result.debug.state_transition_reason = decision.reason;
    result.debug.active_block_reasons =
        decision.state == PedestrianStopState::STOPPED_WAIT
            ? decision.reason
            : "none";
    result.debug.ped_emergency_id =
        nearest_person != nullptr ? nearest_person->id : -1;
    result.debug.ped_foot_x_m =
        nearest_person != nullptr ? nearest_person->world_pos_m.x : 0.0f;
    result.debug.ped_foot_y_m =
        nearest_person != nullptr ? nearest_person->world_pos_m.y : 0.0f;
    result.debug.ped_danger_along_s_m =
        nearest_person != nullptr ? nearest_hit.along_s_m : 0.0f;
    result.debug.ped_danger_lateral_distance_m =
        nearest_person != nullptr &&
                std::isfinite(nearest_hit.lateral_distance_m)
            ? nearest_hit.lateral_distance_m
            : 0.0f;
    result.debug.ped_emergency_allowed_width_m =
        nearest_person != nullptr ? nearest_hit.allowed_width_m : 0.0f;
    if (runtime.pedestrian_behavior.state ==
        PedestrianStopState::STOPPED_WAIT) {
        setPlannerSpeed(result, 0.0f, "pedestrian_stopped_wait", "stop");
        result.obstacle_avoidance_engaged = true;
    } else if (runtime.pedestrian_behavior.state ==
               PedestrianStopState::LAUNCH_COMMITTED) {
        setPlannerSpeed(
            result,
            pedestrianLaunchSpeedMps(input, runtime.pedestrian_launch_source),
            pedestrianLaunchSourceName(runtime.pedestrian_launch_source),
            "launch");
        result.debug.startup_launch_release_event =
            decision.entered_launch;
        result.debug.ped_emergency_suppressed_by_launch = any_hard_hit;
    } else if (runtime.pedestrian_behavior.slow_active) {
        setPlannerSpeed(
            result,
            std::min(result.target_speed_mps,
                     std::max(0.0f, input.pedestrian_slow_speed_mps)),
            "pedestrian_slow",
            "caution");
    }

    syncPedestrianStopDebug(runtime, result);
}
void updatePedestrianLaunchProgress(
    PathPlanningRuntime& runtime,
    PathPlanningResult& result,
    bool launch_motion_allowed,
    bool launch_paused_by_path,
    bool launch_paused_by_external_stop)
{
    if (runtime.pedestrian_behavior.state !=
        PedestrianStopState::LAUNCH_COMMITTED) {
        syncPedestrianStopDebug(runtime, result);
        return;
    }

    result.debug.launch_paused_by_path = launch_paused_by_path;
    result.debug.launch_paused_by_external_stop = launch_paused_by_external_stop;
    const PedestrianBehaviorDecision decision = advancePedestrianLaunch(
        runtime.pedestrian_behavior,
        launch_motion_allowed);
    if (decision.launch_completed) {
        runtime.pedestrian_launch_source = PedestrianLaunchSource::NONE;
        runtime.pedestrian_danger_corridor_servo_locked = false;
        result.debug.state_transition_reason = "launch_window_complete";
    }

    syncPedestrianStopDebug(runtime, result);
}
