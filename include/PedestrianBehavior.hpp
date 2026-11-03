#pragma once

#include <deque>
#include <string>
#include <vector>

enum class PedestrianStopState {
    DRIVING = 0,
    STOPPED_WAIT,
    LAUNCH_COMMITTED
};

enum class PedestrianRouteMotion {
    UNKNOWN = 0,
    APPROACHING,
    MOVING_AWAY
};

enum class PedestrianStopCarSide {
    UNLOCKED = 0,
    CENTER,
    CAR_LEFT,
    CAR_RIGHT
};

enum class PedestrianReleaseRequiredSide {
    ANY = 0,
    LEFT_OUTSIDE,
    RIGHT_OUTSIDE
};

struct PedestrianBehaviorConfig {
    int motion_window_frames = 4;
    float motion_trend_threshold_m = 0.010f;
    float motion_max_jump_m = 0.15f;
    int slow_away_confirm_frames = 2;
    int slow_observation_frames = 6;
    int slow_lost_frames = 3;
    int stop_clear_frames = 3;
    int release_lost_grace_frames = 2;
    float exit_boundary_margin_m = 0.05f;
    float stop_car_side_threshold_m = 0.08f;
    int launch_allowed_motion_frames = 30;
};

struct PedestrianObservationEvidence {
    float x_m = 0.0f;
    float y_m = 0.0f;
    float distance_to_vehicle_m = 0.0f;
    float corridor_boundary_clearance_m = 0.0f;
    bool within_slow_distance = false;
    bool potential_conflict = false;
    bool left_outside = false;
    bool right_outside = false;

    bool insideHardCorridor() const {
        return corridor_boundary_clearance_m <= 0.0f;
    }
};

struct PedestrianFrameEvidence {
    std::vector<PedestrianObservationEvidence> people;
    float near_route_lateral_m = 0.0f;
    bool near_route_lateral_valid = false;
};

struct PedestrianMotionRuntime {
    std::deque<float> clearance_history_m;
    bool has_last_position = false;
    float last_x_m = 0.0f;
    float last_y_m = 0.0f;
    bool last_update_was_jump = false;
};

struct PedestrianBehaviorRuntime {
    PedestrianStopState state = PedestrianStopState::DRIVING;
    bool slow_active = false;
    bool slow_dismissed = false;
    int slow_observation_count = 0;
    int slow_lost_count = 0;
    int slow_away_streak = 0;
    bool has_dismissed_position = false;
    float dismissed_x_m = 0.0f;
    float dismissed_y_m = 0.0f;
    PedestrianRouteMotion motion = PedestrianRouteMotion::UNKNOWN;
    PedestrianMotionRuntime motion_runtime;

    bool seen_inside = false;
    int clear_streak = 0;
    int ped_stop_lost_streak = 0;
    bool exit_evidence = false;
    int exit_lost_grace = 0;
    int preventive_away_streak = 0;
    PedestrianMotionRuntime stop_motion_runtime;
    std::deque<float> stop_car_side_history_m;
    float stop_car_side_metric_m = 0.0f;
    PedestrianStopCarSide locked_stop_car_side =
        PedestrianStopCarSide::UNLOCKED;
    PedestrianReleaseRequiredSide release_required_side =
        PedestrianReleaseRequiredSide::ANY;

    int launch_remaining = 0;
    std::string reason = "driving";
};

struct PedestrianBehaviorDecision {
    PedestrianStopState state = PedestrianStopState::DRIVING;
    PedestrianRouteMotion motion = PedestrianRouteMotion::UNKNOWN;
    bool slow_active = false;
    bool entered_stop = false;
    bool entered_launch = false;
    bool launch_completed = false;
    std::string reason = "driving";
};

void resetPedestrianMotion(PedestrianMotionRuntime& runtime);

PedestrianRouteMotion updatePedestrianMotion(
    float x_m,
    float y_m,
    float corridor_boundary_clearance_m,
    const PedestrianBehaviorConfig& config,
    PedestrianMotionRuntime& runtime);

void resetPedestrianBehavior(PedestrianBehaviorRuntime& runtime);

PedestrianBehaviorDecision updatePedestrianBehavior(
    const PedestrianFrameEvidence& frame,
    const PedestrianBehaviorConfig& config,
    PedestrianBehaviorRuntime& runtime);

PedestrianBehaviorDecision advancePedestrianLaunch(
    PedestrianBehaviorRuntime& runtime,
    bool motion_allowed);

const char* pedestrianStopStateName(PedestrianStopState state);
const char* pedestrianRouteMotionName(PedestrianRouteMotion motion);
const char* pedestrianStopCarSideName(PedestrianStopCarSide side);
const char* pedestrianReleaseRequiredSideName(
    PedestrianReleaseRequiredSide side);
