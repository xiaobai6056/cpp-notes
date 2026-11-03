#include "PedestrianBehavior.hpp"

#include <algorithm>
#include <cmath>

namespace {

constexpr int kStopLostReleaseFrames = 5;

float distanceBetween(float ax, float ay, float bx, float by) {
    return std::hypot(ax - bx, ay - by);
}

const PedestrianObservationEvidence* selectCandidate(
    const PedestrianFrameEvidence& frame,
    bool require_slow_distance) {
    const PedestrianObservationEvidence* best_conflict = nullptr;
    const PedestrianObservationEvidence* best_slow = nullptr;
    for (const auto& person : frame.people) {
        if (require_slow_distance && !person.within_slow_distance) {
            continue;
        }
        if (person.potential_conflict &&
            (!best_conflict || person.distance_to_vehicle_m <
                                   best_conflict->distance_to_vehicle_m)) {
            best_conflict = &person;
        }
        if ((!require_slow_distance || person.within_slow_distance) &&
            (!best_slow || person.distance_to_vehicle_m <
                               best_slow->distance_to_vehicle_m)) {
            best_slow = &person;
        }
    }
    return best_conflict ? best_conflict : best_slow;
}

int conflictCount(const PedestrianFrameEvidence& frame) {
    return static_cast<int>(std::count_if(
        frame.people.begin(), frame.people.end(),
        [](const PedestrianObservationEvidence& person) {
            return person.potential_conflict;
        }));
}

void beginLaunch(const PedestrianBehaviorConfig& config,
                 const char* reason,
                 PedestrianBehaviorRuntime& runtime) {
    runtime.state = PedestrianStopState::LAUNCH_COMMITTED;
    runtime.launch_remaining = std::max(1, config.launch_allowed_motion_frames);
    runtime.slow_active = false;
    runtime.slow_dismissed = false;
    runtime.reason = reason;
}

void beginStop(const char* reason, PedestrianBehaviorRuntime& runtime) {
    runtime.state = PedestrianStopState::STOPPED_WAIT;
    runtime.slow_active = false;
    runtime.slow_dismissed = false;
    runtime.slow_observation_count = 0;
    runtime.slow_lost_count = 0;
    runtime.slow_away_streak = 0;
    runtime.motion = PedestrianRouteMotion::UNKNOWN;
    resetPedestrianMotion(runtime.motion_runtime);
    runtime.seen_inside = false;
    runtime.clear_streak = 0;
    runtime.ped_stop_lost_streak = 0;
    runtime.exit_evidence = false;
    runtime.exit_lost_grace = 0;
    runtime.preventive_away_streak = 0;
    resetPedestrianMotion(runtime.stop_motion_runtime);
    runtime.stop_car_side_history_m.clear();
    runtime.stop_car_side_metric_m = 0.0f;
    runtime.locked_stop_car_side = PedestrianStopCarSide::UNLOCKED;
    runtime.release_required_side = PedestrianReleaseRequiredSide::ANY;
    runtime.reason = reason;
}

void clearSlowObservation(PedestrianBehaviorRuntime& runtime) {
    runtime.slow_active = false;
    runtime.slow_observation_count = 0;
    runtime.slow_lost_count = 0;
    runtime.slow_away_streak = 0;
    runtime.motion = PedestrianRouteMotion::UNKNOWN;
    resetPedestrianMotion(runtime.motion_runtime);
}

const PedestrianObservationEvidence* selectNearestInside(
    const PedestrianFrameEvidence& frame) {
    const PedestrianObservationEvidence* nearest = nullptr;
    for (const auto& person : frame.people) {
        if (!person.insideHardCorridor()) continue;
        if (!nearest || person.distance_to_vehicle_m <
                            nearest->distance_to_vehicle_m) {
            nearest = &person;
        }
    }
    return nearest;
}

const PedestrianObservationEvidence* selectNearest(
    const PedestrianFrameEvidence& frame) {
    const PedestrianObservationEvidence* nearest = nullptr;
    for (const auto& person : frame.people) {
        if (!nearest || person.distance_to_vehicle_m <
                            nearest->distance_to_vehicle_m) {
            nearest = &person;
        }
    }
    return nearest;
}

float medianStopCarSideMetric(const std::deque<float>& history) {
    if (history.empty()) return 0.0f;
    std::vector<float> values(history.begin(), history.end());
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if (values.size() % 2 == 0) {
        return 0.5f * (values[middle - 1] + values[middle]);
    }
    return values[middle];
}

void collectStopCarSideMetric(const PedestrianFrameEvidence& frame,
                              PedestrianBehaviorRuntime& runtime) {
    if (runtime.locked_stop_car_side != PedestrianStopCarSide::UNLOCKED ||
        !frame.near_route_lateral_valid ||
        !std::isfinite(frame.near_route_lateral_m)) {
        return;
    }
    runtime.stop_car_side_history_m.push_back(frame.near_route_lateral_m);
    while (runtime.stop_car_side_history_m.size() > 3) {
        runtime.stop_car_side_history_m.pop_front();
    }
    runtime.stop_car_side_metric_m =
        medianStopCarSideMetric(runtime.stop_car_side_history_m);
}

void lockStopCarSide(const PedestrianBehaviorConfig& config,
                     PedestrianBehaviorRuntime& runtime) {
    if (runtime.locked_stop_car_side != PedestrianStopCarSide::UNLOCKED) {
        return;
    }
    runtime.stop_car_side_metric_m =
        medianStopCarSideMetric(runtime.stop_car_side_history_m);
    const float threshold = std::max(0.0f, config.stop_car_side_threshold_m);
    if (runtime.stop_car_side_history_m.empty() ||
        std::abs(runtime.stop_car_side_metric_m) <= threshold) {
        runtime.locked_stop_car_side = PedestrianStopCarSide::CENTER;
        runtime.release_required_side = PedestrianReleaseRequiredSide::ANY;
    } else if (runtime.stop_car_side_metric_m > threshold) {
        runtime.locked_stop_car_side = PedestrianStopCarSide::CAR_LEFT;
        runtime.release_required_side =
            PedestrianReleaseRequiredSide::RIGHT_OUTSIDE;
    } else {
        runtime.locked_stop_car_side = PedestrianStopCarSide::CAR_RIGHT;
        runtime.release_required_side =
            PedestrianReleaseRequiredSide::LEFT_OUTSIDE;
    }
}

bool satisfiesRequiredReleaseSide(
    const PedestrianObservationEvidence& candidate,
    PedestrianReleaseRequiredSide required_side) {
    switch (required_side) {
        case PedestrianReleaseRequiredSide::LEFT_OUTSIDE:
            return candidate.left_outside;
        case PedestrianReleaseRequiredSide::RIGHT_OUTSIDE:
            return candidate.right_outside;
        case PedestrianReleaseRequiredSide::ANY:
            return true;
    }
    return true;
}

bool isContinuousStopReleaseCandidate(
    const PedestrianObservationEvidence& candidate,
    const PedestrianBehaviorConfig& config,
    const PedestrianMotionRuntime& runtime) {
    return !runtime.has_last_position ||
           distanceBetween(runtime.last_x_m, runtime.last_y_m,
                           candidate.x_m, candidate.y_m) <=
               std::max(0.0f, config.motion_max_jump_m);
}

PedestrianBehaviorDecision makeDecision(
    const PedestrianBehaviorRuntime& runtime,
    bool entered_stop = false,
    bool entered_launch = false,
    bool launch_completed = false) {
    PedestrianBehaviorDecision decision;
    decision.state = runtime.state;
    decision.motion = runtime.motion;
    decision.slow_active = runtime.slow_active;
    decision.entered_stop = entered_stop;
    decision.entered_launch = entered_launch;
    decision.launch_completed = launch_completed;
    decision.reason = runtime.reason;
    return decision;
}

}  // namespace

void resetPedestrianMotion(PedestrianMotionRuntime& runtime) {
    runtime = PedestrianMotionRuntime{};
}

PedestrianRouteMotion updatePedestrianMotion(
    float x_m,
    float y_m,
    float corridor_boundary_clearance_m,
    const PedestrianBehaviorConfig& config,
    PedestrianMotionRuntime& runtime) {
    runtime.last_update_was_jump = false;
    if (runtime.has_last_position &&
        distanceBetween(runtime.last_x_m, runtime.last_y_m, x_m, y_m) >
            std::max(0.0f, config.motion_max_jump_m)) {
        resetPedestrianMotion(runtime);
        runtime.last_update_was_jump = true;
    }
    runtime.has_last_position = true;
    runtime.last_x_m = x_m;
    runtime.last_y_m = y_m;
    runtime.clearance_history_m.push_back(corridor_boundary_clearance_m);
    const int window = std::max(2, config.motion_window_frames);
    while (static_cast<int>(runtime.clearance_history_m.size()) > window) {
        runtime.clearance_history_m.pop_front();
    }
    if (static_cast<int>(runtime.clearance_history_m.size()) < window) {
        return PedestrianRouteMotion::UNKNOWN;
    }
    const float net = runtime.clearance_history_m.back() -
                      runtime.clearance_history_m.front();
    const float threshold = std::max(0.0f, config.motion_trend_threshold_m);
    if (net >= threshold) return PedestrianRouteMotion::MOVING_AWAY;
    if (net <= -threshold) return PedestrianRouteMotion::APPROACHING;
    return PedestrianRouteMotion::UNKNOWN;
}

void resetPedestrianBehavior(PedestrianBehaviorRuntime& runtime) {
    runtime = PedestrianBehaviorRuntime{};
}

PedestrianBehaviorDecision updatePedestrianBehavior(
    const PedestrianFrameEvidence& frame,
    const PedestrianBehaviorConfig& config,
    PedestrianBehaviorRuntime& runtime) {
    if (runtime.state == PedestrianStopState::LAUNCH_COMMITTED) {
        runtime.reason = "launch_committed_ignore_pedestrian";
        return makeDecision(runtime);
    }

    const bool any_inside = std::any_of(
        frame.people.begin(), frame.people.end(),
        [](const PedestrianObservationEvidence& person) {
            return person.insideHardCorridor();
        });
    bool entered_stop = false;
    if (runtime.state == PedestrianStopState::DRIVING && any_inside) {
        beginStop("hard_corridor_inside", runtime);
        entered_stop = true;
    }

    if (runtime.state == PedestrianStopState::STOPPED_WAIT) {
        const int inside_count = static_cast<int>(std::count_if(
            frame.people.begin(), frame.people.end(),
            [](const PedestrianObservationEvidence& person) {
                return person.insideHardCorridor();
            }));
        if (runtime.seen_inside || inside_count > 0) {
            collectStopCarSideMetric(frame, runtime);
        }

        if (inside_count > 0) {
            runtime.ped_stop_lost_streak = 0;
            const auto* candidate = selectNearestInside(frame);
            if (candidate) {
                runtime.motion = updatePedestrianMotion(
                    candidate->x_m, candidate->y_m,
                    candidate->corridor_boundary_clearance_m,
                    config, runtime.stop_motion_runtime);
            } else {
                runtime.motion = PedestrianRouteMotion::UNKNOWN;
            }
            runtime.seen_inside = true;
            runtime.clear_streak = 0;
            runtime.exit_lost_grace = 0;
            runtime.preventive_away_streak = 0;
            runtime.exit_evidence =
                inside_count == 1 && candidate != nullptr &&
                candidate->corridor_boundary_clearance_m >=
                    -std::max(0.0f, config.exit_boundary_margin_m) &&
                candidate->corridor_boundary_clearance_m <= 0.0f &&
                runtime.motion == PedestrianRouteMotion::MOVING_AWAY &&
                !runtime.stop_motion_runtime.last_update_was_jump;
            runtime.reason = runtime.exit_evidence
                                 ? "inside_near_boundary_away"
                                 : "inside_risk_present";
            return makeDecision(runtime, entered_stop);
        }

        if (runtime.seen_inside) {
            const auto* nearest = selectNearest(frame);
            const bool continuous_candidate =
                nearest != nullptr &&
                isContinuousStopReleaseCandidate(
                    *nearest, config, runtime.stop_motion_runtime);
            if (continuous_candidate) {
                runtime.ped_stop_lost_streak = 0;
                runtime.motion = updatePedestrianMotion(
                    nearest->x_m, nearest->y_m,
                    nearest->corridor_boundary_clearance_m,
                    config, runtime.stop_motion_runtime);
                lockStopCarSide(config, runtime);
                if (satisfiesRequiredReleaseSide(
                        *nearest, runtime.release_required_side)) {
                    ++runtime.clear_streak;
                } else {
                    runtime.clear_streak = 0;
                }
                runtime.exit_evidence = false;
                runtime.exit_lost_grace = 0;
                if (runtime.clear_streak >=
                    std::max(1, config.stop_clear_frames)) {
                    beginLaunch(config, "all_visible_people_outside", runtime);
                    return makeDecision(runtime, entered_stop, true);
                }
                runtime.reason = runtime.clear_streak > 0
                                     ? "visible_people_outside"
                                     : "visible_person_wrong_release_side";
                return makeDecision(runtime, entered_stop);
            }

            ++runtime.ped_stop_lost_streak;
            runtime.motion = PedestrianRouteMotion::UNKNOWN;

            const int grace_limit =
                std::max(0, config.release_lost_grace_frames);
            if (runtime.clear_streak > 0 &&
                runtime.exit_lost_grace < grace_limit) {
                ++runtime.exit_lost_grace;
                ++runtime.clear_streak;
                if (runtime.clear_streak >=
                    std::max(1, config.stop_clear_frames)) {
                    beginLaunch(config, "outside_then_lost_clear", runtime);
                    return makeDecision(runtime, entered_stop, true);
                }
                runtime.reason = "outside_then_lost_grace";
                return makeDecision(runtime, entered_stop);
            }
            if (runtime.exit_evidence &&
                runtime.exit_lost_grace < grace_limit &&
                runtime.locked_stop_car_side !=
                    PedestrianStopCarSide::CAR_LEFT &&
                runtime.locked_stop_car_side !=
                    PedestrianStopCarSide::CAR_RIGHT) {
                ++runtime.exit_lost_grace;
                if (grace_limit > 0 &&
                    runtime.exit_lost_grace >= grace_limit) {
                    beginLaunch(config, "near_boundary_away_then_lost", runtime);
                    return makeDecision(runtime, entered_stop, true);
                }
                runtime.reason = "exit_evidence_lost_grace";
                return makeDecision(runtime, entered_stop);
            }
            if (runtime.ped_stop_lost_streak >= kStopLostReleaseFrames) {
                beginLaunch(config, "stop_lost_5_launch", runtime);
                return makeDecision(runtime, entered_stop, true);
            }
            runtime.reason = "lost_without_exit_evidence";
            return makeDecision(runtime, entered_stop);
        }

        if (frame.people.empty()) {
            ++runtime.ped_stop_lost_streak;
        } else {
            runtime.ped_stop_lost_streak = 0;
        }
        const auto* candidate = selectCandidate(frame, false);
        if (candidate) {
            runtime.motion = updatePedestrianMotion(
                candidate->x_m, candidate->y_m,
                candidate->corridor_boundary_clearance_m,
                config, runtime.stop_motion_runtime);
        } else {
            runtime.motion = PedestrianRouteMotion::UNKNOWN;
        }

        if (runtime.ped_stop_lost_streak >= kStopLostReleaseFrames) {
            beginLaunch(config, "stop_lost_5_launch", runtime);
            return makeDecision(runtime, entered_stop, true);
        }

        if (candidate) {
            const int candidate_conflict_count =
                candidate->potential_conflict ? 1 : 0;
            const bool no_other_conflicts =
                conflictCount(frame) == candidate_conflict_count;
            if (runtime.motion == PedestrianRouteMotion::MOVING_AWAY &&
                no_other_conflicts) {
                ++runtime.preventive_away_streak;
            } else {
                runtime.preventive_away_streak = 0;
            }
            if (runtime.preventive_away_streak >=
                std::max(1, config.slow_away_confirm_frames)) {
                beginLaunch(config, "preventive_stop_away", runtime);
                return makeDecision(runtime, entered_stop, true);
            }
        } else {
            runtime.preventive_away_streak = 0;
        }
        runtime.reason = "preventive_stop_waiting_for_away";
        return makeDecision(runtime, entered_stop);
    }

    const auto* candidate = selectCandidate(frame, true);
    if (runtime.slow_dismissed) {
        if (!candidate || candidate->potential_conflict ||
            !candidate->within_slow_distance ||
            !runtime.has_dismissed_position ||
            distanceBetween(runtime.dismissed_x_m, runtime.dismissed_y_m,
                            candidate->x_m, candidate->y_m) >
                config.motion_max_jump_m) {
            runtime.slow_dismissed = false;
            runtime.has_dismissed_position = false;
        } else {
            runtime.dismissed_x_m = candidate->x_m;
            runtime.dismissed_y_m = candidate->y_m;
            runtime.reason = "slow_candidate_dismissed";
            return makeDecision(runtime);
        }
    }

    if (!runtime.slow_active) {
        if (!candidate || !candidate->within_slow_distance) {
            runtime.reason = "driving";
            return makeDecision(runtime);
        }
        runtime.slow_active = true;
        runtime.slow_observation_count = 0;
        runtime.slow_lost_count = 0;
        runtime.slow_away_streak = 0;
        resetPedestrianMotion(runtime.motion_runtime);
    }

    if (!candidate || !candidate->within_slow_distance) {
        ++runtime.slow_lost_count;
        runtime.motion = PedestrianRouteMotion::UNKNOWN;
        runtime.slow_away_streak = 0;
        if (runtime.slow_lost_count >= std::max(1, config.slow_lost_frames)) {
            clearSlowObservation(runtime);
            runtime.reason = "slow_candidate_lost";
        } else {
            runtime.reason = "slow_observing_lost";
        }
        return makeDecision(runtime);
    }

    runtime.slow_lost_count = 0;
    ++runtime.slow_observation_count;
    runtime.motion = updatePedestrianMotion(
        candidate->x_m, candidate->y_m,
        candidate->corridor_boundary_clearance_m,
        config, runtime.motion_runtime);

    const int conflicts = conflictCount(frame);
    if (candidate->potential_conflict &&
        runtime.motion == PedestrianRouteMotion::APPROACHING) {
        beginStop("approaching_potential_conflict", runtime);
        return makeDecision(runtime, true);
    }
    const int candidate_conflict_count =
        candidate->potential_conflict ? 1 : 0;
    const bool no_other_conflicts =
        conflicts == candidate_conflict_count;
    if (runtime.motion == PedestrianRouteMotion::MOVING_AWAY &&
        no_other_conflicts) {
        ++runtime.slow_away_streak;
        if (runtime.slow_away_streak >=
            std::max(1, config.slow_away_confirm_frames)) {
            beginLaunch(config, "slow_away_confirmed", runtime);
            return makeDecision(runtime, false, true);
        }
    } else {
        runtime.slow_away_streak = 0;
    }

    if (runtime.slow_observation_count >=
        std::max(1, config.slow_observation_frames)) {
        if (candidate->potential_conflict) {
            beginStop("slow_unknown_conflict_timeout", runtime);
            return makeDecision(runtime, true);
        }
        runtime.slow_dismissed = true;
        runtime.has_dismissed_position = true;
        runtime.dismissed_x_m = candidate->x_m;
        runtime.dismissed_y_m = candidate->y_m;
        clearSlowObservation(runtime);
        runtime.reason = "slow_no_conflict_dismissed";
        return makeDecision(runtime);
    }

    runtime.reason = candidate->potential_conflict
                         ? "slow_observing_conflict"
                         : "slow_observing_no_conflict";
    return makeDecision(runtime);
}

PedestrianBehaviorDecision advancePedestrianLaunch(
    PedestrianBehaviorRuntime& runtime,
    bool motion_allowed) {
    if (runtime.state != PedestrianStopState::LAUNCH_COMMITTED) {
        return makeDecision(runtime);
    }
    if (motion_allowed) --runtime.launch_remaining;
    if (runtime.launch_remaining <= 0) {
        resetPedestrianBehavior(runtime);
        runtime.reason = "launch_completed";
        return makeDecision(runtime, false, false, true);
    }
    runtime.reason = motion_allowed ? "launch_progress" : "launch_paused";
    return makeDecision(runtime);
}

const char* pedestrianStopStateName(PedestrianStopState state) {
    switch (state) {
        case PedestrianStopState::DRIVING: return "DRIVING";
        case PedestrianStopState::STOPPED_WAIT: return "STOPPED_WAIT";
        case PedestrianStopState::LAUNCH_COMMITTED: return "LAUNCH_COMMITTED";
    }
    return "DRIVING";
}

const char* pedestrianRouteMotionName(PedestrianRouteMotion motion) {
    switch (motion) {
        case PedestrianRouteMotion::APPROACHING: return "approaching";
        case PedestrianRouteMotion::MOVING_AWAY: return "moving_away";
        case PedestrianRouteMotion::UNKNOWN: return "unknown";
    }
    return "unknown";
}

const char* pedestrianStopCarSideName(PedestrianStopCarSide side) {
    switch (side) {
        case PedestrianStopCarSide::UNLOCKED: return "UNLOCKED";
        case PedestrianStopCarSide::CENTER: return "CENTER";
        case PedestrianStopCarSide::CAR_LEFT: return "CAR_LEFT";
        case PedestrianStopCarSide::CAR_RIGHT: return "CAR_RIGHT";
    }
    return "UNLOCKED";
}

const char* pedestrianReleaseRequiredSideName(
    PedestrianReleaseRequiredSide side) {
    switch (side) {
        case PedestrianReleaseRequiredSide::ANY: return "ANY";
        case PedestrianReleaseRequiredSide::LEFT_OUTSIDE:
            return "LEFT_OUTSIDE";
        case PedestrianReleaseRequiredSide::RIGHT_OUTSIDE:
            return "RIGHT_OUTSIDE";
    }
    return "ANY";
}
