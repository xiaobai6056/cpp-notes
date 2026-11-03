#pragma once

#include "PedestrianBehavior.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "ObjectTracker.hpp"

inline constexpr float kPedestrianDangerDefaultNearWidthM = 0.70f;
inline constexpr float kPedestrianDangerDefaultFarWidthM = 0.22f;
inline constexpr float kPedestrianDangerDefaultLengthM = 0.70f;
inline constexpr float kPedestrianDangerDefaultSampleStepM = 0.03f;
inline constexpr float kPedestrianDangerDefaultServoDeadband = 10.0f;
inline constexpr float kPedestrianDangerDefaultServoMaxAbs = 140.0f;

inline bool pedestrianDangerCorridorValid(float near_width_m,
                                          float far_width_m,
                                          float length_m,
                                          float sample_step_m) {
    return std::isfinite(near_width_m) &&
           std::isfinite(far_width_m) &&
           std::isfinite(length_m) &&
           std::isfinite(sample_step_m) &&
           far_width_m > 0.0f &&
           near_width_m >= far_width_m &&
           length_m > 0.0f &&
           sample_step_m > 0.0f;
}

inline float pedestrianDangerCorridorAllowedWidthM(float along_s_m,
                                                   float near_width_m,
                                                   float far_width_m,
                                                   float length_m) {
    if (!pedestrianDangerCorridorValid(
            near_width_m, far_width_m, length_m, 1.0f)) {
        return 0.0f;
    }
    const float ratio = std::clamp(along_s_m / length_m, 0.0f, 1.0f);
    return near_width_m + (far_width_m - near_width_m) * ratio;
}

struct PedestrianDangerCorridorGeometry {
    bool valid = false;
    bool straight = true;
    float servo_output = 0.0f;
    float servo_abs_used = 0.0f;
    float turn_radius_m = 0.0f;
    int turn_sign = 0;
    float near_width_m = kPedestrianDangerDefaultNearWidthM;
    float far_width_m = kPedestrianDangerDefaultFarWidthM;
    float length_m = kPedestrianDangerDefaultLengthM;
    float sample_step_m = kPedestrianDangerDefaultSampleStepM;
    std::vector<cv::Point2f> centerline_world;
    std::vector<cv::Point2f> left_boundary_world;
    std::vector<cv::Point2f> right_boundary_world;
    std::vector<cv::Point2f> polygon_world;
};

struct PedestrianDangerHitResult {
    bool hit = false;
    bool corridor_hit = false;
    // Signed distance to the polygon boundary: negative inside, positive outside.
    float boundary_clearance_m = std::numeric_limits<float>::infinity();
    float along_s_m = 0.0f;
    float lateral_distance_m = std::numeric_limits<float>::infinity();
    float signed_lateral_m = std::numeric_limits<float>::infinity();
    float allowed_width_m = 0.0f;
    cv::Point2f closest_world = cv::Point2f(0.0f, 0.0f);
};

struct PedestrianNormalStripGeometry {
    bool valid = false;
    int track_id = -1;
    cv::Point2f pedestrian_world = cv::Point2f(0.0f, 0.0f);
    cv::Point2f center_world = cv::Point2f(0.0f, 0.0f);
    cv::Point2f tangent_unit = cv::Point2f(0.0f, 1.0f);
    cv::Point2f normal_unit = cv::Point2f(1.0f, 0.0f);
    cv::Point2f forward_edge_left_world = cv::Point2f(0.0f, 0.0f);
    cv::Point2f forward_edge_right_world = cv::Point2f(0.0f, 0.0f);
    float half_width_along_road_m = 0.0f;
    float half_length_across_road_m = 0.40f;
    bool forward_edge_intersects_corridor = false;
    bool pedestrian_inside_corridor = false;
};

// ============================================================================
// Core data models
// ============================================================================
struct CenterlineSample {
    float y = 0.0f;
    float x = 0.0f;
};

enum class BranchPreference {
    PreferLeft = 0,
    ForceRight = 1
};

enum class BranchSide {
    None = 0,
    Main = 1,
    // Compatibility slot used by existing Dragon call sites. In the migrated
    // branch-output chain, this slot carries the main branch, not a traditional
    // left road-boundary line.
    Left = 2,
    Right = 3
};

struct BranchPlanMeta {
    int visible_branch_count = 0;
    bool has_fork = false;
    // During the demo-to-Dragon migration, selected_side describes the selected
    // output slot: Left means main branch, Right means right fork branch.
    BranchSide selected_side = BranchSide::None;
};

struct LineLossTraceFields {
    int line_arrow_object_count = 0;
    int line_raw_guide_count = 0;
    int line_valid_world_guide_count = 0;

    int line_single_anchor_count = 0;
    int line_single_root_count = 0;
    int line_credible_single_count = 0;
    int line_best_single_guide_count = 0;
    float line_best_single_forward_span = 0.0f;

    int line_visible_branch_count = 0;
    std::string line_selected_branch_label = "none";
    int line_selected_branch_guide_count = 0;
    int line_selected_branch_centerline_size = 0;
    bool line_selected_branch_missing = false;

    int line_candidate_size = 0;
    bool line_candidate_usable = false;

    int line_local_route_size = 0;
    int line_local_waypoints_size = 0;
    bool line_local_usable = false;
    bool line_used_candidate_fallback = false;
    std::string line_local_reject_reason = "none";
    int line_final_plan_path_size = 0;
    bool line_final_plan_path_usable = false;

    int line_world_path_size = 0;
    int line_positive_world_y_count = 0;
    bool line_pure_pursuit_target_valid = false;

    std::string line_servo_source = "no_target_hold_or_default";
    std::string line_empty_reason = "unknown";
};

struct CenterlineDebugInfo {
    int input_guides = 0;
    int valid_guides = 0;
    int contiguous_guides = 0;
    int raw_samples = 0;
    int filtered_samples = 0;
    int interp_failures = 0;
    int reject_non_finite = 0;
    int reject_jump = 0;
    int first_dx_break_index = -1;
    float first_dx_break = 0.0f;
    float sample_y_max = 0.0f;
    float sample_y_min = 0.0f;
    float max_center_jump = 0.0f;
    float guide_y_span = 0.0f;
    float adjacent_dx_limit = 0.0f;
    int sample_step_y = 0;
    int shared_guides_available = 0;
    int fork_guides_available = 0;
    int selected_shared_guides = 0;
    int selected_fork_guides = 0;
    int compatible_shared_guides = 0;
    float handoff_dx = 0.0f;
    float handoff_dy = 0.0f;
    float handoff_dx_limit = 0.0f;
    float fork_mean_x = 0.0f;
    float shared_filter_x_limit = 0.0f;
    float bridge_dx = 0.0f;
    std::string guide_selection_mode;
    std::string early_exit_reason;
};

struct BranchDebugInfo {
    int assigned_points = 0;
    int rejected_points = 0;
    int output_branches = 0;
    float main_forward_span = 0.0f;
    float main_max_dx_jump = 0.0f;
    int finalize_input_branch_count = 0;
    bool boundary_final_split = false;
    std::string boundary_final_split_reason = "not_evaluated";
    std::string boundary_reject_reason = "none";
    bool final_output_preserve_labels = false;
    bool final_output_anchor_sort_applied = false;
    std::string output_left_source_ids;
    std::string output_right_source_ids;
    std::string physical_boundary_provider_summary;
    int road_corridor_count = 0;
    bool road_corridor_valid = false;
    float road_corridor_width_mean_m = 0.0f;
    std::string road_corridor_invalid_reason = "none";
    std::string status = "single";
    // Demo-aligned split decision tracking
    bool split_maintain_used = false;
    std::string split_maintain_reason = "none";
    int split_maintain_age = 0;
    bool original_final_split = false;
    std::string original_block_reason = "none";
    bool final_split_after_maintain = false;
    bool was_recently_split_before_decision = false;
    bool visible_split_output_active = false;
    int visible_split_maintain_age = 0;
    std::string split_confirm_mode = "none";
    int split_confirm_count = 0;
    bool split_confirmed = false;
    std::string split_confirm_reject_reason = "none";
    std::string final_semantic_reason = "none";
    bool has_ft_opening_branch_support = false;
    std::string ft_support_reason = "none";
    std::string ft_support_t_ids;
    std::string ft_support_t_fwd;
    std::string ft_support_t_lat;
    float ft_support_t_lat_sp = 0.0f;
    int ft_support_t_n = 0;
    bool ft_support_t_near_ok = false;
    int ft_support_t_near_cov = 0;
    float ft_support_open_branch_score = 0.0f;
    float ft_support_merge_band_score = 0.0f;
    std::string selected_semantic_pair_type = "none";
    std::string selected_semantic_pair_semantic = "none";
    std::string provider_semantic_candidate_source = "none";
    std::string provider_semantic_reason = "none";
    bool usable_opening_evidence = false;
    bool no_anchor_on_main_seen = false;
    bool no_anchor_on_main_preserve_state_only = false;
    bool candidate_reuses_main_forward_seen = false;
    bool candidate_reuses_main_forward_preserve_state_only = false;
    bool active_visible_split_context = false;
    bool split_confirmed_only_context = false;
    bool transient_failure_type_ok = false;
    bool independent_right_fork_evidence = false;
    bool geometry_not_bad = false;
    bool structural_strong_negative = false;
    std::string structural_strong_negative_reason = "none";
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
    float detached_lateral_separation = 0.0f;
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
    bool detached_fresh_entry_gate_evaluated = false;
    bool detached_fresh_entry_gate_allowed = false;
    std::string detached_fresh_entry_gate_reason = "not_evaluated";
    bool detached_fresh_entry_basic_valid = false;
    bool detached_fresh_entry_structural_support = false;
    bool detached_fresh_entry_specific_closing = false;
    std::string detached_fresh_entry_structural_reason = "not_evaluated";
    std::string detached_fresh_entry_closing_reason = "not_evaluated";
    int detached_fresh_entry_support_score = 0;
    int detached_fresh_entry_same_region_closing_count = 0;
    bool stable_double_boundary_recovery_evidence = false;
    bool stable_double_boundary_recovery_confirmed_this_frame = false;
    bool split_recovery_context_active = false;
    int split_recovery_context_age = 0;
    int stable_double_boundary_recovery_count = 0;
    std::string split_recovery_context_reason = "none";
    std::string stable_double_boundary_recovery_reject_reason = "not_evaluated";
    std::string split_recovery_seed_main_ids = "none";
    std::string split_recovery_seed_right_ids = "none";
    bool historical_main_context_active = false;
    std::string historical_main_ids = "none";
    int historical_main_visible_id_count = 0;
    std::string normal_candidate_ids = "none";
    bool normal_matches_historical_main = false;
    float normal_match_previous_right_ratio = 0.0f;
    bool historical_main_guided_grow_attempted = false;
    std::string historical_main_candidate_ids = "none";
    bool historical_main_candidate_generated = false;
    bool historical_main_candidate_credible = false;
    bool historical_main_candidate_usable = false;
    int historical_main_match_common_ids = 0;
    float historical_main_match_ratio = 0.0f;
    bool historical_main_reliable = false;
    std::string historical_main_reject_reason = "not_evaluated";
    bool historical_candidate_appended_to_credible_paths = false;
    bool main_identity_active_split_context = false;
    std::string main_identity_mode = "inactive";
    bool main_identity_stable_dual_line_identity = false;
    std::string main_identity_previous_accepted_main_ids = "none";
    std::string main_identity_previous_main_ids = "none";
    std::string main_identity_previous_right_ids = "none";
    std::string main_identity_raw_top1_ids = "none";
    float main_identity_raw_top1_score = 0.0f;
    std::string main_identity_incumbent_ids = "none";
    float main_identity_incumbent_score = 0.0f;
    int main_identity_incumbent_match_common_ids = 0;
    float main_identity_incumbent_match_ratio = 0.0f;
    bool main_identity_incumbent_usable = false;
    bool main_identity_incumbent_credible = false;
    bool main_identity_incumbent_main_reliable = false;
    std::string main_identity_challenger_ids = "none";
    float main_identity_challenger_score = 0.0f;
    float main_identity_challenger_advantage = 0.0f;
    float main_identity_challenger_match_previous_right_ratio = 0.0f;
    bool main_identity_challenger_matches_previous_right = false;
    bool main_identity_previous_right_veto_active = false;
    bool main_identity_main_takeover_allowed = true;
    std::string main_identity_pending_challenger_ids = "none";
    int main_identity_challenger_confirm_count = 0;
    int main_identity_switch_required_confirm_frames = 0;
    int main_identity_reverse_switch_lock_remaining = 0;
    bool main_identity_switch_blocked = false;
    std::string main_identity_switch_block_reason = "none";
    bool main_identity_switched = false;
    std::string main_identity_switch_reason = "not_evaluated";
    std::string main_identity_final_selected_main_ids = "none";
    bool provider_semantic_not_opening_seen = false;
    bool provider_semantic_used_as_weak_negative = false;
    bool preserve_state_only = false;
    std::string preserve_blocked_reason = "none";
    bool grace_budget_ok = false;
    int consecutive_invalid_visible_pair_count = 0;
    bool structural_hard_gate_pass = false;
    std::string structural_hard_gate_fail_reason = "not_evaluated";
    bool split_maintain_structural_ok = false;
    std::string split_maintain_structural_fail_reason = "not_evaluated";
    bool split_entry_by_semantic_confirm = false;
    bool split_maintain_by_structure = false;
    bool semantic_exit_veto_ignored = false;
    int visible_split_provider_gap_frames = 0;
    bool provider_grace_candidate = false;
    bool provider_grace_used = false;
    bool visible_gate_pass = false;
    std::string visible_gate_fail_reason = "not_evaluated";
    bool maturity_count_pass = true;
    std::string last_confirmed_right_fork_ids = "none";
    std::string current_right_fork_candidate_ids = "none";
    bool suffix_continuation_found = false;
    std::string suffix_continuation_reason = "none";
};

struct ArrowBranch {
    std::vector<ArrowGuide> guides;
    std::string label;
    cv::Scalar color = cv::Scalar(255, 255, 255);
    float anchor_x = 0.0f;
    int shared_prefix_count = 0;
    bool has_poly_fit = false;
    float poly_a = 0.0f;
    float poly_b = 0.0f;
    float poly_c = 0.0f;
    float poly_y_min = 0.0f;
    float poly_y_max = 0.0f;
};

// Demo-aligned types for A2 selector chain.
enum class BoundarySide {
    Unknown = 0,
    Left = -1,
    Right = 1
};

struct BoundaryObservation {
    bool valid = false;
    BoundarySide side = BoundarySide::Unknown;
    std::string source;
    std::vector<ArrowGuide> guides;
    float confidence = 0.0f;
    float root_lateral = 0.0f;
    float root_forward = 0.0f;
    float center_lateral = 0.0f;
    float near_forward = 0.0f;
    float far_forward = 0.0f;
    float heading_deg = 0.0f;
};

enum class BoundarySelectionMode {
    SplitPair = 0,
    LeftOnly,
    RightOnly,
    NoReliableBoundary
};

// Demo-aligned per-side cross-frame boundary track state.
// Mirrors demo BoundaryTrackState in centerline.cpp.
struct BoundaryTrackState {
    BoundarySide side = BoundarySide::Unknown;
    bool active = false;
    float confidence = 0.0f;
    int missed_frames = 0;
    float root_lateral = 0.0f;
    float root_forward = 0.0f;
    float center_lateral = 0.0f;
    float near_forward = 0.0f;
    float far_forward = 0.0f;
    float predicted_lateral = 0.0f;
    float predicted_heading = 0.0f;
    float selected_obs_cost = std::numeric_limits<float>::infinity();
    std::string reject_reason = "none";
    std::string obs_source;
    std::vector<ArrowGuide> guides;
};

// Demo-aligned A2 selector result. Mirrors demo BoundarySelectionResult.
struct BoundarySelectionResult {
    BoundarySelectionMode mode = BoundarySelectionMode::NoReliableBoundary;
    BoundaryObservation left_observation;
    BoundaryObservation right_observation;
    bool left_active = false;
    bool right_active = false;
    int selected_source_rank = -1;
    bool normalized_side_ok = false;
    float width_m = 0.0f;
    std::string width_source = "none";
    std::string reject_reason = "none";
    bool allow_boundary_track_update = false;
    float debug_health_score = -1e9f;
};

struct BoundaryAssociationScore {
    float cost = std::numeric_limits<float>::infinity();
    float predicted_lateral = 0.0f;
    float predicted_heading = 0.0f;
    float overlap = 0.0f;
    bool reach_gap_warning = false;
    std::string reject_reason = "none";
};

struct BoundaryTrackManager {
    bool split_confirmed = false;
    int split_confirm_count = 0;
    std::string split_confirm_mode = "none";
    std::string split_confirm_reject_reason = "none";
    std::string selected_pair_source = "none";
    int selected_pair_rank = -1;
    float selected_pair_width_m = 0.0f;
    float provider_final_width_m = 0.0f;
    float final_pair_width_m = 0.0f;
    std::string final_width_source = "none";
    bool selector_pair_width_out_of_range = false;
    bool selector_split_pair_unavailable = false;
    bool selector_non_physical_v2_candidate = false;
    bool selector_selected_non_physical_v2 = false;
    std::string selected_candidate_provider_type = "none";
    std::string selector_reject_reason = "none";
    float selector_selected_pair_width_m = 0.0f;
    std::string selector_selected_width_source = "none";
    int selector_best_rejected_rank = -1;
    float selector_best_rejected_score = -1e9f;
    std::string selector_best_rejected_reason = "none";
    bool provider_semantic_not_opening_seen = false;
    bool provider_semantic_used_as_weak_negative = false;
    std::string preserve_blocked_reason = "none";
    bool selected_closing_like = false;
    bool selected_closing_suspect = false;
    bool selected_strong_closing_veto = false;
    bool selected_opening_like = false;
    bool selected_stable_split_like = false;
    std::string final_semantic_reason = "none";
    bool visible_split_output_active = false;
    int visible_split_maintain_age = 0;
    int visible_split_provider_gap_frames = 0;
    std::vector<int> last_confirmed_right_fork_ids;
    std::string last_confirmed_right_fork_ids_text = "none";
    int last_confirmed_right_fork_frame = -1;
    std::vector<int> previous_output_main_ids;
    std::vector<int> previous_output_right_ids;
    std::vector<int> previous_accepted_main_ids;
    bool split_recovery_context_active = false;
    int split_recovery_context_age = 0;
    int stable_double_boundary_recovery_count = 0;
    std::vector<int> split_recovery_seed_main_ids;
    std::vector<int> split_recovery_seed_right_ids;
    std::string split_recovery_context_reason = "none";
    std::string stable_double_boundary_recovery_reject_reason = "not_evaluated";
    std::vector<int> pending_main_challenger_ids;
    int pending_main_challenger_confirm_count = 0;
    int main_identity_reverse_lock_remaining = 0;
    std::vector<int> last_switched_from_main_ids;
    std::string selected_right_fork_model = "none";
    bool detached_candidate_valid = false;
    std::string detached_candidate_ids = "none";
    std::string detached_gate_trace = "none";
    float detached_score = 0.0f;
    bool detached_structural_gate_pass = false;
    bool detached_right_side_order_ok = false;
    bool detached_crossing_detected = false;
    bool detached_remerge_detected = false;
    int detached_independent_id_count = 0;
    float detached_main_overlap = 0.0f;
    float detached_main_reuse_ratio = 0.0f;
    float detached_common_forward_overlap = 0.0f;
    int detached_right_chain_count = 0;
    float detached_right_chain_forward_span = 0.0f;
    float detached_right_chain_lateral_span = 0.0f;
    float detached_lateral_separation = 0.0f;
    float detached_right_chain_continuity = 0.0f;
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
    bool detached_fresh_entry_gate_evaluated = false;
    bool detached_fresh_entry_gate_allowed = false;
    std::string detached_fresh_entry_gate_reason = "not_evaluated";
    bool detached_fresh_entry_basic_valid = false;
    bool detached_fresh_entry_structural_support = false;
    bool detached_fresh_entry_specific_closing = false;
    std::string detached_fresh_entry_structural_reason = "not_evaluated";
    std::string detached_fresh_entry_closing_reason = "not_evaluated";
    int detached_fresh_entry_support_score = 0;
    int detached_fresh_entry_same_region_closing_count = 0;
    // Demo-aligned per-side cross-frame track state (replaces B-1 flat fields
    // for A2 selector association scoring).
    BoundaryTrackState left = [] {
        BoundaryTrackState state;
        state.side = BoundarySide::Left;
        return state;
    }();
    BoundaryTrackState right = [] {
        BoundaryTrackState state;
        state.side = BoundarySide::Right;
        return state;
    }();
    // Phase 5: provider semantic gate fields (formal decision path)
    bool frame_semantic_ok = false;
    bool frame_semantic_pending = false;
    bool frame_semantic_suppressed = false;
    std::string frame_semantic_mode = "none";
    std::string frame_semantic_reason = "none";
    bool has_ft_opening_branch_support = false;
    std::string ft_support_reason = "none";
    std::string ft_support_t_ids;
    std::string ft_support_t_fwd;
    std::string ft_support_t_lat;
    float ft_support_t_lat_sp = 0.0f;
    int ft_support_t_n = 0;
    bool ft_support_t_near_ok = false;
    int ft_support_t_near_cov = 0;
    float ft_support_open_branch_score = 0.0f;
    float ft_support_merge_band_score = 0.0f;
    float selected_opening_score = 0.0f;
    float selected_open_branch_score = 0.0f;
    float selected_merge_band_score = 0.0f;
    float selected_transverse_merge_band_score = 0.0f;
    float selected_transverse_opening_branch_score = 0.0f;
    std::string selected_transverse_role = "none";
    float selected_pair_coverage_ratio = 0.0f;
    float selected_pair_unexplained_transverse_risk = 0.0f;
    float selected_pair_semantic_rank_adjust = 0.0f;
    float selected_pair_fake_forward_penalty = 0.0f;
    float selected_pair_forward_side_support_score = 0.0f;
    bool frame_has_strong_closing_veto = false;
    float frame_best_closing_score = 0.0f;
    bool opening_blocked_by_frame_closing_veto = false;
    std::string opening_block_reason = "none";
    float opening_vs_closing_margin = 0.0f;
    bool selected_risk_downgraded_by_frame_opening = false;
    // Phase 5+: full multi-candidate frame-level semantic arbitration
    // (demo-aligned arbitrateFrameSemanticEvidence outputs)
    int semantic_selected_idx = -1;
    int semantic_suppressing_idx = -1;
    int best_opening_idx = -1;
    float best_opening_score = 0.0f;
    int best_closing_idx = -1;
    float best_closing_score = 0.0f;
    int best_merge_idx = -1;
    float best_merge_score = 0.0f;
    int best_stable_idx = -1;
    float best_stable_score = 0.0f;
    float opening_consistency_score = 0.0f;
    float closing_consistency_score = 0.0f;
    std::string opening_consistency_reason = "none";
    std::string closing_consistency_reason = "none";
    // Opening hypothesis cross-frame tracking
    std::string confirmation_region_key = "none";
    std::string opening_hypothesis_id = "none";
    bool confirmation_continued_by_region = false;
    std::string previous_opening_region_key = "none";
    std::string previous_opening_hypothesis_id = "none";
    int next_opening_hypothesis_seq = 1;
    // Competing evidence tracking
    bool competing_semantic_conflict = false;
    std::string competing_conflict_reason = "none";
    // Frame-level closing veto info
    int frame_best_closing_candidate_rank = -1;
    std::string frame_best_closing_pair_type = "none";
    // Per-frame selected-observation active + obs_source fields.
    // computeBoundaryFinalSplitDecision uses BoundaryTrackState active/obs_source
    // for demo-aligned hard gates instead.
    bool left_active = false;
    bool right_active = false;
    std::string left_obs_source = "none";
    std::string right_obs_source = "none";
    // B-1: full demo-aligned cross-frame track association state for A2 selector.
    // These persist across frames and feed association-score computation
    // in evaluateBoundaryPairHealth. They are NOT the per-frame A-1 fields.
    // Mirror of demo BoundaryTrackState (per side).
    bool left_track_active = false;
    bool right_track_active = false;
    float left_track_root_lateral_m = 0.0f;
    float right_track_root_lateral_m = 0.0f;
    float left_track_root_forward_m = 0.0f;
    float right_track_root_forward_m = 0.0f;
    float left_track_near_forward_m = 0.0f;
    float right_track_near_forward_m = 0.0f;
    float left_track_predicted_lateral_m = 0.0f;
    float right_track_predicted_lateral_m = 0.0f;
    float left_track_predicted_heading_deg = 0.0f;
    float right_track_predicted_heading_deg = 0.0f;
    float left_track_confidence = 0.0f;
    float right_track_confidence = 0.0f;
    int left_track_missed_frames = 0;
    int right_track_missed_frames = 0;
    std::vector<ArrowGuide> left_track_guides;
    std::vector<ArrowGuide> right_track_guides;
};

struct BoundaryPairCandidate {
    std::string source = "unknown";
    std::string provider_type = "unknown";
    int provider_rank = -1;
    int band_idx = -1;
    float spread = 0.0f;
    float threshold = 0.0f;
    float excess = 0.0f;
    float center_x = 0.0f;
    std::vector<ArrowGuide> shared_trunk;
    std::vector<ArrowGuide> left_branch;
    std::vector<ArrowGuide> right_branch;
    int left_shared_prefix_count = 0;
    int right_shared_prefix_count = 0;
    int unique_guide_count = 0;
    int left_point_count = 0;
    int right_point_count = 0;
    float trunk_avg_step = 0.0f;
    float left_avg_step = 0.0f;
    float right_avg_step = 0.0f;
    float trunk_max_dx = 0.0f;
    float left_max_dx = 0.0f;
    float right_max_dx = 0.0f;
    float span_balance_ratio = 0.0f;
    bool left_root_ok = false;
    bool right_root_ok = false;
    bool left_near_root_ok = false;
    bool right_near_root_ok = false;
    float left_span_m = 0.0f;
    float right_span_m = 0.0f;
    float left_y_span = 0.0f;
    float right_y_span = 0.0f;
    float weaker_y_span = 0.0f;
    float stronger_y_span = 0.0f;
    float left_bottom_reach_score = 0.0f;
    float right_bottom_reach_score = 0.0f;
    float weaker_bottom_reach_score = 0.0f;
    float stronger_bottom_reach_score = 0.0f;
    float provider_overlap_ratio = 0.0f;
    float weak_far_field_penalty = 0.0f;
    float common_y_overlap = 0.0f;
    float provider_overlap_width_m = 0.0f;
    float provider_near_width_m = 0.0f;
    float provider_overlap_m = 0.0f;
    float provider_pair_confidence = 0.0f;
    float shared_ratio = 0.0f;
    float far_div_pen = 0.0f;
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
    bool ft_opening_candidate = false;
    std::string transverse_t_ids;
    float transverse_near_forward_m = 0.0f;
    float transverse_far_forward_m = 0.0f;
    float transverse_lat_min_m = 0.0f;
    float transverse_lat_max_m = 0.0f;
    float transverse_lat_span_m = 0.0f;
    int transverse_guide_count = 0;
    bool transverse_near_ok = false;
    int transverse_near_cov = 0;
    // [DEBUG_ONLY] Textual primitive trace used in provider/debug summaries.
    std::string transverse_primitive_debug;
    std::string pair_selected_rank_reason = "none";
    std::string geometry_source = "semantic_only";
    std::string selected_right_fork_model = "none";
    bool detached_candidate_valid = false;
    bool detached_structural_gate_pass = false;
    bool detached_right_side_order_ok = false;
    bool detached_crossing_detected = false;
    bool detached_remerge_detected = false;
    int detached_independent_id_count = 0;
    float detached_main_overlap = 0.0f;
    float detached_main_reuse_ratio = 0.0f;
    float detached_common_forward_overlap = 0.0f;
    int detached_right_chain_count = 0;
    float detached_right_chain_forward_span = 0.0f;
    float detached_right_chain_continuity = 0.0f;
    float detached_width_min = 0.0f;
    float detached_width_mean = 0.0f;
    float detached_trend_width_delta = 0.0f;
    float detached_trend_width_slope = 0.0f;
};

// Phase 5+: full multi-candidate frame-level semantic arbitration result
struct FrameSemanticDecision {
    bool semantic_ok = false;
    bool semantic_pending = false;
    bool semantic_suppressed = false;
    std::string semantic_mode = "none";
    std::string semantic_reason = "no_semantic_evidence";
    int selected_semantic_idx = -1;
    int suppressing_idx = -1;
    int best_opening_idx = -1;
    float best_opening_score = 0.0f;
    int best_closing_idx = -1;
    float best_closing_score = 0.0f;
    int best_merge_idx = -1;
    float best_merge_score = 0.0f;
    int best_stable_idx = -1;
    float best_stable_score = 0.0f;
    float opening_consistency_score = 0.0f;
    float closing_consistency_score = 0.0f;
    std::string opening_consistency_reason = "none";
    std::string closing_consistency_reason = "none";
    float opening_vs_closing_margin = 0.0f;
    std::string confirmation_region_key = "none";
    bool competing_semantic_conflict = false;
    std::string competing_conflict_reason = "none";
};

struct FrameClosingVetoInfo {
    float best_closing_score = 0.0f;
    int best_closing_candidate_rank = -1;
    std::string best_closing_pair_type = "none";
    bool has_strong_closing_veto = false;
};

struct PathPlannerRuntimeState {
    BoundaryTrackManager boundary_tracks;
};

struct BranchCenterlineResult {
    std::vector<ArrowGuide> centerline_guides;
    CenterlineDebugInfo debug;
    std::vector<CenterlineSample> centerline;
    std::string fit_mode = "interp";
};

struct RoadCorridorPoint {
    cv::Point2f center_world = cv::Point2f(0.0f, 0.0f);
    cv::Point2f left_world = cv::Point2f(0.0f, 0.0f);
    cv::Point2f right_world = cv::Point2f(0.0f, 0.0f);
    cv::Point2f tangent = cv::Point2f(0.0f, 0.0f);
    cv::Point2f normal = cv::Point2f(0.0f, 0.0f);
    float half_width_m = 0.15f;
};

enum class RoadCorridorRole {
    Accepted = 0,
    // Compatibility slot for existing Dragon planning inputs. The migrated
    // branch pipeline stores the main branch here, not a traditional left line.
    LeftBranch,
    RightBranch
};

struct RoadCorridor {
    bool valid = false;
    RoadCorridorRole role = RoadCorridorRole::Accepted;
    int branch_id = -1;
    float half_width_m = 0.15f;
    std::vector<RoadCorridorPoint> points;
    std::string invalid_reason = "none";
};

struct RoadCorridorSet {
    bool valid = false;
    std::vector<RoadCorridor> corridors;
    std::string invalid_reason = "none";
    float width_min_m = 0.0f;
    float width_mean_m = 0.0f;
    float width_max_m = 0.0f;
};

// ============================================================================
// Local path planning types
// ============================================================================
enum class LocalPlannerMode {
    Disabled = 0,
    Cruise,
    AvoidPrepare,
    AvoidActive,
    AvoidRecover
};

enum class ObstacleAvoidanceState {
    Cruise = 0,
    Prepare,
    Active,
    Recover
};

enum class AvoidScenario {
    None,
    VehicleOnly
};

enum class GoldTransitionState {
    IDLE = 0,
    HARD,
    RECENTER
};

struct PlanningRoutePoint {
    cv::Point2f pos;
    float left = 0.0f;
    float right = 0.0f;
    float half_width = 70.0f;
    float center_residual = 0.0f;
    bool has_bounds = false;
    bool width_clipped = false;
    bool width_limited = false;
};

struct PlanningObject {
    int id = -1;
    ObjectType type = ObjectType::UNKNOWN;
    cv::Point2f pos;
    cv::Point2f velocity;
    cv::Point2f foot_pos;
    cv::Point2f world_pos_m = cv::Point2f(0.0f, 0.0f);
    float distance_to_self_m = 0.0f;
    float bbox_w = 1.0f;
    float bbox_h = 1.0f;
    float footprint_width_m = 0.0f;
    float footprint_length_m = 0.0f;
    float footprint_radius_m = 0.0f;
    bool has_foot_pos = false;
    bool has_world_pos_m = false;
    bool has_footprint_m = false;
    bool current_frame_visible = false;
    bool from_raw_detection = false;
    bool observed_this_frame = false;
};

struct PathPlanningInput {
    std::vector<PlanningRoutePoint> route;
    RoadCorridorSet road_corridors;
    std::vector<PlanningObject> objects;
    std::vector<PlanningObject> raw_person_objects;
    int visible_branch_count = 0;
    cv::Size frame_size;
    bool external_route_decision_valid = false;
    double timestamp_sec = 0.0;
    bool timestamp_valid = false;
    float ego_speed_mps = 0.0f;
    bool ego_speed_valid = false;
    float ego_yaw_deg = 0.0f;
    bool ego_yaw_valid = false;
    float ego_yaw_rate_radps = 0.0f;
    bool ego_yaw_rate_valid = false;
    float pedestrian_cruise_speed_mps = 1.0f;
    bool obstacle_avoid_enable = true;
    bool obstacle_runtime_enabled = true;
    int obstacle_confirm_frames = 3;
    int obstacle_exit_lost_frames = 10;
    int vehicle_recover_hold_frames = 5;
    int vehicle_recover_hold_max_lifetime_frames = 45;
    int obstacle_seen_hold_frames = 3;
    float obstacle_prepare_distance_m = 1.0f;
    float obstacle_activate_distance_m = 0.8f;
    float vehicle_length_m = 0.32f;
    float vehicle_width_m = 0.22f;
    float vehicle_front_m = 0.17f;
    float vehicle_rear_m = 0.15f;
    bool force_left_obstacle_avoidance = false;
    float obstacle_avoid_offset_m = 0.15f;
    float vehicle_avoidance_edge_clearance_m = 0.18f;
    float vehicle_avoidance_max_offset_m = 0.55f;
    float vehicle_curve_avoidance_extra_offset_m = 0.05f;
    float vehicle_avoidance_opposite_steer_threshold_deg = 5.0f;
    int vehicle_avoidance_opposite_steer_hold_frames = 8;
    float steering_angle_deg = 0.0f;
    bool steering_angle_valid = false;
    float obstacle_offset_rate_mps = 0.45f;
    float obstacle_offset_zero_epsilon_m = 0.001f;
    float pedestrian_danger_corridor_near_width_m =
        kPedestrianDangerDefaultNearWidthM;
    float pedestrian_danger_corridor_far_width_m =
        kPedestrianDangerDefaultFarWidthM;
    float pedestrian_danger_corridor_length_m =
        kPedestrianDangerDefaultLengthM;
    float pedestrian_danger_corridor_sample_step_m =
        kPedestrianDangerDefaultSampleStepM;
    float pedestrian_servo_straight_deadband =
        kPedestrianDangerDefaultServoDeadband;
    float pedestrian_servo_radius_max_abs =
        kPedestrianDangerDefaultServoMaxAbs;
    float pedestrian_danger_corridor_servo_smooth_alpha = 0.25f;
    float pedestrian_danger_corridor_servo_max_step = 10.0f;
    bool pedestrian_slow_enable = true;
    float pedestrian_slow_distance_m = 0.8f;
    float pedestrian_slow_speed_mps = 0.25f;
    PedestrianBehaviorConfig pedestrian_behavior_config;
    float pedestrian_normal_strip_half_width_m = 0.20f;
    float pedestrian_stopped_launch_speed_mps = 1.3f;
    float pedestrian_slow_launch_speed_mps = 1.0f;
    float gold_bias_step_m = 0.05f;
    float gold_max_bias_m = 0.20f;
    float gold_capture_compensation_m = 0.03f;
    float gold_capture_tolerance_m = 0.04f;
    float gold_bias_rate_mps = 0.40f;
    float gold_max_new_delta_m = 0.12f;
    int gold_recenter_hold_frames = 10;
    float gold_near_priority_distance_m = 0.30f;
    float gold_late_start_forward_m = 0.18f;
    float gold_late_start_delta_m = 0.09f;
    float gold_center_release_m = 0.02f;
    int path_smooth_window = 5;
    float avoid_lookahead_distance_m = 0.5f;
};

struct ObstacleAvoidanceTrack {
    int id = -1;
    ObjectType type = ObjectType::UNKNOWN;
    int seen_streak = 0;
    int lost_frames = 0;
    bool confirmed = false;
    bool updated = false;
    bool observed_this_frame = false;
    cv::Point2f pixel_foot = cv::Point2f(0.0f, 0.0f);
    cv::Point2f world_pos_m = cv::Point2f(0.0f, 0.0f);
    cv::Point2f last_world_pos_m = cv::Point2f(0.0f, 0.0f);
    cv::Point2f last_observed_world_pos_m = cv::Point2f(0.0f, 0.0f);
    bool has_last_world_pos_m = false;
    bool has_last_observed_world_pos_m = false;
    float footprint_width_m = 0.0f;
    float footprint_length_m = 0.0f;
    float footprint_radius_m = 0.0f;
    bool has_footprint_m = false;
    float distance_to_vehicle_m = 0.0f;
    float point_distance_to_vehicle_m = 0.0f;
    float lateral_from_route_m = 0.0f;
    float center_delta_px = 0.0f;
    double last_timestamp_sec = 0.0;
};

struct GoldBiasRuntime {
    float stable_base_m = 0.0f;
    float target_m = 0.0f;
    float applied_m = 0.0f;
    GoldTransitionState transition_state = GoldTransitionState::IDLE;
    float pending_bias_m = 0.0f;
    int pending_confirm_count = 0;
    int no_executable_gold_frames = 0;
    int target_reached_frames = 0;
};

enum class PedestrianLaunchSource {
    NONE = 0,
    STOPPED_WAIT,
    DRIVING_SLOW
};

struct PathPlanningRuntime {
    ObstacleAvoidanceState obstacle_state = ObstacleAvoidanceState::Cruise;
    AvoidScenario avoid_scenario = AvoidScenario::None;
    std::vector<ObstacleAvoidanceTrack> obstacle_tracks;
    int obstacle_seen_hold_frames = 0;
    PedestrianBehaviorRuntime pedestrian_behavior;
    PedestrianLaunchSource pedestrian_launch_source =
        PedestrianLaunchSource::NONE;
    bool pedestrian_danger_corridor_servo_initialized = false;
    bool pedestrian_danger_corridor_servo_locked = false;
    float pedestrian_danger_corridor_servo_filtered = 0.0f;
    float obstacle_offset_current_m = 0.0f;
    float obstacle_offset_target_m = 0.0f;
    float vehicle_episode_max_required_offset_m = 0.0f;
    float ego_center_lateral_m = 0.0f;
    bool ego_center_lateral_valid = false;
    int obstacle_avoid_side = -1;
    int obstacle_primary_track_id = -1;
    ObjectType obstacle_primary_track_type = ObjectType::UNKNOWN;
    bool vehicle_avoid_side_locked = false;
    int vehicle_locked_avoid_side = 0;
    int vehicle_recover_hold_frames_remaining = 0;
    bool vehicle_recover_hold_started = false;
    int vehicle_recover_hold_age_frames = 0;
    bool vehicle_curve_avoid_boost_episode_active = false;
    bool vehicle_curve_avoid_boost_active_last_frame = false;
    int vehicle_curve_avoid_boost_track_id = -1;
    int vehicle_curve_avoid_boost_side = 0;
    int vehicle_curve_avoid_hold_remaining = 0;
    GoldBiasRuntime gold;
    double last_timestamp_sec = 0.0;
    bool has_last_timestamp = false;
};

struct PathPlanningDebug {
    LocalPlannerMode mode = LocalPlannerMode::Disabled;
    std::string gold_bias_reason = "not_evaluated";
    int gold_visible_count = 0;
    int gold_projectable_count = 0;
    int gold_best_near_capture_count = 0;
    int gold_best_capture_count = 0;
    float gold_best_capture_along_s_sum = 0.0f;
    std::string gold_no_capture_diag;
    bool gold_late_start_rejected = false;
    float gold_late_start_s_m = 0.0f;
    float gold_late_start_delta_m = 0.0f;
    float gold_best_bias_m = 0.0f;
    float gold_stable_base_m = 0.0f;
    std::string gold_transition_state_name = "IDLE";
    float gold_target_bias_m = 0.0f;
    float gold_applied_bias_m = 0.0f;
    float gold_effective_bias_m = 0.0f;
    float gold_pending_bias_m = 0.0f;
    int gold_pending_confirm_count = 0;
    int gold_no_executable_frames = 0;
    int gold_target_reached_frames = 0;
    bool gold_selection_frozen = false;
    int path_points = 0;
    float debug_base_path_x = 0.0f;
    float debug_final_path_x = 0.0f;
    float debug_path_y = 0.0f;
    int debug_dominant_obstacle_id = -1;
    float debug_dominant_obstacle_y = 0.0f;
    std::string debug_dominant_obstacle_type;
    std::string speed_risk_level;
    std::string speed_limit_reason;
    bool route_hold_active = false;
    float target_speed_mps = 1.0f;
    std::string speed_reason;
    PedestrianStopState pedestrian_stop_state = PedestrianStopState::DRIVING;
    std::string pedestrian_stop_state_name = "DRIVING";
    bool slow_candidate = false;
    bool emergency_candidate = false;
    bool blocked_by_path = false;
    bool path_usable = true;
    int launch_remaining = 0;
    std::string pedestrian_launch_source = "none";
    bool launch_paused_by_path = false;
    bool launch_paused_by_external_stop = false;
    std::string planner_speed_source = "cruise";
    std::string final_speed_source = "unknown";
    std::string active_block_reasons = "none";
    std::string state_transition_reason = "none";
    bool pedestrian_observed_this_frame = false;
    bool pedestrian_normal_strip_intersection = false;
    bool pedestrian_route_motion_approaching = false;
    bool pedestrian_route_motion_away = false;
    bool pedestrian_route_motion_unknown = true;
    float pedestrian_boundary_clearance_m = 0.0f;
    float pedestrian_motion_clearance_net_m = 0.0f;
    std::string pedestrian_motion_name = "unknown";
    std::string pedestrian_motion_reason = "no_observed_person";
    std::vector<PedestrianNormalStripGeometry> pedestrian_normal_strips;
    bool pedestrian_seen_inside = false;
    int pedestrian_clear_streak = 0;
    int pedestrian_stop_lost_streak = 0;
    float ped_stop_car_side_metric_m = 0.0f;
    std::string ped_stop_locked_car_side = "UNLOCKED";
    std::string ped_release_required_side = "ANY";
    bool pedestrian_exit_evidence = false;
    int pedestrian_exit_lost_grace = 0;
    int pedestrian_slow_observation_count = 0;
    int pedestrian_slow_lost_count = 0;
    int pedestrian_slow_away_streak = 0;
    std::string reason;
    ObstacleAvoidanceState obstacle_state = ObstacleAvoidanceState::Cruise;
    AvoidScenario avoid_scenario = AvoidScenario::None;
    int obstacle_count = 0;
    int obstacle_active_count = 0;
    int obstacle_primary_id = -1;
    std::string obstacle_primary_type;
    float obstacle_target_pixel_x = 0.0f;
    float obstacle_target_pixel_y = 0.0f;
    float obstacle_distance_m = 0.0f;
    float obstacle_point_distance_m = 0.0f;
    float obstacle_lateral_m = 0.0f;
    int obstacle_avoid_side = 0;
    bool vehicle_avoid_side_locked = false;
    int vehicle_locked_avoid_side = 0;
    bool vehicle_recover_hold_active = false;
    int vehicle_recover_hold_remaining = 0;
    int vehicle_recover_hold_age_frames = 0;
    int vehicle_recover_hold_max_lifetime_frames = 0;
    float vehicle_curve_metric = 0.0f;
    int vehicle_curve_direction = 0;
    bool vehicle_curve_avoid_boost_active = false;
    bool vehicle_opposite_steer_active = false;
    int vehicle_curve_avoid_hold_remaining = 0;
    float obstacle_offset_current_m = 0.0f;
    float obstacle_offset_target_m = 0.0f;
    float final_applied_path_offset_m = 0.0f;
    float realized_offset_m = 0.0f;
    float ego_center_lateral_m = 0.0f;
    float near_route_lateral_m = 0.0f;
    bool near_route_lateral_valid = false;
    int near_route_lateral_sample_count = 0;
    int raw_person_count = 0;
    int startup_launch_protect_remaining = 0;
    bool startup_launch_release_event = false;
    bool ped_emergency_suppressed_by_launch = false;
    bool launch_protect_speed_applied = false;
    int ped_emergency_id = -1;
    float ped_foot_x_m = 0.0f;
    float ped_foot_y_m = 0.0f;
    bool ped_emergency_hit_corridor = false;
    bool ped_emergency_hit = false;
    float ped_danger_corridor_near_width_m = 0.0f;
    float ped_danger_corridor_far_width_m = 0.0f;
    float ped_danger_corridor_length_m = 0.0f;
    float ped_danger_corridor_sample_step_m = 0.0f;
    float ped_danger_servo_raw = 0.0f;
    float ped_danger_servo_output = 0.0f;
    float ped_danger_servo_abs_used = 0.0f;
    bool ped_danger_servo_valid = false;
    bool ped_danger_servo_locked = false;
    std::string ped_danger_servo_source = "none";
    float ped_danger_turn_radius_m = 0.0f;
    int ped_danger_turn_sign = 0;
    bool ped_danger_straight = true;
    float ped_danger_along_s_m = 0.0f;
    float ped_danger_lateral_distance_m = 0.0f;
    float ped_emergency_allowed_width_m = 0.0f;
    PedestrianDangerCorridorGeometry pedestrian_danger_corridor;
    bool cluster_active = false;
    bool avoid_lookahead_active = false;
    float recommended_lookahead_m = 0.0f;
    float ego_speed_mps = 0.0f;
    float ego_yaw_deg = 0.0f;
    float min_static_obstacle_margin_m = 0.0f;
    float min_dynamic_obstacle_margin_m = 0.0f;
    float path_max_curvature = 0.0f;
    float path_min_turn_radius_m = 0.0f;
    float path_max_lateral_step_m = 0.0f;
    bool path_commit_active = false;
    std::string objects_world_trace;
    std::string base_path_world_trace;
    std::string final_path_world_trace;
};

struct PathPlanningResult {
    std::vector<cv::Point2f> waypoints;
    bool planning_enabled = false;
    float speed_factor = 1.0f;
    float target_speed_mps = 1.0f;
    bool pedestrian_emergency_stop = false;
    bool startup_launch_protect_active = false;
    PedestrianStopState pedestrian_stop_state = PedestrianStopState::DRIVING;
    bool pedestrian_stop_state_active = false;
    bool pedestrian_launch_committed = false;
    bool obstacle_avoidance_engaged = false;
    PathPlanningDebug debug;
};

struct ArrowGuideFrameData {
    ArrowGuideDebugInfo debug;
    std::vector<ArrowGuide> guides;
    size_t guides_before_branching = 0;
};

struct BranchPlanningResult {
    BranchDebugInfo debug;
    LineLossTraceFields line_loss;
    std::vector<ArrowBranch> visible_branches;
    RoadCorridorSet road_corridors;
};

struct PolyFit2 {
    float a = 0.0f;
    float b = 0.0f;
    float c = 0.0f;
    bool  valid = false;

    float eval(float y) const { return a * y * y + b * y + c; }
};

// ============================================================================
// Math utilities (inline)
// ============================================================================
inline bool isFiniteFloat(float v) {
    return std::isfinite(v);
}

inline bool isFinitePoint(const cv::Point2f& p) {
    return std::isfinite(p.x) && std::isfinite(p.y);
}

template <typename T>
inline T clampValue(T v, T lo, T hi) {
    return std::max(lo, std::min(v, hi));
}

inline float dotPoint(const cv::Point2f& a, const cv::Point2f& b) {
    return a.x * b.x + a.y * b.y;
}

inline bool sameGuidePoint(const ArrowGuide& a, const ArrowGuide& b) {
    return std::abs(a.pos.x - b.pos.x) < 1e-3f && std::abs(a.pos.y - b.pos.y) < 1e-3f;
}

// ============================================================================
// Key function declarations
// ============================================================================


// --- centerline building ---

std::vector<CenterlineSample> buildCenterlineSamplesFromPoly(
    const PolyFit2& fit,
    float y_max,
    float y_min,
    int frame_w,
    int frame_h);


// --- branch planning ---
BranchPlanningResult planVisibleBranchesForFrame(
    uint32_t frame_id,
    const ArrowGuideFrameData& arrow_frame,
    int arrow_detection_count,
    int frame_w,
    int frame_h,
    PathPlannerRuntimeState& runtime_state);

BranchCenterlineResult buildBranchCenterlineResult(
    const ArrowBranch& branch,
    int frame_w,
    int frame_h);

// Adapter-only: converts the branch planner's road-corridor output into the
// local planner route consumed by Car.cpp. It is not a search-line decision source.
std::vector<PlanningRoutePoint> buildPlanningRouteFromRoadCorridors(
    const RoadCorridorSet& corridor_set,
    int frame_w,
    int frame_h,
    RoadCorridorRole preferred_role = RoadCorridorRole::Accepted,
    bool allow_role_fallback = true);

std::vector<PlanningObject> collectPlanningObjects(
    class ObjectTracker& tracker,
    int frame_idx = -1,
    float gold_position_correction_scale = 1.0f);

std::vector<PlanningObject> collectRawPersonObjects(
    const std::vector<PredictResult>& detections,
    const cv::Size& frame_size,
    float max_forward_m);

// --- local path planning ---
PathPlanningResult planLocalPath(
    const PathPlanningInput& input,
    PathPlanningRuntime& runtime);

void finalizePedestrianBehavior(
    const PathPlanningInput& input,
    float current_servo_output,
    bool current_servo_valid,
    const std::string& current_servo_source,
    float last_applied_servo_output,
    bool last_applied_servo_valid,
    PathPlanningRuntime& runtime,
    PathPlanningResult& result);

PedestrianDangerCorridorGeometry buildPedestrianDangerCorridor(
    float servo_output,
    float near_width_m,
    float far_width_m,
    float length_m,
    float sample_step_m,
    float servo_straight_deadband,
    float servo_radius_max_abs);

PedestrianDangerHitResult evaluatePedestrianDangerCorridor(
    const PedestrianDangerCorridorGeometry& corridor,
    const cv::Point2f& person_world_m);

void updatePedestrianLaunchProgress(
    PathPlanningRuntime& runtime,
    PathPlanningResult& result,
    bool launch_motion_allowed,
    bool launch_paused_by_path,
    bool launch_paused_by_external_stop);

// Resets the complete pedestrian behavior runtime back to clean DRIVING.
// Used by obstacle runtime resets and the Car start-flag stop branch.
void resetPedestrianStopRuntime(PathPlanningRuntime& runtime,
                                const std::string& reason);

void resetGoldBiasRuntime(PathPlanningRuntime& runtime);

// --- vector math helpers ---
bool normalizeVector(const cv::Point2f& v, cv::Point2f& out_unit, float* out_length = nullptr);
bool estimateTailDirection(const std::vector<ArrowGuide>& guides,
                          cv::Point2f& out_unit,
                          float* out_length = nullptr,
                          size_t max_lookback = 3);
float angleBetweenUnitVectorsDeg(const cv::Point2f& a, const cv::Point2f& b);
float forwardScoreFromUnitDirection(const cv::Point2f& unit_dir);



// --- path utilities ---
bool isPathUsable(const std::vector<cv::Point2f>& path, int width, int height);
float turnTightness(const std::vector<cv::Point2f>& pts);
std::vector<cv::Point2f> smoothPath(const std::vector<cv::Point2f>& path, int base_window = 5);
std::vector<cv::Point2f> dedupPath(const std::vector<cv::Point2f>& path);
void drawPath(cv::Mat& frame,
              const std::vector<cv::Point2f>& path,
              const cv::Scalar& color,
              int thickness = 2);

// --- centerline helpers ---
float averageGuideHalfWidth(const std::vector<ArrowGuide>& guides);
// --- world coordinate helper ---
cv::Point2f pixelToWorldMeters(const cv::Point2f& pixel, bool* valid = nullptr);
cv::Point2f worldToPixelMeters(const cv::Point2f& world, bool* valid = nullptr);
float pointToVehicleBodyDistanceMeters(const cv::Point2f& world,
                                       float vehicle_front_m,
                                       float vehicle_rear_m,
                                       float vehicle_width_m);
cv::Point2f correctGoldWorldPosition(
    const cv::Point2f& apparent_world_m,
    float correction_scale);
