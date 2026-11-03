#include "../include/Car.hpp"
#include "../include/HostClock.hpp"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <memory>
#include <poll.h>
#include <sstream>
#include <sys/socket.h>
#include <sys/un.h>
#include <system_error>
#include <utility>
using namespace std;
using namespace cv;

namespace {

constexpr size_t kDebugLogFlushEveryLines = 32;

std::string sanitizePathComponent(const std::string& value) {
    std::string sanitized;
    sanitized.reserve(value.size());
    for (char ch : value) {
        const unsigned char uch = static_cast<unsigned char>(ch);
        if (std::isalnum(uch) || ch == '_' || ch == '-') {
            sanitized.push_back(ch);
        } else {
            sanitized.push_back('_');
        }
    }
    return sanitized;
}

std::string makeCurrentTimestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_tm{};
#if defined(_WIN32)
    localtime_s(&local_tm, &now_time);
#else
    localtime_r(&now_time, &local_tm);
#endif

    std::ostringstream ss;
    ss << std::put_time(&local_tm, "%Y%m%d_%H%M%S");
    return ss.str();
}

std::string getRunTimestamp() {
    const char* env_timestamp = std::getenv("DRAGON_RUN_TIMESTAMP");
    if (env_timestamp != nullptr && env_timestamp[0] != '\0') {
        const std::string sanitized = sanitizePathComponent(env_timestamp);
        if (!sanitized.empty()) {
            return sanitized;
        }
    }
    return makeCurrentTimestamp();
}

std::filesystem::path getWorkspaceSaveDir() {
    const char* override_dir = std::getenv("DRAGON_TIMESTAMPED_SAVE_DIR");
    if (override_dir != nullptr && override_dir[0] != '\0') {
        return std::filesystem::path(override_dir);
    }

    const char* home = std::getenv("HOME");
    const std::filesystem::path home_path =
        (home != nullptr && home[0] != '\0')
            ? std::filesystem::path(home)
            : std::filesystem::path("/home/orangepi");
    return home_path / "workspace" / "save";
}

std::string makeTimestampedSavePath(const std::filesystem::path& save_dir,
                                    const std::string& stem,
                                    const std::string& timestamp,
                                    const std::string& ext) {
    return (save_dir / (stem + "_" + timestamp + ext)).string();
}

void removeLogFileIfExists(const std::string& path, const char* tag) {
    if (path.empty()) {
        return;
    }

    errno = 0;
    if (std::remove(path.c_str()) == 0) {
        std::cout << "[Car] cleared previous " << tag << ": " << path << std::endl;
        return;
    }

    if (errno != ENOENT) {
        std::cerr << "[Car] failed to clear previous " << tag << ": " << path
                  << ", errno=" << errno << std::endl;
    }
}

void clearPreviousPositionLogs(const Params& params) {
    removeLogFileIfExists(params.localizationPayloadLogPath, "position forward log");
    removeLogFileIfExists(params.serialThreadCsvLogPath, "serial telemetry log");
    if (params.imuAdDebugLogEnable) {
        removeLogFileIfExists(params.imuAdDebugLogPath, "IMU AD debug log");
    }
}

bool ensureParentDirectory(const std::string& path) {
    if (path.empty()) {
        return false;
    }

    std::error_code ec;
    const std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (parent.empty() || std::filesystem::exists(parent, ec)) {
        return true;
    }
    return std::filesystem::create_directories(parent, ec) || std::filesystem::exists(parent, ec);
}

bool ensureDirectory(const std::filesystem::path& path) {
    if (path.empty()) {
        return false;
    }

    std::error_code ec;
    if (std::filesystem::exists(path, ec)) {
        return std::filesystem::is_directory(path, ec);
    }
    return std::filesystem::create_directories(path, ec) ||
           std::filesystem::is_directory(path, ec);
}

int requestRoadSignDecisionFromYoloe(int frame_id, const cv::Rect& box, int timeout_ms) {
    const char* socket_path = "/tmp/road_sign_decision.sock";
    constexpr int kRoadSignDecisionTimeoutMs = 10000;
    const int timeout = std::min(kRoadSignDecisionTimeoutMs, std::max(1, timeout_ms));
    const auto request_start = std::chrono::steady_clock::now();

    std::cout << "[RoadSignDecision] start request frame=" << frame_id
              << " 开始OCR识别/API调用请求"
              << " box=(" << box.x << "," << box.y << ","
              << box.x + box.width << "," << box.y + box.height << ")"
              << " timeout_ms=" << timeout << std::endl;

    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        std::cerr << "[RoadSignDecision] socket() failed, errno=" << errno << std::endl;
        return -1;
    }

    auto close_fd = [&]() {
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
    };

    timeval tv{};
    tv.tv_sec = timeout / 1000;
    tv.tv_usec = (timeout % 1000) * 1000;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", socket_path);

    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::cerr << "[RoadSignDecision] connect failed: " << socket_path
                  << ", errno=" << errno << std::endl;
        close_fd();
        return -1;
    }

    nlohmann::json req;
    req["request_id"] = frame_id;
    req["frame_id"] = frame_id;
    req["box"] = {box.x, box.y, box.x + box.width, box.y + box.height};
    req["timeout_ms"] = timeout;
    const std::string payload = req.dump() + "\n";

    const char* data = payload.data();
    size_t remaining = payload.size();
    while (remaining > 0) {
        const ssize_t sent = ::send(fd, data, remaining, 0);
        if (sent <= 0) {
            std::cerr << "[RoadSignDecision] send failed, errno=" << errno << std::endl;
            close_fd();
            return -1;
        }
        data += sent;
        remaining -= static_cast<size_t>(sent);
    }

    std::string response;
    char buffer[256];
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout);
    while (std::chrono::steady_clock::now() < deadline) {
        const auto left_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now()).count();
        pollfd pfd{};
        pfd.fd = fd;
        pfd.events = POLLIN;
        const int pret = ::poll(&pfd, 1, static_cast<int>(std::max<int64_t>(1, left_ms)));
        if (pret <= 0) {
            break;
        }
        const ssize_t n = ::recv(fd, buffer, sizeof(buffer), 0);
        if (n <= 0) {
            break;
        }
        response.append(buffer, buffer + n);
        if (response.find('\n') != std::string::npos) {
            break;
        }
    }

    close_fd();

    const size_t newline = response.find('\n');
    if (newline != std::string::npos) {
        response.resize(newline);
    }
    if (response.empty()) {
        std::cerr << "[RoadSignDecision] empty response" << std::endl;
        return -1;
    }

    try {
        const auto js = nlohmann::json::parse(response);
        const int64_t total_wait_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - request_start).count();
        std::cout << "[RoadSignDecision] response received wait_ms=" << total_wait_ms
                  << " raw=" << response << std::endl;
        const bool ok = js.value("ok", true);
        if (!ok) {
            std::cerr << "[RoadSignDecision] response not ready: "
                      << js.value("error", "ok_false")
                      << ", text=" << js.value("text", "") << std::endl;
            return -1;
        }
        const int decision = js.value("decision", -1);
        if (decision == 0 || decision == 1) {
            std::cout << "[RoadSignDecision] result=" << decision
                      << " api调用结果为=" << decision
                      << " text=" << js.value("text", "")
                      << " ocr_ms=" << js.value("ocr_ms", -1)
                      << " api_ms=" << js.value("api_ms", -1)
                      << " total_ms=" << js.value("total_ms", -1)
                      << " api_used=" << (js.value("api_used", false) ? 1 : 0)
                      << " source=" << js.value("semantic_source", "")
                      << " api_raw=" << js.value("api_raw", "")
                      << std::endl;
            return decision;
        }
    } catch (const std::exception& exc) {
        std::cerr << "[RoadSignDecision] invalid response: " << exc.what()
                  << ", raw=" << response << std::endl;
    }

    return -1;
}

bool openVideoWriterWithFallbacks(cv::VideoWriter& writer, const std::string& path,
                                  int fps, const cv::Size& frame_size)
{
    if (!ensureParentDirectory(path)) {
        std::cerr << "[VideoSave] failed to create parent directory for: "
                  << path << std::endl;
        return false;
    }

    const int codecs[] = {
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
        cv::VideoWriter::fourcc('M', 'P', '4', 'V')
    };

    for (const int fourcc : codecs) {
        if (writer.open(path, fourcc, fps, frame_size, true)) {
            return true;
        }
    }
    return false;
}

cv::Mat prepareFrameForVideoWrite(const cv::Mat& input, const cv::Size& target_size)
{
    if (input.empty()) return cv::Mat();

    cv::Mat output;
    if (input.size() != target_size) {
        cv::resize(input, output, target_size);
    } else {
        output = input.clone();
    }

    if (output.channels() != 3) {
        if (output.channels() == 4) {
            cv::cvtColor(output, output, cv::COLOR_RGBA2RGB);
        } else if (output.channels() == 1) {
            cv::cvtColor(output, output, cv::COLOR_GRAY2RGB);
        }
    }
    return output;
}

bool imageToWorldMeters(const cv::Point2f& pixel, cv::Point2f& world)
{
    static constexpr double h00 = -0.0005731584;
    static constexpr double h01 = 0.0000185282;
    static constexpr double h02 = 0.1769570949;
    static constexpr double h10 = 0.0000257719;
    static constexpr double h11 = 0.0000116754;
    static constexpr double h12 = -0.2010946013;
    static constexpr double h20 = -0.0000148160;
    static constexpr double h21 = -0.0039996717;
    static constexpr double h22 = 1.0000000000;

    const double den = h20 * pixel.x + h21 * pixel.y + h22;
    if (std::abs(den) < 1e-9) {
        return false;
    }

    const double wx = (h00 * pixel.x + h01 * pixel.y + h02) / den;
    const double wy = (h10 * pixel.x + h11 * pixel.y + h12) / den;
    if (!std::isfinite(wx) || !std::isfinite(wy)) {
        return false;
    }

    world = cv::Point2f(static_cast<float>(wx), static_cast<float>(wy));
    return true;
}

struct WorldDetectionMarker {
    cv::Point2f world;
    int type = -1;
};

struct WorldPathProjection {
    bool valid = false;
    cv::Point2f projected_world = cv::Point2f(0.0f, 0.0f);
    cv::Point2f line_projected_world = cv::Point2f(0.0f, 0.0f);
    cv::Point2f tangent_unit = cv::Point2f(0.0f, 1.0f);
    float path_distance_m = 0.0f;
    float along_path_s_m = 0.0f;
    float raw_projection_t = 0.0f;
};

struct GoldTerminalCandidate {
    bool valid = false;
    cv::Point2f world = cv::Point2f(0.0f, 0.0f);
    float distance_to_ego_m = 0.0f;
    float path_distance_m = 0.0f;
    float along_path_s_m = 0.0f;
};

struct GoldTerminalDebugState {
    bool candidate_valid = false;
    bool active = false;
    bool applied = false;
    bool entered = false;
    bool exited = false;
    bool derivative_reset = false;
    cv::Point2f target_world = cv::Point2f(0.0f, 0.0f);
    float distance_to_ego_m = 0.0f;
    float path_distance_m = 0.0f;
    float gold_along_path_s_m = 0.0f;
    float lookahead_along_path_s_m = 0.0f;
    float heading_error_deg = 0.0f;
    float filtered_derivative_deg = 0.0f;
    float servo_output = 0.0f;
    std::string reason = "not_evaluated";
};

struct RoadSignPoseDebugState {
    bool candidate_valid = false;
    bool active = false;
    bool applied = false;
    bool entered = false;
    bool exited = false;
    bool derivative_reset = false;
    cv::Point2f target_world = cv::Point2f(0.0f, 0.0f);
    cv::Point2f target_pixel = cv::Point2f(0.0f, 0.0f);
    float raw_center_x_px = 0.0f;
    float filtered_center_x_px = 0.0f;
    int area = 0;
    float heading_error_deg = 0.0f;
    float filtered_derivative_deg = 0.0f;
    float servo_output = 0.0f;
    std::string reason = "not_evaluated";
};

bool buildRoadSignVirtualTarget(float image_x_px,
                                float radius_m,
                                cv::Point2f& target_world,
                                cv::Point2f& target_pixel)
{
    if (!std::isfinite(image_x_px) || !std::isfinite(radius_m) || radius_m <= 0.0f) {
        return false;
    }

    bool found_lower = false;
    bool found_upper = false;
    cv::Point2f lower_world;
    cv::Point2f lower_pixel;
    cv::Point2f upper_world;
    cv::Point2f upper_pixel;
    cv::Point2f min_world;
    cv::Point2f min_pixel;
    cv::Point2f max_world;
    cv::Point2f max_pixel;
    bool have_min = false;
    bool have_max = false;

    constexpr int kSamples = 720;
    // Do not sample the exact +/-90 degree endpoints. At y == 0 the
    // ground-plane homography is close to its horizon singularity and those
    // points are not useful for an in-frame road-sign center anyway.
    for (int i = 1; i < kSamples; ++i) {
        const float heading_rad =
            -0.5f * static_cast<float>(CV_PI) +
            static_cast<float>(CV_PI) * static_cast<float>(i) /
                static_cast<float>(kSamples);
        const cv::Point2f world(radius_m * std::sin(heading_rad),
                                radius_m * std::cos(heading_rad));
        bool pixel_valid = false;
        const cv::Point2f pixel = worldToPixelMeters(world, &pixel_valid);
        if (!pixel_valid || !std::isfinite(pixel.x) || !std::isfinite(pixel.y)) {
            continue;
        }

        if (!have_min || pixel.x < min_pixel.x) {
            min_world = world;
            min_pixel = pixel;
            have_min = true;
        }
        if (!have_max || pixel.x > max_pixel.x) {
            max_world = world;
            max_pixel = pixel;
            have_max = true;
        }
        if (pixel.x <= image_x_px &&
            (!found_lower || pixel.x > lower_pixel.x)) {
            lower_world = world;
            lower_pixel = pixel;
            found_lower = true;
        }
        if (pixel.x >= image_x_px &&
            (!found_upper || pixel.x < upper_pixel.x)) {
            upper_world = world;
            upper_pixel = pixel;
            found_upper = true;
        }
    }

    if (!have_min || !have_max) {
        return false;
    }
    if (!found_lower) {
        target_world = min_world;
        target_pixel = min_pixel;
        return true;
    }
    if (!found_upper) {
        target_world = max_world;
        target_pixel = max_pixel;
        return true;
    }

    const float pixel_span = upper_pixel.x - lower_pixel.x;
    const float t = std::abs(pixel_span) > 1e-4f
        ? std::clamp((image_x_px - lower_pixel.x) / pixel_span, 0.0f, 1.0f)
        : 0.0f;
    target_world = lower_world + (upper_world - lower_world) * t;
    bool projected_valid = false;
    target_pixel = worldToPixelMeters(target_world, &projected_valid);
    return projected_valid &&
           std::isfinite(target_world.x) && std::isfinite(target_world.y) &&
           std::isfinite(target_pixel.x) && std::isfinite(target_pixel.y) &&
           target_world.y >= 0.0f;
}

struct PedestrianGoldDangerStrip {
    cv::Point2f center_world = cv::Point2f(0.0f, 0.0f);
    cv::Point2f pedestrian_world = cv::Point2f(0.0f, 0.0f);
    cv::Point2f tangent_unit = cv::Point2f(0.0f, 1.0f);
    cv::Point2f normal_unit = cv::Point2f(-1.0f, 0.0f);
    float half_width_along_road_m = 0.0f;
    float half_length_across_road_m = 0.40f;
};

struct PedestrianGoldDangerDebugState {
    float configured_half_width_m = 0.0f;
    int configured_hold_frames = 0;
    bool live = false;
    bool hold_active = false;
    int hold_elapsed_frames = 0;
    int hold_remaining_frames = 0;
    std::string source = "inactive";
    std::vector<PedestrianGoldDangerStrip> strips;
    std::vector<cv::Point2f> ignored_gold_world;
};

struct PedestrianGoldDangerHoldState {
    std::vector<PedestrianGoldDangerStrip> strips;
    int missing_frames = 0;
};

struct VirtualLookaheadState {
    bool has_point = false;
    bool locked = false;
    bool control_active = false;
    int low_arrow_frames = 0;
    cv::Point2f point = cv::Point2f(0.0f, 0.0f);
};

bool lastFiniteForwardPathPoint(const std::vector<cv::Point2f>& world_path,
                                cv::Point2f& out)
{
    for (auto it = world_path.rbegin(); it != world_path.rend(); ++it) {
        if (std::isfinite(it->x) && std::isfinite(it->y) && it->y > 0.0f) {
            out = *it;
            return true;
        }
    }
    return false;
}

bool detectionMarkerPixel(const PredictResult& result, cv::Point2f& pixel)
{
    if (result.width <= 0 || result.height <= 0) {
        return false;
    }

    const float center_x = static_cast<float>(result.x) + static_cast<float>(result.width) * 0.5f;
    const float center_y = static_cast<float>(result.y) + static_cast<float>(result.height) * 0.5f;
    const float foot_y = static_cast<float>(result.y) + static_cast<float>(result.height);

    switch (result.type) {
        case TYPE_ARROW:
        case TYPE_GOLD:
            pixel = cv::Point2f(center_x, center_y);
            return true;
        case TYPE_PEOPLE:
        case TYPE_CAR:
            pixel = cv::Point2f(center_x, foot_y);
            return true;
        default:
            return false;
    }
}

std::vector<WorldDetectionMarker> collectWorldDetectionMarkers(
    const std::vector<PredictResult>& results,
    float gold_position_correction_scale)
{
    std::vector<WorldDetectionMarker> markers;
    markers.reserve(results.size());

    for (const auto& result : results) {
        cv::Point2f pixel;
        if (!detectionMarkerPixel(result, pixel)) {
            continue;
        }

        cv::Point2f world;
        if (!imageToWorldMeters(pixel, world)) {
            continue;
        }
        if (result.type == TYPE_GOLD) {
            world = correctGoldWorldPosition(
                world,
                gold_position_correction_scale);
        }

        markers.push_back(WorldDetectionMarker{world, result.type});
    }

    return markers;
}

std::vector<cv::Point2f> transformPathToWorldMeters(const std::vector<cv::Point2f>& image_path,
                                                     int* invalid_count)
{
    std::vector<cv::Point2f> world_path;
    world_path.reserve(image_path.size());
    int invalid = 0;

    for (const auto& pixel : image_path) {
        cv::Point2f world;
        if (imageToWorldMeters(pixel, world)) {
            world_path.push_back(world);
        } else {
            invalid++;
        }
    }

    if (invalid_count != nullptr) {
        *invalid_count = invalid;
    }
    return world_path;
}

WorldPathProjection projectPointToWorldPath(
    const cv::Point2f& world,
    const std::vector<cv::Point2f>& world_path)
{
    WorldPathProjection best;
    if (!std::isfinite(world.x) || !std::isfinite(world.y) ||
        world_path.size() < 2) {
        return best;
    }

    float cumulative_s_m = 0.0f;
    float best_distance2 = std::numeric_limits<float>::infinity();
    for (size_t i = 0; i + 1 < world_path.size(); ++i) {
        const cv::Point2f& a = world_path[i];
        const cv::Point2f& b = world_path[i + 1];
        if (!std::isfinite(a.x) || !std::isfinite(a.y) ||
            !std::isfinite(b.x) || !std::isfinite(b.y)) {
            continue;
        }

        const cv::Point2f ab = b - a;
        const float segment_length2 = ab.dot(ab);
        if (!std::isfinite(segment_length2) || segment_length2 <= 1e-8f) {
            continue;
        }
        const float segment_length = std::sqrt(segment_length2);
        const float raw_t = (world - a).dot(ab) / segment_length2;
        const float clamped_t = std::clamp(raw_t, 0.0f, 1.0f);
        const cv::Point2f line_projected = a + ab * raw_t;
        const cv::Point2f projected = a + ab * clamped_t;
        const cv::Point2f delta = world - projected;
        const float distance2 = delta.dot(delta);
        if (std::isfinite(distance2) && distance2 < best_distance2) {
            best.valid = true;
            best.projected_world = projected;
            best.line_projected_world = line_projected;
            best.tangent_unit = ab * (1.0f / segment_length);
            best.path_distance_m = std::sqrt(std::max(0.0f, distance2));
            best.along_path_s_m = cumulative_s_m + clamped_t * segment_length;
            best.raw_projection_t = raw_t;
            best_distance2 = distance2;
        }
        cumulative_s_m += segment_length;
    }
    return best;
}

bool pointInsidePedestrianGoldDangerStrip(
    const cv::Point2f& world,
    const PedestrianGoldDangerStrip& strip)
{
    if (!std::isfinite(world.x) || !std::isfinite(world.y)) {
        return false;
    }
    const cv::Point2f delta = world - strip.center_world;
    const float along_road_m = delta.dot(strip.tangent_unit);
    const float across_road_m = delta.dot(strip.normal_unit);
    return std::abs(along_road_m) <=
               strip.half_width_along_road_m + 1e-4f &&
           std::abs(across_road_m) <=
               strip.half_length_across_road_m + 1e-4f;
}

std::vector<PedestrianGoldDangerStrip> buildPedestrianGoldDangerStrips(
    const std::vector<PlanningObject>& raw_person_objects,
    const std::vector<cv::Point2f>& raw_arrow_centerline_world,
    float half_width_along_road_m)
{
    static constexpr float kDangerStripHalfLengthAcrossRoadM = 0.40f;
    std::vector<PedestrianGoldDangerStrip> strips;
    if (half_width_along_road_m <= 0.0f ||
        raw_arrow_centerline_world.size() < 2) {
        return strips;
    }

    strips.reserve(raw_person_objects.size());
    for (const auto& person : raw_person_objects) {
        if (person.type != ObjectType::PERSON ||
            !person.observed_this_frame || !person.current_frame_visible ||
            !person.has_world_pos_m ||
            !std::isfinite(person.world_pos_m.x) ||
            !std::isfinite(person.world_pos_m.y)) {
            continue;
        }

        const WorldPathProjection projection = projectPointToWorldPath(
            person.world_pos_m,
            raw_arrow_centerline_world);
        if (!projection.valid) {
            continue;
        }

        PedestrianGoldDangerStrip strip;
        strip.center_world = projection.line_projected_world;
        strip.pedestrian_world = person.world_pos_m;
        strip.tangent_unit = projection.tangent_unit;
        strip.normal_unit = cv::Point2f(-projection.tangent_unit.y,
                                       projection.tangent_unit.x);
        strip.half_width_along_road_m = half_width_along_road_m;
        strip.half_length_across_road_m =
            kDangerStripHalfLengthAcrossRoadM;
        strips.push_back(strip);
    }
    return strips;
}

void updatePedestrianGoldDangerHold(
    const std::vector<PedestrianGoldDangerStrip>& current_strips,
    bool current_person_visible,
    int hold_frames,
    PedestrianGoldDangerHoldState& hold_state,
    PedestrianGoldDangerDebugState& debug)
{
    hold_frames = std::max(0, hold_frames);
    debug.configured_hold_frames = hold_frames;
    debug.live = !current_strips.empty();
    debug.hold_active = false;
    debug.hold_elapsed_frames = 0;
    debug.hold_remaining_frames = 0;

    if (!current_strips.empty()) {
        hold_state.strips = current_strips;
        hold_state.missing_frames = 0;
        debug.strips = current_strips;
        debug.source = "live";
        return;
    }

    // A newly visible person always invalidates the old held area. If the
    // current raw centerline is insufficient, no stale area is reused.
    if (current_person_visible) {
        hold_state = PedestrianGoldDangerHoldState{};
        debug.strips.clear();
        debug.source = "visible_no_current_strip";
        return;
    }

    if (!hold_state.strips.empty() &&
        hold_state.missing_frames < hold_frames) {
        hold_state.missing_frames++;
        debug.strips = hold_state.strips;
        debug.hold_active = true;
        debug.hold_elapsed_frames = hold_state.missing_frames;
        debug.hold_remaining_frames =
            std::max(0, hold_frames - hold_state.missing_frames);
        debug.source = "held_after_person_lost";
        return;
    }

    hold_state = PedestrianGoldDangerHoldState{};
    debug.strips.clear();
    debug.source = "expired_or_no_history";
}

void filterDangerStripGolds(
    const std::vector<PedestrianGoldDangerStrip>& strips,
    std::vector<PlanningObject>& planning_objects,
    std::vector<cv::Point2f>& ignored_gold_world)
{
    if (strips.empty() || planning_objects.empty()) {
        return;
    }

    std::vector<PlanningObject> filtered;
    filtered.reserve(planning_objects.size());
    for (const auto& object : planning_objects) {
        bool ignored = false;
        if (object.type == ObjectType::GOLD &&
            object.observed_this_frame && object.current_frame_visible &&
            object.has_world_pos_m &&
            std::isfinite(object.world_pos_m.x) &&
            std::isfinite(object.world_pos_m.y)) {
            ignored = std::any_of(
                strips.begin(),
                strips.end(),
                [&](const PedestrianGoldDangerStrip& strip) {
                    return pointInsidePedestrianGoldDangerStrip(
                        object.world_pos_m,
                        strip);
                });
        }
        if (ignored) {
            ignored_gold_world.push_back(object.world_pos_m);
        } else {
            filtered.push_back(object);
        }
    }
    planning_objects.swap(filtered);
}

GoldTerminalCandidate selectNearestTerminalGold(
    const std::vector<PlanningObject>& objects,
    const std::vector<cv::Point2f>& world_path,
    float path_tolerance_m)
{
    GoldTerminalCandidate selected;
    float nearest_distance_m = std::numeric_limits<float>::infinity();
    for (const auto& object : objects) {
        if (object.type != ObjectType::GOLD ||
            !object.observed_this_frame || !object.current_frame_visible ||
            !object.has_world_pos_m ||
            !std::isfinite(object.world_pos_m.x) ||
            !std::isfinite(object.world_pos_m.y) ||
            object.world_pos_m.y <= 0.0f) {
            continue;
        }

        const WorldPathProjection projection =
            projectPointToWorldPath(object.world_pos_m, world_path);
        if (!projection.valid ||
            projection.raw_projection_t < -1e-3f ||
            projection.raw_projection_t > 1.001f ||
            projection.path_distance_m > path_tolerance_m) {
            continue;
        }

        const float distance_m = std::hypot(object.world_pos_m.x,
                                            object.world_pos_m.y);
        if (!std::isfinite(distance_m) || distance_m >= nearest_distance_m) {
            continue;
        }
        selected.valid = true;
        selected.world = object.world_pos_m;
        selected.distance_to_ego_m = distance_m;
        selected.path_distance_m = projection.path_distance_m;
        selected.along_path_s_m = projection.along_path_s_m;
        nearest_distance_m = distance_m;
    }
    return selected;
}

bool selectPurePursuitTarget(const std::vector<cv::Point2f>& world_path,
                             float ahead_distance,
                             cv::Point2f& target,
                             float& target_distance)
{
    cv::Point2f fallback(0.0f, 0.0f);
    float fallback_distance = 0.0f;
    bool has_fallback = false;

    for (const auto& point : world_path) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            continue;
        }
        if (point.y <= 0.0f) {
            continue;
        }

        const float distance = std::hypot(point.x, point.y);
        fallback = point;
        fallback_distance = distance;
        has_fallback = true;

        if (distance >= ahead_distance) {
            target = point;
            target_distance = distance;
            return true;
        }
    }

    if (!has_fallback) {
        return false;
    }

    target = fallback;
    target_distance = fallback_distance;
    return target_distance > 1e-4f;
}

float computePurePursuitSteeringDeg(const cv::Point2f& target,
                                    float target_distance,
                                    float wheel_base)
{
    if (target_distance <= 1e-4f || wheel_base <= 0.0f) {
        return 0.0f;
    }

    const float theta = std::atan2(target.x, target.y);
    const float steering = std::atan2(2.0f * wheel_base * std::sin(theta),
                                      target_distance);
    return steering * 180.0f / static_cast<float>(CV_PI);
}

cv::Point worldToDebugPixel(const cv::Point2f& world,
                            const cv::Point2f& origin,
                            float scale)
{
    return cv::Point(static_cast<int>(std::round(origin.x + world.x * scale)),
                     static_cast<int>(std::round(origin.y - world.y * scale)));
}

bool isInsideWorldView(const cv::Point2f& world,
                       float forward_meters,
                       float rear_meters,
                       float side_meters)
{
    return std::isfinite(world.x) && std::isfinite(world.y) &&
           world.x >= -side_meters && world.x <= side_meters &&
           world.y >= -rear_meters && world.y <= forward_meters;
}

cv::Scalar worldMarkerColor(int type)
{
    switch (type) {
        case TYPE_ARROW:  return cv::Scalar(70, 230, 90);
        case TYPE_PEOPLE: return cv::Scalar(70, 90, 255);
        case TYPE_CAR:    return cv::Scalar(255, 150, 70);
        case TYPE_GOLD:   return cv::Scalar(0, 215, 255);
        default:          return cv::Scalar(230, 230, 230);
    }
}

const char* worldMarkerLabel(int type)
{
    switch (type) {
        case TYPE_ARROW:  return "A";
        case TYPE_PEOPLE: return "P";
        case TYPE_CAR:    return "C";
        case TYPE_GOLD:   return "G";
        default:          return "?";
    }
}

struct VehicleBodyBounds {
    float half_width_m = 0.11f;
    float front_m = 0.17f;
    float rear_m = 0.15f;
};

VehicleBodyBounds makeVehicleBodyBounds(float width_m, float front_m, float rear_m)
{
    VehicleBodyBounds bounds;
    bounds.half_width_m = 0.5f * std::max(0.05f, width_m);
    bounds.front_m = std::max(0.0f, front_m);
    bounds.rear_m = std::max(0.0f, rear_m);
    return bounds;
}

cv::Point2f closestPointOnVehicleBody(const cv::Point2f& world,
                                      const VehicleBodyBounds& body)
{
    const float closest_x =
        std::max(-body.half_width_m, std::min(body.half_width_m, world.x));
    const float closest_y =
        std::max(-body.rear_m, std::min(body.front_m, world.y));
    return cv::Point2f(closest_x, closest_y);
}

void drawVehicleBodyOutline(cv::Mat& view,
                            const cv::Point2f& origin,
                            float scale,
                            const VehicleBodyBounds& body)
{
    const float y_min = -body.rear_m;
    const float y_max = std::max(0.0f, body.front_m);
    if (y_max <= y_min + 1e-4f) {
        return;
    }

    std::vector<cv::Point> outline = {
        worldToDebugPixel(cv::Point2f(-body.half_width_m, y_min), origin, scale),
        worldToDebugPixel(cv::Point2f( body.half_width_m, y_min), origin, scale),
        worldToDebugPixel(cv::Point2f( body.half_width_m, y_max), origin, scale),
        worldToDebugPixel(cv::Point2f(-body.half_width_m, y_max), origin, scale)
    };
    cv::polylines(view, outline, true, cv::Scalar(240, 240, 240), 2, cv::LINE_AA);

    std::ostringstream label;
    label << std::fixed << std::setprecision(2)
          << "body x=+-" << body.half_width_m
          << " y=-" << body.rear_m << ".." << body.front_m;
    cv::putText(view,
                label.str(),
                worldToDebugPixel(cv::Point2f(body.half_width_m + 0.03f, y_max), origin, scale),
                cv::FONT_HERSHEY_SIMPLEX,
                0.42,
                cv::Scalar(230, 230, 230),
                1,
                cv::LINE_AA);
}

cv::Mat drawWorldPathView(const std::vector<cv::Point2f>& world_path,
                          const std::vector<WorldDetectionMarker>& detection_markers,
                          const std::vector<PlanningObject>& raw_person_objects,
                          bool has_target,
                          const cv::Point2f& target,
                          float target_distance,
                          float steering_angle,
                          float servo_output,
                          float applied_speed,
                          float pedestrian_slow_distance_m,
                          const PathPlanningDebug& pedestrian_debug,
                          int invalid_points,
                          int frame_id,
                          bool has_virtual_lookahead,
                          const cv::Point2f& virtual_lookahead,
                          bool virtual_lookahead_active,
                          bool virtual_lookahead_locked,
                          int arrow_count,
                          const GoldTerminalDebugState& gold_terminal,
                          const RoadSignPoseDebugState& road_sign_pose,
                          const PedestrianGoldDangerDebugState& pedestrian_gold_danger,
                          int width,
                          int height,
                          float vehicle_width_m,
                          float vehicle_front_m,
                          float vehicle_rear_m,
                          float forward_meters,
                          float side_meters)
{
    static constexpr float kWorldViewRearMeters = 0.35f;
    width = std::max(width, 320);
    height = std::max(height, 240);
    forward_meters = std::max(forward_meters, 0.5f);
    side_meters = std::max(side_meters, 0.2f);
    const float rear_meters = kWorldViewRearMeters;
    const auto& pedestrian_corridor =
        pedestrian_debug.pedestrian_danger_corridor;

    cv::Mat view(height, width, CV_8UC3, cv::Scalar(18, 22, 24));
    const float scale_x = (width * 0.5f - 24.0f) / side_meters;
    const float scale_y = (height - 54.0f) / (forward_meters + rear_meters);
    const float scale = std::max(1.0f, std::min(scale_x, scale_y));
    const cv::Point2f origin(
        width * 0.5f,
        height - 30.0f - rear_meters * scale);
    const VehicleBodyBounds vehicle_body =
        makeVehicleBodyBounds(vehicle_width_m, vehicle_front_m, vehicle_rear_m);

    for (float x = -side_meters; x <= side_meters + 1e-3f; x += 0.2f) {
        cv::line(view,
                 worldToDebugPixel(cv::Point2f(x, -rear_meters), origin, scale),
                 worldToDebugPixel(cv::Point2f(x, forward_meters), origin, scale),
                 cv::Scalar(44, 50, 52), 1);
    }
    for (float y = 0.0f; y <= forward_meters + 1e-3f; y += 0.2f) {
        cv::line(view,
                 worldToDebugPixel(cv::Point2f(-side_meters, y), origin, scale),
                 worldToDebugPixel(cv::Point2f(side_meters, y), origin, scale),
                 cv::Scalar(44, 50, 52), 1);
    }
    for (float y = -0.2f; y >= -rear_meters - 1e-3f; y -= 0.2f) {
        cv::line(view,
                 worldToDebugPixel(cv::Point2f(-side_meters, y), origin, scale),
                 worldToDebugPixel(cv::Point2f(side_meters, y), origin, scale),
                 cv::Scalar(44, 50, 52), 1);
    }

    cv::line(view,
             worldToDebugPixel(cv::Point2f(0.0f, 0.0f), origin, scale),
             worldToDebugPixel(cv::Point2f(0.0f, forward_meters), origin, scale),
             cv::Scalar(90, 110, 110), 2);
    cv::line(view,
             worldToDebugPixel(cv::Point2f(-side_meters, 0.0f), origin, scale),
             worldToDebugPixel(cv::Point2f(side_meters, 0.0f), origin, scale),
             cv::Scalar(90, 110, 110), 2);

    const cv::Point car = worldToDebugPixel(cv::Point2f(0.0f, 0.0f), origin, scale);
    drawVehicleBodyOutline(view, origin, scale, vehicle_body);

    if (!pedestrian_gold_danger.strips.empty()) {
        cv::Mat danger_overlay = view.clone();
        for (const auto& strip : pedestrian_gold_danger.strips) {
            const cv::Point2f along =
                strip.tangent_unit * strip.half_width_along_road_m;
            const cv::Point2f across =
                strip.normal_unit * strip.half_length_across_road_m;
            const std::vector<cv::Point> polygon = {
                worldToDebugPixel(strip.center_world - along - across,
                                  origin,
                                  scale),
                worldToDebugPixel(strip.center_world + along - across,
                                  origin,
                                  scale),
                worldToDebugPixel(strip.center_world + along + across,
                                  origin,
                                  scale),
                worldToDebugPixel(strip.center_world - along + across,
                                  origin,
                                  scale),
            };
            cv::fillConvexPoly(danger_overlay,
                               polygon,
                               cv::Scalar(0, 0, 255),
                               cv::LINE_AA);
        }
        cv::addWeighted(danger_overlay, 0.25, view, 0.75, 0.0, view);
        for (const auto& strip : pedestrian_gold_danger.strips) {
            const cv::Point2f along =
                strip.tangent_unit * strip.half_width_along_road_m;
            const cv::Point2f across =
                strip.normal_unit * strip.half_length_across_road_m;
            const std::vector<std::vector<cv::Point>> outline{{
                worldToDebugPixel(strip.center_world - along - across,
                                  origin,
                                  scale),
                worldToDebugPixel(strip.center_world + along - across,
                                  origin,
                                  scale),
                worldToDebugPixel(strip.center_world + along + across,
                                  origin,
                                  scale),
                worldToDebugPixel(strip.center_world - along + across,
                                  origin,
                                  scale),
            }};
            cv::polylines(view,
                          outline,
                          true,
                          cv::Scalar(0, 0, 255),
                          2,
                          cv::LINE_AA);
            cv::line(view,
                     worldToDebugPixel(strip.center_world - across,
                                       origin,
                                       scale),
                     worldToDebugPixel(strip.center_world + across,
                                       origin,
                                       scale),
                     cv::Scalar(0, 0, 255),
                     1,
                     cv::LINE_AA);
            std::ostringstream danger_label;
            danger_label << (pedestrian_gold_danger.hold_active
                                 ? "PED DANGER HOLD "
                                 : "PED DANGER LIVE ");
            if (pedestrian_gold_danger.hold_active) {
                danger_label << pedestrian_gold_danger.hold_remaining_frames;
            }
            cv::putText(view,
                        danger_label.str(),
                        worldToDebugPixel(strip.center_world, origin, scale) +
                            cv::Point(8, -8),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.38,
                        cv::Scalar(0, 0, 255),
                        1,
                        cv::LINE_AA);
        }
    }

    if (pedestrian_corridor.valid &&
        pedestrian_corridor.polygon_world.size() >= 3) {
        std::vector<cv::Point> corridor_polygon;
        corridor_polygon.reserve(pedestrian_corridor.polygon_world.size());
        for (const auto& point : pedestrian_corridor.polygon_world) {
            corridor_polygon.push_back(worldToDebugPixel(point, origin, scale));
        }
        cv::Mat corridor_overlay = view.clone();
        cv::fillPoly(corridor_overlay,
                     std::vector<std::vector<cv::Point>>{corridor_polygon},
                     cv::Scalar(30, 50, 220),
                     cv::LINE_AA);
        cv::addWeighted(corridor_overlay, 0.22, view, 0.78, 0.0, view);
        cv::polylines(view,
                      std::vector<std::vector<cv::Point>>{corridor_polygon},
                      true,
                      cv::Scalar(70, 90, 245),
                      2,
                      cv::LINE_AA);

        std::vector<cv::Point> corridor_centerline;
        corridor_centerline.reserve(
            pedestrian_corridor.centerline_world.size());
        for (const auto& point : pedestrian_corridor.centerline_world) {
            corridor_centerline.push_back(
                worldToDebugPixel(point, origin, scale));
        }
        if (corridor_centerline.size() >= 2) {
            cv::polylines(view,
                          std::vector<std::vector<cv::Point>>{
                              corridor_centerline},
                          false,
                          cv::Scalar(80, 210, 255),
                          1,
                          cv::LINE_AA);
        }
    }

    for (const auto& strip : pedestrian_debug.pedestrian_normal_strips) {
        if (!strip.valid) {
            continue;
        }
        const cv::Point2f along =
            strip.tangent_unit * strip.half_width_along_road_m;
        const cv::Point2f across =
            strip.normal_unit * strip.half_length_across_road_m;
        const std::vector<std::vector<cv::Point>> outline{{
            worldToDebugPixel(strip.center_world - along - across,
                              origin,
                              scale),
            worldToDebugPixel(strip.center_world + along - across,
                              origin,
                              scale),
            worldToDebugPixel(strip.center_world + along + across,
                              origin,
                              scale),
            worldToDebugPixel(strip.center_world - along + across,
                              origin,
                              scale),
        }};
        const cv::Scalar strip_color =
            strip.forward_edge_intersects_corridor
                ? cv::Scalar(0, 190, 255)
                : cv::Scalar(255, 210, 60);
        cv::polylines(view,
                      outline,
                      true,
                      strip_color,
                      2,
                      cv::LINE_AA);
        cv::line(view,
                 worldToDebugPixel(strip.forward_edge_left_world,
                                   origin,
                                   scale),
                 worldToDebugPixel(strip.forward_edge_right_world,
                                   origin,
                                   scale),
                 strip.forward_edge_intersects_corridor
                     ? cv::Scalar(0, 0, 255)
                     : strip_color,
                 3,
                 cv::LINE_AA);
        std::ostringstream strip_label;
        strip_label << "NSTRIP " << strip.track_id;
        cv::putText(view,
                    strip_label.str(),
                    worldToDebugPixel(strip.pedestrian_world, origin, scale) +
                        cv::Point(8, 12),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.34,
                    strip_color,
                    1,
                    cv::LINE_AA);
    }

    for (const auto& person : raw_person_objects) {
        if (person.type != ObjectType::PERSON ||
            !person.has_world_pos_m ||
            !person.observed_this_frame ||
            !std::isfinite(person.world_pos_m.x) ||
            !std::isfinite(person.world_pos_m.y) ||
            !std::isfinite(person.distance_to_self_m) ||
            !isInsideWorldView(person.world_pos_m,
                               forward_meters,
                               rear_meters,
                               side_meters)) {
            continue;
        }

        const cv::Point person_pixel =
            worldToDebugPixel(person.world_pos_m, origin, scale);
        cv::line(view,
                 car,
                 person_pixel,
                 cv::Scalar(115, 135, 235),
                 1,
                 cv::LINE_AA);

        const cv::Point2f midpoint_world = person.world_pos_m * 0.5f;
        std::ostringstream pedestrian_distance_label;
        pedestrian_distance_label << std::fixed << std::setprecision(2)
                                  << "Ped " << person.distance_to_self_m << "m";
        cv::putText(view,
                    pedestrian_distance_label.str(),
                    worldToDebugPixel(midpoint_world, origin, scale) + cv::Point(4, -4),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.36,
                    cv::Scalar(145, 165, 255),
                    1,
                    cv::LINE_AA);
    }

    for (size_t i = 1; i < world_path.size(); ++i) {
        const auto& a = world_path[i - 1];
        const auto& b = world_path[i];
        if (!std::isfinite(a.x) || !std::isfinite(a.y) ||
            !std::isfinite(b.x) || !std::isfinite(b.y)) {
            continue;
        }
        cv::line(view, worldToDebugPixel(a, origin, scale),
                 worldToDebugPixel(b, origin, scale),
                 cv::Scalar(40, 220, 90), 2, cv::LINE_AA);
    }

    for (const auto& point : world_path) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            continue;
        }
        cv::circle(view, worldToDebugPixel(point, origin, scale), 2,
                   cv::Scalar(80, 255, 130), -1, cv::LINE_AA);
    }

    for (const auto& marker : detection_markers) {
        if (!isInsideWorldView(marker.world,
                               forward_meters,
                               rear_meters,
                               side_meters)) {
            continue;
        }

        const cv::Point marker_pixel = worldToDebugPixel(marker.world, origin, scale);
        const cv::Scalar color = worldMarkerColor(marker.type);
        cv::circle(view, marker_pixel, 5, color, -1, cv::LINE_AA);
        cv::circle(view, marker_pixel, 8, color, 1, cv::LINE_AA);

        if (marker.type == TYPE_CAR) {
            const cv::Point2f closest =
                closestPointOnVehicleBody(marker.world, vehicle_body);
            const float distance =
                std::hypot(marker.world.x - closest.x, marker.world.y - closest.y);
            if (std::isfinite(distance)) {
                const cv::Point closest_pixel =
                    worldToDebugPixel(closest, origin, scale);
                cv::line(view,
                         marker_pixel,
                         closest_pixel,
                         cv::Scalar(80, 235, 255),
                         1,
                         cv::LINE_AA);
                cv::circle(view, closest_pixel, 4, cv::Scalar(80, 235, 255), -1, cv::LINE_AA);

                const cv::Point2f mid_world((marker.world.x + closest.x) * 0.5f,
                                            (marker.world.y + closest.y) * 0.5f);
                std::ostringstream dist_label;
                dist_label << std::fixed << std::setprecision(2)
                           << "C-body "
                           << distance << "m";
                cv::putText(view,
                            dist_label.str(),
                            worldToDebugPixel(mid_world, origin, scale) + cv::Point(6, -6),
                            cv::FONT_HERSHEY_SIMPLEX,
                            0.42,
                            cv::Scalar(80, 235, 255),
                            1,
                            cv::LINE_AA);
            }
        }

        cv::putText(view,
                    worldMarkerLabel(marker.type),
                    marker_pixel + cv::Point(8, -6),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv::LINE_AA);
    }

    for (const auto& ignored_gold :
         pedestrian_gold_danger.ignored_gold_world) {
        if (!isInsideWorldView(ignored_gold,
                               forward_meters,
                               rear_meters,
                               side_meters)) {
            continue;
        }
        const cv::Point ignored_pixel =
            worldToDebugPixel(ignored_gold, origin, scale);
        const cv::Scalar ignored_red(0, 0, 255);
        cv::circle(view, ignored_pixel, 9, ignored_red, -1, cv::LINE_AA);
        cv::circle(view,
                   ignored_pixel,
                   14,
                   cv::Scalar(255, 255, 255),
                   2,
                   cv::LINE_AA);
        cv::drawMarker(view,
                       ignored_pixel,
                       cv::Scalar(255, 255, 255),
                       cv::MARKER_TILTED_CROSS,
                       16,
                       2,
                       cv::LINE_AA);
        cv::putText(view,
                    "G IGNORED",
                    ignored_pixel + cv::Point(12, -12),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.46,
                    ignored_red,
                    2,
                    cv::LINE_AA);
    }

    if (gold_terminal.applied && gold_terminal.candidate_valid &&
        isInsideWorldView(gold_terminal.target_world,
                          forward_meters,
                          rear_meters,
                          side_meters)) {
        const cv::Point terminal_gold_pixel =
            worldToDebugPixel(gold_terminal.target_world, origin, scale);
        const cv::Scalar terminal_green(40, 255, 40);
        cv::line(view, car, terminal_gold_pixel, terminal_green, 2, cv::LINE_AA);
        cv::circle(view, terminal_gold_pixel, 8, terminal_green, -1, cv::LINE_AA);
        cv::circle(view, terminal_gold_pixel, 13, terminal_green, 2, cv::LINE_AA);
        cv::drawMarker(view,
                       terminal_gold_pixel,
                       cv::Scalar(255, 255, 255),
                       cv::MARKER_CROSS,
                       12,
                       2,
                       cv::LINE_AA);
        cv::putText(view,
                    "G TERMINAL",
                    terminal_gold_pixel + cv::Point(12, -12),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.48,
                    terminal_green,
                    2,
                    cv::LINE_AA);
    }

    if (road_sign_pose.applied &&
        road_sign_pose.candidate_valid &&
        isInsideWorldView(road_sign_pose.target_world,
                          forward_meters,
                          rear_meters,
                          side_meters)) {
        const cv::Point road_sign_virtual_pixel =
            worldToDebugPixel(road_sign_pose.target_world, origin, scale);
        const cv::Scalar road_sign_color(255, 80, 255);
        cv::line(view, car, road_sign_virtual_pixel,
                 road_sign_color, 2, cv::LINE_AA);
        cv::circle(view, road_sign_virtual_pixel, 8,
                   road_sign_color, -1, cv::LINE_AA);
        cv::circle(view, road_sign_virtual_pixel, 13,
                   road_sign_color, 2, cv::LINE_AA);
        cv::drawMarker(view,
                       road_sign_virtual_pixel,
                       cv::Scalar(255, 255, 255),
                       cv::MARKER_DIAMOND,
                       16,
                       2,
                       cv::LINE_AA);
        cv::putText(view,
                    "SIGN VIRTUAL ACTIVE",
                    road_sign_virtual_pixel + cv::Point(12, -12),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.46,
                    road_sign_color,
                    1,
                    cv::LINE_AA);
    }

    if (has_target) {
        cv::Point target_pixel = worldToDebugPixel(target, origin, scale);
        cv::circle(view, target_pixel, 7, cv::Scalar(40, 60, 255), -1, cv::LINE_AA);
        cv::line(view, car, target_pixel, cv::Scalar(70, 120, 255), 1, cv::LINE_AA);
    }

    if (has_virtual_lookahead &&
        std::isfinite(virtual_lookahead.x) &&
        std::isfinite(virtual_lookahead.y)) {
        const cv::Point virtual_pixel =
            worldToDebugPixel(virtual_lookahead, origin, scale);
        const cv::Scalar virtual_color =
            virtual_lookahead_active ? cv::Scalar(0, 180, 255)
                                     : cv::Scalar(255, 80, 210);
        cv::drawMarker(view, virtual_pixel, virtual_color,
                       cv::MARKER_DIAMOND, 18, 2, cv::LINE_AA);
        cv::line(view, car, virtual_pixel, virtual_color, 1, cv::LINE_AA);
        cv::putText(view,
                    virtual_lookahead_active ? "VirtualLA ACTIVE" : "VirtualLA",
                    virtual_pixel + cv::Point(8, -8),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.45,
                    virtual_color,
                    1,
                    cv::LINE_AA);
    }

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3);

    auto drawRightStatusLine = [&](const std::string& text,
                                   int y,
                                   const cv::Scalar& color) {
        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(
            text, cv::FONT_HERSHEY_SIMPLEX, 0.38, 1, &baseline);
        cv::putText(view,
                    text,
                    cv::Point(std::max(4, width - text_size.width - 12), y),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.38,
                    color,
                    1,
                    cv::LINE_AA);
    };
    std::ostringstream right_status;
    right_status << std::fixed << std::setprecision(3)
                 << "NearRouteX: " << std::showpos
                 << pedestrian_debug.near_route_lateral_m << std::noshowpos
                 << "m valid="
                 << (pedestrian_debug.near_route_lateral_valid ? 1 : 0)
                 << " n=" << pedestrian_debug.near_route_lateral_sample_count;
    drawRightStatusLine(right_status.str(), 24, cv::Scalar(80, 255, 255));

    right_status.str("");
    right_status.clear();
    right_status << std::fixed << std::setprecision(3)
                 << "EgoLatOld:  " << std::showpos
                 << pedestrian_debug.ego_center_lateral_m << std::noshowpos
                 << "m";
    drawRightStatusLine(right_status.str(), 44, cv::Scalar(180, 180, 255));

    right_status.str("");
    right_status.clear();
    right_status << "StopSide:   "
                 << (pedestrian_debug.ped_stop_locked_car_side.empty()
                         ? "UNLOCKED"
                         : pedestrian_debug.ped_stop_locked_car_side);
    drawRightStatusLine(right_status.str(), 64, cv::Scalar(255, 220, 120));

    ss << "Frame " << frame_id
       << " world_pts=" << world_path.size()
       << " invalid=" << invalid_points;
    cv::putText(view, ss.str(), cv::Point(12, 24),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(240, 240, 240), 1);

    ss.str("");
    ss.clear();
    if (has_target) {
        ss << "target=(" << target.x << "," << target.y << ")"
           << " Ld=" << target_distance;
    } else {
        ss << "target=N/A";
    }
    cv::putText(view, ss.str(), cv::Point(12, 50),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(120, 220, 255), 1);

    ss.str("");
    ss.clear();
    ss << "steer=" << steering_angle << " servo=" << servo_output;
    cv::putText(view, ss.str(), cv::Point(12, 76),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(80, 255, 160), 1);

    ss.str("");
    ss.clear();
    ss << "virtualLA arrows=" << arrow_count
       << " active=" << (virtual_lookahead_active ? 1 : 0)
       << " locked=" << (virtual_lookahead_locked ? 1 : 0);
    cv::putText(view, ss.str(), cv::Point(12, 102),
                cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(255, 120, 230), 1);

    ss.str("");
    ss.clear();
    const std::string pedestrian_state_name =
        pedestrian_debug.pedestrian_stop_state_name.empty()
            ? "UNKNOWN"
            : pedestrian_debug.pedestrian_stop_state_name;
    std::string pedestrian_motion_name =
        pedestrian_debug.pedestrian_motion_name.empty()
            ? "UNKNOWN"
            : pedestrian_debug.pedestrian_motion_name;
    std::transform(
        pedestrian_motion_name.begin(),
        pedestrian_motion_name.end(),
        pedestrian_motion_name.begin(),
        [](unsigned char ch) {
            return static_cast<char>(std::toupper(ch));
        });
    if (pedestrian_motion_name != "APPROACHING" &&
        pedestrian_motion_name != "MOVING_AWAY") {
        pedestrian_motion_name = "UNKNOWN";
    }
    const cv::Scalar pedestrian_motion_color =
        pedestrian_motion_name == "APPROACHING"
            ? cv::Scalar(0, 0, 255)
            : (pedestrian_motion_name == "MOVING_AWAY"
                   ? cv::Scalar(0, 255, 0)
                   : cv::Scalar(255, 255, 255));

    ss << "PED: " << pedestrian_state_name
       << " SpeedCmd=" << std::fixed << std::setprecision(2) << applied_speed
       << " SlowTh=" << pedestrian_slow_distance_m;
    cv::putText(view,
                ss.str(),
                cv::Point(12, 126),
                cv::FONT_HERSHEY_SIMPLEX,
                0.36,
                cv::Scalar(120, 205, 255),
                1,
                cv::LINE_AA);

    ss.str("");
    ss.clear();
    ss << "MOTION: " << pedestrian_motion_name << " (";
    if (pedestrian_debug.pedestrian_motion_clearance_net_m >= 0.0f) {
        ss << "+";
    }
    ss << std::fixed << std::setprecision(3)
       << pedestrian_debug.pedestrian_motion_clearance_net_m << "m)";
    cv::putText(view,
                ss.str(),
                cv::Point(12, 144),
                cv::FONT_HERSHEY_SIMPLEX,
                0.36,
                pedestrian_motion_color,
                1,
                cv::LINE_AA);

    ss.str("");
    ss.clear();
    ss << "CLEARANCE: " << std::fixed << std::setprecision(3)
       << pedestrian_debug.pedestrian_boundary_clearance_m << "m";
    cv::putText(view,
                ss.str(),
                cv::Point(12, 162),
                cv::FONT_HERSHEY_SIMPLEX,
                0.36,
                cv::Scalar(255, 255, 255),
                1,
                cv::LINE_AA);

    ss.str("");
    ss.clear();
    ss << " PedCorr(N/F/L)="
       << pedestrian_debug.ped_danger_corridor_near_width_m << "/"
       << pedestrian_debug.ped_danger_corridor_far_width_m << "/"
       << pedestrian_debug.ped_danger_corridor_length_m
       << " Servo=" << pedestrian_debug.ped_danger_servo_output
       << " Raw=" << pedestrian_debug.ped_danger_servo_raw
       << " Lock=" << (pedestrian_debug.ped_danger_servo_locked ? "Y" : "N")
       << " Rad="
       << (pedestrian_debug.ped_danger_straight
               ? 0.0f
               : pedestrian_debug.ped_danger_turn_radius_m)
       << " Hit=" << (pedestrian_debug.emergency_candidate ? "Y" : "N");
     cv::putText(view, ss.str(), cv::Point(12, 182),
                 cv::FONT_HERSHEY_SIMPLEX, 0.36, cv::Scalar(120, 205, 255), 1,
                 cv::LINE_AA);

    ss.str("");
    ss.clear();
    const char* terminal_state = gold_terminal.entered && gold_terminal.applied
                                     ? "ENTER"
                                     : (gold_terminal.applied
                                            ? "ACTIVE"
                                            : (gold_terminal.active
                                                   ? "SUPPRESSED"
                                                   : (gold_terminal.exited
                                                          ? "EXIT"
                                                          : "INACTIVE")));
    ss << "GoldTerminalPD=" << terminal_state
       << " reason=" << gold_terminal.reason;
    const cv::Scalar terminal_text_color =
        gold_terminal.applied ? cv::Scalar(40, 255, 40)
                             : (gold_terminal.exited ? cv::Scalar(0, 180, 255)
                                                     : cv::Scalar(170, 170, 170));
    cv::putText(view, ss.str(), cv::Point(12, 202),
                cv::FONT_HERSHEY_SIMPLEX, 0.40, terminal_text_color, 1,
                cv::LINE_AA);

    ss.str("");
    ss.clear();
    ss << std::fixed << std::setprecision(2);
    if (gold_terminal.candidate_valid) {
        ss << "G=(" << gold_terminal.target_world.x << ","
           << gold_terminal.target_world.y << ") d="
           << gold_terminal.distance_to_ego_m << " path="
           << gold_terminal.path_distance_m << " s="
           << gold_terminal.gold_along_path_s_m << "/"
           << gold_terminal.lookahead_along_path_s_m;
    } else {
        ss << "G candidate=N/A";
    }
    cv::putText(view, ss.str(), cv::Point(12, 222),
                cv::FONT_HERSHEY_SIMPLEX, 0.36, terminal_text_color, 1,
                cv::LINE_AA);

    ss.str("");
    ss.clear();
    ss << std::fixed << std::setprecision(2)
       << "Gerr=" << gold_terminal.heading_error_deg
       << " D=" << gold_terminal.filtered_derivative_deg
       << " servo=" << gold_terminal.servo_output
       << " Dreset=" << (gold_terminal.derivative_reset ? 1 : 0);
    cv::putText(view, ss.str(), cv::Point(12, 242),
                 cv::FONT_HERSHEY_SIMPLEX, 0.36, terminal_text_color, 1,
                 cv::LINE_AA);

    ss.str("");
    ss.clear();
    const char* road_sign_pose_state = road_sign_pose.entered && road_sign_pose.applied
        ? "ENTER"
        : (road_sign_pose.applied
               ? "ACTIVE"
               : (road_sign_pose.exited ? "EXIT" : "INACTIVE"));
    ss << std::fixed << std::setprecision(2)
       << "SignPosePD=" << road_sign_pose_state
       << " area=" << road_sign_pose.area;
    if (road_sign_pose.applied) {
        ss << " target=(" << road_sign_pose.target_world.x
           << "," << road_sign_pose.target_world.y << ")"
           << " err=" << road_sign_pose.heading_error_deg
           << " servo=" << road_sign_pose.servo_output;
    }
    ss << " reason=" << road_sign_pose.reason;
    cv::putText(view, ss.str(), cv::Point(12, 262),
                cv::FONT_HERSHEY_SIMPLEX, 0.36,
                road_sign_pose.applied ? cv::Scalar(255, 80, 255)
                                       : cv::Scalar(170, 170, 170),
                1, cv::LINE_AA);

    ss.str("");
    ss.clear();
    ss << std::fixed << std::setprecision(2)
       << "PedGoldDanger strips=" << pedestrian_gold_danger.strips.size()
       << " ignored=" << pedestrian_gold_danger.ignored_gold_world.size()
       << " x=" << pedestrian_gold_danger.configured_half_width_m
       << " length=0.80m"
       << " src=" << pedestrian_gold_danger.source
       << " hold=" << pedestrian_gold_danger.hold_elapsed_frames
       << "/" << pedestrian_gold_danger.configured_hold_frames
       << " rem=" << pedestrian_gold_danger.hold_remaining_frames;
    cv::putText(view, ss.str(), cv::Point(12, 282),
                cv::FONT_HERSHEY_SIMPLEX, 0.36, cv::Scalar(0, 80, 255), 1,
                cv::LINE_AA);

    return view;
}

} // namespace

void Car::BlockingQueueClear(BlockingQueue<DetectionPacket>& BQ)
{
    while(BQ.Size()>0)
    {
        BQ.Take();
    }
}

void Car::BlockingQueueClear(BlockingQueue<VideoSavePacket>& BQ)
{
    while(BQ.Size()>0)
    {
        BQ.Take();
    }
}
void Car::Car_init()
{
    // #################### params loaded #####################
    loadSuccess = loadParams(paramsPath); //!!!!!!!!!!!!
    if (!loadSuccess)
    {
        cerr << "load failed" << endl;
        exit(-1);
    }
    else
    {
        cout << "<<<-------------------?------------------------>>>" << endl;
        cout << "paramsPath: " << paramsPath << endl
             << "modelPath: " << modelPath << endl
             << "debug: "<<params.debug<<endl
             << "save AI: "<<params.saveAI<<endl
             << "timestamped save: " << params.timestampedWorkspaceSaveEnable << endl
             << "Serial_dev: " << Serial_dev << endl
             << "modelPath: " << modelPath << endl;
        // cout << "<<<-------------------------------------------------- ---->>>" << endl;
    }

    clearPreviousPositionLogs(params);

    //////////////////////////// AI init //////////////////////////////////
    std::cout << "[AI] local RKNN init disabled, use shared-memory reader." << std::endl;
    EnableAI = params.AIEnable;
    /////////////////////////////////////////////////////////////////////////////


    ///////////////////////////// serial init //////////////////////////////////
    int check = -1;
    check = uart.Serial_init(Serial_dev);
    if(check < 0)
    {
        if (params.skipMainSerialCheck)
        {
            std::cerr << "[WARN] cannot open main serial port: " << Serial_dev
                      << " , continue because skipMainSerialCheck=true" << std::endl;
        }
        else
        {
            std::cerr << "cannot open port!!!" << std::endl;
            exit(-1);
        }
    }
    else
        cout << "open serial success" << endl;
    uart.blind = false; //false

    {
        if (params.localizationSystemEnable) {
            LocalizationBridge::Config cfg;
            cfg.udp_listen_ip = params.localizationUdpListenIp;
            cfg.udp_listen_port = params.localizationUdpListenPort;
            cfg.udp_target_ip = params.localizationUdpTargetIp.empty()
                ? "192.168.137.137"
                : params.localizationUdpTargetIp;
            cfg.udp_target_port = params.localizationUdpTargetPort;
            cfg.enable_forward = params.localizationSystemEnable;
            cfg.enable_payload_log = params.localizationPayloadLogEnable;
            cfg.payload_log_path = params.localizationPayloadLogPath;
            cfg.use_fixed_z = true;
            cfg.fixed_z_value = 0.15f;
            cfg.yaw_only_mode = true;

            localizationBridge.setConfig(cfg);

            std::cout << "[Car][LocalizationBridge] Init requested. listen="
                      << cfg.udp_listen_ip << ":" << cfg.udp_listen_port
                      << " forward=" << cfg.udp_target_ip << ":" << cfg.udp_target_port
                      << " log=" << (cfg.enable_payload_log ? cfg.payload_log_path : "OFF")
                      << " fixed_z=" << cfg.fixed_z_value
                      << " yaw_source=lower_telemetry"
                      << std::endl;

            localizationBridge.setLowerYawProvider([this](float& yaw_deg) {
                LowerTelemetry telemetry;
                if (this->getLowerTelemetry(telemetry)) {
                    yaw_deg = telemetry.yaw_deg;
                    return true;
                }
                yaw_deg = 0.0f;
                return false;
            });

            if (!localizationBridge.start()) {
                std::cerr << "[Car][LocalizationBridge] start failed, continue without localization forwarding." << std::endl;
            } else {
                std::cout << "[Car][LocalizationBridge] start success, isRunning="
                          << localizationBridge.isRunning() << std::endl;
            }
        } else {
            std::cout << "[Car][Localization] disabled by config; UDP forwarding OFF."
                      << std::endl;
        }
    }

    // #################### IPM init #####################
      // (map1,map2)
    {
        // calibrationPath and ipmMatrixPath set in loadParams 
        cv::FileStorage fs_cam(imgProcessor.calibrationPath, cv::FileStorage::READ);
        if (fs_cam.isOpened()) {
            fs_cam["cameraMatrix"] >> cameraMatrix;
            fs_cam["distCoeffs"] >> distCoeffs;
            fs_cam.release();
            std::cout << "Loaded calibration: " << imgProcessor.calibrationPath << std::endl;
        } else {
            std::cerr << "Failed to open calibration file: " << imgProcessor.calibrationPath << std::endl;
        }

        cv::FileStorage fs_homo(imgProcessor.ipmMatrixPath, cv::FileStorage::READ);
        if (fs_homo.isOpened()) {
            fs_homo["homoMatrix"] >> homoMatrix;
            fs_homo.release();
            std::cout << "Loaded homography: " << imgProcessor.ipmMatrixPath << std::endl;
        } else {
            std::cerr << "Failed to open homography file: " << imgProcessor.ipmMatrixPath << std::endl;
        }

        // share homoMatrix with pathSearcher
        pathSearcher.homoMatrix = homoMatrix;

        imageSize = cv::Size(colsSet, rowsSet);
        if (params.useRemap) {
            if (!cameraMatrix.empty() && !distCoeffs.empty()) {
                initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), cameraMatrix, imageSize, CV_16SC2, map1, map2);
                std::cout << "[Camera] remap enabled." << std::endl;
            } else {
                std::cerr << "[Camera] remap requested but calibration is empty, fallback to direct image." << std::endl;
            }
        } else {
            std::cout << "[Camera] remap disabled (params.useRemap=false)." << std::endl;
        }
    }
    // #################### roadType init #####################
    // #################### display init #####################
    params.display = false;
    params.saveRaw = false;
    std::cout << "[ExternalModel] local display/saveRaw disabled; saveAI/saveWorldView follow config."
              << std::endl;
    std::cout << "[Input] external model mode enabled: skip local capture and consume shmDetectionReader only." << std::endl;

    uart.firstFlag = true;
    initSuccess = true;

    cout<<"Car init done !!!"<< endl;

    objectTracker = std::make_unique<ObjectTracker>(640, 480);
}

void Car::imageProcessor()  // path planning thread
{
    std::cout << "[imageProcessor] thread started, waiting init gate..." << std::endl;
    while (!loadSuccess || !initSuccess) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    std::cout << "[imageProcessor] init gate passed." << std::endl;

    // ========== fixed planning dimensions ==========
    const int PLAN_WIDTH = 640;
    const int PLAN_HEIGHT = 480;
    const float DISPLAY_SCALE_X = 1.0f;
    const float DISPLAY_SCALE_Y = 1.0f;

    int frame_idx = 0;
    const float speed_decision_clear_mps =
        std::max(0.0f, params.speedDecision.clearSpeedMps);
    const float speed_decision_obstacle_mps =
        std::max(0.0f, params.speedDecision.obstacleVisibleSpeedMps);
    const int speed_decision_clear_frames =
        std::max(0, params.speedDecision.clearFrames);
    int speed_decision_clear_count = speed_decision_clear_frames;
    float speed_decision_cruise_mps = speed_decision_clear_mps;

    // Physical-plane pure pursuit control. The path is still searched in image
    // coordinates, then transformed by the calibrated image-to-ground matrix.
    const float base_pure_ahead_distance =
        std::max(0.05f, params.worldPurePursuitParams.aheadDistance);
    const float pure_wheel_base =
        std::max(0.01f, params.worldPurePursuitParams.wheelBase);
    const float servo_output_limit =
        std::fabs(params.worldPurePursuitParams.servoLimit);

    float steering_angle = 0.0f;
    float last_planner_steering_angle_deg = 0.0f;
    bool last_planner_steering_angle_valid = false;
    float last_applied_servo_output = 0.0f;
    bool last_applied_servo_valid = false;
    float servo_output = 0.0f;
    float target_distance = 0.0f;
    cv::Point2f pursuit_target(0.0f, 0.0f);
    bool has_pursuit_target = false;
    int world_invalid_points = 0;
    VirtualLookaheadState virtual_lookahead_state;
    const int virtual_low_arrow_max_count =
        std::max(0, params.virtualLookahead.lowArrowMaxCount);
    const int virtual_recover_arrow_min_count =
        std::max(virtual_low_arrow_max_count + 1,
                 params.virtualLookahead.recoverArrowMinCount);
    const float virtual_activate_distance_m =
        std::max(0.0f, params.virtualLookahead.activateDistanceM);
    const float virtual_fixed_servo_output_abs =
        std::fabs(params.virtualLookahead.fixedServoOutput);
    const float virtual_speed_mps =
        std::max(0.0f, params.speedDecision.virtualLookaheadSpeedMps);
    const float gold_terminal_path_tolerance_m =
        std::max(0.0f,
                 params.goldPathPlanning.terminalPosePathToleranceM);
    const float gold_terminal_activation_distance_m =
        std::max(0.0f,
                 params.goldPathPlanning.terminalPoseActivationDistanceM);
    const float gold_terminal_derivative_alpha = std::clamp(
        params.goldPathPlanning.terminalPoseDerivativeFilterAlpha,
        0.0f,
        1.0f);
    const float gold_terminal_derivative_limit_deg =
        std::max(0.0f,
                 params.goldPathPlanning.terminalPoseDerivativeLimitDeg);
    const float gold_terminal_target_jump_reset_m =
        std::max(0.0f,
                 params.goldPathPlanning.terminalPoseTargetJumpResetM);
    const float gold_terminal_servo_limit = std::min(
        std::fabs(params.goldPathPlanning.terminalPoseServoAbs),
        servo_output_limit);
    const float road_sign_pose_virtual_radius_m =
        std::max(0.05f, params.roadSignFork.poseVirtualRadiusM);
    const float road_sign_pose_center_filter_alpha = std::clamp(
        params.roadSignFork.poseCenterFilterAlpha, 0.0f, 1.0f);
    const float road_sign_pose_center_deadband_px =
        std::max(0.0f, params.roadSignFork.poseCenterDeadbandPx);
    const float road_sign_pose_derivative_alpha = std::clamp(
        params.roadSignFork.poseDerivativeFilterAlpha, 0.0f, 1.0f);
    const float road_sign_pose_derivative_limit_deg =
        std::max(0.0f, params.roadSignFork.poseDerivativeLimitDeg);
    const float road_sign_pose_target_jump_reset_m =
        std::max(0.0f, params.roadSignFork.poseTargetJumpResetM);
    const float road_sign_pose_servo_limit = std::min(
        std::fabs(params.roadSignFork.poseServoAbs),
        servo_output_limit);
    const float pedestrian_gold_danger_strip_half_width_m =
        std::max(
            0.0f,
            params.goldPathPlanning.pedestrianGoldDangerStripHalfWidthM);
    const int pedestrian_gold_danger_strip_hold_frames =
        std::max(
            0,
            params.goldPathPlanning.pedestrianGoldDangerStripHoldFrames);
    bool gold_terminal_pd_initialized = false;
    float gold_terminal_last_error_deg = 0.0f;
    float gold_terminal_filtered_derivative_deg = 0.0f;
    cv::Point2f gold_terminal_previous_target_world(0.0f, 0.0f);
    bool gold_terminal_previous_target_valid = false;
    bool gold_terminal_applied_previous_frame = false;
    bool road_sign_pose_pd_initialized = false;
    bool road_sign_pose_center_initialized = false;
    float road_sign_pose_filtered_center_x_px = 0.0f;
    float road_sign_pose_last_error_deg = 0.0f;
    float road_sign_pose_filtered_derivative_deg = 0.0f;
    cv::Point2f road_sign_pose_previous_target_world(0.0f, 0.0f);
    bool road_sign_pose_previous_target_valid = false;
    bool road_sign_pose_applied_previous_frame = false;

    // ==========  ==========

    std::ofstream servo_log_file;
    size_t servo_log_pending_lines = 0;
    if (params.servoDebugLogEnable) {
        servo_log_file.open(params.servoDebugLogPath, std::ios::out | std::ios::trunc);
        if (!servo_log_file.is_open()) {
            std::cerr << "[]  " << params.servoDebugLogPath << std::endl;
        } else {
            servo_log_file << "========================================" << std::endl;
            servo_log_file << "[World Pure Pursuit Debug]" << std::endl;
            servo_log_file << "- Record transformed path target, steering angle and servo output." << std::endl;
            servo_log_file << "========================================" << std::endl;
            servo_log_file << "Frame,target_x,target_y,target_distance,steering_angle,servo_output,speed,world_pts,invalid_pts,path_size,has_target" << std::endl;
            servo_log_file.flush();
        }
    }

    std::ofstream local_planner_log_file;
    size_t local_planner_log_pending_lines = 0;
    if (params.localPlannerLogEnable) {
        ensureParentDirectory(params.localPlannerLogPath);
        local_planner_log_file.open(
            params.localPlannerLogPath, std::ios::out | std::ios::trunc);
        if (!local_planner_log_file.is_open()) {
            std::cerr << "[LocalPlannerLog] failed to open: "
                      << params.localPlannerLogPath << std::endl;
        }
    }
    if (local_planner_log_file.is_open()) {
        local_planner_log_file << "proc_frame,packet_frame,mode,obstacle_state,avoid_scenario,"
                               << "target_speed,ego_speed,path_points,"
                               << "gold_bias_reason,gold_visible_count,gold_projectable_count,"
                               << "gold_best_near_capture_count,gold_best_capture_count,"
                               << "gold_best_capture_along_s_sum,gold_no_capture_diag,"
                               << "gold_late_start_rejected,"
                               << "gold_late_start_s_m,gold_late_start_delta_m,"
                               << "gold_best_bias_m,gold_stable_base_m,gold_transition_state,"
                               << "gold_target_bias_m,gold_applied_bias_m,"
                               << "gold_effective_bias_m,gold_pending_bias_m,"
                               << "gold_pending_confirm_count,gold_no_executable_frames,"
                               << "gold_target_reached_frames,gold_selection_frozen,"
                               << "traffic_target_speed,final_target_speed,applied_speed,"
                               << "ped_emergency_stop,"
                               << "start_delay_active,speed_override_mode,route_side,"
                               << "avoid_side,vehicle_avoid_side_locked,"
                               << "vehicle_locked_avoid_side,"
                               << "vehicle_curve_metric,vehicle_curve_direction,vehicle_curve_avoid_boost_active,"
                               << "vehicle_opposite_steer_active,vehicle_curve_avoid_hold_remaining,"
                               << "vehicle_recover_hold_active,vehicle_recover_hold_remaining,"
                               << "vehicle_recover_hold_age_frames,vehicle_recover_hold_max_lifetime_frames,"
                               << "obstacle_target_x,obstacle_target_y,"
                               << "base_path_x,final_path_x,path_y,"
                               << "route_hold_active,speed_risk_level,speed_limit_reason,"
                               << "reason,speed_reason,obstacle_count,obstacle_active_count,"
                               << "obstacle_id,obstacle_type,obstacle_distance_m,"
                               << "obstacle_point_distance_m,obstacle_lateral_m,"
                               << "avoid_offset_current_m,avoid_offset_target_m,"
                               << "final_applied_path_offset_m,realized_offset_m,"
                               << "ego_center_lateral_m,near_route_lateral_m,"
                               << "near_route_lateral_valid,near_route_lateral_sample_count,"
                               << "raw_person_count,ped_boundary_clearance_m,"
                               << "ped_motion_clearance_net_m,ped_motion_name,ped_motion_reason,"
                               << "ped_seen_inside,ped_clear_streak,"
                               << "ped_stop_lost_streak,"
                               << "ped_stop_car_side_metric_m,ped_stop_locked_car_side,"
                               << "ped_release_required_side,"
                               << "ped_exit_evidence,ped_exit_lost_grace,"
                               << "ped_slow_observation_count,ped_slow_lost_count,ped_slow_away_streak,"
                               << "startup_launch_protect_remaining,startup_launch_release_event,"
                               << "ped_emergency_suppressed_by_launch,"
                               << "launch_protect_speed_applied,"
                               << "pedestrian_stop_state,pedestrian_launch_source,"
                               << "slow_candidate,emergency_candidate,"
                               << "pedestrian_observed_this_frame,"
                               << "path_usable,blocked_by_path,launch_remaining,"
                               << "launch_paused_by_path,planner_speed_source,"
                               << "final_speed_source,active_block_reasons,"
                               << "state_transition_reason,"
                               << "ped_emergency_id,ped_foot_x_m,ped_foot_y_m,"
                               << "ped_emergency_hit_corridor,ped_emergency_hit,"
                               << "ped_danger_corridor_near_width_m,"
                               << "ped_danger_corridor_far_width_m,"
                               << "ped_danger_corridor_length_m,"
                               << "ped_danger_corridor_sample_step_m,"
                               << "ped_danger_servo_raw,ped_danger_servo_output,"
                               << "ped_danger_servo_locked,ped_danger_servo_abs_used,"
                               << "ped_danger_servo_valid,ped_danger_servo_source,"
                               << "ped_danger_turn_radius_m,ped_danger_turn_sign,"
                               << "ped_danger_straight,ped_danger_along_s_m,"
                               << "ped_danger_lateral_distance_m,"
                               << "ped_emergency_allowed_width_m,"
                               << "cluster_active,avoid_lookahead_active,recommended_lookahead_m,"
                               << "steering_angle,servo_output,"
                               << "gold_terminal_candidate_valid,gold_terminal_active,"
                               << "gold_terminal_applied,gold_terminal_entered,gold_terminal_exited,"
                               << "gold_terminal_derivative_reset,gold_terminal_reason,"
                               << "gold_terminal_x,gold_terminal_y,gold_terminal_distance_m,"
                               << "gold_terminal_path_distance_m,gold_terminal_s_m,"
                               << "gold_terminal_lookahead_s_m,gold_terminal_heading_error_deg,"
                               << "gold_terminal_filtered_derivative_deg,gold_terminal_servo_output,"
                               << "ped_gold_danger_strip_count,ped_gold_ignored_count,"
                               << "ped_gold_danger_half_width_m,"
                               << "ped_gold_danger_live,ped_gold_danger_hold_active,"
                               << "ped_gold_danger_hold_elapsed_frames,"
                               << "ped_gold_danger_hold_remaining_frames,"
                               << "ped_gold_danger_hold_config_frames,ped_gold_danger_source,"
                               << "line_arrow_object_count,line_raw_guide_count,"
                               << "line_valid_world_guide_count,"
                               << "line_single_anchor_count,line_single_root_count,"
                               << "line_credible_single_count,line_best_single_guide_count,"
                               << "line_best_single_forward_span,"
                                << "line_visible_branch_count,line_selected_branch_label,"
                                << "line_selected_branch_guide_count,"
                                << "line_selected_branch_centerline_size,"
                                << "line_candidate_size,line_candidate_usable,"
                                << "line_local_route_size,line_local_waypoints_size,"
                                << "line_local_usable,line_used_candidate_fallback,"
                                << "line_local_reject_reason,"
                                << "line_final_plan_path_size,line_final_plan_path_usable,"
                                << "line_world_path_size,line_positive_world_y_count,"
                                << "line_pure_pursuit_target_valid,"
                                << "line_servo_source,line_empty_reason"
                                << std::endl;
        local_planner_log_file.flush();
    }

    std::ofstream serial_timing_log_file;
    size_t serial_timing_log_pending_lines = 0;
    std::filesystem::path serial_timing_log_path(params.localPlannerLogPath);
    if (serial_timing_log_path.empty()) {
        serial_timing_log_path = "serial_timing.csv";
    } else {
        serial_timing_log_path.replace_filename(
            serial_timing_log_path.stem().string() + "_serial_timing.csv");
    }
    ensureParentDirectory(serial_timing_log_path.string());
    serial_timing_log_file.open(
        serial_timing_log_path.string(), std::ios::out | std::ios::trunc);
    if (serial_timing_log_file.is_open()) {
        serial_timing_log_file
            << "proc_frame,packet_frame,final_cmd_mono_us,applied_speed,"
            << "serial_begin_mono_us,tx_encoded_mono_us,serial_speed,servo,tx_hex,"
            << "serial_end_mono_us,"
            << "serial_write_us,time_since_last_serial_write_us,"
            << "final_to_serial_begin_us,loop_end_mono_us,loop_duration_us,serial_rc\n";
        serial_timing_log_file.flush();
        std::cout << "[SerialTiming] CSV: " << serial_timing_log_path << std::endl;
    } else {
        std::cerr << "[SerialTiming] failed to open: "
                  << serial_timing_log_path << std::endl;
    }

    const auto monotonicUs = [](const std::chrono::steady_clock::time_point& tp) {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            tp.time_since_epoch()).count();
    };
    std::chrono::steady_clock::time_point last_serial_begin_tp{};
    bool last_serial_begin_valid = false;

    std::ofstream dynamic_path_planner_log_file;
    size_t dynamic_path_planner_log_pending_lines = 0;
    if (params.dynamicPathPlannerLogEnable) {
        ensureParentDirectory(params.dynamicPathPlannerLogPath);
        dynamic_path_planner_log_file.open(
            params.dynamicPathPlannerLogPath, std::ios::out | std::ios::trunc);
        if (dynamic_path_planner_log_file.is_open()) {
            dynamic_path_planner_log_file
                << "proc_frame,packet_frame,mode,reason,target_speed,ego_speed,"
                << "path_points,speed_risk_level,speed_limit_reason,speed_reason\n";
            dynamic_path_planner_log_file.flush();
        } else {
            std::cerr << "[DynamicPathPlannerLog] failed to open: "
                      << params.dynamicPathPlannerLogPath << std::endl;
        }
    }

    auto logServoDebug = [&](int frame_id,
                             const cv::Point2f& target,
                             float lookahead,
                             float steer_angle,
                             float servo_out,
                             float speed,
                             int world_pts,
                             int invalid_pts,
                             int path_size,
                             bool has_target) {
        if (!servo_log_file.is_open()) return;

        servo_log_file << frame_id << "," << target.x << "," << target.y << "," << lookahead
                       << "," << steer_angle << "," << servo_out
                       << "," << speed << "," << world_pts << "," << invalid_pts
                       << "," << path_size << "," << (has_target ? 1 : 0) << '\n';
        if (++servo_log_pending_lines >= kDebugLogFlushEveryLines) {
            servo_log_file.flush();
            servo_log_pending_lines = 0;
        }
    };

    auto draw_results_on_frame = [this, DISPLAY_SCALE_X, DISPLAY_SCALE_Y](cv::Mat& img, const std::vector<PredictResult>& results) {
        if (img.empty()) {
            return;
        }

        for (const auto& result : results) {
            if (result.width <= 0 || result.height <= 0) {
                continue;
            }

            // Map detection boxes from planning coordinates to display coordinates.
            const int sx = static_cast<int>(std::round(result.x * DISPLAY_SCALE_X));
            const int sy = static_cast<int>(std::round(result.y * DISPLAY_SCALE_Y));
            const int sw = std::max(1, static_cast<int>(std::round(result.width * DISPLAY_SCALE_X)));
            const int sh = std::max(1, static_cast<int>(std::round(result.height * DISPLAY_SCALE_Y)));

            cv::Rect box(sx, sy, sw, sh);
            cv::rectangle(img, box, detection.getCvcolor(result.type), 2);

            std::string score = std::to_string(result.score);
            std::string label_name = result.label + " [" + score.substr(0, score.find(".") + 3) + "]";
            int text_y = std::max(0, sy - 4);
            cv::putText(img, label_name, cv::Point(sx, text_y), cv::FONT_HERSHEY_PLAIN, 1,
                        cv::Scalar(0, 0, 254), 1);
        }
    };

    enum class RoadSignForkPhase {
        DefaultLeft = 0,
        StopForDecision,
        WaitDecision,
        ArmSelected,
        PassSelectedFork,
        Cooldown
    };

    struct RoadSignObservation {
        bool seen = false;
        bool slowdown_seen = false;
        bool pose_box_valid = false;
        float score = 0.0f;
        int area = 0;
        cv::Rect box;
        float pose_score = 0.0f;
        int pose_area = 0;
        cv::Rect pose_box;
    };

    enum class SpeedLightPhase {
        Cruise = 0,
        RedStop,
        PassAfterGreen,
        Cooldown
    };

    struct SpeedLightObservation {
        bool red_seen = false;
        float red_score = 0.0f;
        int red_bottom_y = 0;
        int red_height = 0;

        bool yellow_seen = false;
        float yellow_score = 0.0f;
        int yellow_bottom_y = 0;
        int yellow_height = 0;

        bool green_seen = false;
        float green_score = 0.0f;
        int green_bottom_y = 0;
        int green_height = 0;
    };

    enum class StopSignStopPhase {
        Idle = 0,
        SeenArmed,
        LostGrace,
        TerminalStop
    };

    struct StopSignObservation {
        bool seen = false;
        float score = 0.0f;
    };

    RoadSignForkPhase road_sign_phase = RoadSignForkPhase::DefaultLeft;
    int road_sign_seen_streak = 0;
    int road_sign_hold_lost_frames = 0;
    int road_sign_slowdown_seen_streak = 0;
    int road_sign_slowdown_lost_frames = 0;
    bool road_sign_slowdown_active = false;
    int road_sign_stop_frames = 0;
    int road_sign_arm_frames = 0;
    int road_sign_fork_seen_streak = 0;
    int road_sign_fork_exit_streak = 0;
    int road_sign_pass_frames = 0;
    int road_sign_pass_sign_lost_frames = 0;
    bool road_sign_pass_had_sign_gap = false;
    int road_sign_cooldown_frames = 0;
    int road_sign_active_lost_frames = 0;
    int road_sign_decision = -1;
    int road_sign_request_frame_id = -1;
    cv::Rect road_sign_request_box;
    std::future<int> road_sign_decision_future;
    bool road_sign_decision_request_sent = false;
    bool road_sign_decision_wait_started = false;
    std::chrono::steady_clock::time_point road_sign_decision_wait_start_tp{};
    int road_sign_wait_log_frames = 0;
    int64_t road_sign_cooldown_left_ms = 0;
    bool road_sign_decision_servo_hold_active = false;
    float road_sign_decision_servo_hold_output = 0.0f;
    bool road_sign_decision_servo_hold_stop_required = false;
    std::chrono::steady_clock::time_point road_sign_decision_servo_hold_start_tp{};
    bool road_sign_first_pass_completed = false;
    int road_sign_first_decision = -1;
    bool road_sign_second_opposite_used = false;
    bool road_sign_second_opposite_active = false;
    bool road_sign_last_visible_split_output_active = false;
    int road_sign_last_visible_split_frame = -1;

    auto roadSignSecondOppositeReady = [&]() -> bool {
        return params.roadSignFork.secondOppositeOnlyEnable &&
               road_sign_first_pass_completed &&
               (road_sign_first_decision == 0 || road_sign_first_decision == 1) &&
               !road_sign_second_opposite_used;
    };

    SpeedLightPhase speed_light_phase = SpeedLightPhase::Cruise;
    int stop_light_seen_streak = 0;
    int green_light_seen_streak = 0;
    int pass_light_exit_streak = 0;
    int speed_light_cooldown_frames = 0;
    auto speedLightStateMachineEnabled = [&]() -> bool {
        return params.speedLight.enable && params.speedLight.stateMachineEnable;
    };

    StopSignStopPhase stop_sign_phase = StopSignStopPhase::Idle;
    int stop_sign_seen_streak = 0;
    bool stop_sign_last_seen_valid = false;
    std::chrono::steady_clock::time_point stop_sign_last_seen_tp{};
    int64_t stop_sign_lost_ms_display = 0;
    // Keep the stop-sign full-stop state machine in place for future use,
    // but disable its runtime effect for now.
    constexpr bool kStopSignStopStateMachineEnabled = false;

    auto roadSignForkPhaseName = [](RoadSignForkPhase phase) -> const char* {
        switch (phase) {
        case RoadSignForkPhase::StopForDecision: return "STOP_FOR_DECISION";
        case RoadSignForkPhase::WaitDecision: return "WAIT_DECISION";
        case RoadSignForkPhase::ArmSelected: return "ARM_SELECTED";
        case RoadSignForkPhase::PassSelectedFork: return "PASS_SELECTED";
        case RoadSignForkPhase::Cooldown: return "COOLDOWN";
        default: return "DEFAULT_LEFT";
        }
    };

    auto branchPreferenceName = [](BranchPreference preference) -> const char* {
        return (preference == BranchPreference::ForceRight) ? "RIGHT" : "LEFT";
    };

    auto branchSideName = [](BranchSide side) -> const char* {
        switch (side) {
        case BranchSide::Main: return "MAIN";
        case BranchSide::Left: return "LEFT";
        case BranchSide::Right: return "RIGHT";
        default: return "NONE";
        }
    };

    auto speedLightPhaseName = [](SpeedLightPhase phase) -> const char* {
        switch (phase) {
        case SpeedLightPhase::RedStop: return "RED_STOP";
        case SpeedLightPhase::PassAfterGreen: return "PASS_GREEN";
        case SpeedLightPhase::Cooldown: return "COOLDOWN";
        default: return "CRUISE";
        }
    };

    auto stopSignStopPhaseName = [](StopSignStopPhase phase) -> const char* {
        switch (phase) {
        case StopSignStopPhase::SeenArmed: return "SEEN";
        case StopSignStopPhase::LostGrace: return "LOST_GRACE";
        case StopSignStopPhase::TerminalStop: return "TERMINAL_STOP";
        default: return "IDLE";
        }
    };

    auto localPlannerModeName = [](LocalPlannerMode mode) -> const char* {
        switch (mode) {
        case LocalPlannerMode::Cruise: return "CRUISE";
        case LocalPlannerMode::AvoidPrepare: return "AVOID_PREPARE";
        case LocalPlannerMode::AvoidActive: return "AVOID_ACTIVE";
        case LocalPlannerMode::AvoidRecover: return "AVOID_RECOVER";
        default: return "DISABLED";
        }
    };

    auto obstacleStateName = [](ObstacleAvoidanceState state) -> const char* {
        switch (state) {
        case ObstacleAvoidanceState::Prepare: return "PREPARE";
        case ObstacleAvoidanceState::Active: return "ACTIVE";
        case ObstacleAvoidanceState::Recover: return "RECOVER";
        default: return "CRUISE";
        }
    };

    auto avoidScenarioName = [](AvoidScenario scenario) -> const char* {
        switch (scenario) {
        case AvoidScenario::VehicleOnly: return "VEHICLE_ONLY";
        default: return "NONE";
        }
    };

    auto resetRoadSignForkState = [&]() {
        road_sign_seen_streak = 0;
        road_sign_hold_lost_frames = 0;
        road_sign_slowdown_seen_streak = 0;
        road_sign_slowdown_lost_frames = 0;
        road_sign_slowdown_active = false;
        road_sign_stop_frames = 0;
        road_sign_arm_frames = 0;
        road_sign_fork_seen_streak = 0;
        road_sign_fork_exit_streak = 0;
        road_sign_pass_frames = 0;
        road_sign_pass_sign_lost_frames = 0;
        road_sign_pass_had_sign_gap = false;
        road_sign_cooldown_frames = 0;
        road_sign_active_lost_frames = 0;
        road_sign_decision = -1;
        road_sign_request_frame_id = -1;
        road_sign_request_box = cv::Rect();
        road_sign_decision_request_sent = false;
        road_sign_decision_wait_started = false;
        road_sign_wait_log_frames = 0;
        road_sign_second_opposite_active = false;
    };

    auto startRoadSignDecisionServoHold = [&](int frame_id) {
        road_sign_decision_servo_hold_active = false;
        road_sign_decision_servo_hold_output = 0.0f;
        road_sign_decision_servo_hold_stop_required = false;

        if (!params.roadSignFork.decisionServoHoldEnable ||
            (road_sign_decision != 0 && road_sign_decision != 1)) {
            return;
        }

        road_sign_decision_servo_hold_output =
            (road_sign_decision == 1)
                ? params.roadSignFork.decisionServoRightOutput
                : params.roadSignFork.decisionServoLeftOutput;
        road_sign_decision_servo_hold_active = true;
        road_sign_decision_servo_hold_stop_required =
            !road_sign_second_opposite_active;
        road_sign_decision_servo_hold_start_tp =
            std::chrono::steady_clock::now();

        std::cout << "[RoadSignFork] decision servo hold start frame=" << frame_id
                  << " decision=" << road_sign_decision
                  << " output=" << road_sign_decision_servo_hold_output
                  << " mode="
                  << (road_sign_decision_servo_hold_stop_required
                          ? "stop_until_encoder_stationary"
                          : "moving_timed")
                  << " hold_ms="
                  << std::max(0, params.roadSignFork.decisionServoHoldMs)
                  << std::endl;
    };

    auto setRoadSignForkPhase = [&](RoadSignForkPhase new_phase, int frame_id) {
        if (road_sign_phase == new_phase) {
            return;
        }

        const RoadSignForkPhase old_phase = road_sign_phase;
        road_sign_phase = new_phase;
        const bool first_pass_completed_now =
            params.roadSignFork.secondOppositeOnlyEnable &&
            old_phase == RoadSignForkPhase::PassSelectedFork &&
            road_sign_phase == RoadSignForkPhase::DefaultLeft &&
            !road_sign_second_opposite_active &&
            !road_sign_first_pass_completed &&
            (road_sign_decision == 0 || road_sign_decision == 1);
        if (first_pass_completed_now) {
            road_sign_first_pass_completed = true;
            road_sign_first_decision = road_sign_decision;
            std::cout << "[RoadSignFork] first pass completed decision="
                      << road_sign_first_decision
                      << " second_opposite_only=ready" << std::endl;
        }
        if (old_phase == RoadSignForkPhase::DefaultLeft &&
            road_sign_phase != RoadSignForkPhase::DefaultLeft) {
            road_sign_active_lost_frames = 0;
        }
        switch (road_sign_phase) {
        case RoadSignForkPhase::DefaultLeft:
            resetRoadSignForkState();
            break;
        case RoadSignForkPhase::StopForDecision:
            road_sign_slowdown_seen_streak = 0;
            road_sign_slowdown_lost_frames = 0;
            road_sign_slowdown_active = false;
            road_sign_stop_frames = 0;
            road_sign_arm_frames = 0;
            road_sign_fork_seen_streak = 0;
            road_sign_fork_exit_streak = 0;
            road_sign_pass_frames = 0;
            road_sign_pass_sign_lost_frames = 0;
            road_sign_pass_had_sign_gap = false;
            road_sign_cooldown_frames = 0;
            break;
        case RoadSignForkPhase::WaitDecision:
            road_sign_slowdown_seen_streak = 0;
            road_sign_slowdown_lost_frames = 0;
            road_sign_slowdown_active = false;
            road_sign_stop_frames = 0;
            road_sign_arm_frames = 0;
            road_sign_fork_seen_streak = 0;
            road_sign_fork_exit_streak = 0;
            road_sign_pass_frames = 0;
            road_sign_pass_sign_lost_frames = 0;
            road_sign_pass_had_sign_gap = false;
            road_sign_cooldown_frames = 0;
            break;
        case RoadSignForkPhase::ArmSelected:
            road_sign_arm_frames = 0;
            road_sign_fork_seen_streak = 0;
            road_sign_fork_exit_streak = 0;
            road_sign_pass_frames = 0;
            road_sign_pass_sign_lost_frames = 0;
            road_sign_pass_had_sign_gap = false;
            road_sign_cooldown_frames = 0;
            startRoadSignDecisionServoHold(frame_id);
            break;
        case RoadSignForkPhase::PassSelectedFork:
            road_sign_fork_seen_streak = 0;
            road_sign_fork_exit_streak = 0;
            road_sign_pass_frames = 0;
            road_sign_pass_sign_lost_frames = 0;
            road_sign_pass_had_sign_gap = false;
            break;
        case RoadSignForkPhase::Cooldown:
            road_sign_seen_streak = 0;
            road_sign_hold_lost_frames = 0;
            road_sign_slowdown_seen_streak = 0;
            road_sign_slowdown_lost_frames = 0;
            road_sign_slowdown_active = false;
            road_sign_stop_frames = 0;
            road_sign_arm_frames = 0;
            road_sign_fork_seen_streak = 0;
            road_sign_fork_exit_streak = 0;
            road_sign_pass_frames = 0;
            road_sign_pass_sign_lost_frames = 0;
            road_sign_pass_had_sign_gap = false;
            road_sign_cooldown_frames = 0;
            break;
        }

        std::cout << "[RoadSignFork] Frame#" << frame_id
                  << " phase=" << roadSignForkPhaseName(road_sign_phase)
                  << std::endl;
    };

    auto extractRoadSignObservation = [&](const std::vector<PredictResult>& results) -> RoadSignObservation {
        RoadSignObservation obs;
        float best_full_score = -1.0f;
        int best_pose_area = -1;
        float best_pose_score = -1.0f;

        auto fillObservationBox = [&](RoadSignObservation& target, const PredictResult& result, int area) {
            target.score = result.score;
            target.area = area;
            target.box = cv::Rect(result.x, result.y, result.width, result.height);
        };

        for (const auto& result : results) {
            if (result.type != TYPE_ROAD_SIGN || result.width <= 0 || result.height <= 0) {
                continue;
            }
            const int area = std::max(0, result.width) * std::max(0, result.height);
            const int sign_area_threshold =
                std::max(0, params.roadSignFork.signAreaThreshold);
            const int slowdown_area_threshold = std::max(
                0,
                std::min(params.roadSignFork.stopSlowdownAreaThreshold,
                         std::max(0, sign_area_threshold - 1)));
            if (area >= slowdown_area_threshold) {
                obs.slowdown_seen = true;
                if (area > best_pose_area ||
                    (area == best_pose_area && result.score > best_pose_score)) {
                    best_pose_area = area;
                    best_pose_score = result.score;
                    obs.pose_box_valid = true;
                    obs.pose_score = result.score;
                    obs.pose_area = area;
                    obs.pose_box = cv::Rect(result.x, result.y, result.width, result.height);
                }
            }
            if (area < std::max(0, params.roadSignFork.signAreaThreshold)) {
                continue;
            }
            if (result.score > best_full_score) {
                best_full_score = result.score;
                obs.seen = true;
                fillObservationBox(obs, result, area);
            }
        }

        return obs;
    };

    auto updateRoadSignDetectionStreak = [&](bool seen) {
        if (seen) {
            road_sign_seen_streak++;
            road_sign_hold_lost_frames = 0;
            return;
        }

        if (road_sign_seen_streak > 0 &&
            road_sign_hold_lost_frames < std::max(0, params.roadSignFork.signLostHoldFrames)) {
            road_sign_hold_lost_frames++;
            return;
        }

        road_sign_seen_streak = 0;
        road_sign_hold_lost_frames = 0;
    };

    auto resetRoadSignSlowdownTracking = [&]() {
        road_sign_slowdown_seen_streak = 0;
        road_sign_slowdown_lost_frames = 0;
        road_sign_slowdown_active = false;
    };

    auto updateRoadSignSlowdownTracking = [&](bool slowdown_seen) {
        if (road_sign_phase != RoadSignForkPhase::DefaultLeft) {
            resetRoadSignSlowdownTracking();
            return;
        }

        if (slowdown_seen) {
            road_sign_slowdown_seen_streak++;
            road_sign_slowdown_lost_frames = 0;
            if (road_sign_slowdown_seen_streak >=
                std::max(1, params.roadSignFork.stopSlowdownStableFrames)) {
                road_sign_slowdown_active = true;
            }
            return;
        }

        road_sign_slowdown_seen_streak = 0;
        if (road_sign_slowdown_active &&
            road_sign_slowdown_lost_frames <
                std::max(0, params.roadSignFork.stopSlowdownHoldFrames)) {
            road_sign_slowdown_lost_frames++;
            return;
        }

        resetRoadSignSlowdownTracking();
    };

    auto beginRoadSignStopForDecision = [&](const RoadSignObservation& obs,
                                            int frame_id,
                                            int source_frame_id) {
        road_sign_request_frame_id = source_frame_id;
        road_sign_request_box = obs.box;
        road_sign_decision_request_sent = false;
        road_sign_decision_wait_started = false;
        road_sign_wait_log_frames = 0;
        std::cout << "[RoadSignFork] stop: road_sign area=" << obs.area
                  << " score=" << obs.score
                  << " proc_frame=" << frame_id
                  << " source_frame=" << source_frame_id
                  << " box=(" << obs.box.x << "," << obs.box.y << ","
                  << obs.box.x + obs.box.width << ","
                  << obs.box.y + obs.box.height << ")"
                  << std::endl;
        setRoadSignForkPhase(RoadSignForkPhase::StopForDecision, frame_id);
    };

    auto beginRoadSignSecondOppositeSelection = [&](const RoadSignObservation& obs,
                                                    int frame_id,
                                                    int source_frame_id) {
        if (road_sign_first_decision != 0 && road_sign_first_decision != 1) {
            return;
        }

        road_sign_decision = 1 - road_sign_first_decision;
        road_sign_second_opposite_used = true;
        road_sign_second_opposite_active = true;
        road_sign_request_frame_id = -1;
        road_sign_request_box = cv::Rect();
        road_sign_decision_request_sent = false;
        road_sign_decision_wait_started = false;
        road_sign_wait_log_frames = 0;
        resetRoadSignSlowdownTracking();

        std::cout << "[RoadSignFork] second opposite selection without OCR"
                  << " first_decision=" << road_sign_first_decision
                  << " decision=" << road_sign_decision
                  << " area=" << obs.area
                  << " score=" << obs.score
                  << " proc_frame=" << frame_id
                  << " source_frame=" << source_frame_id
                  << std::endl;
        setRoadSignForkPhase(RoadSignForkPhase::ArmSelected, frame_id);
    };

    auto updateRoadSignStateBeforePlanner = [&](const RoadSignObservation& obs,
                                                 int frame_id,
                                                 int source_frame_id) -> BranchPreference {
        road_sign_cooldown_left_ms = 0;

        if (!params.roadSignFork.enable) {
            resetRoadSignSlowdownTracking();
            if (road_sign_phase != RoadSignForkPhase::DefaultLeft) {
                setRoadSignForkPhase(RoadSignForkPhase::DefaultLeft, frame_id);
            }
            return BranchPreference::PreferLeft;
        }

        if (params.roadSignFork.secondOppositeOnlyEnable &&
            road_sign_second_opposite_used &&
            road_sign_phase == RoadSignForkPhase::DefaultLeft) {
            resetRoadSignSlowdownTracking();
            return BranchPreference::PreferLeft;
        }

        const bool second_opposite_ready = roadSignSecondOppositeReady();
        updateRoadSignDetectionStreak(obs.seen);
        updateRoadSignSlowdownTracking(obs.slowdown_seen);
        const bool road_sign_stable =
            obs.seen &&
            road_sign_seen_streak >=
                std::max(1, params.roadSignFork.signStableFrames);

        if (road_sign_phase != RoadSignForkPhase::DefaultLeft) {
            if (obs.seen) {
                road_sign_active_lost_frames = 0;
            } else {
                road_sign_active_lost_frames++;
                if (road_sign_active_lost_frames >=
                    std::max(1, params.roadSignFork.activeLostExitFrames)) {
                    std::cout << "[RoadSignFork] active lost road_sign, exit"
                              << " lost_frames=" << road_sign_active_lost_frames
                              << " threshold=" << std::max(1, params.roadSignFork.activeLostExitFrames)
                              << " proc_frame=" << frame_id << std::endl;
                    setRoadSignForkPhase(RoadSignForkPhase::DefaultLeft, frame_id);
                    return BranchPreference::PreferLeft;
                }
            }
        }

        switch (road_sign_phase) {
        case RoadSignForkPhase::DefaultLeft:
            if (road_sign_stable) {
                if (second_opposite_ready) {
                    beginRoadSignSecondOppositeSelection(obs, frame_id, source_frame_id);
                } else {
                    beginRoadSignStopForDecision(obs, frame_id, source_frame_id);
                }
            }
            break;
        case RoadSignForkPhase::StopForDecision:
            road_sign_stop_frames++;
            {
                bool speed_stationary = false;
                LowerTelemetry telemetry;
                if (getLowerTelemetry(telemetry)) {
                    const float speed_mps = telemetry.speed_mps;
                    speed_stationary =
                        std::isfinite(speed_mps) &&
                        std::fabs(speed_mps) <= 0.12f;
                }
                const bool stop_wait_expired =
                    road_sign_stop_frames >= std::max(1, params.roadSignFork.decisionStopFrames);
                if (!speed_stationary && !stop_wait_expired) {
                    std::cout << "[RoadSignFork] waiting before OCR: stop_frame="
                              << road_sign_stop_frames
                              << " speed_stationary=0" << std::endl;
                    break;
                }
                constexpr int kRoadSignDecisionTimeoutMs = 10000;
                const int timeout_ms =
                    std::min(kRoadSignDecisionTimeoutMs, std::max(1, params.roadSignFork.decisionTimeoutMs));
                const int req_frame = road_sign_request_frame_id >= 0 ? road_sign_request_frame_id : frame_id;
                const cv::Rect req_box = road_sign_request_box;
                if (road_sign_decision_request_sent) {
                    std::cout << "[RoadSignFork] OCR/API request already sent, wait only proc_frame="
                              << frame_id
                              << " request_frame=" << req_frame << std::endl;
                    setRoadSignForkPhase(RoadSignForkPhase::WaitDecision, frame_id);
                    break;
                }
                if (!road_sign_decision_wait_started) {
                    road_sign_decision_wait_started = true;
                    road_sign_decision_wait_start_tp = std::chrono::steady_clock::now();
                }
                road_sign_wait_log_frames = 0;
                std::cout << "[RoadSignFork] begin OCR/API wait: 等待 开始OCR识别/API调用 proc_frame=" << frame_id
                          << " request_frame=" << req_frame
                          << " timeout_ms=" << timeout_ms << std::endl;
                road_sign_decision_request_sent = true;
                road_sign_decision_future = std::async(
                    std::launch::async,
                    [req_frame, req_box, timeout_ms]() {
                        return requestRoadSignDecisionFromYoloe(req_frame, req_box, timeout_ms);
                    });
                setRoadSignForkPhase(RoadSignForkPhase::WaitDecision, frame_id);
            }
            break;
        case RoadSignForkPhase::WaitDecision:
            if (road_sign_decision_future.valid() &&
                road_sign_decision_future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                road_sign_decision = road_sign_decision_future.get();
                const int64_t wait_ms =
                    road_sign_decision_wait_started
                        ? std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - road_sign_decision_wait_start_tp).count()
                        : 0;
                std::cout << "[RoadSignFork] OCR/API future ready: OCR/API完成 proc_frame=" << frame_id
                          << " waited_ms=" << wait_ms
                          << " decision=" << road_sign_decision << std::endl;
                if (road_sign_decision != 0 && road_sign_decision != 1) {
                    constexpr int kRoadSignDecisionTimeoutMs = 10000;
                    const int timeout_ms =
                        std::min(kRoadSignDecisionTimeoutMs, std::max(1, params.roadSignFork.decisionTimeoutMs));
                    const bool wait_timeout =
                        road_sign_decision_wait_started &&
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - road_sign_decision_wait_start_tp).count() >= timeout_ms;
                    if (!wait_timeout) {
                        std::cout << "[RoadSignFork] decision not ready, keep stopping without retry waited_ms="
                                  << wait_ms
                                  << std::endl;
                        break;
                    }
                    road_sign_decision = params.roadSignFork.defaultDecision == 0 ? 0 : 1;
                    std::cout << "[RoadSignFork] decision timeout, fallback="
                              << road_sign_decision
                              << " waited_ms=" << wait_ms << std::endl;
                }
                road_sign_decision_wait_started = false;
                setRoadSignForkPhase(RoadSignForkPhase::ArmSelected, frame_id);
            } else if (road_sign_decision_wait_started) {
                road_sign_wait_log_frames++;
                const int64_t wait_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - road_sign_decision_wait_start_tp).count();
                constexpr int kRoadSignDecisionTimeoutMs = 10000;
                const int timeout_ms =
                    std::min(kRoadSignDecisionTimeoutMs, std::max(1, params.roadSignFork.decisionTimeoutMs));
                if (wait_ms >= timeout_ms) {
                    road_sign_decision = params.roadSignFork.defaultDecision == 0 ? 0 : 1;
                    std::cout << "[RoadSignFork] decision timeout, fallback="
                              << road_sign_decision
                              << " waited_ms=" << wait_ms << std::endl;
                    road_sign_decision_wait_started = false;
                    setRoadSignForkPhase(RoadSignForkPhase::ArmSelected, frame_id);
                    break;
                }
                if (road_sign_wait_log_frames == 1 || road_sign_wait_log_frames % 5 == 0) {
                    std::cout << "[RoadSignFork] waiting OCR/API result: 等待 proc_frame="
                              << frame_id << " waited_ms=" << wait_ms << std::endl;
                }
            }
            break;
        case RoadSignForkPhase::ArmSelected: {
            road_sign_arm_frames++;
            return (road_sign_decision == 1)
                ? BranchPreference::ForceRight
                : BranchPreference::PreferLeft;
        }
        case RoadSignForkPhase::PassSelectedFork:
            road_sign_pass_frames++;
            if (obs.seen) {
                const bool passed_long_enough =
                    road_sign_pass_frames >=
                    std::max(1, params.roadSignFork.forkExitFrames);
                if (!road_sign_second_opposite_active &&
                    road_sign_stable &&
                    passed_long_enough &&
                    road_sign_pass_had_sign_gap) {
                    std::cout << "[RoadSignFork] new road_sign while PASS_SELECTED, restart OCR"
                              << " pass_frames=" << road_sign_pass_frames
                              << " lost_frames=" << road_sign_pass_sign_lost_frames
                              << " stable_frames=" << road_sign_seen_streak
                              << " area=" << obs.area
                              << " score=" << obs.score << std::endl;
                    beginRoadSignStopForDecision(obs, frame_id, source_frame_id);
                }
            } else {
                road_sign_pass_sign_lost_frames++;
                if (road_sign_pass_sign_lost_frames >=
                    std::max(1, params.roadSignFork.signLostHoldFrames)) {
                    road_sign_pass_had_sign_gap = true;
                    road_sign_seen_streak = 0;
                    road_sign_hold_lost_frames = 0;
                }
            }
            break;
        case RoadSignForkPhase::Cooldown:
            road_sign_cooldown_frames++;
            break;
        }

        if (road_sign_phase == RoadSignForkPhase::ArmSelected ||
            road_sign_phase == RoadSignForkPhase::PassSelectedFork) {
            return (road_sign_decision == 1) ? BranchPreference::ForceRight : BranchPreference::PreferLeft;
        }
        return BranchPreference::PreferLeft;
    };

    auto updateRoadSignStateAfterPlanner = [&](const BranchPlanMeta& branch_meta, int frame_id) {
        if (!params.roadSignFork.enable) {
            return;
        }

        switch (road_sign_phase) {
        case RoadSignForkPhase::ArmSelected: {
            const bool desired_right = (road_sign_decision == 1);
            const bool selected_route =
                branch_meta.has_fork &&
                branch_meta.visible_branch_count >= 2 &&
                (desired_right
                     ? branch_meta.selected_side == BranchSide::Right
                     : (branch_meta.selected_side == BranchSide::Left ||
                        branch_meta.selected_side == BranchSide::Main));
            if (selected_route) {
                road_sign_fork_seen_streak++;
                if (road_sign_fork_seen_streak >= std::max(1, params.roadSignFork.forkStableFrames)) {
                    setRoadSignForkPhase(RoadSignForkPhase::PassSelectedFork, frame_id);
                }
            } else {
                road_sign_fork_seen_streak = 0;
            }
            break;
        }
        case RoadSignForkPhase::PassSelectedFork:
            if (branch_meta.has_fork && branch_meta.visible_branch_count >= 2) {
                road_sign_fork_exit_streak = 0;
            } else {
                road_sign_fork_exit_streak++;
            }
            break;
        default:
            break;
        }
    };

    auto resetSpeedLightState = [&]() {
        stop_light_seen_streak = 0;
        green_light_seen_streak = 0;
        pass_light_exit_streak = 0;
        speed_light_cooldown_frames = 0;
    };

    auto setSpeedLightPhase = [&](SpeedLightPhase new_phase, int frame_id) {
        if (speed_light_phase == new_phase) {
            return;
        }

        speed_light_phase = new_phase;
        switch (speed_light_phase) {
        case SpeedLightPhase::Cruise:
            resetSpeedLightState();
            break;
        case SpeedLightPhase::RedStop:
            stop_light_seen_streak = 0;
            green_light_seen_streak = 0;
            pass_light_exit_streak = 0;
            break;
        case SpeedLightPhase::PassAfterGreen:
            stop_light_seen_streak = 0;
            green_light_seen_streak = 0;
            pass_light_exit_streak = 0;
            speed_light_cooldown_frames = 0;
            break;
        case SpeedLightPhase::Cooldown:
            stop_light_seen_streak = 0;
            green_light_seen_streak = 0;
            pass_light_exit_streak = 0;
            speed_light_cooldown_frames = 0;
            break;
        }

        std::cout << "[SpeedLight] Frame#" << frame_id
                  << " phase=" << speedLightPhaseName(speed_light_phase)
                  << std::endl;
    };

    auto isTrafficLightCloseEnough = [&](int bottom_y, int height) -> bool {
        const int min_bottom_y = std::max(0, params.speedLight.lightTriggerMinBottomY);
        const int min_height = std::max(0, params.speedLight.lightTriggerMinHeight);
        return bottom_y >= min_bottom_y || height >= min_height;
    };

    auto extractSpeedLightObservation = [&](const std::vector<PredictResult>& results) -> SpeedLightObservation {
        SpeedLightObservation obs;

        for (const auto& result : results) {
            if (result.width <= 0 || result.height <= 0) {
                continue;
            }

            const int bottom_y = result.y + result.height;
            switch (result.type) {
            case TYPE_RED_LIGHT:
                if (!obs.red_seen || result.score > obs.red_score) {
                    obs.red_seen = true;
                    obs.red_score = result.score;
                    obs.red_bottom_y = bottom_y;
                    obs.red_height = result.height;
                }
                break;
            case TYPE_YELLOW_LIGHT:
                if (!obs.yellow_seen || result.score > obs.yellow_score) {
                    obs.yellow_seen = true;
                    obs.yellow_score = result.score;
                    obs.yellow_bottom_y = bottom_y;
                    obs.yellow_height = result.height;
                }
                break;
            case TYPE_GREEN_LIGHT:
                if (!obs.green_seen || result.score > obs.green_score) {
                    obs.green_seen = true;
                    obs.green_score = result.score;
                    obs.green_bottom_y = bottom_y;
                    obs.green_height = result.height;
                }
                break;
            default:
                break;
            }
        }

        return obs;
    };

    auto updateSpeedLightState = [&](const SpeedLightObservation& obs, int frame_id) {
        if (!speedLightStateMachineEnabled()) {
            if (speed_light_phase != SpeedLightPhase::Cruise) {
                setSpeedLightPhase(SpeedLightPhase::Cruise, frame_id);
            }
            return;
        }

        const bool red_close =
            obs.red_seen && isTrafficLightCloseEnough(obs.red_bottom_y, obs.red_height);
        const bool yellow_close =
            obs.yellow_seen && isTrafficLightCloseEnough(obs.yellow_bottom_y, obs.yellow_height);
        const bool green_close =
            obs.green_seen && isTrafficLightCloseEnough(obs.green_bottom_y, obs.green_height);
        const bool stop_light_close = red_close || yellow_close;
        const bool any_light_close = stop_light_close || green_close;

        stop_light_seen_streak = stop_light_close ? (stop_light_seen_streak + 1) : 0;
        green_light_seen_streak =
            (green_close && !stop_light_close) ? (green_light_seen_streak + 1) : 0;

        switch (speed_light_phase) {
        case SpeedLightPhase::Cruise:
            if (stop_light_seen_streak >= std::max(1, params.speedLight.lightStableFrames)) {
                setSpeedLightPhase(SpeedLightPhase::RedStop, frame_id);
            } else if (green_light_seen_streak >= std::max(1, params.speedLight.greenReleaseFrames)) {
                setSpeedLightPhase(SpeedLightPhase::PassAfterGreen, frame_id);
            }
            break;
        case SpeedLightPhase::RedStop:
            if (green_light_seen_streak >= std::max(1, params.speedLight.greenReleaseFrames)) {
                setSpeedLightPhase(SpeedLightPhase::PassAfterGreen, frame_id);
            }
            break;
        case SpeedLightPhase::PassAfterGreen:
            if (stop_light_seen_streak >= std::max(1, params.speedLight.lightStableFrames)) {
                setSpeedLightPhase(SpeedLightPhase::RedStop, frame_id);
            } else if (any_light_close) {
                pass_light_exit_streak = 0;
            } else {
                pass_light_exit_streak++;
                if (pass_light_exit_streak >= std::max(1, params.speedLight.passLightExitFrames)) {
                    setSpeedLightPhase(SpeedLightPhase::Cooldown, frame_id);
                }
            }
            break;
        case SpeedLightPhase::Cooldown:
            if (stop_light_seen_streak >= std::max(1, params.speedLight.lightStableFrames)) {
                setSpeedLightPhase(SpeedLightPhase::RedStop, frame_id);
            } else if (green_light_seen_streak >= std::max(1, params.speedLight.greenReleaseFrames)) {
                setSpeedLightPhase(SpeedLightPhase::PassAfterGreen, frame_id);
            } else if (++speed_light_cooldown_frames >= std::max(1, params.speedLight.cooldownFrames)) {
                setSpeedLightPhase(SpeedLightPhase::Cruise, frame_id);
            }
            break;
        }
    };

    auto resolveSpeedCommand = [&]() -> float {
        if (!speedLightStateMachineEnabled()) {
            return speed_decision_cruise_mps;
        }

        switch (speed_light_phase) {
        case SpeedLightPhase::RedStop:
            return params.speedLight.stopSpeed;
        case SpeedLightPhase::PassAfterGreen:
        case SpeedLightPhase::Cooldown:
        case SpeedLightPhase::Cruise:
        default:
            return speed_decision_cruise_mps;
        }
    };

    auto resetStopSignStopTracking = [&]() {
        stop_sign_seen_streak = 0;
        stop_sign_last_seen_valid = false;
        stop_sign_lost_ms_display = 0;
    };

    auto setStopSignStopPhase = [&](StopSignStopPhase new_phase, int frame_id) {
        if (stop_sign_phase == new_phase) {
            return;
        }

        stop_sign_phase = new_phase;
        switch (stop_sign_phase) {
        case StopSignStopPhase::Idle:
            resetStopSignStopTracking();
            break;
        case StopSignStopPhase::SeenArmed:
            stop_sign_lost_ms_display = 0;
            break;
        case StopSignStopPhase::LostGrace:
            stop_sign_seen_streak = 0;
            break;
        case StopSignStopPhase::TerminalStop:
            stop_sign_seen_streak = 0;
            break;
        }

        std::cout << "[StopSignStop] Frame#" << frame_id
                  << " phase=" << stopSignStopPhaseName(stop_sign_phase)
                  << std::endl;
    };

    auto extractStopSignObservation = [&](const std::vector<PredictResult>& results) -> StopSignObservation {
        StopSignObservation obs;
        float best_score = -1.0f;

        for (const auto& result : results) {
            if (result.type != TYPE_STOP_SIGN || result.width <= 0 || result.height <= 0) {
                continue;
            }
            if (result.score > best_score) {
                best_score = result.score;
                obs.seen = true;
                obs.score = result.score;
            }
        }

        return obs;
    };

    auto updateStopSignStopState = [&](const StopSignObservation& obs,
                                       int frame_id,
                                       const std::chrono::steady_clock::time_point& now) {
        if (!kStopSignStopStateMachineEnabled || !params.stopSignStop.enable) {
            if (stop_sign_phase != StopSignStopPhase::Idle) {
                setStopSignStopPhase(StopSignStopPhase::Idle, frame_id);
            }
            return;
        }

        if (initialStartDelayConsumed) {
            const auto elapsed_since_start_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
            if (elapsed_since_start_ms < std::max(0, params.stopSignStop.startProtectMs)) {
                if (stop_sign_phase != StopSignStopPhase::Idle) {
                    setStopSignStopPhase(StopSignStopPhase::Idle, frame_id);
                } else {
                    resetStopSignStopTracking();
                }
                return;
            }
        }

        if (stop_sign_phase == StopSignStopPhase::TerminalStop) {
            if (stop_sign_last_seen_valid) {
                stop_sign_lost_ms_display =
                    std::chrono::duration_cast<std::chrono::milliseconds>(now - stop_sign_last_seen_tp).count();
            }
            return;
        }

        if (obs.seen) {
            stop_sign_seen_streak++;
            stop_sign_last_seen_tp = now;
            stop_sign_last_seen_valid = true;
            stop_sign_lost_ms_display = 0;
        } else {
            stop_sign_seen_streak = 0;
            if (stop_sign_last_seen_valid) {
                stop_sign_lost_ms_display =
                    std::chrono::duration_cast<std::chrono::milliseconds>(now - stop_sign_last_seen_tp).count();
            } else {
                stop_sign_lost_ms_display = 0;
            }
        }

        switch (stop_sign_phase) {
        case StopSignStopPhase::Idle:
            if (stop_sign_seen_streak >= std::max(1, params.stopSignStop.stableFrames)) {
                setStopSignStopPhase(StopSignStopPhase::SeenArmed, frame_id);
            }
            break;
        case StopSignStopPhase::SeenArmed:
            if (!obs.seen && stop_sign_last_seen_valid) {
                setStopSignStopPhase(StopSignStopPhase::LostGrace, frame_id);
            }
            break;
        case StopSignStopPhase::LostGrace:
            if (obs.seen) {
                setStopSignStopPhase(StopSignStopPhase::SeenArmed, frame_id);
            } else if (stop_sign_last_seen_valid &&
                       stop_sign_lost_ms_display >= std::max(0, params.stopSignStop.stopAfterLostMs)) {
                setStopSignStopPhase(StopSignStopPhase::TerminalStop, frame_id);
            }
            break;
        case StopSignStopPhase::TerminalStop:
            break;
        }
    };

    // Shared pedestrian stop-state machine runtime. Declared before the main
    // loop so the !uart.startFlag stop branch can reset it as well.
    static PathPlanningRuntime pp_runtime;
    const auto pedestrianBehaviorActive = [&]() {
        return pp_runtime.pedestrian_behavior.state !=
                   PedestrianStopState::DRIVING ||
               pp_runtime.pedestrian_behavior.slow_active;
    };
    PedestrianGoldDangerHoldState pedestrian_gold_danger_hold_state;

    while(1)
    {
        // ========== fixed planning dimensions ==========
        if (uart.startFlag && !prevStartFlag && !initialStartDelayConsumed) {
            // record start time
            startTime = std::chrono::steady_clock::now();
            startDelayActive = true;
            initialStartDelayConsumed = true;
            std::cout << "[Car] ? " << START_DELAY_MS << "ms" << std::endl;
        }
        prevStartFlag = uart.startFlag;

        // ========== fixed planning dimensions ==========
        if (startDelayActive) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - startTime).count();
            if (elapsed >= START_DELAY_MS) {
                startDelayActive = false;
                std::cout << "[Car] " << std::endl;
            }
        }

        if(!uart.startFlag)
        {
            resetRoadSignSlowdownTracking();
            if (road_sign_phase != RoadSignForkPhase::DefaultLeft) {
                setRoadSignForkPhase(RoadSignForkPhase::DefaultLeft, frame_idx);
            }
            // Stopping the run is a lifecycle reset, outside normal driving.
            road_sign_decision_servo_hold_active = false;
            road_sign_decision_servo_hold_output = 0.0f;
            road_sign_decision_servo_hold_stop_required = false;
            road_sign_first_pass_completed = false;
            road_sign_first_decision = -1;
            road_sign_second_opposite_used = false;
            road_sign_second_opposite_active = false;
            road_sign_last_visible_split_output_active = false;
            road_sign_last_visible_split_frame = -1;
            road_sign_pose_pd_initialized = false;
            road_sign_pose_center_initialized = false;
            road_sign_pose_last_error_deg = 0.0f;
            road_sign_pose_filtered_derivative_deg = 0.0f;
            road_sign_pose_previous_target_valid = false;
            road_sign_pose_applied_previous_frame = false;
            if (speed_light_phase != SpeedLightPhase::Cruise) {
                setSpeedLightPhase(SpeedLightPhase::Cruise, frame_idx);
            }
            if (stop_sign_phase != StopSignStopPhase::TerminalStop) {
                setStopSignStopPhase(StopSignStopPhase::Idle, frame_idx);
                resetStopSignStopTracking();
            }
            // Do not carry pedestrian observations or STOP/LAUNCH state across
            // a stop cycle. The next start re-evaluates raw geometry.
            resetPedestrianStopRuntime(pp_runtime, "start_flag_stop");
            last_applied_servo_output = 0.0f;
            last_applied_servo_valid = false;
            resetGoldBiasRuntime(pp_runtime);
            pedestrian_gold_danger_hold_state =
                PedestrianGoldDangerHoldState{};
            BlockingQueueClear(detectionPackets);
            BlockingQueueClear(videoSavePackets);
            detection.results.clear();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        // ========== 1.  ==========
        DetectionPacket packet = detectionPackets.Take();
        const auto loop_begin_tp = std::chrono::steady_clock::now();
        cv::Mat current_frame = std::move(packet.frame_640);
        std::vector<PredictResult> current_results = std::move(packet.results);
        const std::vector<PlanningObject> raw_person_objects =
            collectRawPersonObjects(current_results,
                                    cv::Size(PLAN_WIDTH, PLAN_HEIGHT),
                                    params.obstacleAvoidance.prepareDistanceM);
        const std::vector<PlanningObject> danger_strip_person_objects =
            collectRawPersonObjects(
                current_results,
                cv::Size(PLAN_WIDTH, PLAN_HEIGHT),
                std::numeric_limits<float>::max());
        RoadSignObservation road_sign_obs = extractRoadSignObservation(current_results);
        SpeedLightObservation speed_light_obs = extractSpeedLightObservation(current_results);
        StopSignObservation stop_sign_obs = extractStopSignObservation(current_results);
        const auto stop_sign_now = std::chrono::steady_clock::now();
        updateSpeedLightState(speed_light_obs, frame_idx);
        updateStopSignStopState(stop_sign_obs, frame_idx, stop_sign_now);
        std::vector<PredictResult> planning_results;
        planning_results.reserve(current_results.size());
        for (const auto& result : current_results) {
            if (result.type == TYPE_ARROW ||
                result.type == TYPE_PEOPLE ||
                result.type == TYPE_CAR ||
                result.type == TYPE_GOLD) {
                planning_results.push_back(result);
            }
        }
        const bool observed_car_this_frame = std::any_of(
            current_results.begin(),
            current_results.end(),
            [&](const PredictResult& result) {
                return result.type == TYPE_CAR &&
                       result.width > 0 && result.height > 0;
            });
        const bool obstacle_visible_this_frame =
            !raw_person_objects.empty() || observed_car_this_frame;
        if (!params.speedDecision.enable) {
            speed_decision_cruise_mps = speed_decision_clear_mps;
            speed_decision_clear_count = speed_decision_clear_frames;
        } else if (obstacle_visible_this_frame) {
            speed_decision_cruise_mps = speed_decision_obstacle_mps;
            speed_decision_clear_count = 0;
        } else if (speed_decision_clear_count < speed_decision_clear_frames) {
            speed_decision_clear_count++;
            speed_decision_cruise_mps =
                speed_decision_clear_count >= speed_decision_clear_frames
                    ? speed_decision_clear_mps
                    : speed_decision_obstacle_mps;
        } else {
            speed_decision_cruise_mps = speed_decision_clear_mps;
        }

        // ========== 3. path planning: centerline + local ========== 
        std::vector<cv::Point2f> plan_path;
        std::vector<cv::Point2f> raw_arrow_centerline_world;
        std::vector<PlanningObject> tracked_planning_objects;
        PedestrianGoldDangerDebugState pedestrian_gold_danger_debug;
        pedestrian_gold_danger_debug.configured_half_width_m =
            pedestrian_gold_danger_strip_half_width_m;
        pedestrian_gold_danger_debug.configured_hold_frames =
            pedestrian_gold_danger_strip_hold_frames;
        int current_arrow_count = 0;
        const int current_arrow_object_count =
            static_cast<int>(std::count_if(
                planning_results.begin(),
                planning_results.end(),
                [](const PredictResult& result) {
                    return result.type == TYPE_ARROW;
                }));
        LineLossTraceFields line_loss;
        line_loss.line_arrow_object_count = current_arrow_object_count;
        const float planner_cruise_speed_mps =
            std::isfinite(speed_decision_cruise_mps)
                ? std::max(0.0f, speed_decision_cruise_mps)
                : 0.0f;
        float planner_target_speed_mps = planner_cruise_speed_mps;
        bool planner_speed_valid = false;
        bool obstacle_avoidance_engaged = false;
        bool pedestrian_emergency_stop = false;
        PathPlanningDebug planner_debug;
        PathPlanningInput pp_input;
        PathPlanningResult pp_result;
        bool has_pp_result = false;
        const BranchPreference branch_preference =
            updateRoadSignStateBeforePlanner(road_sign_obs, frame_idx, packet.frame_id);
        const bool road_sign_pose_requested =
            params.roadSignFork.poseAdjustEnable &&
            road_sign_phase == RoadSignForkPhase::DefaultLeft &&
            road_sign_slowdown_active &&
            road_sign_obs.pose_box_valid &&
            road_sign_obs.pose_area >=
                std::max(0, params.roadSignFork.stopSlowdownAreaThreshold) &&
            road_sign_obs.pose_area <
                std::max(0, params.roadSignFork.signAreaThreshold);
        if (road_sign_pose_requested && !pedestrianBehaviorActive()) {
            // Road-sign pose control owns steering in this window. Clear all
            // persistent gold planning state even if the object tracker is
            // unavailable on this frame, so an old gold bias cannot survive.
            resetGoldBiasRuntime(pp_runtime);
        }
        BranchPlanMeta branch_meta;
        bool has_branch_meta = false;

        if (params.AIEnable && EnableAI)
        {
            const auto now = std::chrono::steady_clock::now();
            const int64_t timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();

            if (objectTracker) {
                objectTracker->update(planning_results, frame_idx, timestamp_us);
                tracked_planning_objects = collectPlanningObjects(
                    *objectTracker,
                    frame_idx,
                    params.goldPosition.correctionScale);
                if (road_sign_pose_requested) {
                    tracked_planning_objects.erase(
                        std::remove_if(
                            tracked_planning_objects.begin(),
                            tracked_planning_objects.end(),
                            [](const PlanningObject& object) {
                                return object.type == ObjectType::GOLD;
                            }),
                        tracked_planning_objects.end());
                }

                // ---- Step 1: centerline search ----
                // [ALIGN-DEMO] Match demo semantics: tracker_id is the
                // ObjectTracker stable id used by branch continuity.
                std::vector<ArrowGuide> arrow_guides =
                    objectTracker->getArrowGuides(14, 0.02f);
                current_arrow_count = static_cast<int>(arrow_guides.size());
                static PathPlannerRuntimeState branch_runtime;
                ArrowGuideFrameData arrow_frame;
                arrow_frame.guides = arrow_guides;
                arrow_frame.guides_before_branching = arrow_guides.size();
                arrow_frame.debug.active_arrows = static_cast<int>(arrow_guides.size());
                arrow_frame.debug.accepted = static_cast<int>(arrow_guides.size());
                arrow_frame.debug.max_arrows = static_cast<int>(arrow_guides.size());
                BranchPlanningResult branch_plan = planVisibleBranchesForFrame(
                    static_cast<uint32_t>(frame_idx),
                    arrow_frame,
                    current_arrow_object_count,
                    PLAN_WIDTH,
                    PLAN_HEIGHT,
                    branch_runtime);
                road_sign_last_visible_split_output_active =
                    branch_plan.debug.visible_split_output_active;
                road_sign_last_visible_split_frame = frame_idx;
                line_loss = branch_plan.line_loss;
                std::vector<cv::Point2f> candidate;
                bool selected_visible_right_branch = false;
                const bool upper_wants_right =
                    branch_preference == BranchPreference::ForceRight;
                const bool visible_right_branch_available =
                    std::any_of(branch_plan.visible_branches.begin(),
                                branch_plan.visible_branches.end(),
                                [](const ArrowBranch& branch) {
                                    return branch.label == "right";
                                });
                if (!branch_plan.visible_branches.empty()) {
                    branch_meta.visible_branch_count =
                        static_cast<int>(branch_plan.visible_branches.size());
                    branch_meta.has_fork = branch_plan.visible_branches.size() >= 2;
                    const char* target_label = "main";
                    if (upper_wants_right && visible_right_branch_available) {
                        target_label = "right";
                    }
                    const size_t no_selected_branch = branch_plan.visible_branches.size();
                    const size_t selected_idx = [&]() {
                        for (size_t i = 0; i < branch_plan.visible_branches.size(); ++i) {
                            if (branch_plan.visible_branches[i].label == target_label) return i;
                        }
                        return no_selected_branch;
                    }();
                    if (selected_idx != no_selected_branch) {
                        const ArrowBranch& selected_branch =
                            branch_plan.visible_branches[selected_idx];
                        BranchCenterlineResult centerline_result = buildBranchCenterlineResult(
                            selected_branch, PLAN_WIDTH, PLAN_HEIGHT);
                        selected_visible_right_branch =
                            selected_branch.label == "right";
                        line_loss.line_selected_branch_label = selected_branch.label;
                        line_loss.line_selected_branch_guide_count =
                            static_cast<int>(selected_branch.guides.size());
                        line_loss.line_selected_branch_centerline_size =
                            static_cast<int>(centerline_result.centerline.size());
                        candidate.reserve(centerline_result.centerline.size());
                        for (const auto& sample : centerline_result.centerline) {
                            candidate.emplace_back(sample.x, sample.y);
                        }
                        raw_arrow_centerline_world =
                            transformPathToWorldMeters(candidate, nullptr);
                        branch_meta.selected_side =
                            (selected_branch.label == "main" ||
                             selected_branch.label == "left")
                                ? BranchSide::Left
                                : (selected_branch.label == "right"
                                       ? BranchSide::Right
                                       : BranchSide::Main);
                    } else {
                        line_loss.line_selected_branch_label = target_label;
                        line_loss.line_selected_branch_missing = true;
                        branch_meta.selected_side = BranchSide::Main;
                    }
                }
                has_branch_meta = true;
                line_loss.line_candidate_size = static_cast<int>(candidate.size());
                line_loss.line_candidate_usable =
                    isPathUsable(candidate, PLAN_WIDTH, PLAN_HEIGHT);

                const std::vector<PedestrianGoldDangerStrip>
                    current_pedestrian_gold_danger_strips =
                        buildPedestrianGoldDangerStrips(
                            danger_strip_person_objects,
                            raw_arrow_centerline_world,
                            pedestrian_gold_danger_strip_half_width_m);
                updatePedestrianGoldDangerHold(
                    current_pedestrian_gold_danger_strips,
                    !danger_strip_person_objects.empty(),
                    pedestrian_gold_danger_strip_hold_frames,
                    pedestrian_gold_danger_hold_state,
                    pedestrian_gold_danger_debug);
                filterDangerStripGolds(
                    pedestrian_gold_danger_debug.strips,
                    tracked_planning_objects,
                    pedestrian_gold_danger_debug.ignored_gold_world);

                // ---- Step 2: path planning (enable_path_planning config) ----
                if (params.enablePathPlanning) {
                    pp_input.visible_branch_count = branch_meta.visible_branch_count;
                    pp_input.external_route_decision_valid =
                        has_branch_meta && branch_meta.selected_side != BranchSide::None &&
                        !candidate.empty();
                    pp_input.frame_size = cv::Size(PLAN_WIDTH, PLAN_HEIGHT);
                    pp_input.timestamp_sec = static_cast<double>(timestamp_us) / 1000000.0;
                    pp_input.timestamp_valid = true;
                    pp_input.pedestrian_cruise_speed_mps =
                        planner_cruise_speed_mps;
                    pp_input.obstacle_avoid_enable = params.obstacleAvoidance.enable;
                    pp_input.obstacle_runtime_enabled = !startDelayActive;
                    pp_input.obstacle_confirm_frames = params.obstacleAvoidance.confirmFrames;
                    pp_input.obstacle_exit_lost_frames = params.obstacleAvoidance.exitLostFrames;
                    pp_input.vehicle_recover_hold_frames =
                        params.obstacleAvoidance.vehicleAvoidanceRecoverHoldFrames;
                    pp_input.vehicle_recover_hold_max_lifetime_frames =
                        params.obstacleAvoidance.vehicleRecoverHoldMaxLifetimeFrames;
                    pp_input.obstacle_prepare_distance_m =
                        params.obstacleAvoidance.prepareDistanceM;
                    pp_input.obstacle_activate_distance_m =
                        params.obstacleAvoidance.activateDistanceM;
                    pp_input.vehicle_length_m = params.obstacleAvoidance.vehicleLengthM;
                    pp_input.vehicle_width_m = params.obstacleAvoidance.vehicleWidthM;
                    pp_input.vehicle_front_m = params.obstacleAvoidance.vehicleFrontM;
                    pp_input.vehicle_rear_m = params.obstacleAvoidance.vehicleRearM;
                    pp_input.force_left_obstacle_avoidance =
                        params.obstacleAvoidance.forceLeftAvoidance;
                    pp_input.obstacle_avoid_offset_m = params.obstacleAvoidance.avoidOffsetM;
                    pp_input.vehicle_avoidance_edge_clearance_m =
                        params.obstacleAvoidance.vehicleAvoidanceEdgeClearanceM;
                    pp_input.vehicle_avoidance_max_offset_m =
                        params.obstacleAvoidance.vehicleAvoidanceMaxOffsetM;
                    pp_input.vehicle_curve_avoidance_extra_offset_m =
                        params.obstacleAvoidance.vehicleCurveAvoidanceExtraOffsetM;
                    pp_input.vehicle_avoidance_opposite_steer_threshold_deg =
                        params.obstacleAvoidance.vehicleAvoidanceOppositeSteerThresholdDeg;
                    pp_input.vehicle_avoidance_opposite_steer_hold_frames =
                        params.obstacleAvoidance.vehicleAvoidanceOppositeSteerHoldFrames;
                    pp_input.steering_angle_deg = last_planner_steering_angle_deg;
                    pp_input.steering_angle_valid = last_planner_steering_angle_valid;
                    pp_input.obstacle_offset_rate_mps =
                        params.obstacleAvoidance.offsetRateMps;
                    pp_input.obstacle_offset_zero_epsilon_m =
                        params.obstacleAvoidance.offsetZeroEpsilonM;
                    pp_input.pedestrian_danger_corridor_near_width_m =
                        params.obstacleAvoidance.pedestrianDangerCorridorNearWidthM;
                    pp_input.pedestrian_danger_corridor_far_width_m =
                        params.obstacleAvoidance.pedestrianDangerCorridorFarWidthM;
                    pp_input.pedestrian_danger_corridor_length_m =
                        params.obstacleAvoidance.pedestrianDangerCorridorLengthM;
                    pp_input.pedestrian_danger_corridor_sample_step_m =
                        params.obstacleAvoidance.pedestrianDangerCorridorSampleStepM;
                    pp_input.pedestrian_servo_straight_deadband =
                        params.obstacleAvoidance.pedestrianServoStraightDeadband;
                    pp_input.pedestrian_servo_radius_max_abs =
                        params.obstacleAvoidance.pedestrianServoRadiusMaxAbs;
                    pp_input.pedestrian_danger_corridor_servo_smooth_alpha =
                        params.obstacleAvoidance
                            .pedestrianDangerCorridorServoSmoothAlpha;
                    pp_input.pedestrian_danger_corridor_servo_max_step =
                        params.obstacleAvoidance
                            .pedestrianDangerCorridorServoMaxStep;
                    pp_input.pedestrian_slow_enable =
                        params.obstacleAvoidance.pedestrianSlowEnable;
                    pp_input.pedestrian_slow_distance_m =
                        params.obstacleAvoidance.pedestrianSlowDistanceM;
                    pp_input.pedestrian_slow_speed_mps =
                        params.obstacleAvoidance.pedestrianSlowSpeedMps;
                    pp_input.pedestrian_behavior_config.motion_window_frames =
                        params.obstacleAvoidance.pedestrianMotionWindowFrames;
                    pp_input.pedestrian_behavior_config.motion_trend_threshold_m =
                        params.obstacleAvoidance.pedestrianMotionTrendThresholdM;
                    pp_input.pedestrian_behavior_config.motion_max_jump_m =
                        params.obstacleAvoidance.pedestrianMotionMaxJumpM;
                    pp_input.pedestrian_behavior_config.slow_away_confirm_frames =
                        params.obstacleAvoidance.pedestrianSlowAwayConfirmFrames;
                    pp_input.pedestrian_behavior_config.slow_observation_frames =
                        params.obstacleAvoidance.pedestrianSlowObservationFrames;
                    pp_input.pedestrian_behavior_config.slow_lost_frames =
                        params.obstacleAvoidance.pedestrianSlowLostFrames;
                    pp_input.pedestrian_behavior_config.stop_clear_frames =
                        params.obstacleAvoidance.pedestrianStopClearFrames;
                    pp_input.pedestrian_behavior_config.release_lost_grace_frames =
                        params.obstacleAvoidance.pedestrianReleaseLostGraceFrames;
                    pp_input.pedestrian_behavior_config.exit_boundary_margin_m =
                        params.obstacleAvoidance.pedestrianExitBoundaryMarginM;
                    pp_input.pedestrian_behavior_config.stop_car_side_threshold_m =
                        params.obstacleAvoidance
                            .pedestrianStopCarSideThresholdM;
                    pp_input.pedestrian_behavior_config.launch_allowed_motion_frames =
                        params.obstacleAvoidance.startupLaunchProtectFrames;
                    pp_input.pedestrian_normal_strip_half_width_m =
                        pedestrian_gold_danger_strip_half_width_m;
                    pp_input.pedestrian_stopped_launch_speed_mps =
                        params.obstacleAvoidance
                            .pedestrianStoppedLaunchSpeedMps;
                    pp_input.pedestrian_slow_launch_speed_mps =
                        params.obstacleAvoidance.pedestrianSlowLaunchSpeedMps;
                    pp_input.gold_bias_step_m =
                        params.goldPathPlanning.goldBiasStepM;
                    pp_input.gold_max_bias_m =
                        params.goldPathPlanning.goldMaxBiasM;
                    pp_input.gold_capture_compensation_m =
                        params.goldPathPlanning.goldCaptureCompensationM;
                    pp_input.gold_capture_tolerance_m =
                        params.goldPathPlanning.goldCaptureToleranceM;
                    pp_input.gold_bias_rate_mps =
                        params.goldPathPlanning.goldBiasRateMps;
                    pp_input.gold_max_new_delta_m =
                        params.goldPathPlanning.goldMaxNewDeltaM;
                    pp_input.gold_recenter_hold_frames =
                        params.goldPathPlanning.goldRecenterHoldFrames;
                    pp_input.gold_near_priority_distance_m =
                        params.goldPathPlanning.goldNearPriorityDistanceM;
                    pp_input.gold_late_start_forward_m =
                        params.goldPathPlanning.goldLateStartForwardM;
                    pp_input.gold_late_start_delta_m =
                        params.goldPathPlanning.goldLateStartDeltaM;
                    pp_input.gold_center_release_m =
                        params.goldPathPlanning.goldCenterReleaseM;
                    pp_input.path_smooth_window = params.obstacleAvoidance.pathSmoothWindow;
                    pp_input.avoid_lookahead_distance_m =
                        params.obstacleAvoidance.avoidLookaheadDistanceM;
                    LowerTelemetry telemetry;
                    if (getLowerTelemetry(telemetry)) {
                        pp_input.ego_speed_mps = telemetry.speed_mps;
                        pp_input.ego_speed_valid =
                            std::isfinite(pp_input.ego_speed_mps);
                    }

                    pp_input.objects = tracked_planning_objects;
                    pp_input.raw_person_objects = raw_person_objects;

                    pp_input.road_corridors = branch_plan.road_corridors;
                    pp_input.route = buildPlanningRouteFromRoadCorridors(
                        branch_plan.road_corridors,
                        PLAN_WIDTH,
                        PLAN_HEIGHT,
                        branch_meta.selected_side == BranchSide::Left
                            ? RoadCorridorRole::LeftBranch
                            : (branch_meta.selected_side == BranchSide::Right
                                   ? RoadCorridorRole::RightBranch
                                   : RoadCorridorRole::Accepted),
                        !selected_visible_right_branch);
                    if (selected_visible_right_branch &&
                        pp_input.route.empty() &&
                        isPathUsable(candidate, PLAN_WIDTH, PLAN_HEIGHT)) {
                        pp_input.route.reserve(candidate.size());
                        for (const auto& point_px : candidate) {
                            PlanningRoutePoint route_point;
                            route_point.pos = point_px;
                            route_point.left = point_px.x - route_point.half_width;
                            route_point.right = point_px.x + route_point.half_width;
                            route_point.has_bounds = false;
                            pp_input.route.push_back(route_point);
                        }
                    }
                    line_loss.line_local_route_size =
                        static_cast<int>(pp_input.route.size());

                    // ---- Step 3: local path planning ----
                    pp_result = planLocalPath(pp_input, pp_runtime);
                    has_pp_result = true;
                    planner_debug = pp_result.debug;
                    obstacle_avoidance_engaged = pp_result.obstacle_avoidance_engaged;
                    line_loss.line_local_waypoints_size =
                        static_cast<int>(pp_result.waypoints.size());
                    line_loss.line_local_usable =
                        isPathUsable(pp_result.waypoints, PLAN_WIDTH, PLAN_HEIGHT);

                    const bool candidate_usable = line_loss.line_candidate_usable;
                    const bool local_has_waypoints =
                        pp_result.planning_enabled && !pp_result.waypoints.empty();
                    const bool local_usable =
                        local_has_waypoints && line_loss.line_local_usable;

                    if (local_usable) {
                        plan_path = std::move(pp_result.waypoints);
                        stale_path_count = 0;
                    } else if (candidate_usable) {
                        plan_path = std::move(candidate);
                        stale_path_count = 0;
                        if (pp_result.planning_enabled) {
                            line_loss.line_used_candidate_fallback = true;
                            line_loss.line_local_reject_reason =
                                pp_result.waypoints.empty()
                                    ? "local_waypoints_empty"
                                    : "local_waypoints_unusable";
                        }
                    } else {
                        stale_path_count++;
                    }
                } else {
                    
                    if (isPathUsable(candidate, PLAN_WIDTH, PLAN_HEIGHT)) {
                        plan_path = std::move(candidate);
                        stale_path_count = 0;
                    } else {
                        stale_path_count++;
                    }
                }

                if (stale_path_count > 8) {
                    plan_path.clear();
                    stale_path_count = 0;
                }

                if (!isPathUsable(plan_path, PLAN_WIDTH, PLAN_HEIGHT)) {
                    plan_path.clear();
                }
                line_loss.line_final_plan_path_size =
                    static_cast<int>(plan_path.size());
                line_loss.line_final_plan_path_usable =
                    isPathUsable(plan_path, PLAN_WIDTH, PLAN_HEIGHT);

            }
        }

        if (has_branch_meta) {
            updateRoadSignStateAfterPlanner(branch_meta, frame_idx);
        }

        const bool stop_sign_terminal_stop =
            kStopSignStopStateMachineEnabled &&
            (stop_sign_phase == StopSignStopPhase::TerminalStop);
        const bool road_sign_decision_stop =
            (road_sign_phase == RoadSignForkPhase::StopForDecision ||
             road_sign_phase == RoadSignForkPhase::WaitDecision);

        std::vector<cv::Point2f> world_path =
            transformPathToWorldMeters(plan_path, &world_invalid_points);
        line_loss.line_world_path_size = static_cast<int>(world_path.size());
        line_loss.line_positive_world_y_count =
            static_cast<int>(std::count_if(
                world_path.begin(),
                world_path.end(),
                [](const cv::Point2f& point) {
                    return std::isfinite(point.y) && point.y > 0.0f;
                }));
        bool virtual_speed_override = false;
        const bool virtual_had_previous_frame = virtual_lookahead_state.has_point;
        bool virtual_arrows_recovered = false;
        bool virtual_arrows_low = false;
        if (!params.virtualLookahead.enable) {
            virtual_lookahead_state = VirtualLookaheadState{};
        } else {
            cv::Point2f path_endpoint_world(0.0f, 0.0f);
            const bool has_path_endpoint =
                lastFiniteForwardPathPoint(world_path, path_endpoint_world);
            virtual_arrows_recovered =
                current_arrow_count >= virtual_recover_arrow_min_count;
            virtual_arrows_low =
                current_arrow_count <= virtual_low_arrow_max_count;

            if (virtual_arrows_recovered) {
                virtual_lookahead_state.low_arrow_frames = 0;
                if (has_path_endpoint) {
                    virtual_lookahead_state.point = path_endpoint_world;
                    virtual_lookahead_state.has_point = true;
                }
            } else if (virtual_arrows_low) {
                virtual_speed_override = virtual_had_previous_frame || has_path_endpoint;
                virtual_lookahead_state.low_arrow_frames++;
                if (virtual_had_previous_frame) {
                    virtual_lookahead_state.has_point =
                        std::isfinite(virtual_lookahead_state.point.x) &&
                        std::isfinite(virtual_lookahead_state.point.y);
                } else if (has_path_endpoint) {
                    virtual_lookahead_state.point = path_endpoint_world;
                    virtual_lookahead_state.has_point = true;
                }
                virtual_lookahead_state.locked = virtual_lookahead_state.has_point;
            } else if (has_path_endpoint) {
                virtual_lookahead_state.point = path_endpoint_world;
                virtual_lookahead_state.has_point = true;
                virtual_lookahead_state.locked = false;
            }
        }
        pursuit_target = cv::Point2f(0.0f, 0.0f);
        target_distance = 0.0f;
        steering_angle = 0.0f;
        servo_output = 0.0f;
        has_pursuit_target = false;
        bool virtual_fixed_servo_active = false;
        float virtual_fixed_servo_output = 0.0f;

        if (params.worldPurePursuitParams.enable) {
            const float current_ahead_distance =
                planner_debug.avoid_lookahead_active &&
                        planner_debug.recommended_lookahead_m > 0.0f
                    ? planner_debug.recommended_lookahead_m
                    : base_pure_ahead_distance;
            has_pursuit_target = selectPurePursuitTarget(
                world_path, current_ahead_distance, pursuit_target, target_distance);

            const bool virtual_cancel_ready =
                params.virtualLookahead.enable &&
                virtual_arrows_recovered &&
                has_pursuit_target;
            if (virtual_cancel_ready) {
                virtual_lookahead_state.control_active = false;
                virtual_lookahead_state.locked = false;
            } else if (params.virtualLookahead.enable &&
                       virtual_had_previous_frame &&
                       !has_pursuit_target) {
                virtual_lookahead_state.control_active = true;
                virtual_lookahead_state.locked = true;
                virtual_speed_override = true;
            } else if (virtual_arrows_low &&
                       !virtual_lookahead_state.control_active &&
                       virtual_had_previous_frame &&
                       target_distance <= virtual_activate_distance_m) {
                virtual_lookahead_state.control_active = true;
            }
            if (virtual_lookahead_state.control_active) {
                virtual_speed_override = true;
            }
            if (virtual_lookahead_state.control_active &&
                virtual_lookahead_state.has_point) {
                const float virtual_distance =
                    std::hypot(virtual_lookahead_state.point.x,
                               virtual_lookahead_state.point.y);
                if (std::isfinite(virtual_distance) &&
                    virtual_distance > 1e-4f) {
                    const float side_sign =
                        virtual_lookahead_state.point.x >= 0.0f ? 1.0f : -1.0f;
                    virtual_fixed_servo_active = true;
                    virtual_fixed_servo_output =
                        side_sign * virtual_fixed_servo_output_abs;
                }
            }

            if (has_pursuit_target) {
                line_loss.line_servo_source = "pure_pursuit";
                steering_angle = computePurePursuitSteeringDeg(
                    pursuit_target, target_distance, pure_wheel_base);
                if (!debug1) {
                    servo_output = servor.servoControl(steering_angle);
                    if (servo_output > servo_output_limit) servo_output = servo_output_limit;
                    if (servo_output < -servo_output_limit) servo_output = -servo_output_limit;
                }
            }
            if (virtual_fixed_servo_active) {
                line_loss.line_servo_source = "other";
                steering_angle = virtual_fixed_servo_output;
                servo_output = virtual_fixed_servo_output;
                if (servo_output > servo_output_limit) servo_output = servo_output_limit;
                if (servo_output < -servo_output_limit) servo_output = -servo_output_limit;
            }
        }

        const float road_sign_pose_safety_block_distance_m =
            std::max(0.0f, params.roadSignFork.poseSafetyBlockDistanceM);
        const bool road_sign_pose_near_person = std::any_of(
            raw_person_objects.begin(),
            raw_person_objects.end(),
            [&](const PlanningObject& person) {
                return person.type == ObjectType::PERSON &&
                       person.observed_this_frame &&
                       person.current_frame_visible &&
                       person.has_world_pos_m &&
                       pointToVehicleBodyDistanceMeters(
                           person.world_pos_m,
                           params.obstacleAvoidance.vehicleFrontM,
                           params.obstacleAvoidance.vehicleRearM,
                           params.obstacleAvoidance.vehicleWidthM) <=
                           road_sign_pose_safety_block_distance_m;
            });
        const bool road_sign_pose_near_vehicle = std::any_of(
            tracked_planning_objects.begin(),
            tracked_planning_objects.end(),
            [&](const PlanningObject& object) {
                return object.type == ObjectType::CAR &&
                       object.observed_this_frame &&
                       object.current_frame_visible &&
                       object.has_world_pos_m &&
                       pointToVehicleBodyDistanceMeters(
                           object.world_pos_m,
                           params.obstacleAvoidance.vehicleFrontM,
                           params.obstacleAvoidance.vehicleRearM,
                           params.obstacleAvoidance.vehicleWidthM) <=
                           road_sign_pose_safety_block_distance_m;
            });
        const bool road_sign_pose_safety_blocked =
            road_sign_pose_near_person ||
            road_sign_pose_near_vehicle;

        RoadSignPoseDebugState road_sign_pose_debug;
        road_sign_pose_debug.area = road_sign_obs.pose_area;
        if (!params.roadSignFork.poseAdjustEnable) {
            road_sign_pose_debug.reason = "disabled";
        } else if (!road_sign_pose_requested ||
                   startDelayActive ||
                   stop_sign_terminal_stop ||
                   road_sign_decision_stop) {
            road_sign_pose_debug.reason = "outside_adjust_window";
        } else {
            road_sign_pose_debug.raw_center_x_px =
                static_cast<float>(road_sign_obs.pose_box.x) +
                static_cast<float>(road_sign_obs.pose_box.width) * 0.5f;
            if (!road_sign_pose_center_initialized) {
                road_sign_pose_filtered_center_x_px =
                    road_sign_pose_debug.raw_center_x_px;
                road_sign_pose_center_initialized = true;
            } else {
                road_sign_pose_filtered_center_x_px =
                    road_sign_pose_center_filter_alpha *
                        road_sign_pose_filtered_center_x_px +
                    (1.0f - road_sign_pose_center_filter_alpha) *
                        road_sign_pose_debug.raw_center_x_px;
            }
            road_sign_pose_debug.filtered_center_x_px =
                road_sign_pose_filtered_center_x_px;
            road_sign_pose_debug.candidate_valid = buildRoadSignVirtualTarget(
                road_sign_pose_filtered_center_x_px,
                road_sign_pose_virtual_radius_m,
                road_sign_pose_debug.target_world,
                road_sign_pose_debug.target_pixel);
            if (!road_sign_pose_debug.candidate_valid) {
                road_sign_pose_debug.reason = "virtual_target_failed";
            } else if (road_sign_pose_safety_blocked) {
                // Keep the target calculation available for diagnostics, but
                // do not apply or render it while obstacle safety has priority.
                road_sign_pose_debug.reason = road_sign_pose_near_person
                    ? "blocked_near_pedestrian"
                    : "blocked_near_vehicle";
            } else {
                road_sign_pose_debug.active = true;
                road_sign_pose_debug.reason = "active";
            }
        }

        road_sign_pose_debug.entered =
            road_sign_pose_debug.active &&
            !road_sign_pose_applied_previous_frame;
        if (road_sign_pose_debug.active) {
            bool center_pixel_valid = false;
            const cv::Point2f center_pixel = worldToPixelMeters(
                cv::Point2f(0.0f, road_sign_pose_virtual_radius_m),
                &center_pixel_valid);
            const bool within_deadband =
                center_pixel_valid &&
                std::abs(road_sign_pose_filtered_center_x_px - center_pixel.x) <=
                    road_sign_pose_center_deadband_px;
            const float road_sign_heading_error_deg = within_deadband
                ? 0.0f
                : std::atan2(road_sign_pose_debug.target_world.x,
                             road_sign_pose_debug.target_world.y) *
                      180.0f / static_cast<float>(CV_PI);
            const bool target_jump_reset =
                road_sign_pose_previous_target_valid &&
                cv::norm(road_sign_pose_debug.target_world -
                         road_sign_pose_previous_target_world) >
                    road_sign_pose_target_jump_reset_m;
            const bool reset_derivative =
                within_deadband ||
                !road_sign_pose_pd_initialized ||
                road_sign_pose_debug.entered ||
                target_jump_reset;
            float raw_derivative_deg = 0.0f;
            if (reset_derivative) {
                road_sign_pose_filtered_derivative_deg = 0.0f;
                road_sign_pose_debug.derivative_reset = true;
            } else {
                raw_derivative_deg = std::clamp(
                    road_sign_heading_error_deg -
                        road_sign_pose_last_error_deg,
                    -road_sign_pose_derivative_limit_deg,
                    road_sign_pose_derivative_limit_deg);
                road_sign_pose_filtered_derivative_deg =
                    road_sign_pose_derivative_alpha *
                        road_sign_pose_filtered_derivative_deg +
                    (1.0f - road_sign_pose_derivative_alpha) *
                        raw_derivative_deg;
            }

            const float pose_servo_output =
                params.roadSignFork.poseKp * road_sign_heading_error_deg +
                params.roadSignFork.poseKd *
                    road_sign_pose_filtered_derivative_deg;
            servo_output = std::clamp(
                pose_servo_output,
                -road_sign_pose_servo_limit,
                road_sign_pose_servo_limit);
            steering_angle = road_sign_heading_error_deg;
            line_loss.line_servo_source = "road_sign_pose_pd";
            road_sign_pose_debug.applied = true;
            road_sign_pose_debug.reason = within_deadband
                ? "center_deadband"
                : "active";
            road_sign_pose_debug.heading_error_deg =
                road_sign_heading_error_deg;
            road_sign_pose_debug.filtered_derivative_deg =
                road_sign_pose_filtered_derivative_deg;
            road_sign_pose_debug.servo_output = servo_output;
            road_sign_pose_last_error_deg = road_sign_heading_error_deg;
            road_sign_pose_pd_initialized = true;
            road_sign_pose_previous_target_world =
                road_sign_pose_debug.target_world;
            road_sign_pose_previous_target_valid = true;
        } else {
            road_sign_pose_pd_initialized = false;
            road_sign_pose_center_initialized = false;
            road_sign_pose_last_error_deg = 0.0f;
            road_sign_pose_filtered_derivative_deg = 0.0f;
            road_sign_pose_previous_target_valid = false;
        }
        road_sign_pose_debug.exited =
            !road_sign_pose_debug.applied &&
            road_sign_pose_applied_previous_frame;
        road_sign_pose_applied_previous_frame =
            road_sign_pose_debug.applied;

        GoldTerminalDebugState gold_terminal_debug;
        GoldTerminalCandidate gold_terminal_candidate;
        if (!pedestrianBehaviorActive()) {
            gold_terminal_candidate = selectNearestTerminalGold(
                tracked_planning_objects,
                world_path,
                gold_terminal_path_tolerance_m);
        }
        gold_terminal_debug.candidate_valid = gold_terminal_candidate.valid;
        if (gold_terminal_candidate.valid) {
            gold_terminal_debug.target_world = gold_terminal_candidate.world;
            gold_terminal_debug.distance_to_ego_m =
                gold_terminal_candidate.distance_to_ego_m;
            gold_terminal_debug.path_distance_m =
                gold_terminal_candidate.path_distance_m;
            gold_terminal_debug.gold_along_path_s_m =
                gold_terminal_candidate.along_path_s_m;
        }

        WorldPathProjection lookahead_projection;
        const WorldPathProjection ego_projection =
            projectPointToWorldPath(cv::Point2f(0.0f, 0.0f), world_path);
        if (has_pursuit_target) {
            lookahead_projection =
                projectPointToWorldPath(pursuit_target, world_path);
        }
        if (gold_terminal_candidate.valid && ego_projection.valid) {
            gold_terminal_debug.gold_along_path_s_m =
                gold_terminal_candidate.along_path_s_m -
                ego_projection.along_path_s_m;
        }
        if (lookahead_projection.valid && ego_projection.valid) {
            gold_terminal_debug.lookahead_along_path_s_m =
                lookahead_projection.along_path_s_m -
                ego_projection.along_path_s_m;
        }

        if (road_sign_pose_debug.applied) {
            gold_terminal_debug.reason = "suppressed_by_road_sign_pose";
        } else if (pedestrianBehaviorActive()) {
            gold_terminal_debug.reason = "pedestrian_behavior_active";
        } else if (!params.goldPathPlanning.terminalPoseEnable) {
            gold_terminal_debug.reason = "disabled";
        } else if (!gold_terminal_candidate.valid) {
            gold_terminal_debug.reason = "no_candidate";
        } else if (!has_pursuit_target || !lookahead_projection.valid ||
                   !ego_projection.valid) {
            gold_terminal_debug.reason = "no_lookahead";
        } else if (gold_terminal_debug.gold_along_path_s_m <= 0.0f) {
            gold_terminal_debug.reason = "gold_at_or_behind_ego";
        } else if (gold_terminal_debug.gold_along_path_s_m >=
                   gold_terminal_debug.lookahead_along_path_s_m) {
            gold_terminal_debug.reason = "gold_at_or_beyond_lookahead";
        } else if (gold_terminal_candidate.distance_to_ego_m >=
                   gold_terminal_activation_distance_m) {
            gold_terminal_debug.reason = "outside_activation_distance";
        } else {
            gold_terminal_debug.active = true;
            gold_terminal_debug.reason = "active";
        }
        gold_terminal_debug.entered =
            gold_terminal_debug.active && !gold_terminal_applied_previous_frame;

        if (gold_terminal_debug.active) {
            const float terminal_heading_error_deg =
                std::atan2(gold_terminal_candidate.world.x,
                           gold_terminal_candidate.world.y) *
                180.0f / static_cast<float>(CV_PI);
            const bool target_jump_reset =
                gold_terminal_previous_target_valid &&
                cv::norm(gold_terminal_candidate.world -
                         gold_terminal_previous_target_world) >
                    gold_terminal_target_jump_reset_m;
            const bool reset_derivative =
                !gold_terminal_pd_initialized ||
                gold_terminal_debug.entered ||
                target_jump_reset;
            float terminal_raw_derivative_deg = 0.0f;
            if (reset_derivative) {
                gold_terminal_filtered_derivative_deg = 0.0f;
                gold_terminal_debug.derivative_reset = true;
            } else {
                terminal_raw_derivative_deg = std::clamp(
                    terminal_heading_error_deg -
                        gold_terminal_last_error_deg,
                    -gold_terminal_derivative_limit_deg,
                    gold_terminal_derivative_limit_deg);
                gold_terminal_filtered_derivative_deg =
                    gold_terminal_derivative_alpha *
                        gold_terminal_filtered_derivative_deg +
                    (1.0f - gold_terminal_derivative_alpha) *
                        terminal_raw_derivative_deg;
            }

            const float terminal_servo_output =
                params.goldPathPlanning.terminalPoseKp *
                    terminal_heading_error_deg +
                params.goldPathPlanning.terminalPoseKd *
                    gold_terminal_filtered_derivative_deg;
            servo_output = std::clamp(terminal_servo_output,
                                      -gold_terminal_servo_limit,
                                      gold_terminal_servo_limit);
            steering_angle = terminal_heading_error_deg;
            line_loss.line_servo_source = "gold_terminal_pose";
            gold_terminal_debug.applied = true;
            gold_terminal_debug.reason = "active";
            gold_terminal_debug.heading_error_deg =
                terminal_heading_error_deg;
            gold_terminal_debug.filtered_derivative_deg =
                gold_terminal_filtered_derivative_deg;
            gold_terminal_debug.servo_output = servo_output;
            gold_terminal_last_error_deg = terminal_heading_error_deg;
            gold_terminal_pd_initialized = true;
            gold_terminal_previous_target_world =
                gold_terminal_candidate.world;
            gold_terminal_previous_target_valid = true;
        } else {
            gold_terminal_pd_initialized = false;
            gold_terminal_last_error_deg = 0.0f;
            gold_terminal_filtered_derivative_deg = 0.0f;
            gold_terminal_previous_target_valid = false;
        }
        if (stop_sign_terminal_stop) {
            line_loss.line_servo_source = "other";
            steering_angle = 0.0f;
            servo_output = 0.0f;
        }
        if (road_sign_decision_servo_hold_active) {
            bool release_servo_hold = false;
            if (road_sign_decision_servo_hold_stop_required) {
                LowerTelemetry telemetry;
                release_servo_hold =
                    getLowerTelemetry(telemetry) &&
                    std::isfinite(telemetry.speed_mps) &&
                    std::fabs(telemetry.speed_mps) <= 0.10f;
            } else {
                const int hold_ms =
                    std::max(0, params.roadSignFork.decisionServoHoldMs);
                const int64_t elapsed_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() -
                        road_sign_decision_servo_hold_start_tp).count();
                release_servo_hold = elapsed_ms >= hold_ms;
            }
            if (release_servo_hold) {
                road_sign_decision_servo_hold_active = false;
                road_sign_decision_servo_hold_output = 0.0f;
                road_sign_decision_servo_hold_stop_required = false;
            }

            // The first OCR-selected pass keeps the existing stop-until-
            // stationary behavior. The second opposite-only pass applies the
            // same directional output while moving, then releases it by time.
            if (road_sign_decision_servo_hold_active) {
                servo_output = road_sign_decision_servo_hold_output;
                if (servo_output > servo_output_limit) servo_output = servo_output_limit;
                if (servo_output < -servo_output_limit) servo_output = -servo_output_limit;
                steering_angle = servo_output;
                line_loss.line_servo_source = "other";
            }
        }
        if (gold_terminal_debug.active &&
            line_loss.line_servo_source != "gold_terminal_pose") {
            gold_terminal_debug.applied = false;
            gold_terminal_debug.reason = "suppressed_by_upper_control";
            gold_terminal_pd_initialized = false;
            gold_terminal_last_error_deg = 0.0f;
            gold_terminal_filtered_derivative_deg = 0.0f;
            gold_terminal_previous_target_valid = false;
        }
        gold_terminal_debug.servo_output =
            gold_terminal_debug.applied ? servo_output : 0.0f;
        gold_terminal_debug.entered =
            gold_terminal_debug.applied &&
            !gold_terminal_applied_previous_frame;
        gold_terminal_debug.exited =
            !gold_terminal_debug.applied &&
            gold_terminal_applied_previous_frame;
        gold_terminal_applied_previous_frame = gold_terminal_debug.applied;
        last_planner_steering_angle_deg = steering_angle;
        last_planner_steering_angle_valid =
            std::isfinite(steering_angle) &&
            has_pursuit_target &&
            (line_loss.line_servo_source == "pure_pursuit" ||
             line_loss.line_servo_source == "gold_terminal_pose" ||
             line_loss.line_servo_source == "road_sign_pose_pd");
        line_loss.line_pure_pursuit_target_valid = has_pursuit_target;
        bool pedestrian_control_target_valid = false;
        if (has_pp_result) {
            if (virtual_fixed_servo_active) {
                pedestrian_control_target_valid =
                    virtual_lookahead_state.has_point &&
                    std::isfinite(virtual_lookahead_state.point.x) &&
                    std::isfinite(virtual_lookahead_state.point.y);
            } else if (has_pursuit_target) {
                pedestrian_control_target_valid = true;
            }
            const bool current_servo_for_corridor_valid =
                std::isfinite(servo_output) &&
                line_loss.line_servo_source != "no_target_hold_or_default";
            finalizePedestrianBehavior(
                pp_input,
                servo_output,
                current_servo_for_corridor_valid,
                line_loss.line_servo_source,
                last_applied_servo_output,
                last_applied_servo_valid,
                pp_runtime,
                pp_result);
            if (current_servo_for_corridor_valid) {
                last_applied_servo_output = servo_output;
                last_applied_servo_valid = true;
            }
            planner_target_speed_mps = pp_result.target_speed_mps;
            planner_speed_valid =
                std::isfinite(planner_target_speed_mps) && planner_target_speed_mps >= 0.0f;
            planner_debug = pp_result.debug;
            pedestrian_emergency_stop = pp_result.pedestrian_emergency_stop;
        }

        const float traffic_target_speed = stop_sign_terminal_stop ? 0.0f : resolveSpeedCommand();
        const bool traffic_speed_override =
            speedLightStateMachineEnabled() &&
            speed_light_phase == SpeedLightPhase::RedStop;
        const float planner_or_global_speed =
            planner_speed_valid ? planner_target_speed_mps : speed_decision_cruise_mps;
        const bool upper_stop_active =
            startDelayActive ||
            stop_sign_terminal_stop ||
            road_sign_decision_stop ||
            (road_sign_decision_servo_hold_active &&
             road_sign_decision_servo_hold_stop_required) ||
            (traffic_speed_override && traffic_target_speed <= 0.0f);
        const bool pedestrian_stop_wait =
            has_pp_result &&
            pp_result.pedestrian_stop_state == PedestrianStopState::STOPPED_WAIT;
        const bool pedestrian_launch_committed =
            has_pp_result &&
            pp_result.pedestrian_stop_state == PedestrianStopState::LAUNCH_COMMITTED;
        const bool planner_path_unavailable =
            has_pp_result &&
            (!planner_speed_valid ||
             ((pp_result.pedestrian_stop_state == PedestrianStopState::STOPPED_WAIT ||
               pp_result.pedestrian_stop_state == PedestrianStopState::LAUNCH_COMMITTED) &&
              !pedestrian_control_target_valid));
        std::string final_speed_source = "planner";
        float target_speed = planner_or_global_speed;
        if (traffic_speed_override) {
            target_speed = std::min(traffic_target_speed, target_speed);
            final_speed_source = "traffic_light";
        }
        if (upper_stop_active) {
            if (stop_sign_terminal_stop) {
                target_speed = 0.0f;
                final_speed_source = "stop_sign";
            } else if (road_sign_decision_stop) {
                target_speed = 0.0f;
                final_speed_source = "road_sign_stop";
            } else if (startDelayActive) {
                target_speed = 0.0f;
                final_speed_source = "start_delay";
            } else {
                target_speed = 0.0f;
                final_speed_source = "traffic_light";
            }
        } else if (planner_path_unavailable) {
            target_speed = 0.0f;
            final_speed_source = "path_unavailable";
        } else if (pedestrian_launch_committed) {
            target_speed = std::max(0.0f, planner_target_speed_mps);
            final_speed_source = planner_debug.pedestrian_launch_source;
        } else if (pedestrian_stop_wait) {
            target_speed = 0.0f;
            final_speed_source = "pedestrian_stopped_wait";
        }
        if (road_sign_slowdown_active && !road_sign_decision_stop &&
            !pedestrian_stop_wait) {
            target_speed = std::min(
                target_speed,
                std::max(0.0f, params.roadSignFork.stopSlowdownSpeedMps));
            final_speed_source = "road_sign_slowdown";
        }
        float applied_speed = target_speed;
        if (virtual_speed_override &&
            !stop_sign_terminal_stop &&
            !road_sign_decision_stop &&
            !startDelayActive &&
            !planner_path_unavailable &&
            !pedestrian_stop_wait &&
            !pedestrian_launch_committed) {
            target_speed = virtual_speed_mps;
            applied_speed = virtual_speed_mps;
            final_speed_source = "virtual_speed_override";
        }
        if (has_pp_result) {
            const bool launch_paused_by_path =
                pedestrian_launch_committed && planner_path_unavailable;
            const bool launch_paused_by_external_stop =
                pedestrian_launch_committed && upper_stop_active;
            const bool launch_motion_allowed =
                pedestrian_launch_committed &&
                !upper_stop_active &&
                !planner_path_unavailable &&
                applied_speed > 0.01f;
            updatePedestrianLaunchProgress(
                pp_runtime,
                pp_result,
                launch_motion_allowed,
                launch_paused_by_path,
                launch_paused_by_external_stop);
            planner_debug = pp_result.debug;
            planner_debug.final_speed_source = final_speed_source;
        }
        uart.speed = applied_speed;
        const auto final_cmd_tp = std::chrono::steady_clock::now();

        const float serial_speed = startDelayActive ? 0.0f : uart.speed;
        std::chrono::steady_clock::time_point serial_begin_tp{};
        std::chrono::steady_clock::time_point serial_end_tp{};
        int64_t time_since_last_serial_write_us = -1;
        int serial_rc = 0;
        int64_t tx_encoded_mono_us = 0;
        std::array<uint8_t, 11> tx_frame{};
        bool serial_write_attempted = false;
        if (!debug1 && uart.startFlag)
        {
            serial_begin_tp = std::chrono::steady_clock::now();
            time_since_last_serial_write_us =
                last_serial_begin_valid
                    ? std::chrono::duration_cast<std::chrono::microseconds>(
                          serial_begin_tp - last_serial_begin_tp).count()
                    : -1;
            last_serial_begin_tp = serial_begin_tp;
            last_serial_begin_valid = true;

            if (startDelayActive) {
                serial_rc = uart.Serial_writeWithSpeed(servo_output, serial_speed, 11);
            } else {
                serial_rc = uart.Serial_write(servo_output, 11);
            }
            serial_end_tp = std::chrono::steady_clock::now();
            tx_encoded_mono_us = uart.lastTxEncodedMonoUs();
            tx_frame = uart.lastTxFrame();
            serial_write_attempted = true;
        }

        auto resolveLineEmptyReason = [](const LineLossTraceFields& trace) -> const char* {
            if (trace.line_arrow_object_count == 0) return "no_arrow_objects";
            if (trace.line_raw_guide_count == 0) return "no_raw_guides";
            if (trace.line_valid_world_guide_count == 0) return "no_valid_world_guides";
            if (trace.line_single_anchor_count == 0 ||
                trace.line_single_root_count == 0) {
                return "no_single_anchor";
            }
            if (trace.line_credible_single_count == 0) return "no_credible_single";
            if (trace.line_visible_branch_count == 0) return "visible_branch_empty";
            if (trace.line_selected_branch_missing) return "selected_branch_missing";
            if (trace.line_selected_branch_centerline_size == 0) {
                return "selected_centerline_empty";
            }
            if (trace.line_selected_branch_centerline_size == 1) {
                return "selected_centerline_too_short";
            }
            if (trace.line_candidate_size == 0) return "candidate_empty";
            if (trace.line_candidate_size < 2 || !trace.line_candidate_usable) {
                return "candidate_unusable";
            }
            if (trace.line_final_plan_path_size == 0 &&
                trace.line_candidate_usable &&
                trace.line_local_route_size == 0) {
                return "route_unavailable";
            }
            if (trace.line_final_plan_path_size == 0 &&
                trace.line_local_route_size > 0 &&
                trace.line_local_waypoints_size == 0) {
                return "local_waypoints_empty";
            }
            if (trace.line_final_plan_path_size == 0 &&
                trace.line_local_waypoints_size > 0 &&
                !trace.line_local_usable) {
                return "local_waypoints_unusable";
            }
            if (trace.line_final_plan_path_size == 0 ||
                !trace.line_final_plan_path_usable) {
                return "final_plan_unusable";
            }
            if (trace.line_world_path_size == 0) return "world_path_empty";
            if (trace.line_positive_world_y_count == 0) return "no_positive_world_y";
            if (!trace.line_pure_pursuit_target_valid) {
                return "pure_pursuit_target_invalid";
            }
            if (trace.line_used_candidate_fallback) {
                return "local_fallback_candidate";
            }
            return "ok";
        };
        line_loss.line_empty_reason = resolveLineEmptyReason(line_loss);

        if (local_planner_log_file.is_open()) {
            const char* speed_override_mode =
                stop_sign_terminal_stop
                    ? "stop_sign_terminal_stop"
                    : (startDelayActive
                           ? "start_delay_zero"
                           : (virtual_speed_override
                                  ? "virtual_lookahead_speed"
                                  : (traffic_speed_override ? "traffic_min_active" : "planner_or_global")));
            local_planner_log_file << std::fixed << std::setprecision(2)
                                   << frame_idx << ","
                                   << packet.frame_id << ","
                                   << localPlannerModeName(planner_debug.mode) << ","
                                   << obstacleStateName(planner_debug.obstacle_state) << ","
                                   << avoidScenarioName(planner_debug.avoid_scenario) << ","
                                   << planner_debug.target_speed_mps << ","
                                   << planner_debug.ego_speed_mps << ","
                                   << planner_debug.path_points << ","
                                   << planner_debug.gold_bias_reason << ","
                                   << planner_debug.gold_visible_count << ","
                                   << planner_debug.gold_projectable_count << ","
                                   << planner_debug.gold_best_near_capture_count << ","
                                   << planner_debug.gold_best_capture_count << ","
                                   << planner_debug.gold_best_capture_along_s_sum << ","
                                   << planner_debug.gold_no_capture_diag << ","
                                   << (planner_debug.gold_late_start_rejected ? 1 : 0) << ","
                                   << planner_debug.gold_late_start_s_m << ","
                                   << planner_debug.gold_late_start_delta_m << ","
                                   << planner_debug.gold_best_bias_m << ","
                                   << planner_debug.gold_stable_base_m << ","
                                   << planner_debug.gold_transition_state_name << ","
                                   << planner_debug.gold_target_bias_m << ","
                                   << planner_debug.gold_applied_bias_m << ","
                                   << planner_debug.gold_effective_bias_m << ","
                                   << planner_debug.gold_pending_bias_m << ","
                                   << planner_debug.gold_pending_confirm_count << ","
                                   << planner_debug.gold_no_executable_frames << ","
                                   << planner_debug.gold_target_reached_frames << ","
                                   << (planner_debug.gold_selection_frozen ? 1 : 0) << ","
                                   << traffic_target_speed << ","
                                   << target_speed << ","
                                   << applied_speed << ","
                                   << (pedestrian_emergency_stop ? 1 : 0) << ","
                                   << (startDelayActive ? 1 : 0) << ","
                                   << speed_override_mode << ","
                                   << branchSideName(branch_meta.selected_side) << ","
                                    << planner_debug.obstacle_avoid_side << ","
                                    << (planner_debug.vehicle_avoid_side_locked ? 1 : 0) << ","
                                    << planner_debug.vehicle_locked_avoid_side << ","
                                    << planner_debug.vehicle_curve_metric << ","
                                    << planner_debug.vehicle_curve_direction << ","
                                    << (planner_debug.vehicle_curve_avoid_boost_active ? 1 : 0) << ","
                                    << (planner_debug.vehicle_opposite_steer_active ? 1 : 0) << ","
                                    << planner_debug.vehicle_curve_avoid_hold_remaining << ","
                                    << (planner_debug.vehicle_recover_hold_active ? 1 : 0) << ","
                                    << planner_debug.vehicle_recover_hold_remaining << ","
                                    << planner_debug.vehicle_recover_hold_age_frames << ","
                                    << planner_debug.vehicle_recover_hold_max_lifetime_frames << ","
                                    << planner_debug.obstacle_target_pixel_x << ","
                                   << planner_debug.obstacle_target_pixel_y << ","
                                   << planner_debug.debug_base_path_x << ","
                                   << planner_debug.debug_final_path_x << ","
                                   << planner_debug.debug_path_y << ","
                                   << (planner_debug.route_hold_active ? 1 : 0) << ","
                                   << planner_debug.speed_risk_level << ","
                                   << planner_debug.speed_limit_reason << ","
                                   << planner_debug.reason << ","
                                   << planner_debug.speed_reason << ","
                                   << planner_debug.obstacle_count << ","
                                   << planner_debug.obstacle_active_count << ","
                                   << planner_debug.obstacle_primary_id << ","
                                   << planner_debug.obstacle_primary_type << ","
                                   << planner_debug.obstacle_distance_m << ","
                                   << planner_debug.obstacle_point_distance_m << ","
                                   << planner_debug.obstacle_lateral_m << ","
                                   << planner_debug.obstacle_offset_current_m << ","
                                   << planner_debug.obstacle_offset_target_m << ","
                                   << planner_debug.final_applied_path_offset_m << ","
                                   << planner_debug.realized_offset_m << ","
                                   << planner_debug.ego_center_lateral_m << ","
                                   << planner_debug.near_route_lateral_m << ","
                                   << (planner_debug.near_route_lateral_valid ? 1 : 0) << ","
                                   << planner_debug.near_route_lateral_sample_count << ","
                                   << planner_debug.raw_person_count << ","
                                   << planner_debug.pedestrian_boundary_clearance_m << ","
                                   << planner_debug.pedestrian_motion_clearance_net_m << ","
                                   << planner_debug.pedestrian_motion_name << ","
                                   << planner_debug.pedestrian_motion_reason << ","
                                   << (planner_debug.pedestrian_seen_inside ? 1 : 0) << ","
                                   << planner_debug.pedestrian_clear_streak << ","
                                   << planner_debug.pedestrian_stop_lost_streak << ","
                                   << planner_debug.ped_stop_car_side_metric_m << ","
                                   << planner_debug.ped_stop_locked_car_side << ","
                                   << planner_debug.ped_release_required_side << ","
                                   << (planner_debug.pedestrian_exit_evidence ? 1 : 0) << ","
                                   << planner_debug.pedestrian_exit_lost_grace << ","
                                   << planner_debug.pedestrian_slow_observation_count << ","
                                   << planner_debug.pedestrian_slow_lost_count << ","
                                   << planner_debug.pedestrian_slow_away_streak << ","
                                   << planner_debug.startup_launch_protect_remaining << ","
                                   << (planner_debug.startup_launch_release_event ? 1 : 0) << ","
                                   << (planner_debug.ped_emergency_suppressed_by_launch ? 1 : 0) << ","
                                   << (planner_debug.launch_protect_speed_applied ? 1 : 0) << ","
                                   << planner_debug.pedestrian_stop_state_name << ","
                                   << planner_debug.pedestrian_launch_source << ","
                                   << (planner_debug.slow_candidate ? 1 : 0) << ","
                                   << (planner_debug.emergency_candidate ? 1 : 0) << ","
                                   << (planner_debug.pedestrian_observed_this_frame ? 1 : 0) << ","
                                   << (planner_debug.path_usable ? 1 : 0) << ","
                                   << (planner_debug.blocked_by_path ? 1 : 0) << ","
                                   << planner_debug.launch_remaining << ","
                                   << (planner_debug.launch_paused_by_path ? 1 : 0) << ","
                                   << planner_debug.planner_speed_source << ","
                                   << planner_debug.final_speed_source << ","
                                   << planner_debug.active_block_reasons << ","
                                   << planner_debug.state_transition_reason << ","
                                   << planner_debug.ped_emergency_id << ","
                                   << planner_debug.ped_foot_x_m << ","
                                   << planner_debug.ped_foot_y_m << ","
                                   << (planner_debug.ped_emergency_hit_corridor ? 1 : 0) << ","
                                   << (planner_debug.ped_emergency_hit ? 1 : 0) << ","
                                   << planner_debug.ped_danger_corridor_near_width_m << ","
                                   << planner_debug.ped_danger_corridor_far_width_m << ","
                                   << planner_debug.ped_danger_corridor_length_m << ","
                                   << planner_debug.ped_danger_corridor_sample_step_m << ","
                                   << planner_debug.ped_danger_servo_raw << ","
                                   << planner_debug.ped_danger_servo_output << ","
                                   << (planner_debug.ped_danger_servo_locked ? 1 : 0) << ","
                                   << planner_debug.ped_danger_servo_abs_used << ","
                                   << (planner_debug.ped_danger_servo_valid ? 1 : 0) << ","
                                   << planner_debug.ped_danger_servo_source << ","
                                   << planner_debug.ped_danger_turn_radius_m << ","
                                   << planner_debug.ped_danger_turn_sign << ","
                                   << (planner_debug.ped_danger_straight ? 1 : 0) << ","
                                   << planner_debug.ped_danger_along_s_m << ","
                                   << planner_debug.ped_danger_lateral_distance_m << ","
                                   << planner_debug.ped_emergency_allowed_width_m << ","
                                   << (planner_debug.cluster_active ? 1 : 0) << ","
                                   << (planner_debug.avoid_lookahead_active ? 1 : 0) << ","
                                   << planner_debug.recommended_lookahead_m << ","
                                   << steering_angle << ","
                                   << servo_output << ","
                                   << (gold_terminal_debug.candidate_valid ? 1 : 0) << ","
                                   << (gold_terminal_debug.active ? 1 : 0) << ","
                                   << (gold_terminal_debug.applied ? 1 : 0) << ","
                                   << (gold_terminal_debug.entered ? 1 : 0) << ","
                                   << (gold_terminal_debug.exited ? 1 : 0) << ","
                                   << (gold_terminal_debug.derivative_reset ? 1 : 0) << ","
                                   << gold_terminal_debug.reason << ","
                                   << gold_terminal_debug.target_world.x << ","
                                   << gold_terminal_debug.target_world.y << ","
                                   << gold_terminal_debug.distance_to_ego_m << ","
                                   << gold_terminal_debug.path_distance_m << ","
                                   << gold_terminal_debug.gold_along_path_s_m << ","
                                   << gold_terminal_debug.lookahead_along_path_s_m << ","
                                   << gold_terminal_debug.heading_error_deg << ","
                                   << gold_terminal_debug.filtered_derivative_deg << ","
                                   << gold_terminal_debug.servo_output << ","
                                   << pedestrian_gold_danger_debug.strips.size() << ","
                                   << pedestrian_gold_danger_debug.ignored_gold_world.size() << ","
                                   << pedestrian_gold_danger_debug.configured_half_width_m << ","
                                   << (pedestrian_gold_danger_debug.live ? 1 : 0) << ","
                                   << (pedestrian_gold_danger_debug.hold_active ? 1 : 0) << ","
                                   << pedestrian_gold_danger_debug.hold_elapsed_frames << ","
                                   << pedestrian_gold_danger_debug.hold_remaining_frames << ","
                                   << pedestrian_gold_danger_debug.configured_hold_frames << ","
                                   << pedestrian_gold_danger_debug.source << ","
                                   << line_loss.line_arrow_object_count << ","
                                   << line_loss.line_raw_guide_count << ","
                                   << line_loss.line_valid_world_guide_count << ","
                                   << line_loss.line_single_anchor_count << ","
                                   << line_loss.line_single_root_count << ","
                                   << line_loss.line_credible_single_count << ","
                                   << line_loss.line_best_single_guide_count << ","
                                   << line_loss.line_best_single_forward_span << ","
                                   << line_loss.line_visible_branch_count << ","
                                   << line_loss.line_selected_branch_label << ","
                                   << line_loss.line_selected_branch_guide_count << ","
                                   << line_loss.line_selected_branch_centerline_size << ","
                                   << line_loss.line_candidate_size << ","
                                   << (line_loss.line_candidate_usable ? 1 : 0) << ","
                                   << line_loss.line_local_route_size << ","
                                   << line_loss.line_local_waypoints_size << ","
                                   << (line_loss.line_local_usable ? 1 : 0) << ","
                                   << (line_loss.line_used_candidate_fallback ? 1 : 0) << ","
                                   << line_loss.line_local_reject_reason << ","
                                   << line_loss.line_final_plan_path_size << ","
                                   << (line_loss.line_final_plan_path_usable ? 1 : 0) << ","
                                   << line_loss.line_world_path_size << ","
                                   << line_loss.line_positive_world_y_count << ","
                                   << (line_loss.line_pure_pursuit_target_valid ? 1 : 0) << ","
                                   << line_loss.line_servo_source << ","
                                   << line_loss.line_empty_reason << "\n";
            if (++local_planner_log_pending_lines >= kDebugLogFlushEveryLines) {
                local_planner_log_file.flush();
                local_planner_log_pending_lines = 0;
            }
        }

        if (dynamic_path_planner_log_file.is_open()) {
            dynamic_path_planner_log_file << std::fixed << std::setprecision(2)
                                          << frame_idx << ","
                                          << packet.frame_id << ","
                                          << localPlannerModeName(planner_debug.mode) << ","
                                          << planner_debug.reason << ","
                                          << planner_debug.target_speed_mps << ","
                                          << planner_debug.ego_speed_mps << ","
                                          << planner_debug.path_points << ","
                                          << planner_debug.speed_risk_level << ","
                                          << planner_debug.speed_limit_reason << ","
                                          << planner_debug.speed_reason << "\n";
            if (++dynamic_path_planner_log_pending_lines >= kDebugLogFlushEveryLines) {
                dynamic_path_planner_log_file.flush();
                dynamic_path_planner_log_pending_lines = 0;
            }
        }

        cv::Mat world_path_frame;
        if (params.worldPurePursuitParams.saveWorldView) {
            const std::vector<WorldDetectionMarker> world_detection_markers =
                collectWorldDetectionMarkers(
                    current_results,
                    params.goldPosition.correctionScale);
            world_path_frame = drawWorldPathView(
                world_path,
                world_detection_markers,
                raw_person_objects,
                has_pursuit_target,
                pursuit_target,
                target_distance,
                steering_angle,
                servo_output,
                applied_speed,
                params.obstacleAvoidance.pedestrianSlowDistanceM,
                planner_debug,
                world_invalid_points,
                frame_idx,
                virtual_lookahead_state.has_point,
                virtual_lookahead_state.point,
                virtual_lookahead_state.control_active,
                virtual_lookahead_state.locked,
                current_arrow_count,
                gold_terminal_debug,
                road_sign_pose_debug,
                pedestrian_gold_danger_debug,
                params.worldPurePursuitParams.worldViewWidth,
                params.worldPurePursuitParams.worldViewHeight,
                params.obstacleAvoidance.vehicleWidthM,
                params.obstacleAvoidance.vehicleFrontM,
                params.obstacleAvoidance.vehicleRearM,
                params.worldPurePursuitParams.worldViewForwardMeters,
                params.worldPurePursuitParams.worldViewSideMeters);
            if (!world_path_frame.empty()) {
                std::ostringstream road_sign_world_ss;
                road_sign_world_ss << "RoadSignFork: sign="
                                   << (road_sign_obs.seen ? "Y" : "N")
                                   << " pref=" << branchPreferenceName(branch_preference)
                                   << " phase=" << roadSignForkPhaseName(road_sign_phase)
                                   << " branch=" << branchSideName(branch_meta.selected_side)
                                   << " forks=" << branch_meta.visible_branch_count
                                   << " decision=" << road_sign_decision
                                   << " cd_ms=" << road_sign_cooldown_left_ms
                                   << " hold=" << (road_sign_decision_servo_hold_active ? 1 : 0);
                cv::putText(world_path_frame,
                            road_sign_world_ss.str(),
                            cv::Point(12, 270),
                            cv::FONT_HERSHEY_SIMPLEX,
                            0.55,
                            cv::Scalar(255, 230, 90),
                            1,
                            cv::LINE_AA);
            }
        }

        // ========== 4. unified render (320x240) ==========
        const bool render_or_save_debug_view =
            params.display || params.saveRaw || params.saveAI ||
            params.worldPurePursuitParams.saveWorldView;
        if (render_or_save_debug_view) {
        const bool has_source_frame = !current_frame.empty();
        if (!has_source_frame)
        {
            current_frame = cv::Mat::zeros(480, 640, CV_8UC3);
        }
        if (!current_frame.empty())
        {
            cv::Mat raw_frame;
            if (params.saveRaw && has_source_frame) {
                raw_frame = current_frame.clone();
            }

            // current_frame as drawing canvas
            
            if (!current_results.empty()) {
                draw_results_on_frame(current_frame, current_results);
            }

            
            if (!plan_path.empty()) {
                std::vector<cv::Point2f> display_path;
                display_path.reserve(plan_path.size());
                for (const auto& pt : plan_path) {
                    display_path.push_back(cv::Point2f(
                        pt.x * DISPLAY_SCALE_X,
                        pt.y * DISPLAY_SCALE_Y
                    ));
                }
                drawPath(current_frame, display_path, cv::Scalar(0, 255, 0), 2);
            }

            std::string info = "Frame: " + std::to_string(frame_idx) +
                              "  Packet: " + std::to_string(packet.frame_id) +
                              "  Objects: " + std::to_string(current_results.size()) +
                              "  Path: " + (plan_path.empty() ? "No" : "Yes");
            cv::putText(current_frame, info, cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
            if (!has_source_frame) {
                cv::putText(current_frame, "Canvas Debug View", cv::Point(10, 450),
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(180, 180, 180), 1);
            }

            // ervo_output
            std::ostringstream servo_ss;
            servo_ss << std::fixed << std::setprecision(1);
            servo_ss << "Servo: " << servo_output;
            cv::putText(current_frame, servo_ss.str(), cv::Point(10, 60),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);

            // ========== 5. save result video (640x480) ==========
            std::ostringstream error_ss;
            error_ss << std::fixed << std::setprecision(2);
            if (has_pursuit_target) {
                error_ss << "Target: (" << pursuit_target.x << "," << pursuit_target.y << ")";
            } else {
                error_ss << "Target: N/A";
            }
            cv::putText(current_frame, error_ss.str(), cv::Point(10, 90),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 1);

            std::ostringstream trend_ss;
            trend_ss << std::fixed << std::setprecision(2);
            if (has_pursuit_target) {
                trend_ss << "Steer: " << steering_angle << " Ld: " << target_distance;
            } else {
                trend_ss << "Steer: N/A";
            }
            cv::putText(current_frame, trend_ss.str(), cv::Point(10, 120),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 200, 0), 1);

            std::ostringstream road_sign_ss;
            road_sign_ss << "ForkSM: " << roadSignForkPhaseName(road_sign_phase)
                          << " cd_ms=" << road_sign_cooldown_left_ms
                          << " hold=" << (road_sign_decision_servo_hold_active ? 1 : 0)
                          << " pref=" << branchPreferenceName(branch_preference)
                          << " branch=" << branchSideName(branch_meta.selected_side)
                          << " cnt=" << branch_meta.visible_branch_count;
            cv::putText(current_frame, road_sign_ss.str(), cv::Point(10, 150),
                       cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 220, 120), 1);

            std::ostringstream speed_light_ss;
            speed_light_ss << std::fixed << std::setprecision(2);
            speed_light_ss << "SpeedLightSM: " << speedLightPhaseName(speed_light_phase)
                           << " traffic=" << traffic_target_speed
                           << " plan=" << planner_target_speed_mps
                           << " final=" << target_speed
                           << " applied=" << applied_speed;
            cv::putText(current_frame, speed_light_ss.str(), cv::Point(10, 180),
                       cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(120, 255, 180), 1);

            std::ostringstream stop_sign_ss;
            stop_sign_ss << "StopSignSM: " << stopSignStopPhaseName(stop_sign_phase)
                         << " seen=" << (stop_sign_obs.seen ? 1 : 0)
                         << " lost_ms=" << stop_sign_lost_ms_display
                         << " stop_ms=" << std::max(0, params.stopSignStop.stopAfterLostMs);
            if (initialStartDelayConsumed) {
                const auto start_protect_elapsed_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(stop_sign_now - startTime).count();
                const int start_protect_left_ms =
                    std::max(0, std::max(0, params.stopSignStop.startProtectMs) -
                                    static_cast<int>(start_protect_elapsed_ms));
                stop_sign_ss << " protect_ms=" << start_protect_left_ms;
            }
            cv::putText(current_frame, stop_sign_ss.str(), cv::Point(10, 210),
                       cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(80, 120, 255), 1);

            std::ostringstream local_planner_ss;
            local_planner_ss << std::fixed << std::setprecision(2);
            local_planner_ss << "LocalPlanner: mode=" << localPlannerModeName(planner_debug.mode)
                             << " obs=" << obstacleStateName(planner_debug.obstacle_state)
                             << " scene=" << avoidScenarioName(planner_debug.avoid_scenario)
                             << " target=" << planner_debug.target_speed_mps
                             << " off=" << planner_debug.obstacle_offset_current_m
                             << " realized=" << planner_debug.realized_offset_m;
            cv::putText(current_frame, local_planner_ss.str(), cv::Point(10, 240),
                       cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(255, 180, 80), 1);

            if (params.saveRaw || params.saveAI || params.worldPurePursuitParams.saveWorldView) {
                VideoSavePacket save_packet;
                save_packet.frame_id = packet.frame_id;

                if (params.saveRaw && has_source_frame) {
                    save_packet.raw_frame = std::move(raw_frame);
                }
                if (params.saveAI) {
                    save_packet.result_frame = current_frame;
                }
                if (params.worldPurePursuitParams.saveWorldView && !world_path_frame.empty()) {
                    save_packet.world_frame = std::move(world_path_frame);
                }

                if (!save_packet.raw_frame.empty() || !save_packet.result_frame.empty() ||
                    !save_packet.world_frame.empty()) {
                    videoSavePackets.Put(std::move(save_packet));
                }
            }

            // ========== 6. display ==========
            if (params.display)
            {
                imshow("AI Result", current_frame);
                cv::waitKey(1);
            }
        }
        }

        // ========== 7. Post-TX serial and world-pursuit debug logging ==========
        if (serial_write_attempted)
        {
            const int64_t start_delay_elapsed_ms =
                startDelayActive
                    ? std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - startTime).count()
                    : 0;
            std::ostringstream tx_hex;
            if (tx_encoded_mono_us >= monotonicUs(serial_begin_tp) &&
                tx_encoded_mono_us <= monotonicUs(serial_end_tp)) {
                tx_hex << std::uppercase << std::hex << std::setfill('0');
                for (size_t i = 0; i < tx_frame.size(); ++i) {
                    if (i > 0) {
                        tx_hex << ' ';
                    }
                    tx_hex << std::setw(2) << static_cast<unsigned int>(tx_frame[i]);
                }
            } else {
                tx_hex << "NA";
            }

            if (startDelayActive) {
                static int delay_log_count = 0;
                if (delay_log_count % 30 == 0) {
                    std::cout << "[Car] ?.. ? " << start_delay_elapsed_ms
                              << "ms / " << START_DELAY_MS << "ms" << std::endl;
                }
                delay_log_count++;
            }

            logServoDebug(frame_idx,
                          pursuit_target,
                          target_distance,
                          steering_angle,
                          servo_output,
                          serial_speed,
                          static_cast<int>(world_path.size()),
                          world_invalid_points,
                          static_cast<int>(plan_path.size()),
                          has_pursuit_target);

            const auto loop_end_tp = std::chrono::steady_clock::now();
            if (serial_timing_log_file.is_open()) {
                serial_timing_log_file
                    << frame_idx << ','
                    << packet.frame_id << ','
                    << monotonicUs(final_cmd_tp) << ','
                    << std::fixed << std::setprecision(3) << applied_speed << ','
                    << monotonicUs(serial_begin_tp) << ','
                    << tx_encoded_mono_us << ','
                    << serial_speed << ','
                    << servo_output << ','
                    << tx_hex.str() << ','
                    << monotonicUs(serial_end_tp) << ','
                    << std::chrono::duration_cast<std::chrono::microseconds>(
                           serial_end_tp - serial_begin_tp).count() << ','
                    << time_since_last_serial_write_us << ','
                    << std::chrono::duration_cast<std::chrono::microseconds>(
                           serial_begin_tp - final_cmd_tp).count() << ','
                    << monotonicUs(loop_end_tp) << ','
                    << std::chrono::duration_cast<std::chrono::microseconds>(
                           loop_end_tp - loop_begin_tp).count() << ','
                    << serial_rc << '\n';
                if (++serial_timing_log_pending_lines >= kDebugLogFlushEveryLines) {
                    serial_timing_log_file.flush();
                    serial_timing_log_pending_lines = 0;
                }
            }

        }

        // Keep frame index continuous at the end of each processing loop.
        frame_idx++;
    }
}

void Car::videoSaver()
{
    std::cout << "[videoSaver] thread started, waiting init gate..." << std::endl;
    while (!loadSuccess || !initSuccess) {
        usleep(1000);
    }
    std::cout << "[videoSaver] init gate passed." << std::endl;

    if (!params.saveRaw && !params.saveAI && !params.worldPurePursuitParams.saveWorldView) {
        std::cout << "[videoSaver] saveRaw/saveAI/worldView disabled, thread exiting." << std::endl;
        return;
    }

    constexpr int kFrameFps = 30;
    const cv::Size kSaveSize(640, 480);

    cv::VideoWriter raw_writer;
    if (params.saveRaw &&
        !openVideoWriterWithFallbacks(raw_writer, debugViewPath, kFrameFps, kSaveSize)) {
        std::cerr << "[WARNING] failed to open raw(debugView) writer: "
                  << debugViewPath << std::endl;
    }

    cv::VideoWriter result_writer;
    if (params.saveAI &&
        !openVideoWriterWithFallbacks(result_writer, aiViewPath, kFrameFps, kSaveSize)) {
        std::cerr << "[WARNING] failed to open AI result writer: "
                  << aiViewPath << std::endl;
    }

    cv::VideoWriter world_writer;
    const cv::Size world_save_size(
        std::max(320, params.worldPurePursuitParams.worldViewWidth),
        std::max(240, params.worldPurePursuitParams.worldViewHeight));
    if (params.worldPurePursuitParams.saveWorldView &&
        !openVideoWriterWithFallbacks(world_writer, worldPathViewPath, kFrameFps, world_save_size)) {
        std::cerr << "[WARNING] failed to open world path view writer: "
                  << worldPathViewPath << std::endl;
    }

    while (1)
    {
        if (!uart.startFlag) {
            BlockingQueueClear(videoSavePackets);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        VideoSavePacket packet = videoSavePackets.Take();

        if (params.saveRaw && raw_writer.isOpened() && !packet.raw_frame.empty()) {
            cv::Mat raw_save_frame = prepareFrameForVideoWrite(packet.raw_frame, kSaveSize);
            if (!raw_save_frame.empty()) {
                raw_writer << raw_save_frame;
            }
        }

        if (params.saveAI && result_writer.isOpened() && !packet.result_frame.empty()) {
            cv::Mat result_save_frame = prepareFrameForVideoWrite(packet.result_frame, kSaveSize);
            if (!result_save_frame.empty()) {
                result_writer << result_save_frame;
            }
        }

        if (params.worldPurePursuitParams.saveWorldView &&
            world_writer.isOpened() && !packet.world_frame.empty()) {
            cv::Mat world_save_frame = prepareFrameForVideoWrite(packet.world_frame, world_save_size);
            if (!world_save_frame.empty()) {
                world_writer << world_save_frame;
            }
        }
    }
}

void Car::serialPort() 
{
    while (!loadSuccess || !initSuccess) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    std::cout << "[serialPort] init gate passed, waiting IMU bytes on "
              << Serial_dev << std::endl;

    enum State {
        WAIT_HEAD,
        WAIT_FF,
        WAIT_F1,
        WAIT_LEN,
        WAIT_DATA
    } state = WAIT_HEAD;
    constexpr size_t kHeaderLen = 4;      // AA FF F1 LEN
    constexpr size_t kChecksumLen = 2;    // sc + ac
    constexpr uint8_t kPayloadLen = 8;    // speed_raw + yaw_raw
    constexpr size_t kMaxFrameLen = kHeaderLen + kPayloadLen + kChecksumLen;

    uint8_t buffer[kMaxFrameLen] = {0};
    size_t idx = 0;
    uint8_t data_len = 0;
    uint8_t sc = 0, ac = 0;
    uint8_t resByte = 0;
    int rret = -1;

    host_clock::ensureInitT0();
    std::ofstream rx_timing_log_file;
    size_t rx_timing_log_pending_lines = 0;
    std::filesystem::path rx_timing_log_path(params.localPlannerLogPath);
    if (rx_timing_log_path.empty()) {
        rx_timing_log_path = "rx_timing.csv";
    } else {
        rx_timing_log_path.replace_filename(
            rx_timing_log_path.stem().string() + "_rx_timing.csv");
    }
    ensureParentDirectory(rx_timing_log_path.string());
    rx_timing_log_file.open(
        rx_timing_log_path.string(), std::ios::out | std::ios::trunc);
    if (rx_timing_log_file.is_open()) {
        rx_timing_log_file << "rx_mono_us,speed_raw,speed_mps\n";
        rx_timing_log_file.flush();
        std::cout << "[RxTiming] CSV: " << rx_timing_log_path << std::endl;
    } else {
        std::cerr << "[RxTiming] failed to open: "
                  << rx_timing_log_path << std::endl;
    }

    const bool serial_csv_log_enabled =
        params.serialThreadCsvLogEnable;
    if (serial_csv_log_enabled) {
        ensureParentDirectory(params.serialThreadCsvLogPath);
        imu_time_align_log_.open(params.serialThreadCsvLogPath, std::ios::out | std::ios::app);
        if (imu_time_align_log_.is_open()) {
            imu_time_align_pending_lines_ = 0;
            imu_time_align_log_.seekp(0, std::ios::end);
            if (imu_time_align_log_.tellp() == static_cast<std::streampos>(0)) {
                imu_time_align_log_
                    << "t_host_sec,host_unix_sec,payload_len,speed_raw,yaw_raw,speed_mps,yaw_deg\n";
                imu_time_align_log_.flush();
            }
            std::cout << "[Car] serial telemetry CSV: " << params.serialThreadCsvLogPath << std::endl;
        } else {
            std::cerr << "[Car] serial telemetry log open failed: " << params.serialThreadCsvLogPath
                      << std::endl;
        }
    }

    std::ofstream imu_ad_debug_log;
    size_t imu_ad_debug_pending_lines = 0;
    if (params.imuAdDebugLogEnable) {
        ensureParentDirectory(params.imuAdDebugLogPath);
        imu_ad_debug_log.open(params.imuAdDebugLogPath, std::ios::out | std::ios::app);
        if (imu_ad_debug_log.is_open()) {
            imu_ad_debug_log.seekp(0, std::ios::end);
            if (imu_ad_debug_log.tellp() == static_cast<std::streampos>(0)) {
                imu_ad_debug_log
                    << "t_host_sec,host_unix_sec,payload_len,"
                    << "speed_raw,yaw_raw,speed_mps,yaw_deg\n";
                imu_ad_debug_log.flush();
            }
            std::cout << "[Car] IMU AD debug CSV: " << params.imuAdDebugLogPath << std::endl;
        } else {
            std::cerr << "[Car] IMU AD debug log open failed: " << params.imuAdDebugLogPath
                      << std::endl;
        }
    }

    while (1)
    {
        rret = uart.receiveBytes(resByte, 50);
        if (rret != 0) {
            continue;
        }

        switch (state) {
            case WAIT_HEAD:
                if (resByte == 0xAA) {
                    buffer[0] = 0xAA;
                    idx = 1;
                    state = WAIT_FF;
                }
                break;
            case WAIT_FF:
                if (resByte == 0xFF) {
                    buffer[idx++] = resByte;
                    state = WAIT_F1;
                } else {
                    state = WAIT_HEAD;
                }
                break;
            case WAIT_F1:
                if (resByte == 0xF1) {
                    buffer[idx++] = resByte;
                    state = WAIT_LEN;
                } else {
                    state = WAIT_HEAD;
                }
                break;
            case WAIT_LEN:
                data_len = resByte;
                buffer[idx++] = resByte;
                if (data_len == kPayloadLen) {
                    state = WAIT_DATA;
                } else {
                    state = WAIT_HEAD;
                    idx = 0;
                }
                break;
            case WAIT_DATA:
                if (idx >= sizeof(buffer)) {
                    state = WAIT_HEAD;
                    idx = 0;
                    break;
                }
                buffer[idx++] = resByte;
                {
                    const size_t frame_len = kHeaderLen + static_cast<size_t>(data_len) + kChecksumLen;
                    if (idx < frame_len) {
                        break;
                    }

                    sc = 0; ac = 0;
                    for (size_t i = 0; i < kHeaderLen + static_cast<size_t>(data_len); ++i) {
                        sc += buffer[i];
                        ac += sc;
                    }
                    if (sc == buffer[kHeaderLen + static_cast<size_t>(data_len)] &&
                        ac == buffer[kHeaderLen + static_cast<size_t>(data_len) + 1]) {
                        
                        int32_t vals[2] = {0};
                        for (size_t i = 0; i < 2; ++i) {
                            const size_t base = kHeaderLen + i * 4;
                            const uint32_t raw = static_cast<uint32_t>(buffer[base]) |
                                                 (static_cast<uint32_t>(buffer[base + 1]) << 8) |
                                                 (static_cast<uint32_t>(buffer[base + 2]) << 16) |
                                                 (static_cast<uint32_t>(buffer[base + 3]) << 24);
                            vals[i] = static_cast<int32_t>(raw);
                        }

                        const int32_t speed_raw = vals[0];
                        const int32_t yaw_raw = vals[1];
                        const float speed_mps = static_cast<float>(speed_raw) / 50000.0f;
                        const float yaw_deg = static_cast<float>(yaw_raw) / 1000.0f;
                        const int64_t rx_mono_us =
                            std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now().time_since_epoch()).count();
                        const uint64_t timestamp_ms = static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::system_clock::now().time_since_epoch()).count());
                        {
                            std::lock_guard<std::mutex> lock(lower_telemetry_mutex_);
                            lower_telemetry_.speed_mps = static_cast<float>(speed_raw) / 50000.0f;
                            lower_telemetry_.yaw_deg = static_cast<float>(yaw_raw) / 1000.0f;
                            lower_telemetry_.timestamp_ms = timestamp_ms;
                            lower_telemetry_.valid = true;
                        }

                        host_clock::ensureInitT0();
                        const double t_host = host_clock::steadyNowSec();
                        const double t_wall = host_clock::wallUnixSec();

                        if (serial_csv_log_enabled && imu_time_align_log_.is_open()) {
                            imu_time_align_log_ << std::fixed << std::setprecision(15) << t_host << ','
                                                << t_wall << ',' << static_cast<int>(data_len) << ','
                                                << speed_raw << ',' << yaw_raw << ','
                                                << speed_mps << ',' << yaw_deg << '\n';
                            if (++imu_time_align_pending_lines_ >= 32) {
                                imu_time_align_log_.flush();
                                imu_time_align_pending_lines_ = 0;
                            }
                        }

                        if (params.imuAdDebugLogEnable && imu_ad_debug_log.is_open()) {
                            imu_ad_debug_log << std::fixed << std::setprecision(15)
                                             << t_host << ',' << t_wall << ','
                                             << static_cast<int>(data_len) << ','
                                             << speed_raw << ',' << yaw_raw << ','
                                             << std::setprecision(6) << speed_mps << ',' << yaw_deg
                                             << '\n';
                            if (++imu_ad_debug_pending_lines >= 32) {
                                imu_ad_debug_log.flush();
                                imu_ad_debug_pending_lines = 0;
                            }
                        }

                        if (rx_timing_log_file.is_open()) {
                            rx_timing_log_file
                                << rx_mono_us << ','
                                << speed_raw << ','
                                << std::fixed << std::setprecision(6) << speed_mps
                                << '\n';
                            if (++rx_timing_log_pending_lines >= kDebugLogFlushEveryLines) {
                                rx_timing_log_file.flush();
                                rx_timing_log_pending_lines = 0;
                            }
                        }
                    }
                    state = WAIT_HEAD;
                    idx = 0;
                }
                break;
        }
    }
}

void Car::shmDetectionReader() {
    std::cout << "[shmDetectionReader] thread started, waiting init gate..." << std::endl;
    while (!loadSuccess || !initSuccess) {
        usleep(1000);
    }
    std::cout << "[shmDetectionReader] init gate passed." << std::endl;

    shmReader_ = std::make_unique<yoloe::SharedMemoryReader>();
    shmReaderRunning_.store(false);

    while (!shmReader_->init(SHM_NAME, 100)) {
        std::cerr << "[ShmReader] wait writer ..." << std::endl;
        usleep(200000);
    }

    shmReaderRunning_.store(true);
    std::cout << "[ShmReader] connected, reading detections from shared memory." << std::endl;

    uint32_t last_frame_id = 0;

    while (1)
    {
        if(!uart.startFlag)
        {
            BlockingQueueClear(detectionPackets);
            detection.results.clear();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        yoloe::DetectionResult shm_result;
        if (!shmReader_->read(shm_result, last_frame_id)) {
            usleep(2000);
            continue;
        }
        last_frame_id = shm_result.frame_id;

        DetectionPacket packet;
        packet.frame_id = static_cast<int>(shm_result.frame_id);
        packet.results.clear();

        if (!shm_result.image_data.empty() && shm_result.width > 0 && shm_result.height > 0) {
            cv::Mat rgb(static_cast<int>(shm_result.height), static_cast<int>(shm_result.width), CV_8UC3,
                        const_cast<uint8_t*>(shm_result.image_data.data()));
            cv::cvtColor(rgb, packet.frame_640, cv::COLOR_RGB2BGR);
            if (packet.frame_640.cols != 640 || packet.frame_640.rows != 480) {
                cv::resize(packet.frame_640, packet.frame_640, cv::Size(640, 480));
            }
        } else {
            packet.frame_640.release();
        }

        packet.results.reserve(shm_result.boxes.size());
        for (const auto& box : shm_result.boxes) {
            PredictResult result;
            result.label = box.class_name;
            result.score = box.confidence;
            result.x = static_cast<int>(std::round(box.x1));
            result.y = static_cast<int>(std::round(box.y1));
            result.width = static_cast<int>(std::round(box.x2 - box.x1));
            result.height = static_cast<int>(std::round(box.y2 - box.y1));
            // type:
            // 0=arrow,1=people,2=car,4=door,5=road_sign,
            // 6=speed_limit_sign,7=red,8=yellow,9=green,10=stop
            result.type = static_cast<int>(box.class_id);
            if (result.type < 0 || result.type > 10) {
                continue;
            }

            if (result.width <= 0 || result.height <= 0) {
                continue;
            }

            packet.results.push_back(result);
        }

        detectionPackets.Put(std::move(packet));
    }
}

bool Car::loadParams(string jsonPath)
{
    int flag;
    params = paramsInit(jsonPath, &flag);

    if(flag==0)
    {
        return false;
    }

    if (params.timestampedWorkspaceSaveEnable) {
        const std::filesystem::path save_dir = getWorkspaceSaveDir();
        const std::string run_timestamp = getRunTimestamp();
        aiViewPath = makeTimestampedSavePath(save_dir, "aiView", run_timestamp, ".avi");
        debugViewPath = makeTimestampedSavePath(save_dir, "debugView", run_timestamp, ".avi");
        worldPathViewPath = makeTimestampedSavePath(save_dir, "worldPathView", run_timestamp, ".avi");
        params.localPlannerLogPath =
            makeTimestampedSavePath(save_dir, "local_planner", run_timestamp, ".log");

        if (!ensureDirectory(save_dir)) {
            std::cerr << "[Car] failed to create timestamped save directory: "
                      << save_dir << std::endl;
        } else {
            std::cout << "[Car] timestamped save enabled. dir=" << save_dir
                      << " timestamp=" << run_timestamp << std::endl;
        }
    }

    pathSearcher.params = params;
    imgProcessor.params = params;

    debug1 = params.debug;
    imgProcessor.calibrationPath = workfolderPath + "/res/calibration/calibration.xml";
    imgProcessor.ipmMatrixPath = workfolderPath + "/res/IPM/homography.xml";
    modelPath ="/home/orangepi/workspace/Dragon/res/model/yolov3_mobilenet_v1/yolov5_rknn/yolov5.rknn";


    servor.Pbase = params.servoParams.Pbase;
    servor.P3base = params.servoParams.P3base;
    servor.Ibase = params.servoParams.Ibase;
    servor.Dbase = params.servoParams.Dbase;
    servor.D3base = params.servoParams.D3base;
    servor.PdPbase = params.servoParams.PdPbase;
    servor.PdP3base = params.servoParams.PdP3base;
    servor.PdDbase = params.servoParams.PdDbase;
    servor.PdD3base = params.servoParams.PdD3base;
    servor.FuzzyEnable = params.servoParams.FuzzyEnable;
    servor.PolyEnable = params.servoParams.PolyEnable;
    servor.weight = params.servoParams.weight;

    uart.speed = std::max(0.0f, params.speedDecision.clearSpeedMps);
    uart.startFlag = params.StartFlag;
    // legacy parameter initializations removed (bridge, parking, hamburger, charge, workplace, cone)
    // those modules are no longer part of the build, data kept here has no object definitions.


    pathSearcher.ROAD_WIDTH = params.pathSearchParams.ROAD_WIDTH;
    pathSearcher.pixel_per_meter = params.pathSearchParams.pixel_per_meter;
    pathSearcher.edgeMaximum = params.pathSearchParams.edgeMaximum;
    pathSearcher.kl = params.pathSearchParams.kl;
    pathSearcher.bl = params.pathSearchParams.bl;
    pathSearcher.kr = params.pathSearchParams.kr;
    pathSearcher.br = params.pathSearchParams.br;

    std::cout << "[Car] World Pure Pursuit Params:" << std::endl;
    std::cout << "[Car]   enable: " << params.worldPurePursuitParams.enable << std::endl;
    std::cout << "[Car]   aheadDistance: " << params.worldPurePursuitParams.aheadDistance << std::endl;
    std::cout << "[Car]   wheelBase: " << params.worldPurePursuitParams.wheelBase << std::endl;
    std::cout << "[Car]   servoLimit: " << params.worldPurePursuitParams.servoLimit << std::endl;
    std::cout << "[Car]   saveWorldView: " << params.worldPurePursuitParams.saveWorldView << std::endl;
    std::cout << "[Car] Servo Params:" << std::endl;
    std::cout << "[Car]   Pbase: " << servor.Pbase << std::endl;
    std::cout << "[Car]   Dbase: " << servor.Dbase << std::endl;
    std::cout << "[Car]   PolyEnable: " << servor.PolyEnable << std::endl;
    std::cout << "[Car]   FuzzyEnable: " << servor.FuzzyEnable << std::endl;

    return true;
}
