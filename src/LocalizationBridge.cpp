#include "../include/LocalizationBridge.hpp"
#include "../include/json.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/socket.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>

namespace {

constexpr size_t kLogFlushEveryLines = 32;

double hostUnixSec() {
    using clock = std::chrono::system_clock;
    const auto now = clock::now().time_since_epoch();
    return std::chrono::duration<double>(now).count();
}

bool readPositionFromJson(const nlohmann::json& js, double& x, double& y) {
    const std::string type = js.value("type", "");
    if (type == "ac_xy") {
        if (!js.contains("x") || !js["x"].is_number() ||
            !js.contains("y") || !js["y"].is_number()) {
            return false;
        }
        x = js["x"].get<double>();
        y = js["y"].get<double>();
        return std::isfinite(x) && std::isfinite(y);
    }

    if (type == "robot_position") {
        if (!js.contains("pos") || !js["pos"].is_array() ||
            js["pos"].size() < 3 ||
            !js["pos"][0].is_number() ||
            !js["pos"][2].is_number()) {
            return false;
        }
        x = js["pos"][0].get<double>();
        y = js["pos"][2].get<double>();
        return std::isfinite(x) && std::isfinite(y);
    }

    return false;
}

} // namespace

LocalizationBridge::LocalizationBridge() : LocalizationBridge(Config{}) {}

LocalizationBridge::LocalizationBridge(const Config& config) : config_(config) {}

LocalizationBridge::~LocalizationBridge() {
    stop();
}

void LocalizationBridge::setLowerYawProvider(
    const std::function<bool(float& yaw_deg)>& provider) {
    std::lock_guard<std::mutex> lock(yaw_provider_mutex_);
    lower_yaw_provider_ = provider;
}

void LocalizationBridge::setConfig(const Config& config) {
    if (running_.load()) {
        std::cerr << "[LocalizationBridge] setConfig ignored: bridge is running" << std::endl;
        return;
    }
    config_ = config;
}

bool LocalizationBridge::openUdp() {
    udp_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (udp_fd_ < 0) {
        std::cerr << "[LocalizationBridge] create udp socket failed, errno="
                  << errno << " " << std::strerror(errno) << std::endl;
        return false;
    }

    sockaddr_in bind_addr{};
    bind_addr.sin_family = AF_INET;
    bind_addr.sin_port = htons(static_cast<uint16_t>(config_.udp_listen_port));
    if (config_.udp_listen_ip.empty() || config_.udp_listen_ip == "0.0.0.0") {
        bind_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (inet_pton(AF_INET, config_.udp_listen_ip.c_str(),
                         &bind_addr.sin_addr) != 1) {
        std::cerr << "[LocalizationBridge] invalid listen ip="
                  << config_.udp_listen_ip << std::endl;
        close(udp_fd_);
        udp_fd_ = -1;
        return false;
    }

    if (bind(udp_fd_, reinterpret_cast<const sockaddr*>(&bind_addr),
             sizeof(bind_addr)) != 0) {
        std::cerr << "[LocalizationBridge] udp bind failed: "
                  << config_.udp_listen_ip << ":" << config_.udp_listen_port
                  << ", errno=" << errno << " " << std::strerror(errno)
                  << std::endl;
        close(udp_fd_);
        udp_fd_ = -1;
        return false;
    }
    timeval timeout{};
    timeout.tv_sec = 0;
    timeout.tv_usec = 100000;
    setsockopt(udp_fd_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    return true;
}

bool LocalizationBridge::openPayloadLog() {
    if (!config_.enable_payload_log) {
        return true;
    }
    std::lock_guard<std::mutex> lock(payload_log_mutex_);
    payload_log_stream_.open(config_.payload_log_path, std::ios::out | std::ios::app);
    if (!payload_log_stream_.is_open()) {
        std::cerr << "[LocalizationBridge] open payload log failed: "
                  << config_.payload_log_path << std::endl;
        return false;
    }
    payload_log_pending_lines_ = 0;
    return true;
}

bool LocalizationBridge::start() {
    if (running_.load()) return true;
    if (!openUdp()) return false;
    if (!openPayloadLog()) {
        closeAll();
        return false;
    }

    rx_count_ = 0;
    parse_ok_count_ = 0;
    parse_fail_count_ = 0;
    udp_send_ok_count_ = 0;
    udp_send_fail_count_ = 0;

    running_.store(true);
    worker_ = std::thread(&LocalizationBridge::workerLoop, this);
    std::cout << "[LocalizationBridge] started, input=UDP JSON "
              << config_.udp_listen_ip << ":" << config_.udp_listen_port
              << ", forward=" << config_.udp_target_ip << ":"
              << config_.udp_target_port
              << (config_.enable_payload_log
                      ? (", log=" + config_.payload_log_path)
                      : ", log=OFF")
              << std::endl;
    return true;
}

void LocalizationBridge::stop() {
    if (!running_.load()) return;
    running_.store(false);
    if (worker_.joinable()) worker_.join();
    closeAll();
    std::cout << "[LocalizationBridge] stopped" << std::endl;
}

void LocalizationBridge::closeAll() {
    if (udp_fd_ >= 0) {
        close(udp_fd_);
        udp_fd_ = -1;
    }
    {
        std::lock_guard<std::mutex> lock(payload_log_mutex_);
        if (payload_log_stream_.is_open()) {
            payload_log_stream_.flush();
            payload_log_stream_.close();
        }
        payload_log_pending_lines_ = 0;
    }
}

void LocalizationBridge::workerLoop() {
    uint8_t buffer[2048] = {0};
    sockaddr_in src_addr{};
    socklen_t src_len = sizeof(src_addr);

    while (running_.load()) {
        const ssize_t n = recvfrom(
            udp_fd_,
            buffer,
            sizeof(buffer),
            0,
            reinterpret_cast<sockaddr*>(&src_addr),
            &src_len);

        if (n > 0) {
            rx_count_++;
            handleIncomingJson(buffer, static_cast<size_t>(n));
        } else if (n == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        } else {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            std::cerr << "[LocalizationBridge] udp recv error, errno="
                      << errno << " " << std::strerror(errno) << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

bool LocalizationBridge::handleIncomingJson(const uint8_t* data, size_t len) {
    if (data == nullptr || len == 0) {
        return false;
    }

    const std::string text(reinterpret_cast<const char*>(data), len);
    const auto js = nlohmann::json::parse(text, nullptr, false);
    if (js.is_discarded() || !js.is_object()) {
        parse_fail_count_++;
        std::cerr << "[LocalizationBridge] json parse failed, bytes="
                  << len << std::endl;
        return false;
    }

    double x = 0.0;
    double y = 0.0;
    if (!readPositionFromJson(js, x, y)) {
        parse_fail_count_++;
        std::cerr << "[LocalizationBridge] unsupported localization payload: "
                  << text << std::endl;
        return false;
    }

    float lower_yaw_deg = 0.0f;
    {
        std::function<bool(float& yaw_deg)> provider;
        {
            std::lock_guard<std::mutex> lock(yaw_provider_mutex_);
            provider = lower_yaw_provider_;
        }
        if (provider) {
            provider(lower_yaw_deg);
        }
    }

    float pitch_deg = 0.0f;
    const float yaw_deg = -lower_yaw_deg;
    float roll_deg = 0.0f;
    if (config_.yaw_only_mode) {
        pitch_deg = 0.0f;
        roll_deg = 0.0f;
    }

    parse_ok_count_++;
    const double timestamp_sec =
        js.contains("timestamp") && js["timestamp"].is_number()
            ? js["timestamp"].get<double>()
            : hostUnixSec();
    const double out_z =
        config_.use_fixed_z
            ? static_cast<double>(config_.fixed_z_value)
            : (js.contains("z") && js["z"].is_number() ? js["z"].get<double>() : 0.0);

    nlohmann::json out;
    out["type"] = "robot_position";
    out["pos"] = {x, out_z, y};
    out["euler"] = {pitch_deg, yaw_deg, roll_deg};
    out["seq"] = js.value("seq", static_cast<int64_t>(parse_ok_count_));
    out["timestamp"] = timestamp_sec;
    out["host_unix_sec"] = hostUnixSec();
    out["source"] = "ac313_udp_json_lower_yaw";

    return forwardJson(out.dump());
}

void LocalizationBridge::appendPayloadLog(const std::string& payload) {
    if (!config_.enable_payload_log) {
        return;
    }

    std::lock_guard<std::mutex> lock(payload_log_mutex_);
    if (!payload_log_stream_.is_open()) {
        return;
    }
    payload_log_stream_ << payload << '\n';
    if (++payload_log_pending_lines_ >= kLogFlushEveryLines) {
        payload_log_stream_.flush();
        payload_log_pending_lines_ = 0;
    }
}

bool LocalizationBridge::forwardJson(const std::string& payload) {
    appendPayloadLog(payload);
    if (!config_.enable_forward) {
        return true;
    }
    if (udp_fd_ < 0) {
        udp_send_fail_count_++;
        return false;
    }

    sockaddr_in target{};
    target.sin_family = AF_INET;
    target.sin_port = htons(static_cast<uint16_t>(config_.udp_target_port));
    if (inet_pton(AF_INET, config_.udp_target_ip.c_str(), &target.sin_addr) != 1) {
        udp_send_fail_count_++;
        std::cerr << "[LocalizationBridge] invalid target ip="
                  << config_.udp_target_ip << std::endl;
        return false;
    }

    const ssize_t sent = sendto(
        udp_fd_,
        payload.data(),
        payload.size(),
        0,
        reinterpret_cast<const sockaddr*>(&target),
        sizeof(target));
    if (sent != static_cast<ssize_t>(payload.size())) {
        udp_send_fail_count_++;
        std::cerr << "[LocalizationBridge] udp send failed, payload_len="
                  << payload.size() << ", errno=" << errno << " "
                  << std::strerror(errno) << std::endl;
        return false;
    }

    udp_send_ok_count_++;
    return true;
}
