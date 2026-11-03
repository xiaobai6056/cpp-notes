#pragma once

#include <atomic>
#include <cstdint>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

class LocalizationBridge {
public:
    struct Config {
        std::string udp_listen_ip = "0.0.0.0";
        int udp_listen_port = 9001;
        std::string udp_target_ip = "192.168.137.137";
        int udp_target_port = 9005;
        bool enable_forward = true;
        bool enable_payload_log = false;
        std::string payload_log_path =
            "/home/orangepi/workspace/Dragon/res/Videos/localization_forward.log";
        bool use_fixed_z = true;
        float fixed_z_value = 0.15f;
        bool yaw_only_mode = true;
    };

    LocalizationBridge();
    explicit LocalizationBridge(const Config& config);
    ~LocalizationBridge();

    bool start();
    void stop();
    bool isRunning() const { return running_.load(); }
    void setConfig(const Config& config);

    void setLowerYawProvider(const std::function<bool(float& yaw_deg)>& provider);

private:
    Config config_;
    std::atomic<bool> running_{false};
    std::thread worker_;
    int udp_fd_ = -1;

    uint64_t rx_count_ = 0;
    uint64_t parse_ok_count_ = 0;
    uint64_t parse_fail_count_ = 0;
    uint64_t udp_send_ok_count_ = 0;
    uint64_t udp_send_fail_count_ = 0;

    std::ofstream payload_log_stream_;
    mutable std::mutex payload_log_mutex_;
    size_t payload_log_pending_lines_ = 0;

    std::function<bool(float& yaw_deg)> lower_yaw_provider_;
    mutable std::mutex yaw_provider_mutex_;

    bool openUdp();
    bool openPayloadLog();
    void closeAll();
    void workerLoop();
    bool handleIncomingJson(const uint8_t* data, size_t len);
    bool forwardJson(const std::string& payload);
    void appendPayloadLog(const std::string& payload);
};
