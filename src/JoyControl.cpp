#include "JoyControl.hpp"
#include <fcntl.h>      // open
#include <unistd.h>     // close, read
#include <cmath>        // abs
#include <iostream>

// 轴号配置
#define AXIS_LEFT_Y 1   // 速度控制轴
#define AXIS_RIGHT_X 2  // 方向控制轴 (部分手柄可能是 2)

JoyControl::JoyControl(const std::string& dev_path) 
    : device_path_(dev_path), fd_(-1), 
      deadzone_(4000),      // 默认死区
      max_speed_(3.0f),   // 默认速度最大值 100
      max_turn_(150.0f),     // 默认转向最大值 45 (假设是角度)
      current_speed_(0.0f), 
      current_turn_(0.0f) 
{
}

JoyControl::~JoyControl() {
    if (fd_ >= 0) close(fd_);
}

bool JoyControl::init() {
    fd_ = open(device_path_.c_str(), O_RDONLY | O_NONBLOCK);
    if (fd_ < 0) return false;
    return true;
}

void JoyControl::update() {
    if (fd_ < 0) return;

    struct js_event js;
    
    while (read(fd_, &js, sizeof(js)) > 0) {
        if ((js.type & ~JS_EVENT_INIT) == JS_EVENT_AXIS) {
            
            if (js.number == AXIS_LEFT_Y) {
                // 处理速度：传入 max_speed_，且反转符号(invert=true)
                current_speed_ = mapValue(js.value, max_speed_, false); 
            }
            else if (js.number == AXIS_RIGHT_X) {
                // 处理转向：传入 max_turn_，不反转(invert=false)
                current_turn_ = mapValue(js.value, max_turn_, false);
            }
        }
    }
}

// 修改后的通用映射函数，接受 target_max 参数
float JoyControl::mapValue(int raw_value, float target_max, bool invert) {
    // 1. 死区检测
    if (std::abs(raw_value) < deadzone_) {
        return 0.0f;
    }

    // 2. 归一化 (0.0 - 1.0)
    float normalized = static_cast<float>(raw_value) / 32767.0f;

    // 3. 符号处理
    if (invert) {
        normalized = -normalized;
    }

    // 4. 映射到具体的目标范围 (Speed 或 Turn)
    return normalized * target_max;
}

float JoyControl::getSpeed() const { return current_speed_; }
float JoyControl::getTurn() const { return current_turn_; }

bool JoyControl::isActive() const {
    return (std::abs(current_speed_) > 0.1f || std::abs(current_turn_) > 0.1f);
}

void JoyControl::setDeadzone(int deadzone) {
    if (deadzone < 0) deadzone = 0;
    if (deadzone > 30000) deadzone = 30000;
    deadzone_ = deadzone;
}

void JoyControl::setMaxSpeed(float max_val) {
    max_speed_ = max_val;
}

void JoyControl::setMaxTurn(float max_val) {
    max_turn_ = max_val;
}