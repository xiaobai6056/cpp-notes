#ifndef JOYCONTROL_HPP
#define JOYCONTROL_HPP

#include <string>
#include <linux/joystick.h>

class JoyControl {
public:
    /**
     * @brief 构造函数
     * @param dev_path 手柄设备路径，默认为 /dev/input/js0
     */
    JoyControl(const std::string& dev_path = "/dev/input/js0");

    ~JoyControl();

    bool init();
    void update();

    float getSpeed() const;
    float getTurn() const;
    bool isActive() const;

    // --- 参数设置接口 ---

    /**
     * @brief 设置死区 (0 - 32767)
     */
    void setDeadzone(int deadzone);

    /**
     * @brief 单独设置【速度】的最大输出值
     * 例如：设为 100.0，则 getSpeed() 返回 -100.0 到 100.0
     */
    void setMaxSpeed(float max_val);

    /**
     * @brief 单独设置【转向】的最大输出值
     * 例如：舵机最大角度 45度，则设为 45.0，getTurn() 返回 -45.0 到 45.0
     */
    void setMaxTurn(float max_val);

private:
    std::string device_path_;
    int fd_;
    
    // 配置参数
    int deadzone_;
    float max_speed_; // 速度最大值
    float max_turn_;  // 转向最大值

    // 当前状态值
    float current_speed_;
    float current_turn_;

    /**
     * @brief 内部映射函数
     * @param raw_value 原始读数
     * @param target_max 目标最大值 (根据是速度还是转向传入不同值)
     * @param invert 是否反转符号
     */
    float mapValue(int raw_value, float target_max, bool invert = false);
};

#endif // JOYCONTROL_HPP