#pragma once

#include <chrono>
#include <mutex>

namespace host_clock {

struct ClockState {
    std::mutex mtx;
    bool inited = false;
    std::chrono::steady_clock::time_point t0{};
};

inline ClockState& state() {
    static ClockState s;
    return s;
}

inline void ensureInitT0() {
    auto& st = state();
    std::lock_guard<std::mutex> lock(st.mtx);
    if (!st.inited) {
        st.t0 = std::chrono::steady_clock::now();
        st.inited = true;
    }
}

inline double steadyNowSec() {
    auto& st = state();
    std::lock_guard<std::mutex> lock(st.mtx);
    if (!st.inited) {
        st.t0 = std::chrono::steady_clock::now();
        st.inited = true;
        return 0.0;
    }
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - st.t0).count();
}

inline double wallUnixSec() {
    return std::chrono::duration<double>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

} // namespace host_clock
