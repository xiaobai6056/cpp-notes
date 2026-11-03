/**
 * shared_memory_reader.cpp - 共享内存读取器实现
 * 
 * 高性能、低延迟的进程间通信
 * 使用 mmap 实现零拷贝数据读取
 */

#include "shared_memory.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>
#include <thread>
#include <chrono>

namespace yoloe {

// 业务类别名称表（与共享内存 class_id 顺序一致）
const char* SharedMemoryReader::CLASS_NAMES[] = {
    "arrow",
    "people",
    "car",
    "gold",
    "door",
    "road_sign",
    "speed_limit_sign",
    "red",
    "yellow",
    "green",
    "stop"
};

SharedMemoryReader::SharedMemoryReader()
    : connected_(false), shm_fd_(-1), shm_ptr_(nullptr) {
}

SharedMemoryReader::~SharedMemoryReader() {
    close();
}

bool SharedMemoryReader::init(const std::string& shm_name, int timeout_ms) {
    // 打开共享内存
    shm_fd_ = shm_open(shm_name.c_str(), O_RDONLY, 0666);
    if (shm_fd_ < 0) {
        fprintf(stderr, "[ShmReader] Failed to open shared memory '%s': %s\n",
                shm_name.c_str(), strerror(errno));
        fprintf(stderr, "[ShmReader] Please make sure the Python detection program is running.\n");
        return false;
    }

    // 映射共享内存
    shm_ptr_ = mmap(NULL, SHM_TOTAL_SIZE, PROT_READ, MAP_SHARED, shm_fd_, 0);
    if (shm_ptr_ == MAP_FAILED) {
        fprintf(stderr, "[ShmReader] Failed to mmap shared memory: %s\n", strerror(errno));
        close();
        return false;
    }

    connected_ = true;
    fprintf(stdout, "[ShmReader] Shared memory reader initialized successfully.\n");
    fflush(stdout);
    return true;
}

bool SharedMemoryReader::hasNewData() {
    if (!connected_ || !shm_ptr_) return false;

    ShmHeader* header = static_cast<ShmHeader*>(shm_ptr_);
    return header->magic == SHM_MAGIC && header->valid == 1;
}

bool SharedMemoryReader::read(DetectionResult& result, uint32_t last_frame_id) {
    if (!connected_ || !shm_ptr_) return false;

    SharedData* data = static_cast<SharedData*>(shm_ptr_);
    ShmHeader* header = &data->header;

    // 检查数据有效性
    if (header->magic != SHM_MAGIC) {
        return false;  // 共享内存未初始化
    }

    // 内存屏障，确保读取最新数据
    __sync_synchronize();

    if (header->valid != 1) {
        return false;  // 数据未就绪
    }

    // 读取头部信息
    // Polling usually observes the same frame several times. Reject it before
    // allocating vectors or copying the optional image payload.
    if (header->frame_id == last_frame_id) {
        return false;
    }

    result.frame_id = header->frame_id;
    result.timestamp_ns = header->timestamp_ns;
    result.width = header->image_width;
    result.height = header->image_height;

    // 复制图像数据
    size_t img_size = header->image_size;
    if (img_size > 0 && img_size <= SHM_IMAGE_SIZE) {
        result.image_data.resize(img_size);
        memcpy(result.image_data.data(), data->image_data, img_size);
    }

    // 复制检测框
    uint32_t count = header->detection_count;
    result.boxes.clear();

    for (uint32_t i = 0; i < count && i < MAX_DETECTIONS; i++) {
        ShmDetection& det = data->detections[i];
        DetectionResult::Box box;
        box.x1 = det.x1;
        box.y1 = det.y1;
        box.x2 = det.x2;
        box.y2 = det.y2;
        box.confidence = det.confidence;
        box.class_id = det.class_id;
        constexpr size_t kClassNameCount = 11;
        if (det.class_id < kClassNameCount) {
            box.class_name = CLASS_NAMES[det.class_id];
        } else {
            box.class_name = "unknown";
        }
        result.boxes.push_back(box);
    }

    return true;
}

void SharedMemoryReader::close() {
    if (shm_ptr_) {
        munmap(shm_ptr_, SHM_TOTAL_SIZE);
        shm_ptr_ = nullptr;
    }
    if (shm_fd_ >= 0) {
        ::close(shm_fd_);  // 使用 :: 显式调用 POSIX close()
        shm_fd_ = -1;
    }
    connected_ = false;
}

} // namespace yoloe
