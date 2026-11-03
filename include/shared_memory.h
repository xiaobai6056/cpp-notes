/**
 * shared_memory.h - 共享内存数据结构定义（Python <-> C++统一协议）
 *
 * 内存布局:
 *   [Header 64B][Image Data][Detection[0] 24B]...[Detection[MAX-1] 24B]
 */

#ifndef SHARED_MEMORY_H
#define SHARED_MEMORY_H

#include <cstdint>
#include <cstdlib>

#ifdef __cplusplus
extern "C" {
#endif

// ============== 常量定义 ==============

// 共享内存名称 (/dev/shm/ 下的文件名)
#define SHM_NAME "/yoloe_det_shm"

// 最大检测目标数量
#define MAX_DETECTIONS 64

// 图像固定尺寸（统一为 640x640）
#define MAX_IMAGE_WIDTH 640
#define MAX_IMAGE_HEIGHT 640
#define MAX_IMAGE_SIZE (MAX_IMAGE_WIDTH * MAX_IMAGE_HEIGHT * 3)  // RGB 格式

// 共享内存布局大小
#define SHM_HEADER_SIZE 64
#define SHM_IMAGE_SIZE MAX_IMAGE_SIZE
#define SHM_DETECTION_SIZE (MAX_DETECTIONS * sizeof(ShmDetection))
#define SHM_TOTAL_SIZE (SHM_HEADER_SIZE + SHM_IMAGE_SIZE + SHM_DETECTION_SIZE)

// 魔数
#define SHM_MAGIC 0xDEADBEEF

// ============== 数据结构 ==============

/**
 * 单个检测框结构体 (24字节)
 * 与 Python 端格式 '<fffffB3x' 严格一致
 */
typedef struct __attribute__((packed)) {
    float x1, y1;        // 左上角坐标
    float x2, y2;        // 右下角坐标
    float confidence;    // 置信度
    uint8_t class_id;    // 类别ID
    uint8_t padding[3];  // 对齐填充
} ShmDetection;

/**
 * 共享内存头部 (64字节)
 * 与 Python 端格式 '<IIIIQIIII24x' 严格一致
 */
typedef struct __attribute__((packed)) {
    volatile uint32_t magic;            // 0xDEADBEEF
    volatile uint32_t frame_id;         // 帧ID
    volatile uint32_t valid;            // 数据有效标志 (1=有效, 0=写入中)
    volatile uint32_t detection_count;  // 检测目标数量
    volatile uint64_t timestamp_ns;     // 时间戳（纳秒）
    volatile uint32_t image_width;      // 图像宽度
    volatile uint32_t image_height;     // 图像高度
    volatile uint32_t image_channels;   // 图像通道数（固定3=RGB）
    volatile uint32_t image_size;       // 图像字节数
    uint8_t padding[24];                // 填充到64B
} ShmHeader;

/**
 * 检测结果共享数据
 */
typedef struct {
    ShmHeader header;
    uint8_t image_data[SHM_IMAGE_SIZE];
    ShmDetection detections[MAX_DETECTIONS];
} SharedData;

#ifdef __cplusplus
}
#endif

// ============== C++ 封装类声明 ==============

#ifdef __cplusplus

#include <string>
#include <vector>

namespace yoloe {

struct DetectionResult {
    uint32_t frame_id;
    uint64_t timestamp_ns;
    uint32_t width;
    uint32_t height;
    std::vector<uint8_t> image_data;  // RGB

    struct Box {
        float x1, y1, x2, y2;
        float confidence;
        uint8_t class_id;
        std::string class_name;
    };

    std::vector<Box> boxes;
};

class SharedMemoryReader {
public:
    SharedMemoryReader();
    ~SharedMemoryReader();

    bool init(const std::string& shm_name = SHM_NAME, int timeout_ms = 100);
    bool read(DetectionResult& result, uint32_t last_frame_id = UINT32_MAX);
    bool hasNewData();
    void close();

    bool isConnected() const { return connected_; }

private:
    bool connected_;
    int shm_fd_;
    void* shm_ptr_;

    static const char* CLASS_NAMES[];
};

} // namespace yoloe

#endif // __cplusplus

#endif // SHARED_MEMORY_H
