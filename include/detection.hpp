
#pragma once
#include <sys/time.h>
#include <sys/stat.h>
#include <cstdio>
#include <functional>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <cstdlib>
#include <memory>
#include <cctype>
#include <stdlib.h>
#include "common.hpp"
#include "yolov5.h"
using namespace std;

struct NDTensor
{
    std::vector<int64_t> shape;
    std::vector<float> buffer;
    std::vector<uint8_t> u8_buffer;

    NDTensor() = default;
    explicit NDTensor(const std::vector<int64_t> &shape_) : shape(shape_)
    {
        int64_t count = 1;
        for (auto dim : shape)
        {
            count *= dim;
        }
        if (count > 0)
        {
            buffer.resize(static_cast<size_t>(count), 0.0f);
        }
    }

    float *value()
    {
        return buffer.empty() ? nullptr : buffer.data();
    }

    const float *value() const
    {
        return buffer.empty() ? nullptr : buffer.data();
    }

    uint8_t *bytes()
    {
        return u8_buffer.empty() ? nullptr : u8_buffer.data();
    }

    const uint8_t *bytes() const
    {
        return u8_buffer.empty() ? nullptr : u8_buffer.data();
    }
};

/**
 * @brief 目标检测结果
 *
 */
struct PredictResult
{
    int type;          // ID
    std::string label; // 标签
    float score;       // 置信度
    int x;             // 坐标
    int y;             // 坐标
    int width;         // 尺寸
    int height;        // 尺寸
};

class Detection
{
private:
    std::vector<std::string> labels;
    rknn_app_context_t app_ctx_{};
    object_detect_result_list od_results_{};
    bool initialized_ = false;
    bool init_log_printed_ = false;
    std::string model_file_;

    static bool fileExists(const std::string &path)
    {
        struct stat st;
        return !path.empty() && stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
    }

    static std::string dirnameOf(const std::string &path)
    {
        auto pos = path.find_last_of('/');
        if (pos == std::string::npos)
        {
            return std::string();
        }
        if (pos == 0)
        {
            return "/";
        }
        return path.substr(0, pos);
    }

    static std::string toLower(std::string text)
    {
        std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch)
                       { return static_cast<char>(std::tolower(ch)); });
        return text;
    }

    static std::string trim(const std::string &text)
    {
        const auto begin = text.find_first_not_of(" \t\r\n");
        if (begin == std::string::npos)
        {
            return std::string();
        }
        const auto end = text.find_last_not_of(" \t\r\n");
        return text.substr(begin, end - begin + 1);
    }

    std::string resolveModelPath(const std::string &pathModel)
    {
        if (pathModel.size() >= 5 && pathModel.substr(pathModel.size() - 5) == ".rknn" && fileExists(pathModel))
        {
            return pathModel;
        }

        const std::vector<std::string> candidates = {
            pathModel + "/yolov5.rknn",
            pathModel + "/model/yolov5.rknn",
            pathModel + "/best.rknn",
            pathModel + "/model.rknn"};

        for (const auto &candidate : candidates)
        {
            if (fileExists(candidate))
            {
                return candidate;
            }
        }
        return std::string();
    }

    void loadLabels(const std::string &pathModel, const std::string &modelFile)
    {
        labels.clear();
        std::vector<std::string> candidates;
        candidates.push_back(pathModel + "/label_list.txt");
        candidates.push_back(pathModel + "/labellist.txt");

        const std::string modelDir = dirnameOf(modelFile);
        if (!modelDir.empty())
        {
            candidates.push_back(modelDir + "/label_list.txt");
            candidates.push_back(modelDir + "/labellist.txt");
        }

        for (const auto &pathLabels : candidates)
        {
            std::ifstream file(pathLabels);
            if (!file.is_open())
            {
                continue;
            }

            std::string line;
            while (getline(file, line))
            {
                line = trim(line);
                if (!line.empty())
                {
                    labels.push_back(line);
                }
            }
            file.close();
            if (!labels.empty())
            {
                return;
            }
        }
    }

    int mapTypeByLabel(const std::string &label, int fallback) const
    {
        const std::string key = toLower(label);
        
        // 新模型标签映射 (arrow, people, car, gold)
        if (key == LABEL_ARROW || key == "arrow")
            return TYPE_ARROW;
        if (key == LABEL_PEOPLE || key == "people" || key == "person")
            return TYPE_PEOPLE;
        if (key == LABEL_CAR || key == "car")
            return TYPE_CAR;
        if (key == LABEL_GOLD || key == "gold")
            return TYPE_GOLD;

        return fallback;
    }

    bool doInference(const image_buffer_t &imgBuffer)
    {
        // std::cout << "[DEBUG] Detection::doInference: Enter" << std::endl;
        
        if (!initialized_)
        {
            std::cerr << "[ERROR] Detection::doInference: not initialized" << std::endl;
            return false;
        }

        od_results_.count = 0;
        image_buffer_t input = imgBuffer;
        
        // 打印输入信息
        // std::cout << "[DEBUG] Detection::doInference: input width=" << input.width 
        //           << ", height=" << input.height 
        //           << ", format=" << input.format
        //           << ", size=" << input.size << std::endl;
        
        if (input.virt_addr == nullptr)
        {
            std::cerr << "[ERROR] Detection::doInference: virt_addr is null!" << std::endl;
            return false;
        }
        
        if (input.width <= 0 || input.height <= 0 || input.size <= 0)
        {
            std::cerr << "[ERROR] Detection::doInference: invalid input parameters!" << std::endl;
            return false;
        }
        
        // 检查前几个字节（调试用）
#if 0
        // std::cout << "[DEBUG] Detection::doInference: first 10 bytes: ";
        for (int i = 0; i < 10 && i < input.size; i++)
        {
            printf("%02x ", input.virt_addr[i]);
        }
        std::cout << std::endl;
#endif
        
        int ret = inference_yolov5_model(&app_ctx_, &input, &od_results_);
        
        // std::cout << "[DEBUG] Detection::doInference: inference_yolov5_model ret=" << ret 
        //           << ", results count=" << od_results_.count << std::endl;
                  
        if (ret != 0)
        {
            std::cerr << "[ERROR] Detection::doInference: inference failed with code " << ret << std::endl;
            od_results_.count = 0;
            return false;
        }
        return true;
    }

public:
    vector<PredictResult> results;
    const int ARROW_CLASS_ID = 0;

    // 暴露标签映射方法供外部使用（共享内存读取线程调用）
    int mapTypeByLabelPublic(const std::string& label, int fallback) const {
        return mapTypeByLabel(label, fallback);
    }

    std::vector<cv::Point> getArrowCenters() {
        std::vector<cv::Point> centers;
        for (const auto &result : results) {
            std::string lbl = toLower(result.label);
            if (result.type == ARROW_CLASS_ID || lbl == "arrow") {
                centers.emplace_back(result.x + result.width / 2, result.y + result.height / 2);
            }
        }
        return centers;
    }

    float calculateIOU(const PredictResult &box1, const PredictResult &box2) {
        int x1 = std::max(box1.x, box2.x);
        int y1 = std::max(box1.y, box2.y);
        int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
        int y2 = std::min(box1.y + box1.height, box2.y + box2.height);

        if (x2 <= x1 || y2 <= y1) return 0.0f;

        int intersection = (x2 - x1) * (y2 - y1);
        int area1 = box1.width * box1.height;
        int area2 = box2.width * box2.height;
        int union_area = area1 + area2 - intersection;
        if (union_area <= 0) return 0.0f;
        return static_cast<float>(intersection) / static_cast<float>(union_area);
    }

    void nms(std::vector<PredictResult> &detections, float iou_threshold = 0.45f) {
        if (detections.empty()) return;
        std::sort(detections.begin(), detections.end(), [](const PredictResult &a, const PredictResult &b) {
            return a.score > b.score;
        });
        std::vector<bool> keep(detections.size(), true);
        for (size_t i = 0; i < detections.size(); ++i) {
            if (!keep[i]) continue;
            for (size_t j = i + 1; j < detections.size(); ++j) {
                if (!keep[j]) continue;
                float iou = calculateIOU(detections[i], detections[j]);
                if (iou > iou_threshold) keep[j] = false;
            }
        }
        std::vector<PredictResult> filtered;
        for (size_t i = 0; i < detections.size(); ++i) if (keep[i]) filtered.push_back(detections[i]);
        detections.swap(filtered);
    }

    void drawBoxes(cv::Mat &img, bool showScores = true) {
        if (img.empty()) return;
        for (const auto &result : results) {
            if (result.width <= 0 || result.height <= 0) continue;
            cv::Rect box(result.x, result.y, result.width, result.height);
            cv::Rect image_bounds(0, 0, img.cols, img.rows);
            box = box & image_bounds;
            if (box.width <= 0 || box.height <= 0) continue;
            cv::Scalar color = getCvcolor(result.type);
            cv::rectangle(img, box, color, 2);
            if (showScores) {
                std::ostringstream oss;
                oss << result.label << ": " << std::fixed << std::setprecision(2) << result.score;
                cv::putText(img, oss.str(), cv::Point(box.x, std::max(0, box.y - 5)), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            }
        }
    }

    ~Detection()
    {
        deinit();
    }

    void deinit()
    {
        if (initialized_)
        {
            release_yolov5_model(&app_ctx_);
            initialized_ = false;
        }
        model_file_.clear();
    }

    bool init (const std::string pathModel)
    {
        // std::cout << "[DEBUG] Detection::init: Enter with path=" << pathModel << std::endl;
        
        deinit();
        model_file_ = resolveModelPath(pathModel);
        if (model_file_.empty())
        {
            std::cout << "RKNN model not found from path: " << pathModel << std::endl;
            return false;
        }
        
        // std::cout << "[DEBUG] Detection::init: Model file resolved to: " << model_file_ << std::endl;

        loadLabels(pathModel, model_file_);
        // std::cout << "[DEBUG] Detection::init: Loaded " << labels.size() << " labels" << std::endl;

        int ret = init_yolov5_model(model_file_.c_str(), &app_ctx_);
        if (ret != 0)
        {
            std::cout << "init_yolov5_model failed: " << model_file_ << std::endl;
            return false;
        }

        initialized_ = true;
        init_log_printed_ = false;
        
        // std::cout << "[DEBUG] Detection::init: Success" << std::endl;
        return true;
    }

    void inference(cv::Mat img)
    {
        auto feeds = preprocess(img, {img.rows, img.cols});
        run(*feeds);
        render();
    }

    std::shared_ptr<std::unordered_map<std::string, NDTensor>> preprocess(
    cv::Mat frame,
    const std::vector<int64_t> &input_size)
{
    auto ret = std::make_shared<std::unordered_map<std::string, NDTensor>>();
    
    // 调试输出：preprocess开始
    //std::cout << "[DEBUG] AI预处理: 开始 - 输入尺寸: " << frame.cols << "x" << frame.rows << " - 通道: " << frame.channels() << std::endl;
    
    if (frame.empty())
    {
        std::cerr << "[ERROR] Detection::preprocess: Input frame is empty!" << std::endl;
        return ret;
    }
    
    // std::cout << "[DEBUG] Detection::preprocess: Input frame size=" << frame.cols << "x" << frame.rows 
    //           << ", channels=" << frame.channels() << std::endl;
    
    // ========== 关键修改：Resize到模型输入尺寸 ==========
    int target_width = 640;
    int target_height = 640;
    
    // 如果传入了input_size，使用传入的尺寸
    if (input_size.size() >= 2)
    {
        target_height = static_cast<int>(input_size[0]);
        target_width = static_cast<int>(input_size[1]);
    }
    
    // std::cout << "[DEBUG] Detection::preprocess: Target size=" << target_width << "x" << target_height << std::endl;
    
    cv::Mat resized_frame;
    if (frame.cols != target_width || frame.rows != target_height)
    {
        cv::resize(frame, resized_frame, cv::Size(target_width, target_height));
        // std::cout << "[DEBUG] Detection::preprocess: Resized to " << target_width << "x" << target_height << std::endl;
    }
    else
    {
        resized_frame = frame;
    }
    
    // 调试输出：尺寸调整后
    //std::cout << "[DEBUG] AI预处理: 尺寸调整后 - 尺寸: " << resized_frame.cols << "x" << resized_frame.rows << " - 通道: " << resized_frame.channels() << std::endl;
    // ==================================================
    
    cv::Mat rgb;
    try {
        cv::cvtColor(resized_frame, rgb, cv::COLOR_BGR2RGB);
        // std::cout << "[DEBUG] Detection::preprocess: cvtColor success" << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "[ERROR] Detection::preprocess: cvtColor failed: " << e.what() << std::endl;
        return ret;
    }
    
    if (rgb.empty())
    {
        std::cerr << "[ERROR] Detection::preprocess: rgb image is empty after conversion!" << std::endl;
        return ret;
    }
    
    // 调试输出：颜色转换后
    //std::cout << "[DEBUG] AI预处理: 颜色转换后 - 尺寸: " << rgb.cols << "x" << rgb.rows << " - 通道: " << rgb.channels() << std::endl;
    
    if (!rgb.isContinuous())
    {
        rgb = rgb.clone();
    }

    // 使用resize后的尺寸创建tensor
    NDTensor imgTensor({rgb.rows, rgb.cols, rgb.channels()});
    
    size_t expected_size = rgb.total() * rgb.elemSize();
    if (expected_size == 0)
    {
        std::cerr << "[ERROR] Detection::preprocess: Expected size is 0!" << std::endl;
        return ret;
    }
    
    imgTensor.u8_buffer.assign(rgb.data, rgb.data + expected_size);
    
    // std::cout << "[DEBUG] Detection::preprocess: Success, buffer size=" 
    //           << imgTensor.u8_buffer.size() << std::endl;
    
    (*ret)["image"] = std::move(imgTensor);
    return ret;
}

    void run(const std::unordered_map<std::string, NDTensor> &feeds)
    {
        // std::cout << "[DEBUG] Detection::run: Enter" << std::endl;
        
        auto imageIt = feeds.find("image");
        if (imageIt == feeds.end())
        {
            std::cerr << "[ERROR] Detection::run: 'image' not found in feeds" << std::endl;
            od_results_.count = 0;
            return;
        }
        
        if (imageIt->second.u8_buffer.empty())
        {
            std::cerr << "[ERROR] Detection::run: image tensor buffer is empty" << std::endl;
            od_results_.count = 0;
            return;
        }
        
        if (imageIt->second.shape.size() < 2)
        {
            std::cerr << "[ERROR] Detection::run: invalid shape, size=" << imageIt->second.shape.size() << std::endl;
            od_results_.count = 0;
            return;
        }

        if (!initialized_)
        {
            if (!init_log_printed_)
            {
                std::cout << "Detection not initialized, skip RKNN inference." << std::endl;
                init_log_printed_ = true;
            }
            od_results_.count = 0;
            return;
        }

        const NDTensor &imgTensor = imageIt->second;
        
        // 打印tensor信息
        // std::cout << "[DEBUG] Detection::run: tensor shape=";
        // for (auto dim : imgTensor.shape) std::cout << dim << " ";
        // std::cout << ", buffer size=" << imgTensor.u8_buffer.size() << std::endl;
        
        if (imgTensor.bytes() == nullptr)
        {
            std::cerr << "[ERROR] Detection::run: imgTensor.bytes() is null!" << std::endl;
            od_results_.count = 0;
            return;
        }
        
        image_buffer_t input{};
        input.width = static_cast<int>(imgTensor.shape[1]);
        input.height = static_cast<int>(imgTensor.shape[0]);
        input.width_stride = input.width;
        input.height_stride = input.height;
        input.format = IMAGE_FORMAT_RGB888;
        input.size = static_cast<int>(imgTensor.u8_buffer.size());
        input.fd = 0;
        input.virt_addr = const_cast<unsigned char *>(imgTensor.bytes());
        
        // 检查input参数
        // std::cout << "[DEBUG] Detection::run: input width=" << input.width 
        //           << ", height=" << input.height 
        //           << ", size=" << input.size 
        //           << ", virt_addr=" << (void*)input.virt_addr << std::endl;
        
        if (input.width <= 0 || input.height <= 0 || input.size <= 0)
        {
            std::cerr << "[ERROR] Detection::run: invalid input parameters!" << std::endl;
            od_results_.count = 0;
            return;
        }
        
        if (input.virt_addr == nullptr)
        {
            std::cerr << "[ERROR] Detection::run: virt_addr is null!" << std::endl;
            od_results_.count = 0;
            return;
        }

        doInference(input);
    }

    NDTensor get_output(int index)
    {
        (void)index;
        return NDTensor();
    }

    void render()
    {
        // std::cout << "[DEBUG] Detection::render: Enter, od_results_.count=" << od_results_.count << std::endl;
        
        results.clear();
        for (int i = 0; i < od_results_.count; ++i)
        {
            const object_detect_result &det = od_results_.results[i];
            PredictResult result;
            result.score = det.prop;
            result.x = det.box.left;
            result.y = det.box.top;
            result.width = det.box.right - det.box.left;
            result.height = det.box.bottom - det.box.top;

            if (result.width <= 0 || result.height <= 0)
            {
                // std::cout << "[DEBUG] Detection::render: Skipping invalid box: width=" 
                //           << result.width << ", height=" << result.height << std::endl;
                continue;
            }

            if (det.cls_id >= 0 && det.cls_id < static_cast<int>(labels.size()))
            {
                result.label = labels[det.cls_id];
            }
            else
            {
                result.label = std::to_string(det.cls_id);
            }
            result.type = mapTypeByLabel(result.label, det.cls_id);

            std::cout << "[DEBUG][render] cls_id=" << det.cls_id
                      << " label=" << result.label
                      << " type=" << result.type
                      << " score=" << result.score
                      << std::endl;

            // std::cout << "[DEBUG] Detection::render: Adding result: " << result.label 
            //           << ", score=" << result.score << ", box=" << result.x << "," 
            //           << result.y << " " << result.width << "x" << result.height << std::endl;
                      
            results.push_back(result);
        }
        
        // std::cout << "[DEBUG] Detection::render: Final results count=" << results.size() << std::endl;
    }

    void drawBox(Mat &img)
    {
        if (img.empty())
        {
            std::cerr << "[WARNING] Detection::drawBox: Input image is empty!" << std::endl;
            return;
        }
        
        for (int i = 0; i < results.size(); i++)
        {
            PredictResult result = results[i];

            auto score = std::to_string(result.score);
            int pointY = result.y - 20;
            if (pointY < 0)
                pointY = 0;
            
            // 确保矩形在图像范围内
            if (result.x >= 0 && result.y >= 0 && 
                result.x + result.width <= img.cols && 
                result.y + result.height <= img.rows)
            {
                cv::Rect rectText(result.x, pointY, result.width, 20);
                cv::rectangle(img, rectText, getCvcolor(result.type), -1);
                std::string label_name = result.label + " [" + score.substr(0, score.find(".") + 3) + "]";
                cv::Rect rect(result.x, result.y, result.width, result.height);
                cv::rectangle(img, rect, getCvcolor(result.type), 1);
                cv::putText(img, label_name, Point(result.x, result.y), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 254), 1);
            }
        }
    }

    /**
     * @brief 获取Opencv颜色
     *
     * @param index 序号
     * @return cv::Scalar
     */
    cv::Scalar getCvcolor(int index)
    {
        switch (index)
        {
        case 0:
            return cv::Scalar(0, 255, 0); // 绿
            break;
        case 1:
            return cv::Scalar(255, 255, 0); // 天空蓝
            break;
        case 2:
            return cv::Scalar(0, 0, 255); // 大红
            break;
        case 3:
            return cv::Scalar(0, 250, 250); // 大黄
            break;
        case 4:
            return cv::Scalar(250, 0, 250); // 粉色
            break;
        case 5:
            return cv::Scalar(0, 102, 255); // 橙黄
            break;
        case 6:
            return cv::Scalar(255, 0, 0); // 深蓝
            break;
        case 7:
            return cv::Scalar(255, 255, 255); // 大白
            break;
        case 8:
            return cv::Scalar(247, 43, 113);
            break;
        case 9:
            return cv::Scalar(40, 241, 245);
            break;
        case 10:
            return cv::Scalar(237, 226, 19);
            break;
        case 11:
            return cv::Scalar(245, 117, 233);
            break;
        case 12:
            return cv::Scalar(55, 13, 19);
            break;
        case 13:
            return cv::Scalar(255, 255, 255);
            break;
        case 14:
            return cv::Scalar(237, 226, 19);
            break;
        case 15:
            return cv::Scalar(0, 255, 0);
            break;
        default:
            return cv::Scalar(255, 0, 0);
            break;
        }
    }
};
