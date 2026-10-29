
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace nvinfer1;

// ==================== 配置 ====================

// 模型输入尺寸
constexpr int INPUT_W = 640;
constexpr int INPUT_H = 640;

// 后处理阈值
constexpr float CONF_THRESHOLD = 0.25f;  // 置信度阈值
constexpr float NMS_THRESHOLD = 0.70f;   // NMS 的 IoU 阈值

// 类别名称，索引对应模型输出的类别编号
const std::array<std::string, 3> CLASSNAME = {
    "earbuds", "toothpick", "dice"
};

// TensorRT Engine 文件路径
const std::string ENGINE_PATH =
    "/home/yq/my_project/Python/YOLO/deeplearning/"
    "ultralytics-8.3.163/results/yolo11n/weights/best.engine";


// ==================== 数据结构 ====================

// 保存 Tensor 的名称、字节数和形状
struct TensorInfo {
    std::string name;
    std::size_t bytes = 0;
    Dims shape{};
};

// 保存一个检测框的信息
struct Detection {
    float cx = 0;       // 中心点横坐标
    float cy = 0;       // 中心点纵坐标
    float w = 0;        // 检测框宽度
    float h = 0;        // 检测框高度

    cv::Point2f lt;     // 左上角坐标
    cv::Point2f rb;     // 右下角坐标

    float area = 0;     // 检测框面积
    float scores = 0;   // 最高类别分数
    int classId = 0;    // 预测类别编号
};

// 保存 Letterbox 的变换参数，供后续坐标还原使用
struct LetterboxInfo {
    float scale = 1.0f;
    float padX = 0.0f;
    float padY = 0.0f;
};


// ====== 3. CUDA 和 TensorRT 资源管理 ======

// TensorRT 日志记录器
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // 只输出 WARNING 及以上级别的日志
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << '\n';
        }
    }
};

// GPU 内存删除器
struct CudaDeleter {
    void operator()(void* ptr) const noexcept {
        if (ptr) {
            // ptr 是 cudaMalloc 返回的 GPU 地址
            cudaFree(ptr);
        }
    }
};

using CudaPtr = std::unique_ptr<void, CudaDeleter>;

// CUDA Stream 的 RAII 管理器
struct CudaStreamGuard {
    cudaStream_t stream = nullptr;

    CudaStreamGuard() = default;

    // 禁止复制，避免两个对象重复销毁同一个 Stream
    CudaStreamGuard(const CudaStreamGuard&) = delete;
    CudaStreamGuard& operator=(const CudaStreamGuard&) = delete;

    ~CudaStreamGuard() {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
};


// ==================== Engine 加载 ====================

std::unique_ptr<ICudaEngine> loadEngine(IRuntime* runtime, const std::string& path) {
    // 二进制读取，并将文件指针定位到文件末尾
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "无法打开 Engine 文件：" << path << '\n';
        return nullptr;
    }

    // 获取 Engine 文件大小，检测是否为空
    auto fileSize = file.tellg();
    if (fileSize <= 0) {
        std::cerr << "Engine 文件为空或读取失败！\n";
        return nullptr;
    }

    // 获取 Engine 文件的二进制数据
    std::vector<char> engineData(static_cast<std::size_t>(fileSize));

    // 将文件指针移回开头
    file.seekg(0);

    // 读取完整文件(文件源, 大小)
    if (!file.read(
            engineData.data(),  
            static_cast<std::streamsize>(engineData.size()))) {
        std::cerr << "读取 Engine 文件失败！\n";
        return nullptr;
    }

    // 将序列化数据反序列化为 ICudaEngine
    std::unique_ptr<ICudaEngine> engine(
        runtime->deserializeCudaEngine(
            engineData.data(),
            engineData.size()
        )
    );
    if (!engine) {
        std::cerr << "Engine 反序列化失败！\n";
        return nullptr;
    }

    std::cout << "Engine loaded successfully!\n";
    return engine;
}


// ==================== 获取输入或输出 Tensor 信息 ====================

// 获取各种 Tensor 的相关信息
std::optional<TensorInfo> getTensorInfo(ICudaEngine* engine, TensorIOMode targetMode) {
    std::optional<TensorInfo> result;

    // 遍历 Engine 中的所有输入输出 Tensor
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);

        // 只处理当前需要的输入或输出 Tensor
        if (engine->getTensorIOMode(name) != targetMode) {
            continue;
        }
        // 当前程序只支持一个输入 Tensor 和一个输出 Tensor
        if (result.has_value()) {
            std::cerr << "当前程序不支持多个同方向 Tensor！\n";
            return std::nullopt;
        }
        // 当前输出解析和输入预处理均按照 FP32 编写
        if (engine->getTensorDataType(name) != DataType::kFLOAT) {
            std::cerr << "当前程序仅支持 FP32 Tensor！\n";
            return std::nullopt;
        }

        // 获取 Tensor 形状
        auto shape = engine->getTensorShape(name);

        if (shape.nbDims <= 0) {
            return std::nullopt;
        }

        // 各维度相乘，得到 Tensor 的元素总数
        std::size_t totalElements = 1;

        for (int j = 0; j < shape.nbDims; ++j) {
            if (shape.d[j] <= 0) {
                std::cerr << "不支持动态或无效 Tensor 维度！\n";
                return std::nullopt;
            }

            totalElements *= static_cast<std::size_t>(shape.d[j]);
        }

        // FP32 每个元素占 sizeof(float) 个字节
        result = TensorInfo{
            name,
            totalElements * sizeof(float),
            shape
        };
    }

    // 如果没有找到目标 Tensor，返回空 optional
    return result;
}


// ==================== 图像预处理 ====================

// 将摄像头读取的 BGR 图像转换为模型需要的 NCHW FP32 数据
bool preprocess(const cv::Mat& img, std::vector<float>& nchwData, LetterboxInfo& info) {
    if (img.empty()) {
        return false;
    }

    // ====== Letterbox 尺寸处理 ======
    // 计算等比例缩放系数，保证图像完整放入 640×640
    const float scale = std::min(static_cast<float>(INPUT_W) / img.cols, static_cast<float>(INPUT_H) / img.rows);

    // 计算缩放后的图像尺寸
    const int resizedW = static_cast<int>(std::round(img.cols * scale));
    const int resizedH = static_cast<int>(std::round(img.rows * scale));

    // 等比例缩放
    cv::Mat resized;
    cv::resize(
        img,
        resized,
        cv::Size(resizedW, resizedH),
        0,
        0,
        cv::INTER_LINEAR
    );

    // 计算上下左右需要填充的像素数
    const int padW = INPUT_W - resizedW;
    const int padH = INPUT_H - resizedH;

    const int left = padW / 2;
    const int right = padW - left;
    const int top = padH / 2;
    const int bottom = padH - top;

    // 填充为固定尺寸的正方形
    cv::Mat padded;
    cv::copyMakeBorder(
        resized,
        padded,
        top, bottom,
        left, right,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114)
    );

    // 记录实际填充偏移，后续需要用它还原检测框坐标
    info.scale = scale;
    info.padX = static_cast<float>(left);
    info.padY = static_cast<float>(top);

    // ====== BGR转RGB 通道处理 ======
    cv::Mat rgb;
    cv::cvtColor(
        padded,
        rgb,
        cv::COLOR_BGR2RGB
    );

    // ====== uint8 转换为 FP32 + 像素值归一化 ======
    cv::Mat imgFloat;
    rgb.convertTo(
        imgFloat,
        CV_32F,
        1.0 / 255.0
    );

    // ====== HWC 转 NCHW ======
    std::vector<cv::Mat> channels;
    cv::split(imgFloat, channels);
    if (channels.size() != 3) {
        return false;
    }

    // 清除上一帧数据，保留已有容量以减少重复内存分配
    nchwData.clear();
    nchwData.reserve(imgFloat.total() * 3);

    // 按 R、G、B 顺序，逐行将每个通道追加到一维数组
    for (const auto& channel : channels) {
        for (int r = 0; r < channel.rows; ++r) {
            const float* row = channel.ptr<float>(r);

            nchwData.insert(
                nchwData.end(),
                row,
                row + channel.cols
            );
        }
    }

    // 此时数组包含 3×640×640 个 float
    return true;
}


// ==================== 模型输出解析 + 置信度初筛 ====================

// 当前模型输出形状为 (1, 7, 8400)
// 前 4 个通道为 cx、cy、w、h，后 3 个通道为类别分数
std::vector<Detection> parseCandidates(const std::vector<float>& modelOutput, int candidateCount) {
    std::vector<Detection> candidates;

    // 遍历每个候选框
    for (int i = 0; i < candidateCount; ++i) {

        // 先将类别 0 的分数作为当前最高分
        float maxScore = modelOutput[4 * candidateCount + i];
        int classId = 0;

        // 遍历每个类别
        for (int c = 1; c < static_cast<int>(CLASSNAME.size()); ++c) {
            // 获取第c个类别的分数
            float score = modelOutput[(4 + c) * candidateCount + i];
            // 筛选
            if (score > maxScore) {
                maxScore = score;
                classId = c;
            }
        }

        // 过滤置信度不足或无效的分数
        if (!std::isfinite(maxScore) || maxScore < CONF_THRESHOLD) {
            continue;
        }

        // ====== 组织初步检测结果 ======
        Detection det{};

        // 按照输出张量的通道布局读取候选框坐标
        det.cx = modelOutput[0 * candidateCount + i];
        det.cy = modelOutput[1 * candidateCount + i];
        det.w = modelOutput[2 * candidateCount + i];
        det.h = modelOutput[3 * candidateCount + i];

        // 防止异常坐标或非正宽高进入后处理
        if (!std::isfinite(det.cx) ||
            !std::isfinite(det.cy) ||
            !std::isfinite(det.w) ||
            !std::isfinite(det.h) ||
            det.w <= 0 ||
            det.h <= 0) {
            continue;
        }

        // 计算
        det.lt = cv::Point2f(det.cx - det.w / 2.0f, det.cy - det.h / 2.0f);
        det.rb = cv::Point2f(det.cx + det.w / 2.0f, det.cy + det.h / 2.0f);

        det.area = det.w * det.h;
        det.scores = maxScore;
        det.classId = classId;

        // 只保存通过初步置信度筛选的候选框
        candidates.push_back(det);
    }

    return candidates;
}


// ==================== IoU ====================

float calculateIoU(const Detection& a, const Detection& b) {
    // ====== 计算交集面积 ======
    // 如果两个框没有重叠，宽度或高度会被限制为 0
    float wI = std::max(0.0f, std::min(a.rb.x, b.rb.x) - std::max(a.lt.x, b.lt.x));
    float hI = std::max(0.0f, std::min(a.rb.y, b.rb.y) - std::max(a.lt.y, b.lt.y));
    float areaI = wI * hI;

    // ====== 并集面积 ======
    float areaU = a.area + b.area - areaI;

    // 防止非法面积导致除零
    if (areaU <= 0.0f) {
        return 0.0f;
    }

    return areaI / areaU;
}


// ==================== NMS ====================

std::vector<Detection> NMS(std::vector<Detection> candidates) {
    // ====== 按置信度从高到低排序 ======
    std::sort(candidates.begin(), candidates.end(), 
    [](const Detection& a, const Detection& b) {
            return a.scores > b.scores;
        });

    // 保存最终保留的检测框
    std::vector<Detection> kept;
    kept.reserve(candidates.size());

    // ====== 依次处理候选框 ======
    for (const auto& candidate : candidates) {
        bool shouldKeep = true;

        // 只与已经保留的框比较
        for (const auto& selected : kept) {

            // 不同类别之间不互相抑制
            if (candidate.classId != selected.classId) {
                continue;
            }

            // 计算当前候选框与已保留框的重叠程度
            float iou = calculateIoU(candidate, selected);

            // 同类检测框重叠过高，则抑制当前候选框
            if (iou > NMS_THRESHOLD) {
                shouldKeep = false;
                break;
            }
        }

        // 没有被抑制，加入最终结果
        if (shouldKeep) {
            kept.push_back(candidate);
        }
    }

    return kept;
}


// ==================== 坐标还原与绘图 ====================

void drawDetections(cv::Mat& img, const std::vector<Detection>& detections, const LetterboxInfo& info) {
    // 遍历每一个检测结果
    for (const auto& det : detections) {
        // ====== 将模型输入画布上的坐标还原到原始图像 ======
        // 先减去填充偏移，再除以缩放比例
        float x1 = (det.lt.x - info.padX) / info.scale;
        float y1 = (det.lt.y - info.padY) / info.scale;

        float x2 = (det.rb.x - info.padX) / info.scale;
        float y2 = (det.rb.y - info.padY) / info.scale;

        // 限制坐标范围，防止检测框超出图像边界
        x1 = std::clamp(x1, 0.0f, static_cast<float>(img.cols - 1));
        y1 = std::clamp(y1, 0.0f, static_cast<float>(img.rows - 1));
        
        x2 = std::clamp(x2, 0.0f, static_cast<float>(img.cols - 1));
        y2 = std::clamp(y2, 0.0f, static_cast<float>(img.rows - 1));

        // 跳过无有效面积的框
        if (x2 <= x1 || y2 <= y1) {
            continue;
        }

        // ====== 绘制框与文字 ======
        // OpenCV 绘图使用整数像素坐标
        cv::Point p1(cvRound(x1), cvRound(y1));
        cv::Point p2(cvRound(x2), cvRound(y2));

        // 绘制绿色矩形框
        cv::rectangle(
            img, p1, p2,
            cv::Scalar(0, 255, 0),
            2
        );

        // 拼接类别名称与置信度
        std::string label = CLASSNAME[det.classId] + " " + cv::format("%.2f", det.scores);

        // 将文字放在检测框上方，避免超出图片顶部
        cv::putText(
            img,
            label,
            cv::Point(p1.x, std::max(20, p1.y - 8)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(255, 255, 255),
            2
        );
    }
}


int main()
{
    // ====== 1. TensorRT 初始化 ======
    Logger logger;

    // 创建runtime
    std::unique_ptr<IRuntime> runtime(createInferRuntime(logger)); 
    if (!runtime) {
        std::cerr << "创建 Runtime 失败！\n";
        return 1;
    }

    // 创建engine
    auto engine = loadEngine(runtime.get(), ENGINE_PATH);
    if (!engine) {
        return 1;
    }

    // 创建执行上下文
    std::unique_ptr<IExecutionContext> context(
        engine->createExecutionContext()
    );
    if (!context) {
        std::cerr << "创建 ExecutionContext 失败！\n";
        return 1;
    }

    // ====== 2. 获取 Tensor 信息 ======
    auto inputInfo = getTensorInfo(
        engine.get(),
        TensorIOMode::kINPUT
    );

    auto outputInfo = getTensorInfo(
        engine.get(),
        TensorIOMode::kOUTPUT
    );

    if (!inputInfo || !outputInfo) {
        std::cerr << "获取 Tensor 信息失败！\n";
        return 1;
    }

    // 当前预处理要求输入形状为 (1, 3, 640, 640)
    const Dims& inShape = inputInfo->shape;

    if (inShape.nbDims != 4 ||
        inShape.d[0] != 1 ||
        inShape.d[1] != 3 ||
        inShape.d[2] != INPUT_H ||
        inShape.d[3] != INPUT_W) {
        std::cerr << "输入 Tensor 形状不符合要求！\n";
        return 1;
    }

    // 当前后处理要求输出形状为 (1, 7, N)
    const Dims& outShape = outputInfo->shape;

    if (outShape.nbDims != 3 ||
        outShape.d[0] != 1 ||
        outShape.d[1] != 4 + static_cast<int>(CLASSNAME.size()) ||
        outShape.d[2] <= 0) {
        std::cerr << "输出 Tensor 形状不符合要求！\n";
        return 1;
    }

    // 获取候选框数量，例如当前模型为 8400
    const int candidateCount = outShape.d[2];

    // ====== 3. 分配输入、输出 GPU 内存 ======
    // 输入
    void* rawInput = nullptr;
    auto status = cudaMalloc(&rawInput, inputInfo->bytes);
    if (status != cudaSuccess) {
        std::cerr << "输入 GPU 内存分配失败："
                  << cudaGetErrorString(status) << '\n';
        return 1;
    }
    CudaPtr deviceInput(rawInput);

    // 输出
    void* rawOutput = nullptr;
    status = cudaMalloc(&rawOutput, outputInfo->bytes);
    if (status != cudaSuccess) {
        std::cerr << "输出 GPU 内存分配失败："
                  << cudaGetErrorString(status) << '\n';
        return 1;
    }
    CudaPtr deviceOutput(rawOutput);

    // ====== 4 将 GPU 地址绑定到 Tensor ======
    // 输入
    if (!context->setTensorAddress(
            inputInfo->name.c_str(),
            deviceInput.get())) {
        std::cerr << "绑定输入 Tensor 失败！\n";
        return 1;
    }

    // 输出
    if (!context->setTensorAddress(
            outputInfo->name.c_str(),
            deviceOutput.get())) {
        std::cerr << "绑定输出 Tensor 失败！\n";
        return 1;
    }

    // ====== 5 创建 CUDA Stream ======
    CudaStreamGuard streamGuard;

    status = cudaStreamCreate(&streamGuard.stream);
    if (status != cudaSuccess) {
        std::cerr << "创建 CUDA Stream 失败："
                  << cudaGetErrorString(status) << '\n';
        return 1;
    }

    // ====== 6 创建 CPU 端输入输出容器 ======
    std::vector<float> inputHost;

    // 预留输入空间，减少后续重复扩容
    inputHost.reserve(inputInfo->bytes / sizeof(float));

    // 输出容器大小固定，每帧覆盖其中的数据
    std::vector<float> outputHost(outputInfo->bytes / sizeof(float));

    // ====== 7. 打开摄像头 ======
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "无法打开摄像头！\n";
        return 1;
    }
    std::cout << "摄像头检测已启动，按 q 退出。\n";

    cv::Mat frame;

    // ====== 8. 实时推理循环 ======
    while (true) {
        // 读取当前帧
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "读取摄像头画面失败！\n";
            break;
        }

        // 图像预处理
        LetterboxInfo letterboxInfo;
        if (!preprocess(frame, inputHost, letterboxInfo)) {
            std::cerr << "图像预处理失败！\n";
            break;
        }

        // 检查 CPU 输入数据大小是否与模型输入匹配
        if (inputHost.size() * sizeof(float) !=
            inputInfo->bytes) {
            std::cerr << "输入数据大小不匹配！\n";
            break;
        }

        // 将输入数据从 CPU 复制到 GPU
        status = cudaMemcpyAsync(
            deviceInput.get(),
            inputHost.data(),
            inputInfo->bytes,
            cudaMemcpyHostToDevice,
            streamGuard.stream
        );
        if (status != cudaSuccess) {
            std::cerr << "输入数据复制失败："
                      << cudaGetErrorString(status) << '\n';
            break;
        }

        // 提交 TensorRT 推理任务
        if (!context->enqueueV3(streamGuard.stream)) {
            std::cerr << "TensorRT 推理提交失败！\n";
            break;
        }

        // 将输出数据从 GPU 复制回 CPU
        status = cudaMemcpyAsync(
            outputHost.data(),
            deviceOutput.get(),
            outputInfo->bytes,
            cudaMemcpyDeviceToHost,
            streamGuard.stream
        );

        if (status != cudaSuccess) {
            std::cerr << "输出数据复制失败：" << cudaGetErrorString(status) << '\n';
            break;
        }

        // 等待输入复制、推理、输出复制全部完成
        status = cudaStreamSynchronize(streamGuard.stream);
        if (status != cudaSuccess) {
            std::cerr << "CUDA Stream 同步失败："
                      << cudaGetErrorString(status) << '\n';
            break;
        }

        // 解析模型输出，并进行初步置信度筛选
        auto candidates = parseCandidates(
            outputHost,
            candidateCount
        );

        // 执行 NMS，过滤同一类别中高度重叠的检测框
        auto finalDetections = NMS(std::move(candidates));

        // 坐标还原，并在当前原始图像上绘制检测框
        drawDetections(
            frame,
            finalDetections,
            letterboxInfo
        );

        // 显示实时检测结果
        cv::imshow("YOLO TensorRT Camera", frame);

        // 按 q 退出实时检测
        if ((cv::waitKey(1) & 0xFF) == 'q') {
            break;
        }
    }

    // ====== 9. 释放资源 ======

    // 确保已经提交的 GPU 任务完成
    cudaStreamSynchronize(streamGuard.stream);

    // 释放摄像头并关闭显示窗口
    cap.release();
    cv::destroyAllWindows();

    std::cout << "摄像头检测已结束。\n";

    return 0;
}
