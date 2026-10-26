#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeBase.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/cvdef.h>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <ostream>
#include <string>
#include <vector>
#include <cuda_runtime_api.h>
#include <optional>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace nvinfer1;

const std::array<std::string, 3> CLASSNAME {"earbuds", "toothpick", "dice"};
constexpr float CONF_THRESHOLD = 0.25f;
constexpr float NMS_THRESHOLD = 0.7f;
const std::string IMG_PATH = "";
const std::string OUTPUT_PATH = "";
const std::string MODEL_PATH = "";

class Logger : public ILogger
{
public:
    // 重写log函数
    void log(Severity severity, char const* msg) noexcept override
    {
        // 仅输出报错相关
        if (severity == Severity::kINTERNAL_ERROR || severity == Severity::kERROR || severity == Severity::kWARNING)
        {
            std::cout << msg << std::endl;
        }

        return;
    }
};

// 检测结果结构体
struct Detection 
{
    float cx;
    float cy;
    float w;
    float h;
    cv::Point2f lt;
    cv::Point2f rb;
    float area;
    float scores;
    int classId;
};

// Cuda删除器
struct CudaDeleter
{
    void operator()(void* ptr) const noexcept
    {
        if (ptr)
        {
            auto status = cudaFree(ptr);
            if (status != cudaSuccess)
            {
                std::cerr << "释放 GPU 内存失败："
                          << cudaGetErrorString(status) << '\n';
            }
        }
    }
};

// 输入信息
struct TensorInfo
{
    std::string name;
    std::size_t bytes;
};

using CudaPtr = std::unique_ptr<void, CudaDeleter>;

// 查看模型I/O Tensor的信息
void checkModelIO(ICudaEngine* engine)
{
    if (engine == nullptr)
    {
        std::cerr << "Engine is nullptr!" << '\n';
        return;
    }

    std::cout << "====== 模型IO信息 ======" << std::endl;
    std::cout << "Engine Tensor 总数: " << engine->getNbIOTensors() << std::endl;

    // 遍历每个Tensor
    for (int32_t i  = 0; i < engine->getNbIOTensors(); i++)
    {
        int64_t total = 1;
        int64_t total_byte = 0;

        // 获取名称
        auto name(engine->getIOTensorName(i));
        std::cout << "Tensor [" << i << "] name: " << name << std::endl;

        // 获取I/O模式
        auto mode = engine->getTensorIOMode(name);
        if (mode == TensorIOMode::kNONE) 
        {
            std::cout << "Tensor [" << i << "] I/O mode: None" << std::endl;
        }
        else if (mode == TensorIOMode::kINPUT) {
            std::cout << "Tensor [" << i << "] I/O mode: Input" << std::endl;
        }
        else if (mode == TensorIOMode::kOUTPUT) {
            std::cout << "Tensor [" << i << "] I/O mode: Output" << std::endl;
        }
        else {
            std::cerr << "Tensor [" << i << "] I/O mode: Unknow" << std::endl;
        }

        // 获取shape
        auto shape = engine->getTensorShape(name);
        std::cout << "Tensor [" << i << "] 总维度数量：" << shape.nbDims << std::endl;
        std::cout << "Tensor [" << i << "] 形状：" << std::endl;
        for (int32_t j = 0; j < shape.nbDims; j++)
        {
            if (shape.d[j] < 0)
            {
                std::cout << "存在动态维度，暂时无法计算实际大小\n";
            }

            std::cout << "d[" << j << "]: " << shape.d[j] << std::endl;
            total *= shape.d[j];
        }
        std::cout << "Tensor [" << i << "] 元素总数: " << total << std::endl;
        
        // 查询数据类型
        auto type = engine->getTensorDataType(name);
        if (type == DataType::kFLOAT) 
        {
            total_byte = total * 4;
            std::cout << "Tensor [" << i << "]  数据类型: FP32\n"
                      << "Tensor [" << i << "]  数据总大小: " << total_byte << "字节" << std::endl;
        }
        else if (type == DataType::kHALF) {
            total_byte = total * 2;
            std::cout << "Tensor [" << i << "]  数据类型: FP16\n"
                      << "Tensor [" << i << "]  数据总大小: " << total * 2 << "字节" << std::endl;

        }
        else if (type == DataType::kINT8) {
            total_byte = total * 1;
            std::cout << "Tensor [" << i << "]  数据类型: INT8\n"
                      << "Tensor [" << i << "]  数据总大小: " << total * 1 << "字节" << std::endl;

        }
        else {
            std::cout << "Tensor [" << i << "] 数据类型: others" << std::endl;
        }

        std::cout << "\n====\n" << std::endl;
    }
}

// 加载并创建engine
std::unique_ptr<ICudaEngine> loadEngine(IRuntime* runtime)
{
    // ====== 读取Engine文件 ======
    const std::string enginePath = MODEL_PATH;

    // 二进制读取
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);

    if (!file) {
        std::cerr << "Failed to open engine file\n";
        return nullptr;
    }

    // 确保文件里真有东西
    auto fileSize = file.tellg();
    if (fileSize <= 0) {
        std::cerr << "Engine file is empty or unreadable\n";
        return nullptr;
    }

    std::vector<char> engineData(
        static_cast<std::size_t>(fileSize)
    );

    file.seekg(0);

    if (!file.read(engineData.data(), fileSize)) {
        std::cerr << "Failed to read engine file\n";
        return nullptr;
    }

    std::unique_ptr<ICudaEngine> engine(
    runtime->deserializeCudaEngine(
        engineData.data(),
        engineData.size()
        )
    );

    if (!engine)
    {
        std::cerr << "Loading engine failed\n";
        return nullptr;
    }

    std::cout << "Engine loaded successfully!" << std::endl;
    return engine;
}

// 获取输入Tensor信息
std::optional<TensorInfo> getInputInfo(ICudaEngine* engine)
{
    TensorInfo inputInfo{};

    // 计算输入Tensor字节数
    for (int32_t i  = 0; i < engine->getNbIOTensors(); i++)
    {
        int64_t totalElements = 1;

        // 获取名称
        auto name(engine->getIOTensorName(i));
        
        // 获取I/O模式
        auto mode = engine->getTensorIOMode(name);
        if (mode == TensorIOMode::kINPUT) 
        {
            inputInfo.name = name;
            // 获取shape
            auto shape = engine->getTensorShape(name);

            for (int32_t j = 0; j < shape.nbDims; j++)
            {
                totalElements *= shape.d[j];
            }
            
            // 查询数据类型
            auto type = engine->getTensorDataType(name);
            if (type == DataType::kFLOAT) 
            {
                inputInfo.bytes = static_cast<std::size_t>(totalElements * sizeof(float));
                return inputInfo;
            }
            else if (type == DataType::kHALF) {
                inputInfo.bytes = static_cast<std::size_t>(totalElements * 2);
                return inputInfo;
            }
            else if (type == DataType::kINT8) {
                inputInfo.bytes = static_cast<std::size_t>(totalElements * 1);
                return inputInfo;
            }
            else {
                std::cerr << "不支持的输入 Tensor 数据类型！" << '\n';
                return std::nullopt;
            }
        }
        else {
            continue;
        }
    }

    std::cerr << "不存在输入Tensor！" << std::endl;
    return std::nullopt;
}

// 获取输出Tensor信息
std::optional<TensorInfo> getOutputInfo(ICudaEngine* engine)
{
    TensorInfo outputInfo{};

    // 计算输出 Tensor 字节数
    for (int32_t i  = 0; i < engine->getNbIOTensors(); i++)
    {
        int64_t totalElements = 1;

        // 获取名称
        auto name(engine->getIOTensorName(i));
        
        // 获取I/O模式
        auto mode = engine->getTensorIOMode(name);
        if (mode == TensorIOMode::kOUTPUT) 
        {
            outputInfo.name = name;
            // 获取shape
            auto shape = engine->getTensorShape(name);

            for (int32_t j = 0; j < shape.nbDims; j++)
            {
                totalElements *= shape.d[j];
            }
            
            // 查询数据类型
            auto type = engine->getTensorDataType(name);
            if (type == DataType::kFLOAT) 
            {
                outputInfo.bytes = static_cast<std::size_t>(totalElements * sizeof(float));
                return outputInfo;
            }
            else if (type == DataType::kHALF) {
                outputInfo.bytes = static_cast<std::size_t>(totalElements * 2);
                return outputInfo;
            }
            else if (type == DataType::kINT8) {
                outputInfo.bytes = static_cast<std::size_t>(totalElements * 1);
                return outputInfo;
            }
            else {
                std::cerr << "不支持的输出 Tensor 数据类型！" << '\n';
                return std::nullopt;
            }
        }
        else {
            continue;
        }
    }

    std::cerr << "不存在输出Tensor！" << std::endl;
    return std::nullopt;
}

// 输出图像信息
void printImageInfo(const cv::Mat& img) 
{
    std::cout << "====== 图像信息 ======\n"
              << "图像宽度：" << img.cols << "\n"
              << "图像高度：" << img.rows << "\n"
              << "通道数：" << img.channels() << "\n" 
              << "数据深度编码：" << img.depth() << "\n";

    if (img.type() == CV_8UC3)
    {
        std::cout << "类型名称：CV_8UC3\n";
    }
    else if (img.type() == CV_32FC3)
    {
        std::cout << "类型名称：CV_32FC3\n";
    }
    else if (img.type() == CV_32FC1)
    {
        std::cout << "类型名称：CV_32FC1\n";
    }
    else 
    {
        std::cout << "类型名称：Others\n";
    }
}

// 等比缩放
cv::Mat letterbox(cv::Mat& img)
{
    cv::Mat resized;
    // 求缩放比例
    float rate = std::min(640.0 / img.cols, 640.0 / img.rows);

    // 缩放
    cv::resize(img, resized, cv::Size(rate  * img.cols, rate * img.rows));

    // 填充
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, (640 - resized.rows) / 2, (640 - resized.rows) / 2, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    return padded;
}

// HWC转NCHW(1, 3, 640, 640)
std::optional<std::vector<float>> HWC2NCHW(const cv::Mat& img) 
{
    // 分类色彩通道
    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    if (channels.size() != 3) {
        std::cerr << "输入图像 通道数 不符合规范！\n";
        return std::nullopt; 
    }

    // 查看一下信息
    for (auto& mat : channels) {
        printImageInfo(mat);
    }

    // 把三个通道的矩阵填入nchw里
    std::vector<float> nchwData;
    nchwData.reserve(640 * 640 * 3);
    for (auto& channel : channels) {
        for (int r = 0; r < channel.rows; r++) {
            nchwData.insert(nchwData.end(), channel.ptr<float>(r), channel.ptr<float>(r) + channel.cols);
        }
    }

    std::cout << "\nNCHW数据量：" << nchwData.size() << "\n";

    return nchwData;
}

// 打印推理结果
void printModelOutput(const std::vector<float>& modelOutput, const int i) 
{
    std::cout << "====== 模型输出信息 ======\n";
    std::cout << "候选框[" << i << "] 中心点cx：" << modelOutput[0 * 8400 + i] << "\n"
              << "候选框[" << i << "] 中心点cy：" << modelOutput[1 * 8400 + i] << "\n"
              << "候选框[" << i << "] 宽度w：" << modelOutput[2 * 8400 + i] << "\n"
              << "候选框[" << i << "] 高度h：" << modelOutput[3 * 8400 + i] << "\n"
              << "候选框[" << i << "] 类别 0 分数：" << modelOutput[4 * 8400 + i] << "\n"
              << "候选框[" << i << "] 类别 1 分数：" << modelOutput[5 * 8400 + i] << "\n"
              << "候选框[" << i << "] 类别 2 分数：" << modelOutput[6 * 8400 + i] << "\n";
              
    float maxScore = std::max(modelOutput[4 * 8400 + i], std::max(modelOutput[5 * 8400 + i], modelOutput[6 * 8400 + i]));
    std::cout << "候选框[" << i << "] 最高分：" << maxScore << "\n";
    if (maxScore < CONF_THRESHOLD) {
        std::cout << "候选框[" << i << "] 置信度过低，将被过滤\n";
        return;
    }
    else if (maxScore == modelOutput[4 * 8400 + i]) {
        std::cout << "候选框[" << i << "] 最可能类别：" << CLASSNAME[0] << "\n";
    }
    else if (maxScore == modelOutput[5 * 8400 + i]) {
        std::cout << "候选框[" << i << "] 最可能类别：" << CLASSNAME[1] << "\n";
    }
    else {
        std::cout << "候选框[" << i << "] 最可能类别：" << CLASSNAME[2] << "\n";
    }
}

// 获取推理结果
std::optional<Detection> parseCandidate(const std::vector<float>& modelOutput, int i)
{
    Detection res{};

    res.scores = std::max(modelOutput[4 * 8400 + i], std::max(modelOutput[5 * 8400 + i], modelOutput[6 * 8400 + i]));

    if (res.scores < CONF_THRESHOLD) {
        return std::nullopt;
    }
    else if (res.scores == modelOutput[4 * 8400 + i]) {
        res.classId = 0;
    }
    else if (res.scores == modelOutput[5 * 8400 + i]) {
        res.classId = 1;
    }
    else {
        res.classId = 2;
    }

    res.cx = modelOutput[0 * 8400 + i];
    res.cy = modelOutput[1 * 8400 + i];
    res.w = modelOutput[2 * 8400 + i];
    res.h = modelOutput[3 * 8400 + i];
    res.lt.x = res.cx - res.w / 2;
    res.rb.x = res.cx + res.w / 2;
    res.lt.y = res.cy - res.h / 2;
    res.rb.y = res.cy + res.h / 2;

    res.area = res.w * res.h;

    return res;
}

// 计算 IoU
float calculateIoU(const Detection& a, const Detection& b) 
{
    // 交集
    float wI = std::max(0.0f, std::min(a.rb.x, b.rb.x) - std::max(a.lt.x, b.lt.x));
    float hI = std::max(0.0f, std::min(a.rb.y, b.rb.y) - std::max(a.lt.y, b.lt.y));
    float areaI = wI * hI;

    // 并集
    float areaU = a.area + b.area - areaI;
    if (areaU == 0.0f) {
        std::cerr << "两框并集面积为0\n";
        return -1.0f;
    }

    return areaI / areaU;
}

// NMS筛选
std::vector<Detection> NMS(std::vector<Detection> candidates)
{
    // 按分数从大到小排序
    std::sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) {
       return a.scores > b.scores;
    });
    // for (const auto& candidate : candidates) {
    //     std::cout << candidate.scores << "\n";
    // }

    std::vector<Detection> kept;
    kept.reserve(candidates.size());

    for (const auto& candidate : candidates) {
        bool shouldKeep = true;

        for (const auto& selected : kept) {
            // 不同类别之间不互相抑制
            if (candidate.classId != selected.classId) {
                continue;
            }

            // 计算当前候选框与已保留框的 IoU
            float iou = calculateIoU(candidate, selected);

            if (iou > NMS_THRESHOLD) {
                shouldKeep = false;
                break;
            }
        }

        // 没有被抑制，保留当前候选框
        if (shouldKeep) {
            kept.push_back(candidate);
        }
    }

    return kept;
}

// 绘制检测框和类别名称
void drawRect(cv::Mat& img, const cv::Point2f& lt, const cv::Point2f& rb, int classId, float score)
{
    // 将坐标转换为整数像素坐标
    cv::Point p1(
        static_cast<int>(lt.x),
        static_cast<int>(lt.y)
    );

    cv::Point p2(
        static_cast<int>(rb.x),
        static_cast<int>(rb.y)
    );

    // 绘制绿色检测框
    cv::rectangle(img, p1, p2, cv::Scalar(0, 255, 0), 3);

    // 构造类别名称和置信度文本
    std::string label = CLASSNAME[classId] + " " + cv::format("%.2f", score);

    // 防止文字绘制位置超出图片顶部
    int textY = std::max(20, p1.y - 8);

    // 绘制白色文字
    cv::putText(
        img,
        label,
        cv::Point(p1.x, textY),
        cv::FONT_HERSHEY_SIMPLEX,
        0.7,
        cv::Scalar(255, 255, 255),
        2
    );
}

// 坐标还原
void idk(cv::Mat& img, std::vector<Detection>& detections) 
{
    const float scale = std::min(
        640.0f / img.cols,
        640.0f / img.rows
    );

    const float padX = (640.0f - img.cols * scale) / 2.0f;  // 左右各自填充像素数
    const float padY = (640.0f - img.rows * scale) / 2.0f;  // 上下各自填充像素数

    for (const auto& det : detections)
    {
        // 去掉填充
        float x1 = (det.lt.x - padX) / scale;
        float y1 = (det.lt.y - padY) / scale;
        float x2 = (det.rb.x - padX) / scale;
        float y2 = (det.rb.y - padY) / scale;

        // 限制在原图范围内
        x1 = std::clamp(x1, 0.0f, static_cast<float>(img.cols - 1));
        y1 = std::clamp(y1, 0.0f, static_cast<float>(img.rows - 1));
        x2 = std::clamp(x2, 0.0f, static_cast<float>(img.cols - 1));
        y2 = std::clamp(y2, 0.0f, static_cast<float>(img.rows - 1));

        std::cout << "\n类别：" << CLASSNAME[det.classId]
                << "，置信度：" << det.scores << '\n'
                << "左上角：(" << x1 << ", " << y1 << ")\n"
                << "右下角：(" << x2 << ", " << y2 << ")\n";

        drawRect(img, cv::Point2f(x1, y1), cv::Point2f(x2, y2), det.classId, det.scores);
    }
}

int main() 
{
    // ====== 1. 创建Logger对象 ======
    Logger logger;

    // 创建TensorRT Runtime，接收logger
    std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));
    if (!runtime) {
        std::cerr << "Creating runtime failed\n";
        return 1;
    }

    // ====== 2. 创建Engine对象 ======
    auto engine = loadEngine(runtime.get());
    if (!engine) {
        std::cerr << "Loading engine failed\n";
        return 1;
    }

    // 查看I/O Tensor信息
    checkModelIO(engine.get());

    // ====== 3. 创建上下文Context对象 ======
    std::unique_ptr<IExecutionContext> context(engine->createExecutionContext());
    if (!context) {
        std::cerr << "Creating context failed\n";
        return 1;
    }
    std::cout << "Context created successfully!" << std::endl;
    
    // ====== 4. 分配绑定 GPU 内存给输入Tensor ======
    auto inputInfo = getInputInfo(engine.get());
    if (!inputInfo.has_value()) {
        return 1;
    }
    
    // 注册一个空指针
    void* rawInput = nullptr;
    auto inputStatus = cudaMalloc(&rawInput, inputInfo->bytes);    // 注册 GPU 内存
    if (inputStatus != cudaSuccess) {
        std::cerr << cudaGetErrorString(inputStatus) << '\n';
        return 1;
    }
    std::cout << "成功为输入 Tensor 分配 GPU 内存！" << std::endl;

    CudaPtr deviceInput(rawInput);  // 自动管理释放

    // 绑定内存到 Tensor
    if (!context->setTensorAddress(inputInfo->name.c_str(), deviceInput.get())) {
        std::cerr << "内存绑定失败！" << std::endl;
        return 1;
    }

    // ====== 5. 分配绑定 GPU 内存给输出Tensor ======
    // 获取输出 Tensor 信息
    auto outputInfo = getOutputInfo(engine.get());
    if (!outputInfo.has_value()) {
        return 1;
    }

    // 分配内存
    void* rawOutput = nullptr;
    auto outputStatus = cudaMalloc(&rawOutput, outputInfo->bytes);    // 注册 GPU 内存
    if (outputStatus != cudaSuccess) {
        std::cerr << cudaGetErrorString(outputStatus) << '\n';
        return 1;
    }
    std::cout << "成功为输出 Tensor 分配 GPU 内存！" << std::endl;

    CudaPtr deviceOutput(rawOutput);  // 自动管理释放

    // 绑定内存到 Tensor
    if (!context->setTensorAddress(outputInfo->name.c_str(), deviceOutput.get())) {
        std::cerr << "内存绑定失败！" << std::endl;
        return 1;
    }

    // ====== 6. 读取图像 ======
    std::string imgPath = IMG_PATH;
    cv::Mat img = cv::imread(imgPath);
    if (img.empty()) {
        std::cerr << "图像读取失败！" << std::endl;
        return 1;
    }

    printImageInfo(img);  // 输出图像信息

    // ====== 7. 图像预处理 ======
    // 缩放
    auto padded = letterbox(img);
    printImageInfo(padded);

    // 颜色空间转换BGR -> RGB
    cv::Mat imgRGB;
    cv::cvtColor(padded, imgRGB, cv::COLOR_BGR2RGB);
    printImageInfo(imgRGB);

    // 像素归一化
    cv::Mat imgFloat;
    imgRGB.convertTo(imgFloat, CV_32F, 1.0 / 255.0);
    printImageInfo(imgFloat);

    // HWC转NCHW
    auto nchwData = HWC2NCHW(imgFloat);
    if (!nchwData.has_value()) {
        return 1;
    }
    if (nchwData->size() * sizeof(float) != inputInfo->bytes) {
        std::cerr << "nchw字节数错误！\n";
        return 1;
    }

    // ====== 8. 把数据从 CPU 复制到 GPU ======
    auto deviceCopyStatus =  cudaMemcpy(deviceInput.get(), nchwData->data(), inputInfo->bytes, cudaMemcpyHostToDevice);
    if (deviceCopyStatus != cudaSuccess) {
        std::cerr << cudaGetErrorString(deviceCopyStatus) << "\n"; 
        return 1;
    }
    std::cout << "图像数据已成功复制到 GPU！\n";

    // ====== 9. 执行模型推理 ======
    // 创建 CUDA Stream
    cudaStream_t stream = nullptr;
    auto streamStatus = cudaStreamCreate(&stream);
    if (streamStatus != cudaSuccess) {
        std::cerr << "创建 CUDA Stream 失败：" << cudaGetErrorString(streamStatus) << "\n";
        return 1;
    }

    // 将推理任务到 CUDA Stream
    if (!context->enqueueV3(stream)) {
        std::cerr << "提交 TensorRT 推理任务失败！\n";
        return 1;
    }
    std::cout << "TensorRT 推理任务已提交！\n";

    // 等待 GPU 上的任务完成
    auto syncStatus = cudaStreamSynchronize(stream);
    if (syncStatus != cudaSuccess) {
        std::cerr << cudaGetErrorString(syncStatus) << "\n";
        return 1;
    }
    std::cout << "TensorRT 推理完成！\n";

    // ====== 10. 将输出复制回 CPU ======
    // 创建接收容器
    std::vector<float> modelOutput(outputInfo->bytes / sizeof(float));
    
    // 复制回 CPU
    auto hostCopyStatus = cudaMemcpy(modelOutput.data(), deviceOutput.get(), outputInfo->bytes, cudaMemcpyDeviceToHost);
    if (hostCopyStatus != cudaSuccess) {
        std::cerr << cudaGetErrorString(hostCopyStatus) << "\n";
        return 1;
    }
    std::cout << "推理输出已成功复制回 CPU！\n";

    // printModelOutput(modelOutput, 0);

    // ====== 11. 后处理 ======
    // 初步置信度筛选
    std::vector<Detection> candidates;
    for (int i = 0; i < 8400; i++) {
        auto candidate = parseCandidate(modelOutput, i);
        if (!candidate.has_value()) {
            // std::cout << "候选框[" << i << "] 置信度过低，被过滤\n";
            continue;
        }

        candidates.push_back(*candidate);
    }
    std::cout << "剩余候选框数量：" << candidates.size() << "\n";

    // NMS筛选
    auto finalDetections = NMS(candidates);
    std::cout << "NMS 前候选框数量：" << candidates.size() << '\n';
    std::cout << "NMS 后候选框数量：" << finalDetections.size() << '\n';

    // 检测框坐标还原
    idk(img, finalDetections);

    std::string outputPath = OUTPUT_PATH;
    cv::imwrite(outputPath, img);

    return 0; 
}