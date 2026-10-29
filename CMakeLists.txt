cmake_minimum_required(VERSION 3.22)

project(tensorrt_cpp_demo LANGUAGES CXX)

# 使用 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 查找 CUDA Toolkit
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS
    core
    imgcodecs
    imgproc
    videoio
    highgui
)

# 查找 TensorRT 头文件和运行库
find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS /usr/include/x86_64-linux-gnu /usr/include
    REQUIRED
)

find_library(TENSORRT_LIBRARY
    NAMES nvinfer
    REQUIRED
)

# 创建可执行程序
add_executable(tensorrt_cpp_demo src/main.cpp)

# 配置头文件搜索路径
target_include_directories(tensorrt_cpp_demo PRIVATE
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# 链接 TensorRT 和 CUDA Runtime
target_link_libraries(tensorrt_cpp_demo PRIVATE
    ${TENSORRT_LIBRARY}
    CUDA::cudart
    ${OpenCV_LIBS}
)

# ====== TensorRT 摄像头实时检测 ======

add_executable(camera src/camera.cpp)

target_include_directories(camera PRIVATE
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(camera PRIVATE
    ${TENSORRT_LIBRARY}
    CUDA::cudart
    ${OpenCV_LIBS}
)