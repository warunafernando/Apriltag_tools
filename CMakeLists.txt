cmake_minimum_required(VERSION 3.10)
project(AprilTagDebugTools)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find Qt5
find_package(Qt5 REQUIRED COMPONENTS Core Widgets)

# Find AprilTag libraries
# Use pkg-config if available (like the main project)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(APRILTAG apriltag)
endif()

# If pkg-config didn't work, try manual find
if(NOT APRILTAG_FOUND)
    find_path(APRILTAG_INCLUDE_DIR apriltag/apriltag.h
        PATHS
        ${CMAKE_SOURCE_DIR}/../src/apriltag_cgpadwick
        /usr/local/include
        /usr/include
    )

    find_library(APRILTAG_LIBRARY
        NAMES apriltag
        PATHS
        ${CMAKE_SOURCE_DIR}/../src/apriltag_cgpadwick
        ${CMAKE_SOURCE_DIR}/../src/apriltags_cuda/build
        /usr/local/lib
        /usr/lib
        /usr/lib/x86_64-linux-gnu
    )
endif()

# Set Qt5 to use MOC automatically
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Try to find CUDA
find_package(CUDA QUIET)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../src/apriltags_cuda/tools
    ${CMAKE_SOURCE_DIR}/../src/apriltags_cuda/src
    /home/nav/HuaTengLinuxSDK_V2.1.0.49_250630/linuxSDK_V2.1.0.49_250630/include
    /usr/local/include
    /usr/local/include/apriltag
    /usr/local/include/apriltag/common
)

# Include CUDA directories if available
if(CUDA_FOUND)
    include_directories(
        ${CUDA_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/../src/apriltags_cuda/build/cccl/libcudacxx/include
        ${CMAKE_SOURCE_DIR}/../src/apriltags_cuda/build/cccl/thrust
        /usr/local/include
        /usr/local/include/apriltag
        /usr/local/include/apriltag/common
    )
endif()

if(APRILTAG_FOUND)
    include_directories(${APRILTAG_INCLUDE_DIRS})
else()
    include_directories(
        ${APRILTAG_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/../src/apriltag_cgpadwick
    )
endif()

# Algorithm classes (compile as CUDA if available)
if(APRILTAG_CUDA_LIBRARY AND CUDA_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
    # Compile fast_apriltag_algorithm.cu as CUDA
    set_source_files_properties(fast_apriltag_algorithm.cu PROPERTIES
        LANGUAGE CUDA
        COMPILE_OPTIONS "-Xcudafe;--diag_suppress=611")
endif()

# AprilTag Debug GUI
# Compile as CUDA when available (Qt MOC will still work)
if(APRILTAG_CUDA_LIBRARY AND CUDA_FOUND)
    enable_language(CUDA)
    # Set CUDA architecture (same as apriltags_cuda project)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
    add_executable(apriltag_debug_gui 
        apriltag_debug_gui.cu
        apriltag_algorithm_factory.cu
        cpu_apriltag_algorithm.cpp
        fast_apriltag_algorithm.cu
    )
    set_source_files_properties(apriltag_algorithm_factory.cu PROPERTIES
        LANGUAGE CUDA
        COMPILE_OPTIONS "-Xcudafe;--diag_suppress=611")
    # Suppress nvcc warning 611 (same as video_visualize_fixed.cu)
    set_source_files_properties(apriltag_debug_gui.cu PROPERTIES
        COMPILE_OPTIONS "-Xcudafe;--diag_suppress=611")
else()
    add_executable(apriltag_debug_gui 
        apriltag_debug_gui.cu
        apriltag_algorithm_factory.cu
        cpu_apriltag_algorithm.cpp
    )
endif()

# Try to find MindVision SDK library
find_library(MVCAMERA_LIBRARY
    NAMES MVSDK CameraApi libCameraApi MVCamera
    PATHS
    /home/nav/HuaTengLinuxSDK_V2.1.0.49_250630/linuxSDK_V2.1.0.49_250630/lib/x64
    /home/nav/HuaTengLinuxSDK_V2.1.0.49_250630/linuxSDK_V2.1.0.49_250630/lib
    /usr/lib
    /usr/local/lib
    /usr/lib/x86_64-linux-gnu
)

if(APRILTAG_FOUND)
    target_link_libraries(apriltag_debug_gui
        ${OpenCV_LIBS}
        ${APRILTAG_LIBRARIES}
        Qt5::Core
        Qt5::Widgets
    )
else()
    target_link_libraries(apriltag_debug_gui
        ${OpenCV_LIBS}
        ${APRILTAG_LIBRARY}
        Qt5::Core
        Qt5::Widgets
    )
endif()

# Link MindVision SDK if found (optional)
if(MVCAMERA_LIBRARY)
    message(STATUS "Found MindVision SDK: ${MVCAMERA_LIBRARY}")
    target_link_libraries(apriltag_debug_gui ${MVCAMERA_LIBRARY})
    target_compile_definitions(apriltag_debug_gui PRIVATE HAVE_MINDVISION_SDK)
else()
    message(STATUS "MindVision SDK library not found. MindVision camera support will be disabled.")
endif()

# Try to find and link CUDA AprilTag library
find_library(APRILTAG_CUDA_LIBRARY
    NAMES apriltag_cuda
    PATHS
    ${CMAKE_SOURCE_DIR}/../src/apriltags_cuda/build
    /usr/local/lib
    /usr/lib
)

if(APRILTAG_CUDA_LIBRARY AND CUDA_FOUND)
    message(STATUS "Found CUDA AprilTag library: ${APRILTAG_CUDA_LIBRARY}")
    
    # Find glog - same as video_visualize_fixed
    find_package(glog QUIET)
    
    # Link CUDA libraries with glog AFTER (dependency order)
    # Note: CUDA Driver API (libcuda.so) is needed for primary context management
    find_library(CUDA_DRIVER_LIBRARY cuda PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/cuda/lib64)
    target_link_libraries(apriltag_debug_gui 
        ${APRILTAG_CUDA_LIBRARY}
        ${CUDA_LIBRARIES}
        ${CUDA_CUDART_LIBRARY}
    )
    # Link CUDA Driver API if found (for primary context management)
    if(CUDA_DRIVER_LIBRARY)
        target_link_libraries(apriltag_debug_gui ${CUDA_DRIVER_LIBRARY})
    endif()
    
    # Link glog AFTER CUDA library (required by libapriltag_cuda.a)
    if(glog_FOUND)
        target_link_libraries(apriltag_debug_gui glog::glog)
    else()
        find_library(GLOG_LIBRARY glog PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib)
        if(GLOG_LIBRARY)
            target_link_libraries(apriltag_debug_gui ${GLOG_LIBRARY})
        endif()
    endif()
    
    target_compile_definitions(apriltag_debug_gui PRIVATE HAVE_CUDA_APRILTAG)
    
    message(STATUS "CUDA AprilTag support enabled")
else()
    message(STATUS "CUDA AprilTag library not found. CUDA algorithm will be disabled.")
    if(NOT CUDA_FOUND)
        message(STATUS "CUDA package not found")
    endif()
    if(NOT APRILTAG_CUDA_LIBRARY)
        message(STATUS "apriltag_cuda library not found")
    endif()
endif()

# Set output directory
set_target_properties(apriltag_debug_gui PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

