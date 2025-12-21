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

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/../src/apriltags_cuda/tools
    /home/nav/HuaTengLinuxSDK_V2.1.0.49_250630/linuxSDK_V2.1.0.49_250630/include
)

if(APRILTAG_FOUND)
    include_directories(${APRILTAG_INCLUDE_DIRS})
else()
    include_directories(
        ${APRILTAG_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/../src/apriltag_cgpadwick
    )
endif()

# AprilTag Debug GUI
add_executable(apriltag_debug_gui apriltag_debug_gui.cpp)

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

# Set output directory
set_target_properties(apriltag_debug_gui PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

