#include "cpu_apriltag_algorithm.h"
#include "apriltag/tag36h11.h"
#include "apriltag/common/image_u8.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <sstream>
#include <iomanip>

using namespace cv;

CpuAprilTagAlgorithm::CpuAprilTagAlgorithm()
    : width_(0)
    , height_(0)
    , initialized_(false)
    , tf_(nullptr)
    , td_(nullptr)
    , frame_count_(0)
{
    // Create tag family and detector
    tf_ = tag36h11_create();
    td_ = apriltag_detector_create();
    if (td_ && tf_) {
        apriltag_detector_add_family(td_, tf_);
        td_->quad_decimate = 1.0;
        td_->quad_sigma = 0.0;
        td_->refine_edges = 1;
        td_->decode_sharpening = 0.25;
        td_->nthreads = 4;
        td_->wp = workerpool_create(4);
    }
}

CpuAprilTagAlgorithm::~CpuAprilTagAlgorithm() {
    cleanup();
}

bool CpuAprilTagAlgorithm::initialize(int width, int height) {
    if (width <= 0 || height <= 0) {
        return false;
    }
    
    width_ = width;
    height_ = height;
    initialized_ = true;
    
    // Reset timing stats
    resetTimingStats();
    
    return true;
}

zarray_t* CpuAprilTagAlgorithm::processFrame(const cv::Mat& gray_frame, bool mirror) {
    if (!initialized_ || !td_ || gray_frame.empty()) {
        return nullptr;
    }
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // Reset last frame timing
    TimingStats frame_timing;
    frame_timing.reset();
    
    // Step 1: Handle grayscale conversion (input should already be grayscale)
    auto convert_start = std::chrono::high_resolution_clock::now();
    cv::Mat gray = gray_frame;
    if (gray_frame.channels() == 3) {
        cv::cvtColor(gray_frame, gray, cv::COLOR_BGR2GRAY);
    }
    auto convert_end = std::chrono::high_resolution_clock::now();
    frame_timing.grayscale_convert_ms = 
        std::chrono::duration<double, std::milli>(convert_end - convert_start).count();
    
    // Step 2: Apply mirroring if requested
    auto mirror_start = std::chrono::high_resolution_clock::now();
    cv::Mat gray_for_detection;
    if (mirror) {
        cv::flip(gray, gray_for_detection, 1);  // 1 = horizontal flip
    } else {
        gray_for_detection = gray;
    }
    auto mirror_end = std::chrono::high_resolution_clock::now();
    frame_timing.mirror_ms = 
        std::chrono::duration<double, std::milli>(mirror_end - mirror_start).count();
    
    // Step 3: Run detection
    auto detect_start = std::chrono::high_resolution_clock::now();
    
    image_u8_t im = {
        .width = gray_for_detection.cols,
        .height = gray_for_detection.rows,
        .stride = gray_for_detection.cols,
        .buf = gray_for_detection.data
    };
    
    zarray_t* detections = apriltag_detector_detect(td_, &im);
    
    auto detect_end = std::chrono::high_resolution_clock::now();
    frame_timing.detection_ms = 
        std::chrono::duration<double, std::milli>(detect_end - detect_start).count();
    
    // Record number of detections
    if (detections) {
        frame_timing.num_detections = zarray_size(detections);
    } else {
        frame_timing.num_detections = 0;
    }
    
    // Update total time
    auto total_end = std::chrono::high_resolution_clock::now();
    frame_timing.total_ms = 
        std::chrono::duration<double, std::milli>(total_end - total_start).count();
    
    // Update shared timing data with lock (minimize lock time)
    {
        std::lock_guard<std::mutex> lock(timing_mutex_);
        last_frame_timing_ = frame_timing;
        accumulated_stats_.grayscale_convert_ms += frame_timing.grayscale_convert_ms;
        accumulated_stats_.mirror_ms += frame_timing.mirror_ms;
        accumulated_stats_.detection_ms += frame_timing.detection_ms;
        accumulated_stats_.total_ms += frame_timing.total_ms;
        frame_count_++;
    }
    
    return detections;
}

void CpuAprilTagAlgorithm::cleanup() {
    initialized_ = false;
    
    if (td_) {
        apriltag_detector_destroy(td_);
        td_ = nullptr;
    }
    
    if (tf_) {
        tag36h11_destroy(tf_);
        tf_ = nullptr;
    }
}

CpuAprilTagAlgorithm::TimingStats CpuAprilTagAlgorithm::getLastFrameTiming() const {
    std::lock_guard<std::mutex> lock(timing_mutex_);
    return last_frame_timing_;
}

CpuAprilTagAlgorithm::TimingStats CpuAprilTagAlgorithm::getAverageTiming() const {
    std::lock_guard<std::mutex> lock(timing_mutex_);
    TimingStats avg;
    
    if (frame_count_ > 0) {
        avg.grayscale_convert_ms = accumulated_stats_.grayscale_convert_ms / frame_count_;
        avg.mirror_ms = accumulated_stats_.mirror_ms / frame_count_;
        avg.detection_ms = accumulated_stats_.detection_ms / frame_count_;
        avg.total_ms = accumulated_stats_.total_ms / frame_count_;
    }
    
    return avg;
}

void CpuAprilTagAlgorithm::resetTimingStats() const {
    std::lock_guard<std::mutex> lock(timing_mutex_);
    accumulated_stats_.reset();
    frame_count_ = 0;
}

std::string CpuAprilTagAlgorithm::getTimingReport() const {
    // Get all data while holding lock (don't call getAverageTiming to avoid double-lock)
    TimingStats avg;
    TimingStats last_frame;
    int frame_count;
    TimingStats accum;
    
    {
        std::lock_guard<std::mutex> lock(timing_mutex_);
        last_frame = last_frame_timing_;
        frame_count = frame_count_;
        accum = accumulated_stats_;
    }
    
    // Calculate average without holding lock
    if (frame_count > 0) {
        avg.grayscale_convert_ms = accum.grayscale_convert_ms / frame_count;
        avg.mirror_ms = accum.mirror_ms / frame_count;
        avg.detection_ms = accum.detection_ms / frame_count;
        avg.total_ms = accum.total_ms / frame_count;
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    
    // Calculate FPS
    double fps = (avg.total_ms > 0.0) ? (1000.0 / avg.total_ms) : 0.0;
    
    oss << "=== CPU AprilTag Timing Analysis ===\n\n";
    
    // Average timing breakdown
    oss << "Average Timing (last " << frame_count << " frames):\n";
    oss << "  Total: " << avg.total_ms << " ms (" << fps << " FPS)\n\n";
    
    oss << "Timing Breakdown:\n";
    if (avg.total_ms > 0.0) {
        oss << "  Grayscale Convert: " << avg.grayscale_convert_ms << " ms "
            << "(" << (100.0 * avg.grayscale_convert_ms / avg.total_ms) << "%)\n";
        oss << "  Mirror: " << avg.mirror_ms << " ms "
            << "(" << (100.0 * avg.mirror_ms / avg.total_ms) << "%)\n";
        oss << "  Detection: " << avg.detection_ms << " ms "
            << "(" << (100.0 * avg.detection_ms / avg.total_ms) << "%)\n";
    }
    
    oss << "\nStatistics:\n";
    oss << "  Last Frame Detections: " << last_frame.num_detections << "\n";
    
    return oss.str();
}

