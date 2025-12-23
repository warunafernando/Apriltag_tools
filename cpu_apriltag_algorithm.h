#ifndef CPU_APRILTAG_ALGORITHM_H
#define CPU_APRILTAG_ALGORITHM_H

#include "apriltag_algorithm.h"
#include "apriltag/apriltag.h"
#include <vector>
#include <chrono>
#include <string>
#include <mutex>

/**
 * CPU-based AprilTag algorithm implementation.
 * 
 * This uses the standard OpenCV/CPU-based AprilTag detector with detailed timing analysis.
 */
class CpuAprilTagAlgorithm : public AprilTagAlgorithm {
public:
    // Timing statistics structure
    struct TimingStats {
        double grayscale_convert_ms = 0.0;      // Time to convert to grayscale (if needed)
        double mirror_ms = 0.0;                  // Time to mirror image (if enabled)
        double detection_ms = 0.0;               // Time for apriltag_detector_detect
        double total_ms = 0.0;                   // Total processFrame time
        
        // Frame statistics
        int num_detections = 0;
        
        // Reset all counters
        void reset() {
            grayscale_convert_ms = 0.0;
            mirror_ms = 0.0;
            detection_ms = 0.0;
            total_ms = 0.0;
            num_detections = 0;
        }
    };
    
    CpuAprilTagAlgorithm();
    ~CpuAprilTagAlgorithm() override;
    
    bool initialize(int width, int height) override;
    zarray_t* processFrame(const cv::Mat& gray_frame, bool mirror) override;
    void cleanup() override;
    std::string getName() const override { return "OpenCV CPU (AprilTag)"; }
    bool requiresCuda() const override { return false; }
    
    // Timing analysis methods
    TimingStats getLastFrameTiming() const;
    TimingStats getAverageTiming() const;
    void resetTimingStats() const;
    std::string getTimingReport() const;
    
private:
    // Dimensions
    int width_;
    int height_;
    bool initialized_;
    
    // AprilTag detector
    apriltag_family_t* tf_;
    apriltag_detector_t* td_;
    
    // Timing data
    mutable TimingStats last_frame_timing_;
    mutable TimingStats accumulated_stats_;
    mutable int frame_count_;
    mutable std::mutex timing_mutex_;
};

#endif // CPU_APRILTAG_ALGORITHM_H

