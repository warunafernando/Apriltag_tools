#ifndef FAST_APRILTAG_ALGORITHM_H
#define FAST_APRILTAG_ALGORITHM_H

#include "apriltag_algorithm.h"

#ifdef HAVE_CUDA_APRILTAG
#include "../src/apriltags_cuda/src/apriltag_gpu.h"
#include "apriltag.h"
#include "g2d.h"
#include <vector>
#include <memory>
#include <chrono>
#include <string>

// Forward declaration for DecodeTagsFromQuads
namespace frc971 {
namespace apriltag {
    void DecodeTagsFromQuads(const std::vector<QuadCorners> &quad_corners,
                             const uint8_t *gray_buf, int width, int height,
                             apriltag_detector_t *td,
                             const CameraMatrix &camera_matrix,
                             const DistCoeffs &distortion_coefficients,
                             zarray_t *detections,
                             zarray_t *poly0,
                             zarray_t *poly1);
}
}

/**
 * Fast AprilTag algorithm implementation.
 * 
 * This is a direct port of the algorithm from video_visualize_fixed.cu.
 * It uses GPU-accelerated detection followed by CPU-based tag decoding.
 */
class FastAprilTagAlgorithm : public AprilTagAlgorithm {
public:
    // Timing statistics structure
    struct TimingStats {
        double detect_gpu_only_ms = 0.0;      // Stage 1: DetectGpuOnly
        double fit_quads_ms = 0.0;             // Stage 2: FitQuads
        double mirror_ms = 0.0;                // Stage 3: Mirror operations
        double copy_gray_ms = 0.0;             // Stage 4: CopyGrayHostTo
        double decode_tags_ms = 0.0;           // Stage 5: DecodeTagsFromQuads
        double scale_coords_ms = 0.0;          // Stage 6: Scale coordinates
        double filter_duplicates_ms = 0.0;     // Stage 7: Filter duplicates
        double total_ms = 0.0;                 // Total processFrame time
        
        // GPU detector internal timing (from GpuDetector)
        double gpu_cuda_ops_ms = 0.0;          // CUDA operations time from detector
        double gpu_cpu_decode_ms = 0.0;        // CPU decode time from detector (if available)
        
        // Frame statistics
        int num_quads = 0;
        int num_detections_before_filter = 0;
        int num_detections_after_filter = 0;
        
        // Reset all counters
        void reset() {
            detect_gpu_only_ms = 0.0;
            fit_quads_ms = 0.0;
            mirror_ms = 0.0;
            copy_gray_ms = 0.0;
            decode_tags_ms = 0.0;
            scale_coords_ms = 0.0;
            filter_duplicates_ms = 0.0;
            total_ms = 0.0;
            gpu_cuda_ops_ms = 0.0;
            gpu_cpu_decode_ms = 0.0;
            num_quads = 0;
            num_detections_before_filter = 0;
            num_detections_after_filter = 0;
        }
        
        // Get formatted string representation
        std::string toString() const;
    };
    
    FastAprilTagAlgorithm();
    ~FastAprilTagAlgorithm() override;
    
    bool initialize(int width, int height) override;
    zarray_t* processFrame(const cv::Mat& gray_frame, bool mirror) override;
    void cleanup() override;
    std::string getName() const override { return "Fast AprilTag"; }
    bool requiresCuda() const override { return true; }
    
    // Timing analysis methods
    TimingStats getLastFrameTiming() const { return last_frame_timing_; }
    TimingStats getAverageTiming() const;
    void resetTimingStats() const;
    std::string getTimingReport() const;
    
private:
    // Dimensions
    int width_;
    int height_;
    bool initialized_;
    
    // GPU detector
    frc971::apriltag::GpuDetector* gpu_detector_;
    
    // CPU decode detector
    apriltag_family_t* tf_gpu_;
    apriltag_detector_t* td_gpu_;
    apriltag_detector_t* td_for_gpu_;  // Detector passed to GpuDetector (GpuDetector doesn't take ownership)
    
    // Camera calibration (loaded from fisheye calibration)
    frc971::apriltag::CameraMatrix gpu_cam_;
    frc971::apriltag::DistCoeffs gpu_dist_;
    
    // Reusable buffer to avoid per-frame allocations
    std::vector<uint8_t> gray_host_buffer_;  // Reused for CopyGrayHostTo (resized by CopyGrayHostTo if needed)
    
    // Minimum distance for duplicate filtering (pixels)
    double min_distance_;
    
    // Timing tracking
    mutable TimingStats last_frame_timing_;      // Last frame's timing
    mutable TimingStats accumulated_stats_;      // Accumulated stats for averaging
    mutable int frame_count_;                    // Number of frames processed
    
    // Helper functions ported from video_visualize_fixed.cu
    bool isValidDetection(apriltag_detection_t* det, int width, int height);
    std::vector<apriltag_detection_t*> filterDuplicates(const zarray_t* detections, int width, int height, double min_distance = 50.0);
    void mirrorQuadCoordinates(std::vector<frc971::apriltag::QuadCorners>& quads, int width);
    void loadCalibration(double& fx, double& fy, double& cx, double& cy,
                        double& k1, double& k2, double& p1, double& p2, double& k3);
    void scaleDetectionCoordinates(apriltag_detection_t* det, double decimate_factor);
};

#else
// Stub implementation when CUDA is not available
class FastAprilTagAlgorithm : public AprilTagAlgorithm {
public:
    FastAprilTagAlgorithm() {}
    ~FastAprilTagAlgorithm() override {}
    bool initialize(int width, int height) override { return false; }
    zarray_t* processFrame(const cv::Mat& gray_frame, bool mirror) override { return nullptr; }
    void cleanup() override {}
    std::string getName() const override { return "Fast AprilTag (CUDA required)"; }
    bool requiresCuda() const override { return true; }
};
#endif // HAVE_CUDA_APRILTAG

#endif // FAST_APRILTAG_ALGORITHM_H

