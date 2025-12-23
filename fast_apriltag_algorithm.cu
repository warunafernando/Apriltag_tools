#ifdef HAVE_CUDA_APRILTAG

#include "fast_apriltag_algorithm.h"
#include "../src/apriltags_cuda/src/apriltag_utils.h"
#include "apriltag/tag36h11.h"
#include "apriltag/common/workerpool.h"
#include "g2d.h"  // For g2d_polygon_create_zeros and g2d_polygon_destroy
#include <opencv2/opencv.hpp>
#include <map>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include <sstream>
#include <iomanip>

using namespace cv;
using namespace std;

FastAprilTagAlgorithm::FastAprilTagAlgorithm()
    : width_(0), height_(0), initialized_(false),
      gpu_detector_(nullptr),
      tf_gpu_(nullptr), td_gpu_(nullptr), td_for_gpu_(nullptr),
      min_distance_(50.0),
      frame_count_(0) {
    accumulated_stats_.reset();
    last_frame_timing_.reset();
}

FastAprilTagAlgorithm::~FastAprilTagAlgorithm() {
    cleanup();
}

bool FastAprilTagAlgorithm::isValidDetection(apriltag_detection_t* det, int width, int height) {
    // Check center coordinates
    if (det->c[0] < 0 || det->c[0] >= width || det->c[1] < 0 || det->c[1] >= height) {
        return false;
    }
    // Check all corner coordinates
    for (int i = 0; i < 4; i++) {
        if (det->p[i][0] < 0 || det->p[i][0] >= width || 
            det->p[i][1] < 0 || det->p[i][1] >= height) {
            return false;
        }
    }
    return true;
}

std::vector<apriltag_detection_t*> FastAprilTagAlgorithm::filterDuplicates(
    const zarray_t* detections, int width, int height, double min_distance) {
    
    std::vector<apriltag_detection_t*> filtered;
    std::vector<apriltag_detection_t*> all_dets;
    
    // Extract all detections
    for (int i = 0; i < zarray_size(detections); i++) {
        apriltag_detection_t* det;
        zarray_get(const_cast<zarray_t*>(detections), i, &det);
        all_dets.push_back(det);
    }
    
    // Group by tag ID and keep only the best VALID one per ID
    std::map<int, apriltag_detection_t*> best_by_id;
    
    for (auto* det : all_dets) {
        // Only consider valid detections (within image bounds)
        if (!isValidDetection(det, width, height)) {
            continue;
        }
        
        auto it = best_by_id.find(det->id);
        if (it == best_by_id.end()) {
            // First valid detection with this ID
            best_by_id[det->id] = det;
        } else {
            // Compare decision margins - keep the one with higher margin
            if (det->decision_margin > it->second->decision_margin) {
                best_by_id[det->id] = det;
            }
        }
    }
    
    // Convert map to vector
    for (auto& pair : best_by_id) {
        filtered.push_back(pair.second);
    }
    
    return filtered;
}

void FastAprilTagAlgorithm::mirrorQuadCoordinates(
    std::vector<frc971::apriltag::QuadCorners>& quads, int width) {
    
    for (auto& quad : quads) {
        // Mirror x coordinates for all 4 corners
        for (int i = 0; i < 4; i++) {
            quad.corners[i][0] = width - 1 - quad.corners[i][0];
        }
        // Swap corners to maintain correct orientation: 0<->1, 2<->3
        float temp[2];
        temp[0] = quad.corners[0][0]; temp[1] = quad.corners[0][1];
        quad.corners[0][0] = quad.corners[1][0]; quad.corners[0][1] = quad.corners[1][1];
        quad.corners[1][0] = temp[0]; quad.corners[1][1] = temp[1];
        
        temp[0] = quad.corners[2][0]; temp[1] = quad.corners[2][1];
        quad.corners[2][0] = quad.corners[3][0]; quad.corners[2][1] = quad.corners[3][1];
        quad.corners[3][0] = temp[0]; quad.corners[3][1] = temp[1];
    }
}

void FastAprilTagAlgorithm::scaleDetectionCoordinates(apriltag_detection_t* det, double decimate_factor) {
    if (decimate_factor <= 1.0) return;
    
    if (decimate_factor != 1.5) {
        // Standard scaling: (x - 0.5) * factor + 0.5
        for (int j = 0; j < 4; j++) {
            det->p[j][0] = (det->p[j][0] - 0.5) * decimate_factor + 0.5;
            det->p[j][1] = (det->p[j][1] - 0.5) * decimate_factor + 0.5;
        }
        det->c[0] = (det->c[0] - 0.5) * decimate_factor + 0.5;
        det->c[1] = (det->c[1] - 0.5) * decimate_factor + 0.5;
    } else {
        // For 1.5x decimation, simple multiplication
        for (int j = 0; j < 4; j++) {
            det->p[j][0] *= decimate_factor;
            det->p[j][1] *= decimate_factor;
        }
        det->c[0] *= decimate_factor;
        det->c[1] *= decimate_factor;
    }
}

void FastAprilTagAlgorithm::loadCalibration(double& fx, double& fy, double& cx, double& cy,
                                            double& k1, double& k2, double& p1, double& p2, double& k3) {
    // Default calibration values (will be overridden if fisheye calibration is available)
    fx = 905.495617;
    fy = 609.916016;
    cx = 907.909470;
    cy = 352.682645;
    k1 = k2 = p1 = p2 = k3 = 0.0;
    
    // TODO: Load from fisheye calibration if available
    // For now, use defaults (calibration loading will be handled by GUI and passed to algorithm)
    // This matches video_visualize_fixed.cu behavior when no calibration file is provided
}

bool FastAprilTagAlgorithm::initialize(int width, int height) {
    if (initialized_) {
        cleanup();
    }
    
    width_ = width;
    height_ = height;
    
    // Validate dimensions - must be even numbers for CUDA decimation
    if (width <= 0 || height <= 0 || width > 10000 || height > 10000) {
        std::cerr << "FastAprilTagAlgorithm: Invalid frame dimensions: " << width << "x" << height << std::endl;
        return false;
    }
    
    // Ensure dimensions are even (required for quad_decimate = 2.0)
    if (width % 2 != 0) width_ = (width / 2) * 2;
    if (height % 2 != 0) height_ = (height / 2) * 2;
    
    // Setup tag family
    tf_gpu_ = tag36h11_create();
    if (!tf_gpu_) {
        std::cerr << "FastAprilTagAlgorithm: Failed to create tag family" << std::endl;
        return false;
    }
    
    // Create detector for GpuDetector (same as 'td' in video_visualize_fixed.cu)
    td_for_gpu_ = apriltag_detector_create();
    if (!td_for_gpu_) {
        tag36h11_destroy(tf_gpu_);
        tf_gpu_ = nullptr;
        std::cerr << "FastAprilTagAlgorithm: Failed to create tag detector" << std::endl;
        return false;
    }
    
    apriltag_detector_add_family(td_for_gpu_, tf_gpu_);
    td_for_gpu_->quad_decimate = 2.0;  // CUDA requires 2.0
    td_for_gpu_->quad_sigma = 0.0;  // Match video_visualize_fixed.cu
    td_for_gpu_->refine_edges = 1;  // Match video_visualize_fixed.cu (true)
    td_for_gpu_->debug = false;  // Match video_visualize_fixed.cu
    td_for_gpu_->nthreads = 1;  // Match video_visualize_fixed.cu default
    td_for_gpu_->wp = workerpool_create(td_for_gpu_->nthreads);
    
    // Create SEPARATE detector for CPU decode (exactly like td_cpu in video_visualize_fixed.cu)
    td_gpu_ = apriltag_detector_create();
    if (!td_gpu_) {
        apriltag_detector_destroy(td_for_gpu_);
        tag36h11_destroy(tf_gpu_);
        tf_gpu_ = nullptr;
        td_for_gpu_ = nullptr;
        std::cerr << "FastAprilTagAlgorithm: Failed to create CPU decode detector" << std::endl;
        return false;
    }
    apriltag_detector_add_family(td_gpu_, tf_gpu_);
    // Copy settings from GPU detector (same as video_visualize_fixed.cu does)
    td_gpu_->quad_decimate = td_for_gpu_->quad_decimate;
    td_gpu_->quad_sigma = td_for_gpu_->quad_sigma;
    td_gpu_->refine_edges = td_for_gpu_->refine_edges;
    td_gpu_->debug = td_for_gpu_->debug;
    td_gpu_->nthreads = td_for_gpu_->nthreads;
    td_gpu_->wp = workerpool_create(td_gpu_->nthreads);
    
    // Load calibration
    double fx, fy, cx, cy, k1, k2, p1, p2, k3;
    loadCalibration(fx, fy, cx, cy, k1, k2, p1, p2, k3);
    gpu_cam_ = frc971::apriltag::CameraMatrix{fx, fy, cx, cy};
    gpu_dist_ = frc971::apriltag::DistCoeffs{k1, k2, p1, p2, k3};
    
    // Create GPU detector
    try {
        gpu_detector_ = new frc971::apriltag::GpuDetector(width_, height_, td_for_gpu_, gpu_cam_, gpu_dist_);
        std::cerr << "FastAprilTagAlgorithm: GPU detector initialized successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "FastAprilTagAlgorithm: Exception creating GpuDetector: " << e.what() << std::endl;
        apriltag_detector_destroy(td_for_gpu_);
        apriltag_detector_destroy(td_gpu_);
        tag36h11_destroy(tf_gpu_);
        td_for_gpu_ = nullptr;
        td_gpu_ = nullptr;
        tf_gpu_ = nullptr;
        return false;
    } catch (...) {
        std::cerr << "FastAprilTagAlgorithm: Unknown exception creating GpuDetector" << std::endl;
        apriltag_detector_destroy(td_for_gpu_);
        apriltag_detector_destroy(td_gpu_);
        tag36h11_destroy(tf_gpu_);
        td_for_gpu_ = nullptr;
        td_gpu_ = nullptr;
        tf_gpu_ = nullptr;
        return false;
    }
    
    // Pre-allocate reusable buffer for gray image (will be resized by CopyGrayHostTo if needed)
    gray_host_buffer_.reserve(width_ * height_);
    gray_host_buffer_.resize(width_ * height_);
    // Note: polygon arrays (poly0, poly1) are created per-frame to avoid threading/state issues
    
    initialized_ = true;
    return true;
}

zarray_t* FastAprilTagAlgorithm::processFrame(const cv::Mat& gray_frame, bool mirror) {
    using namespace std::chrono;
    auto total_start = high_resolution_clock::now();
    
    // Reset timing for this frame
    last_frame_timing_.reset();
    
    if (!initialized_) {
        // Silently return - this is expected before delayed initialization completes
        return nullptr;
    }
    
    // Safety check: ensure GPU detector is valid
    if (!gpu_detector_) {
        std::cerr << "FastAprilTagAlgorithm: ERROR - GPU detector is null but initialized_ is true!" << std::endl;
        return nullptr;
    }
    
    // Validate frame (from video_visualize_fixed.cu process_frame lambda)
    if (gray_frame.empty() || gray_frame.data == nullptr) {
        std::cerr << "FastAprilTagAlgorithm: Invalid frame (empty or null data)" << std::endl;
        return nullptr;
    }
    if (gray_frame.cols != width_ || gray_frame.rows != height_) {
        std::cerr << "FastAprilTagAlgorithm: Frame size mismatch: " << gray_frame.cols << "x" << gray_frame.rows 
                  << " expected " << width_ << "x" << height_ << std::endl;
        return nullptr;
    }
    if (!gray_frame.isContinuous()) {
        std::cerr << "FastAprilTagAlgorithm: Frame is not contiguous" << std::endl;
        return nullptr;
    }
    if (gray_frame.type() != CV_8UC1) {
        std::cerr << "FastAprilTagAlgorithm: Frame is not grayscale (CV_8UC1)" << std::endl;
        return nullptr;
    }
    
    // Get GPU detector timing before operations (to calculate delta)
    double prev_gpu_cuda_total = gpu_detector_->GetCudaOperationsDurationMs();
    
    // Stage 1: GPU-only detection (from video_visualize_fixed.cu line 1115)
    auto stage1_start = high_resolution_clock::now();
    try {
        gpu_detector_->DetectGpuOnly(gray_frame.data);
    } catch (const std::exception& e) {
        std::cerr << "FastAprilTagAlgorithm: Exception in DetectGpuOnly: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "FastAprilTagAlgorithm: Unknown exception in DetectGpuOnly" << std::endl;
        return nullptr;
    }
    auto stage1_end = high_resolution_clock::now();
    last_frame_timing_.detect_gpu_only_ms = duration<double, milli>(stage1_end - stage1_start).count();
    
    // Update GPU CUDA operations timing (delta from detector)
    double cur_gpu_cuda_total = gpu_detector_->GetCudaOperationsDurationMs();
    last_frame_timing_.gpu_cuda_ops_ms = cur_gpu_cuda_total - prev_gpu_cuda_total;
    
    // Stage 2: Fit quads (full resolution, from video_visualize_fixed.cu line 1155)
    auto stage2_start = high_resolution_clock::now();
    // Make a copy since FitQuads() returns const reference and we may need to modify it
    std::vector<frc971::apriltag::QuadCorners> quads_fullres = gpu_detector_->FitQuads();
    auto stage2_end = high_resolution_clock::now();
    last_frame_timing_.fit_quads_ms = duration<double, milli>(stage2_end - stage2_start).count();
    last_frame_timing_.num_quads = static_cast<int>(quads_fullres.size());
    
    // Stage 3: Mirror handling (from video_visualize_fixed.cu lines 1159-1178)
    auto stage3_start = high_resolution_clock::now();
    if (mirror) {
        try {
            gpu_detector_->MirrorGrayImageOnGpu();
        } catch (const std::exception& e) {
            std::cerr << "FastAprilTagAlgorithm: Exception in MirrorGrayImageOnGpu: " << e.what() << std::endl;
            return nullptr;
        } catch (...) {
            std::cerr << "FastAprilTagAlgorithm: Unknown exception in MirrorGrayImageOnGpu" << std::endl;
            return nullptr;
        }
        // Also mirror quad coordinates (on CPU, they're small)
        mirrorQuadCoordinates(quads_fullres, width_);
    }
    auto stage3_end = high_resolution_clock::now();
    last_frame_timing_.mirror_ms = duration<double, milli>(stage3_end - stage3_start).count();
    
    // Stage 4: Copy full-resolution gray image from GPU (from video_visualize_fixed.cu line 1183)
    auto stage4_start = high_resolution_clock::now();
    // Reuse pre-allocated buffer (CopyGrayHostTo will resize if needed)
    try {
        gpu_detector_->CopyGrayHostTo(gray_host_buffer_);
    } catch (const std::exception& e) {
        std::cerr << "FastAprilTagAlgorithm: Exception in CopyGrayHostTo: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "FastAprilTagAlgorithm: Unknown exception in CopyGrayHostTo" << std::endl;
        return nullptr;
    }
    auto stage4_end = high_resolution_clock::now();
    last_frame_timing_.copy_gray_ms = duration<double, milli>(stage4_end - stage4_start).count();
    
    // Stage 5: Decode tags (from video_visualize_fixed.cu lines 954-956)
    auto stage5_start = high_resolution_clock::now();
    // Create polygon arrays per-frame (like video_visualize_fixed.cu does in decode thread)
    zarray_t* poly0 = g2d_polygon_create_zeros(4);
    zarray_t* poly1 = g2d_polygon_create_zeros(4);
    zarray_t* detections = zarray_create(sizeof(apriltag_detection_t*));
    frc971::apriltag::DecodeTagsFromQuads(
        quads_fullres, gray_host_buffer_.data(), gpu_detector_->Width(), gpu_detector_->Height(),
        td_gpu_, gpu_cam_, gpu_dist_, detections, poly0, poly1);
    // Note: poly0 and poly1 are small temporary arrays, will be cleaned up when function returns
    auto stage5_end = high_resolution_clock::now();
    last_frame_timing_.decode_tags_ms = duration<double, milli>(stage5_end - stage5_start).count();
    last_frame_timing_.num_detections_before_filter = zarray_size(detections);
    
    // Stage 6: Scale coordinates if needed (from video_visualize_fixed.cu lines 969-977)
    auto stage6_start = high_resolution_clock::now();
    const double gpu_decimate = td_for_gpu_->quad_decimate;
    if (gpu_decimate > 1.0) {
        for (int i = 0; i < zarray_size(detections); i++) {
            apriltag_detection_t* det;
            zarray_get(detections, i, &det);
            scaleDetectionCoordinates(det, gpu_decimate);
        }
    }
    auto stage6_end = high_resolution_clock::now();
    last_frame_timing_.scale_coords_ms = duration<double, milli>(stage6_end - stage6_start).count();
    
    // Stage 7: Filter duplicates (from video_visualize_fixed.cu lines 986-990)
    auto stage7_start = high_resolution_clock::now();
    std::vector<apriltag_detection_t*> filtered = filterDuplicates(
        detections, width_, height_, min_distance_);
    
    // Destroy detections that are NOT in the filtered list
    for (int i = 0; i < zarray_size(detections); i++) {
        apriltag_detection_t* det;
        zarray_get(detections, i, &det);
        bool is_filtered = false;
        for (apriltag_detection_t* f : filtered) {
            if (f == det) {
                is_filtered = true;
                break;
            }
        }
        if (!is_filtered) {
            apriltag_detection_destroy(det);
        }
    }
    zarray_destroy(detections);
    
    // Create new array with filtered detections
    detections = zarray_create(sizeof(apriltag_detection_t*));
    for (auto* det : filtered) {
        zarray_add(detections, &det);
    }
    auto stage7_end = high_resolution_clock::now();
    last_frame_timing_.filter_duplicates_ms = duration<double, milli>(stage7_end - stage7_start).count();
    last_frame_timing_.num_detections_after_filter = zarray_size(detections);
    
    // Calculate total time
    auto total_end = high_resolution_clock::now();
    last_frame_timing_.total_ms = duration<double, milli>(total_end - total_start).count();
    
    // Accumulate statistics for averaging
    accumulated_stats_.detect_gpu_only_ms += last_frame_timing_.detect_gpu_only_ms;
    accumulated_stats_.fit_quads_ms += last_frame_timing_.fit_quads_ms;
    accumulated_stats_.mirror_ms += last_frame_timing_.mirror_ms;
    accumulated_stats_.copy_gray_ms += last_frame_timing_.copy_gray_ms;
    accumulated_stats_.decode_tags_ms += last_frame_timing_.decode_tags_ms;
    accumulated_stats_.scale_coords_ms += last_frame_timing_.scale_coords_ms;
    accumulated_stats_.filter_duplicates_ms += last_frame_timing_.filter_duplicates_ms;
    accumulated_stats_.total_ms += last_frame_timing_.total_ms;
    accumulated_stats_.gpu_cuda_ops_ms += last_frame_timing_.gpu_cuda_ops_ms;
    frame_count_++;
    
    return detections;
}

void FastAprilTagAlgorithm::cleanup() {
    // Cleanup reusable buffers
    gray_host_buffer_.clear();
    gray_host_buffer_.shrink_to_fit();
    
    if (gpu_detector_) {
        delete gpu_detector_;
        gpu_detector_ = nullptr;
    }
    
    if (td_for_gpu_) {
        apriltag_detector_destroy(td_for_gpu_);
        td_for_gpu_ = nullptr;
    }
    
    if (td_gpu_) {
        apriltag_detector_destroy(td_gpu_);
        td_gpu_ = nullptr;
    }
    
    if (tf_gpu_) {
        tag36h11_destroy(tf_gpu_);
        tf_gpu_ = nullptr;
    }
    
    initialized_ = false;
    width_ = 0;
    height_ = 0;
}

// TimingStats::toString implementation
std::string FastAprilTagAlgorithm::TimingStats::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "Timing Breakdown:\n";
    oss << "  Stage 1 - DetectGpuOnly:     " << std::setw(8) << detect_gpu_only_ms << " ms\n";
    oss << "  Stage 2 - FitQuads:          " << std::setw(8) << fit_quads_ms << " ms\n";
    oss << "  Stage 3 - Mirror:            " << std::setw(8) << mirror_ms << " ms\n";
    oss << "  Stage 4 - CopyGrayHostTo:    " << std::setw(8) << copy_gray_ms << " ms\n";
    oss << "  Stage 5 - DecodeTags:        " << std::setw(8) << decode_tags_ms << " ms\n";
    oss << "  Stage 6 - ScaleCoords:       " << std::setw(8) << scale_coords_ms << " ms\n";
    oss << "  Stage 7 - FilterDuplicates:  " << std::setw(8) << filter_duplicates_ms << " ms\n";
    oss << "  GPU CUDA Operations:         " << std::setw(8) << gpu_cuda_ops_ms << " ms\n";
    oss << "  ---------------------------------\n";
    oss << "  TOTAL:                       " << std::setw(8) << total_ms << " ms\n";
    oss << "  FPS:                         " << std::setw(8) << (total_ms > 0 ? 1000.0 / total_ms : 0.0) << "\n";
    oss << "\nStatistics:\n";
    oss << "  Quads detected:              " << num_quads << "\n";
    oss << "  Detections (before filter):  " << num_detections_before_filter << "\n";
    oss << "  Detections (after filter):   " << num_detections_after_filter << "\n";
    return oss.str();
}

// FastAprilTagAlgorithm timing methods implementation
FastAprilTagAlgorithm::TimingStats FastAprilTagAlgorithm::getAverageTiming() const {
    TimingStats avg;
    if (frame_count_ > 0) {
        avg.detect_gpu_only_ms = accumulated_stats_.detect_gpu_only_ms / frame_count_;
        avg.fit_quads_ms = accumulated_stats_.fit_quads_ms / frame_count_;
        avg.mirror_ms = accumulated_stats_.mirror_ms / frame_count_;
        avg.copy_gray_ms = accumulated_stats_.copy_gray_ms / frame_count_;
        avg.decode_tags_ms = accumulated_stats_.decode_tags_ms / frame_count_;
        avg.scale_coords_ms = accumulated_stats_.scale_coords_ms / frame_count_;
        avg.filter_duplicates_ms = accumulated_stats_.filter_duplicates_ms / frame_count_;
        avg.total_ms = accumulated_stats_.total_ms / frame_count_;
        avg.gpu_cuda_ops_ms = accumulated_stats_.gpu_cuda_ops_ms / frame_count_;
    }
    return avg;
}

void FastAprilTagAlgorithm::resetTimingStats() const {
    accumulated_stats_.reset();
    frame_count_ = 0;
}

std::string FastAprilTagAlgorithm::getTimingReport() const {
    std::ostringstream oss;
    oss << "=== Fast AprilTag Timing Analysis ===\n";
    oss << "NOTE: This measures ONLY processFrame() execution time.\n";
    oss << "Detection FPS includes additional overhead (frame cloning,\n";
    oss << "conversion, drawing, etc.) which explains the difference.\n\n";
    
    if (frame_count_ > 0) {
        TimingStats avg = getAverageTiming();
        
        // Show timing breakdown with last frame statistics
        oss << "Timing Breakdown:\n";
        oss << std::fixed << std::setprecision(3);
        oss << "  Stage 1 - DetectGpuOnly:     " << std::setw(8) << avg.detect_gpu_only_ms << " ms\n";
        oss << "  Stage 2 - FitQuads:          " << std::setw(8) << avg.fit_quads_ms << " ms\n";
        oss << "  Stage 3 - Mirror:            " << std::setw(8) << avg.mirror_ms << " ms\n";
        oss << "  Stage 4 - CopyGrayHostTo:    " << std::setw(8) << avg.copy_gray_ms << " ms\n";
        oss << "  Stage 5 - DecodeTags:        " << std::setw(8) << avg.decode_tags_ms << " ms\n";
        oss << "  Stage 6 - ScaleCoords:       " << std::setw(8) << avg.scale_coords_ms << " ms\n";
        oss << "  Stage 7 - FilterDuplicates:  " << std::setw(8) << avg.filter_duplicates_ms << " ms\n";
        oss << "  GPU CUDA Operations:         " << std::setw(8) << avg.gpu_cuda_ops_ms << " ms\n";
        oss << "  ---------------------------------\n";
        oss << "  TOTAL:                       " << std::setw(8) << avg.total_ms << " ms\n";
        oss << "  FPS:                         " << std::setw(8) << (avg.total_ms > 0 ? 1000.0 / avg.total_ms : 0.0) << "\n";
        
        // Statistics with last frame values
        oss << "\nStatistics:\n";
        oss << "  Quads detected (last frame):        " << last_frame_timing_.num_quads << "\n";
        oss << "  Detections (before filter, last):  " << last_frame_timing_.num_detections_before_filter << "\n";
        oss << "  Detections (after filter, last):   " << last_frame_timing_.num_detections_after_filter << "\n";
        
        // Calculate percentages
        oss << "\nPercentage Breakdown (Average):\n";
        if (avg.total_ms > 0) {
            oss << std::fixed << std::setprecision(1);
            oss << "  DetectGpuOnly:     " << std::setw(5) << (avg.detect_gpu_only_ms / avg.total_ms * 100) << "%\n";
            oss << "  FitQuads:          " << std::setw(5) << (avg.fit_quads_ms / avg.total_ms * 100) << "%\n";
            oss << "  Mirror:            " << std::setw(5) << (avg.mirror_ms / avg.total_ms * 100) << "%\n";
            oss << "  CopyGrayHostTo:    " << std::setw(5) << (avg.copy_gray_ms / avg.total_ms * 100) << "%\n";
            oss << "  DecodeTags:        " << std::setw(5) << (avg.decode_tags_ms / avg.total_ms * 100) << "%\n";
            oss << "  ScaleCoords:       " << std::setw(5) << (avg.scale_coords_ms / avg.total_ms * 100) << "%\n";
            oss << "  FilterDuplicates:  " << std::setw(5) << (avg.filter_duplicates_ms / avg.total_ms * 100) << "%\n";
            oss << "  GPU CUDA Ops:      " << std::setw(5) << (avg.gpu_cuda_ops_ms / avg.total_ms * 100) << "%\n";
        }
    } else {
        oss << "No frames processed yet.\n";
    }
    
    return oss.str();
}

#endif // HAVE_CUDA_APRILTAG

