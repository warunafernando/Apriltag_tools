#ifdef HAVE_CUDA_APRILTAG

#include "fast_apriltag_algorithm.h"
#include "../src/apriltags_cuda/src/apriltag_utils.h"
#include "apriltag/tag36h11.h"
#include "apriltag/common/workerpool.h"
#include "g2d.h"  // For g2d_polygon_create_zeros and zarray_destroy
#include <opencv2/opencv.hpp>
#include <map>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <cuda_runtime.h>
#include <future>

using namespace cv;
using namespace std;

FastAprilTagAlgorithm::FastAprilTagAlgorithm()
    : width_(0), height_(0), initialized_(false),
      gpu_detector_(nullptr),
      tf_gpu_(nullptr), td_gpu_(nullptr), td_for_gpu_(nullptr),
      min_distance_(10.0),  // Reduced to 10.0 pixels - only filter extremely close duplicates (same physical tag)
      frame_count_(0),
      frames_seen_(0),
      worker_running_(false) {
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
    std::vector<apriltag_detection_t*> valid_dets;
    
    // First, collect valid detections (within image bounds)
    for (int i = 0; i < zarray_size(detections); i++) {
        apriltag_detection_t* det;
        zarray_get(const_cast<zarray_t*>(detections), i, &det);
        if (isValidDetection(det, width, height)) {
            valid_dets.push_back(det);
        }
    }
    
    if (valid_dets.empty()) {
        return filtered;
    }
    
    // Sort by decision_margin (best first) so we keep the best detection when filtering
    std::sort(valid_dets.begin(), valid_dets.end(),
              [](apriltag_detection_t* a, apriltag_detection_t* b) {
                  return a->decision_margin > b->decision_margin;
              });
    
    // Filter duplicates: for detections with same ID, only remove if they are within min_distance
    // This allows multiple detections with the same ID if they are far enough apart
    // Made less strict: only filter if very close (within min_distance threshold)
    for (auto* det : valid_dets) {
        bool is_duplicate = false;
        
        // Check against already filtered detections
        for (auto* existing : filtered) {
            // Only filter if same ID AND within min_distance
            if (existing->id == det->id && existing->family == det->family) {
                // Calculate center-to-center distance
                double dx = det->c[0] - existing->c[0];
                double dy = det->c[1] - existing->c[1];
                double distance = std::sqrt(dx * dx + dy * dy);
                
                // Only filter if they're very close (within min_distance threshold)
                // This allows multiple detections of the same tag if they're far apart
                if (distance < min_distance) {
                    is_duplicate = true;
                    std::cerr << "[DEBUG] Filtering duplicate: ID=" << det->id 
                              << ", distance=" << std::fixed << std::setprecision(1) << distance 
                              << "px (min_distance=" << min_distance << "px)"
                              << ", existing_margin=" << std::fixed << std::setprecision(1) << existing->decision_margin 
                              << ", new_margin=" << std::fixed << std::setprecision(1) << det->decision_margin 
                              << ", existing_center=(" << std::fixed << std::setprecision(1) << existing->c[0] 
                              << "," << existing->c[1] << ")"
                              << ", new_center=(" << std::fixed << std::setprecision(1) << det->c[0] 
                              << "," << det->c[1] << ")" << std::endl;
                    break;
                }
            }
        }
        
        if (!is_duplicate) {
            filtered.push_back(det);
        }
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
    // Multiple GpuDetector instances or recreations can cause CUDA context issues
    if (initialized_) {
        if (width == width_ && height == height_) {
            return true;  // Already initialized with same dimensions, no need to recreate
        }
        cleanup();
    }
    
    width_ = width;
    height_ = height;
    
    // Validate dimensions - must be even numbers for CUDA decimation
    if (width <= 0 || height <= 0 || width > 10000 || height > 10000) {
        return false;
    }
    
    // Ensure dimensions are even (required for quad_decimate = 2.0)
    if (width % 2 != 0) width_ = (width / 2) * 2;
    if (height % 2 != 0) height_ = (height / 2) * 2;
    
    // Setup tag family
    tf_gpu_ = tag36h11_create();
    if (!tf_gpu_) {
        return false;
    }
    
    // Create detector for GpuDetector (same as 'td' in video_visualize_fixed.cu)
    td_for_gpu_ = apriltag_detector_create();
    if (!td_for_gpu_) {
        tag36h11_destroy(tf_gpu_);
        tf_gpu_ = nullptr;
        return false;
    }
    
    apriltag_detector_add_family(td_for_gpu_, tf_gpu_);
    td_for_gpu_->quad_decimate = 2.0;  // CUDA requires 2.0
    td_for_gpu_->quad_sigma = 0.0;  // Match video_visualize_fixed.cu
    td_for_gpu_->refine_edges = 1;  // Match video_visualize_fixed.cu (true)
    td_for_gpu_->decode_sharpening = 0.5;  // Increased from 0.25 for better decoding sensitivity
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
    
    // Set default quad threshold parameters (more sensitive defaults for better detection)
    // These will be updated by updateDetectorParameters when user applies settings
    td_for_gpu_->qtp.min_cluster_pixels = 4;  // More sensitive (lower = detects smaller clusters)
    td_for_gpu_->qtp.max_line_fit_mse = 12.0;  // More lenient (higher = allows more imperfect quads)
    td_for_gpu_->qtp.cos_critical_rad = cos(10.0 * M_PI / 180.0);  // More angle tolerance
    td_for_gpu_->qtp.min_white_black_diff = 4;  // More sensitive (lower = detects lower contrast tags)
    td_gpu_->qtp.min_cluster_pixels = td_for_gpu_->qtp.min_cluster_pixels;
    td_gpu_->qtp.max_line_fit_mse = td_for_gpu_->qtp.max_line_fit_mse;
    td_gpu_->qtp.cos_critical_rad = td_for_gpu_->qtp.cos_critical_rad;
    td_gpu_->qtp.min_white_black_diff = td_for_gpu_->qtp.min_white_black_diff;
    
    // Load calibration
    double fx, fy, cx, cy, k1, k2, p1, p2, k3;
    loadCalibration(fx, fy, cx, cy, k1, k2, p1, p2, k3);
    gpu_cam_ = frc971::apriltag::CameraMatrix{fx, fy, cx, cy};
    gpu_dist_ = frc971::apriltag::DistCoeffs{k1, k2, p1, p2, k3};
    
    // Pre-allocate reusable buffer for gray image (will be resized by CopyGrayHostTo if needed)
    gray_host_buffer_.reserve(width_ * height_);
    gray_host_buffer_.resize(width_ * height_);
    
    // This matches standalone pattern where detector is created in the processing thread
    gpu_detector_ = nullptr;  // Will be created in worker thread
    
    // This isolates CUDA context from Qt event loop, matching standalone's working pattern
    worker_running_ = true;
    cuda_worker_thread_ = std::thread(&FastAprilTagAlgorithm::cudaWorkerThread, this);
    
    initialized_ = true;
    return true;
}

zarray_t* FastAprilTagAlgorithm::processFrame(const cv::Mat& gray_frame, bool mirror) {
    std::cerr.flush();
    
    // Basic frame validation
    if (gray_frame.empty() || gray_frame.data == nullptr || 
        !gray_frame.isContinuous() || gray_frame.type() != CV_8UC1) {
        return nullptr;
    }
    
    // Frame counter for logging
    int frame_num = frames_seen_.fetch_add(1);
    
    // Get frame dimensions and ensure they're even (required for quad_decimate = 2.0)
    int frame_width = gray_frame.cols;
    int frame_height = gray_frame.rows;
    if (frame_width % 2 != 0) frame_width = (frame_width / 2) * 2;
    if (frame_height % 2 != 0) frame_height = (frame_height / 2) * 2;
    
    // Handle size changes: reinitialize if dimensions changed
    if (!initialized_ || frame_width != width_ || frame_height != height_) {
        if (initialized_) {
            // Size changed - need to reinitialize
            cleanup();
        }
        
        width_ = frame_width;
        height_ = frame_height;
        
        if (!initialize(width_, height_)) {
            return nullptr;
        }
    }
    
    // Handle frame size mismatch: create resized copy if needed
    cv::Mat frame_to_process = gray_frame;
    if (gray_frame.cols != width_ || gray_frame.rows != height_) {
        // Frame doesn't match - resize it to match detector dimensions
        cv::resize(gray_frame, frame_to_process, cv::Size(width_, height_));
    }
    
    // Ensure frame is continuous (required for direct memory access)
    if (!frame_to_process.isContinuous()) {
        frame_to_process = frame_to_process.clone();
    }
    
    // This isolates CUDA context from Qt event loop
    if (!worker_running_.load()) {
        return nullptr;
    }
    
    // Limit queue size to prevent memory buildup
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (frame_queue_.size() > 3) {
            return nullptr;  // Drop frame if queue is too full
        }
    }
    
    auto job = std::make_unique<FrameJob>();
    // Clone the processed frame (may be resized) to ensure data stays valid during async CUDA operations
    // The cloned frame will stay in the FrameJob struct until processing completes
    job->frame = frame_to_process.clone();
    if (job->frame.empty() || !job->frame.isContinuous()) {
        return nullptr;
    }
    job->mirror = mirror;
    auto future = job->result_promise.get_future();
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        frame_queue_.push(std::move(job));
    }
    queue_cv_.notify_one();
    
    // Wait for result (with timeout to prevent deadlock)
    // Use longer timeout since CUDA operations can take time
    auto status = future.wait_for(std::chrono::milliseconds(5000));
    if (status == std::future_status::timeout) {
        // Remove timed-out job from queue
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (!frame_queue_.empty()) {
                frame_queue_.pop();  // Remove the timed-out job
            }
        }
        return nullptr;
    }
    
    return future.get();
}

// Process frame directly using the detector (same pattern as video_visualize_fixed.cu)
zarray_t* FastAprilTagAlgorithm::processFrameDirect(const cv::Mat& gray_frame, bool mirror) {
    using namespace std::chrono;
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;
    
    auto total_start = high_resolution_clock::now();
    
    last_frame_timing_.reset();
    
    // Validate frame (matching standalone program validation exactly)
    if (gray_frame.empty() || gray_frame.data == nullptr) {
        return nullptr;
    }
    if (gray_frame.cols != width_ || gray_frame.rows != height_) {
        return nullptr;
    }
    if (!gray_frame.isContinuous()) {
        return nullptr;
    }
    if (gray_frame.type() != CV_8UC1) {
        return nullptr;
    }
    
    // Validate GPU detector
    if (!gpu_detector_) {
        return nullptr;
    }
    
    // Stage 1: GPU-only detection (same as video_visualize_fixed.cu)
    // Use frame data directly like standalone program (frame stays in scope during call)
    auto stage1_start = high_resolution_clock::now();
    
    // Check CUDA context before calling (CUDA operations require valid context)
    // Note: CUDA errors don't throw C++ exceptions, so we can't catch them with try/catch
    
    // Clear any previous CUDA errors
    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess && cuda_err != cudaErrorNotReady) {
    }
    
    // Check CUDA device and context before calling
    int current_device = -1;
    cuda_err = cudaGetDevice(&current_device);
    if (cuda_err != cudaSuccess) {
        return nullptr;
    }
    
    // Check CUDA memory info
    size_t free_mem = 0, total_mem = 0;
    cuda_err = cudaMemGetInfo(&free_mem, &total_mem);
    if (cuda_err == cudaSuccess) {
    }
    
    if (gray_frame.data == nullptr) {
        return nullptr;
    }
    
    // Validate frame data is accessible
    if (gray_frame.data == nullptr) {
        return nullptr;
    }
    
    // Validate frame matches detector dimensions
    if (gray_frame.cols != width_ || gray_frame.rows != height_) {
        return nullptr;
    }
    
    // Ensure frame is continuous (required for direct memory access)
    if (!gray_frame.isContinuous()) {
        return nullptr;
    }
    
    // Clear any pending CUDA errors
    cudaGetLastError();
    
    std::cerr.flush();  // Force flush before potentially crashing call
    
    try {
        // Call DetectGpuOnly - this is where the crash happens
        gpu_detector_->DetectGpuOnly(gray_frame.data);
        
        
        // Immediately check for CUDA errors (async operations may have failed)
        cuda_err = cudaGetLastError();
        if (cuda_err != cudaSuccess) {
            return nullptr;
        }
        
        // Verify CUDA context is still valid
        int device_after = -1;
        cuda_err = cudaGetDevice(&device_after);
        if (cuda_err != cudaSuccess) {
            return nullptr;
        }
        
    } catch (const std::exception& e) {
        return nullptr;
    } catch (...) {
        return nullptr;
    }
    
    // The image pointer must stay valid until FitQuads() synchronizes the stream
    // Syncing here too early might interfere with the async operations
    auto stage1_end = high_resolution_clock::now();
    last_frame_timing_.detect_gpu_only_ms = duration<double, std::milli>(stage1_end - stage1_start).count();
    
    // Stage 2: Fit quads
    // This ensures the async memcpy from DetectGpuOnly completes
    // After FitQuads returns, the image pointer is safe to reuse
    auto stage2_start = high_resolution_clock::now();
    
    std::vector<frc971::apriltag::QuadCorners> quads_fullres;
    try {
        quads_fullres = gpu_detector_->FitQuads();
    } catch (const std::exception& e) {
        return nullptr;
    } catch (...) {
        return nullptr;
    }
    
    // This ensures all async operations from DetectGpuOnly are complete
    cuda_err = cudaDeviceSynchronize();
    if (cuda_err != cudaSuccess) {
        return nullptr;
    }
    
    auto stage2_end = high_resolution_clock::now();
    last_frame_timing_.fit_quads_ms = duration<double, std::milli>(stage2_end - stage2_start).count();
    last_frame_timing_.num_quads = static_cast<int>(quads_fullres.size());
    
    // Store quads for debug visualization (before filtering)
    last_frame_quads_ = quads_fullres;
    
    // Stage 3: Mirror handling
    auto stage3_start = high_resolution_clock::now();
    if (mirror) {
        try {
            gpu_detector_->MirrorGrayImageOnGpu();
            mirrorQuadCoordinates(quads_fullres, width_);
        } catch (const std::exception& e) {
            return nullptr;
        } catch (...) {
            return nullptr;
        }
    }
    auto stage3_end = high_resolution_clock::now();
    last_frame_timing_.mirror_ms = duration<double, std::milli>(stage3_end - stage3_start).count();
    
    // Stage 4: Copy gray image to host
    auto stage4_start = high_resolution_clock::now();
    try {
        gpu_detector_->CopyGrayHostTo(gray_host_buffer_);
    } catch (const std::exception& e) {
        return nullptr;
    } catch (...) {
        return nullptr;
    }
    auto stage4_end = high_resolution_clock::now();
    last_frame_timing_.copy_gray_ms = duration<double, std::milli>(stage4_end - stage4_start).count();
    
    // Stage 5: Decode tags
    auto stage5_start = high_resolution_clock::now();
    zarray_t* poly0 = g2d_polygon_create_zeros(4);
    zarray_t* poly1 = g2d_polygon_create_zeros(4);
    zarray_t* detections = zarray_create(sizeof(apriltag_detection_t*));
    
    frc971::apriltag::DecodeTagsFromQuads(
        quads_fullres, gray_host_buffer_.data(), width_, height_,
        td_gpu_, gpu_cam_, gpu_dist_, detections, poly0, poly1);
    
    // Cleanup polygon arrays
    zarray_destroy(poly0);
    zarray_destroy(poly1);
    
    auto stage5_end = high_resolution_clock::now();
    last_frame_timing_.decode_tags_ms = duration<double, std::milli>(stage5_end - stage5_start).count();
    last_frame_timing_.num_detections_before_filter = zarray_size(detections);
    
    // Debug output (comprehensive, not tag-specific)
    int num_quads = static_cast<int>(quads_fullres.size());
    int num_detections = zarray_size(detections);
    double decode_rate = (num_quads > 0) ? (100.0 * num_detections / num_quads) : 0.0;
    std::cerr << "[DEBUG] Decode stage: " << num_quads << " quads -> " << num_detections 
              << " detections (decode_rate=" << std::fixed << std::setprecision(1) << decode_rate << "%)" << std::endl;
    
    if (num_quads > num_detections) {
        int failed_quads = num_quads - num_detections;
        std::cerr << "[DEBUG] WARNING: " << failed_quads 
                  << " quads failed to decode (decode_rate=" << decode_rate << "%)" << std::endl;
    }
    
    // Stage 6: Scale coordinates (if decimation was used)
    auto stage6_start = high_resolution_clock::now();
    if (td_gpu_ && td_gpu_->quad_decimate > 1.0) {
        for (int i = 0; i < zarray_size(detections); i++) {
            apriltag_detection_t* det;
            zarray_get(detections, i, &det);
            scaleDetectionCoordinates(det, td_gpu_->quad_decimate);
        }
    }
    auto stage6_end = high_resolution_clock::now();
    last_frame_timing_.scale_coords_ms = duration<double, std::milli>(stage6_end - stage6_start).count();
    
    // Stage 7: Filter duplicates
    auto stage7_start = high_resolution_clock::now();
    
    // Debug: Log all detections before filtering
    std::vector<int> detected_ids;
    std::vector<double> detected_margins;
    std::vector<Point2f> detected_centers;
    for (int i = 0; i < zarray_size(detections); i++) {
        apriltag_detection_t* det;
        zarray_get(detections, i, &det);
        detected_ids.push_back(det->id);
        detected_margins.push_back(det->decision_margin);
        detected_centers.push_back(Point2f(det->c[0], det->c[1]));
    }
    std::cerr << "[DEBUG] Before filtering: " << zarray_size(detections) << " detections";
    if (!detected_ids.empty()) {
        std::cerr << " - IDs=[";
        for (size_t i = 0; i < detected_ids.size(); i++) {
            std::cerr << detected_ids[i] << "(m=" << std::fixed << std::setprecision(1) << detected_margins[i] 
                      << ", c=(" << std::fixed << std::setprecision(1) << detected_centers[i].x 
                      << "," << detected_centers[i].y << "))";
            if (i < detected_ids.size() - 1) std::cerr << ", ";
        }
        std::cerr << "]";
    }
    std::cerr << std::endl;
    
    std::vector<apriltag_detection_t*> filtered = filterDuplicates(detections, width_, height_, min_distance_);
    
    // Debug: Log filtered detections
    std::vector<int> filtered_ids;
    std::vector<double> filtered_margins;
    for (auto* det : filtered) {
        filtered_ids.push_back(det->id);
        filtered_margins.push_back(det->decision_margin);
    }
    std::cerr << "[DEBUG] After filtering: " << filtered.size() << " detections";
    if (!filtered_ids.empty()) {
        std::cerr << " - IDs=[";
        for (size_t i = 0; i < filtered_ids.size(); i++) {
            std::cerr << filtered_ids[i] << "(m=" << std::fixed << std::setprecision(1) << filtered_margins[i] << ")";
            if (i < filtered_ids.size() - 1) std::cerr << ", ";
        }
        std::cerr << "]";
    }
    std::cerr << std::endl;
    
    // Check if any detections were lost during filtering
    if (zarray_size(detections) > filtered.size()) {
        std::cerr << "[DEBUG] Filtering removed " << (zarray_size(detections) - filtered.size()) 
                  << " duplicate detections" << std::endl;
    }
    
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
    
    // Create new zarray with filtered detections
    zarray_t* result = zarray_create(sizeof(apriltag_detection_t*));
    for (apriltag_detection_t* det : filtered) {
        zarray_add(result, &det);
    }
    
    auto stage7_end = high_resolution_clock::now();
    last_frame_timing_.filter_duplicates_ms = duration<double, std::milli>(stage7_end - stage7_start).count();
    last_frame_timing_.num_detections_after_filter = filtered.size();
    
    // Calculate total time
    auto total_end = high_resolution_clock::now();
    last_frame_timing_.total_ms = duration<double, std::milli>(total_end - total_start).count();
    
    // Update accumulated stats
    accumulated_stats_.detect_gpu_only_ms += last_frame_timing_.detect_gpu_only_ms;
    accumulated_stats_.fit_quads_ms += last_frame_timing_.fit_quads_ms;
    accumulated_stats_.mirror_ms += last_frame_timing_.mirror_ms;
    accumulated_stats_.copy_gray_ms += last_frame_timing_.copy_gray_ms;
    accumulated_stats_.decode_tags_ms += last_frame_timing_.decode_tags_ms;
    accumulated_stats_.scale_coords_ms += last_frame_timing_.scale_coords_ms;
    accumulated_stats_.filter_duplicates_ms += last_frame_timing_.filter_duplicates_ms;
    accumulated_stats_.total_ms += last_frame_timing_.total_ms;
    frame_count_++;
    
    return result;
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

// This isolates CUDA context from Qt event loop, matching standalone's working pattern
void FastAprilTagAlgorithm::cudaWorkerThread() {
    
    // CUDA contexts are thread-local, so we need to ensure proper initialization
    int device_count = 0;
    cudaError_t init_err = cudaGetDeviceCount(&device_count);
    if (init_err != cudaSuccess || device_count == 0) {
        worker_running_ = false;
        return;
    }
    
    // Set device 0 explicitly to ensure context is created
    init_err = cudaSetDevice(0);
    if (init_err != cudaSuccess) {
        worker_running_ = false;
        return;
    }
    
    // Verify device is set
    int current_device = -1;
    init_err = cudaGetDevice(&current_device);
    if (init_err != cudaSuccess || current_device != 0) {
        worker_running_ = false;
        return;
    }
    
    // Create GpuDetector in worker thread (CUDA contexts are thread-local)
    if (!gpu_detector_) {
        try {
            gpu_detector_ = new frc971::apriltag::GpuDetector(width_, height_, td_for_gpu_, gpu_cam_, gpu_dist_);
            
            // Verify CUDA context is still valid after creation
            init_err = cudaGetDevice(&current_device);
            if (init_err != cudaSuccess || current_device != 0) {
            }
        } catch (const std::exception& e) {
            worker_running_ = false;
            return;
        } catch (...) {
            worker_running_ = false;
            return;
        }
    }
    
    while (worker_running_.load()) {
        std::unique_ptr<FrameJob> job;
        
        // Wait for a frame job
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { 
                return !frame_queue_.empty() || !worker_running_.load(); 
            });
            
            if (!worker_running_.load()) {
                break;
            }
            
            if (frame_queue_.empty()) {
                continue;
            }
            
            job = std::move(frame_queue_.front());
            frame_queue_.pop();
        }
        
        int device = -1;
        cudaError_t ctx_err = cudaGetDevice(&device);
        if (ctx_err != cudaSuccess) {
            job->result_promise.set_value(nullptr);
            continue;
        }
        
        // This is critical - if async operations from previous frame are still pending,
        // they can cause crashes when DetectGpuOnly starts new async operations
        cudaError_t pre_sync = cudaDeviceSynchronize();
        if (pre_sync != cudaSuccess) {
        }
        cudaStreamSynchronize(0);  // Also sync default stream
        cudaGetLastError();  // Clear any errors
        
        // Validate gpu_detector_ before processing
        if (!gpu_detector_) {
            job->result_promise.set_value(nullptr);
            continue;
        }
        
        // This resets internal state and might prevent accumulation issues
        static int frame_count_in_worker = 0;
        frame_count_in_worker++;
        if (frame_count_in_worker <= 5) {
            try {
                gpu_detector_->ReinitializeDetections();
            } catch (const std::exception& e) {
            } catch (...) {
            }
        }
        
        // Validate frame
        if (job->frame.empty() || job->frame.data == nullptr) {
            job->result_promise.set_value(nullptr);
            continue;
        }
        
        // Store frame data pointer to verify it stays valid
        uint8_t* frame_data_ptr = job->frame.data;
        size_t frame_size = job->frame.rows * job->frame.cols;
        
        // Process frame in this dedicated CUDA thread (like standalone)
        zarray_t* result = nullptr;
        try {
            // Verify frame data is still valid before processing
            if (job->frame.data != frame_data_ptr || job->frame.data == nullptr) {
                job->result_promise.set_value(nullptr);
                continue;
            }
            
            result = processFrameDirect(job->frame, job->mirror);
            
            // Verify frame data is still valid after processing
            if (job->frame.data != frame_data_ptr) {
            }
        } catch (const std::exception& e) {
            result = nullptr;
        } catch (...) {
            result = nullptr;
        }
        
        // Return result (even if nullptr)
        // Note: zarray_t* is just a pointer, so it's safe to pass through promise
        try {
            job->result_promise.set_value(result);
        } catch (const std::exception& e) {
        } catch (...) {
        }
        
        // GpuDetector uses a single stream, so we need to ensure it's idle
        // Check if there are any pending operations on the default stream
        cudaError_t stream_status = cudaStreamQuery(0);  // Query default stream
        if (stream_status == cudaErrorNotReady) {
        }
        
        // Verify CUDA device is still accessible after processing
        int device_check = -1;
        cudaError_t device_err = cudaGetDevice(&device_check);
        if (device_err != cudaSuccess) {
        } else if (device_check != 0) {
        }
    }
    
}

void FastAprilTagAlgorithm::updateDetectorParameters(
    double quad_decimate,
    double quad_sigma,
    bool refine_edges,
    double decode_sharpening,
    int nthreads,
    int min_cluster_pixels,
    double max_line_fit_mse,
    double critical_angle_degrees,
    int min_white_black_diff) {
    
    // Note: quad_decimate must be 2.0 for CUDA, so we ignore the parameter
    // but we can update other parameters
    
    if (td_for_gpu_) {
        td_for_gpu_->quad_sigma = static_cast<float>(quad_sigma);
        td_for_gpu_->refine_edges = refine_edges ? 1 : 0;
        td_for_gpu_->decode_sharpening = decode_sharpening;
        td_for_gpu_->nthreads = nthreads;
        
        // Update quad threshold parameters
        td_for_gpu_->qtp.min_cluster_pixels = min_cluster_pixels;
        td_for_gpu_->qtp.max_line_fit_mse = static_cast<float>(max_line_fit_mse);
        td_for_gpu_->qtp.cos_critical_rad = cos(critical_angle_degrees * M_PI / 180.0);
        td_for_gpu_->qtp.min_white_black_diff = min_white_black_diff;
        
        // Recreate worker pool if thread count changed
        if (td_for_gpu_->wp) {
            workerpool_destroy(td_for_gpu_->wp);
        }
        td_for_gpu_->wp = workerpool_create(td_for_gpu_->nthreads);
    }
    
    if (td_gpu_) {
        // Sync parameters with GPU detector (for CPU decode stage)
        td_gpu_->quad_decimate = td_for_gpu_->quad_decimate;  // Keep 2.0 for CUDA
        td_gpu_->quad_sigma = td_for_gpu_->quad_sigma;
        td_gpu_->refine_edges = td_for_gpu_->refine_edges;
        td_gpu_->decode_sharpening = td_for_gpu_->decode_sharpening;
        td_gpu_->nthreads = td_for_gpu_->nthreads;
        
        // Update quad threshold parameters
        td_gpu_->qtp.min_cluster_pixels = td_for_gpu_->qtp.min_cluster_pixels;
        td_gpu_->qtp.max_line_fit_mse = td_for_gpu_->qtp.max_line_fit_mse;
        td_gpu_->qtp.cos_critical_rad = td_for_gpu_->qtp.cos_critical_rad;
        td_gpu_->qtp.min_white_black_diff = td_for_gpu_->qtp.min_white_black_diff;
        
        // Recreate worker pool if thread count changed
        if (td_gpu_->wp) {
            workerpool_destroy(td_gpu_->wp);
        }
        td_gpu_->wp = workerpool_create(td_gpu_->nthreads);
    }
    
    std::cerr << "Fast AprilTag: Updated detector parameters - "
              << "quad_sigma=" << quad_sigma
              << ", refine_edges=" << refine_edges
              << ", decode_sharpening=" << decode_sharpening
              << ", nthreads=" << nthreads
              << ", min_cluster_pixels=" << min_cluster_pixels
              << ", max_line_fit_mse=" << max_line_fit_mse
              << ", critical_angle=" << critical_angle_degrees
              << ", min_white_black_diff=" << min_white_black_diff << std::endl;
}

void FastAprilTagAlgorithm::cleanup() {
    
    if (worker_running_.load()) {
        worker_running_ = false;
        queue_cv_.notify_all();  // Wake up worker thread
        
        if (cuda_worker_thread_.joinable()) {
            cuda_worker_thread_.join();
        }
    }
    
    // Clear any remaining jobs
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!frame_queue_.empty()) {
            auto job = std::move(frame_queue_.front());
            frame_queue_.pop();
            job->result_promise.set_value(nullptr);  // Signal failure
        }
    }
    
    // Destroy detector (created in worker thread)
    if (gpu_detector_) {
        delete gpu_detector_;
        gpu_detector_ = nullptr;
    }
    
    // Cleanup reusable buffers
    gray_host_buffer_.clear();
    gray_host_buffer_.shrink_to_fit();
    
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

#endif // HAVE_CUDA_APRILTAG
