// GPU Detector Wrapper
// This file provides a C++ interface to the CUDA-based GpuDetector
// It will be conditionally compiled only when CUDA is available

#ifdef HAVE_CUDA_APRILTAG

#include "gpu_detector_wrapper.h"
#include "../src/apriltags_cuda/src/apriltag_gpu.h"
#include <apriltag/apriltag.h>
#include <apriltag/tag36h11.h>

namespace {

// Wrapper struct to hold GPU detector (opaque pointer approach)
struct GpuDetectorWrapper {
    frc971::apriltag::GpuDetector* detector;
    apriltag_detector_t* td;
    apriltag_family_t* tf;
};

}  // namespace

extern "C" {

void* gpu_detector_create(int width, int height, 
                          double fx, double fy, double cx, double cy,
                          double k1, double k2, double p1, double p2, double k3) {
    GpuDetectorWrapper* wrapper = new GpuDetectorWrapper;
    
    // Create detector
    wrapper->tf = tag36h11_create();
    wrapper->td = apriltag_detector_create();
    apriltag_detector_add_family(wrapper->td, wrapper->tf);
    wrapper->td->quad_decimate = 2.0;  // CUDA requires 2.0
    wrapper->td->quad_sigma = 0.0;
    wrapper->td->refine_edges = 1;
    wrapper->td->nthreads = 4;
    wrapper->td->wp = workerpool_create(4);
    
    // Create camera matrix and distortion coefficients
    frc971::apriltag::CameraMatrix cam{fx, fy, cx, cy};
    frc971::apriltag::DistCoeffs dist{k1, k2, p1, p2, k3};
    
    // Create GPU detector
    wrapper->detector = new frc971::apriltag::GpuDetector(width, height, wrapper->td, cam, dist);
    
    return wrapper;
}

void gpu_detector_destroy(void* handle) {
    if (!handle) return;
    
    GpuDetectorWrapper* wrapper = static_cast<GpuDetectorWrapper*>(handle);
    
    if (wrapper->detector) {
        delete wrapper->detector;
        wrapper->detector = nullptr;
    }
    if (wrapper->td) {
        apriltag_detector_destroy(wrapper->td);
        wrapper->td = nullptr;
    }
    if (wrapper->tf) {
        tag36h11_destroy(wrapper->tf);
        wrapper->tf = nullptr;
    }
    
    delete wrapper;
}

void gpu_detector_detect(void* handle, const uint8_t* image_data) {
    if (!handle) return;
    
    GpuDetectorWrapper* wrapper = static_cast<GpuDetectorWrapper*>(handle);
    if (wrapper->detector) {
        wrapper->detector->Detect(image_data);
    }
}

const zarray_t* gpu_detector_get_detections(void* handle) {
    if (!handle) return nullptr;
    
    GpuDetectorWrapper* wrapper = static_cast<GpuDetectorWrapper*>(handle);
    if (wrapper->detector) {
        return wrapper->detector->Detections();
    }
    return nullptr;
}

}  // extern "C"

#endif  // HAVE_CUDA_APRILTAG



