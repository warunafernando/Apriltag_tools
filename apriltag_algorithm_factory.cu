#include "apriltag_algorithm_factory.h"
#include "apriltag/common/image_u8.h"
#include <memory>

#ifdef HAVE_CUDA_APRILTAG
#include "fast_apriltag_algorithm.h"
#endif

// Forward declarations for existing algorithm wrappers (to be created)
// For now, we'll create stub implementations

class CpuAprilTagAlgorithm : public AprilTagAlgorithm {
public:
    bool initialize(int width, int height) override { return true; }
    zarray_t* processFrame(const cv::Mat& gray_frame, bool mirror) override { return nullptr; }
    void cleanup() override {}
    std::string getName() const override { return "OpenCV CPU (AprilTag)"; }
    bool requiresCuda() const override { return false; }
private:
    // TODO: Implement wrapper for existing CPU detection
};

class CudaAprilTagAlgorithm : public AprilTagAlgorithm {
public:
    bool initialize(int width, int height) override { return true; }
    zarray_t* processFrame(const cv::Mat& gray_frame, bool mirror) override { return nullptr; }
    void cleanup() override {}
    std::string getName() const override { return "CUDA GPU (AprilTag)"; }
    bool requiresCuda() const override { return true; }
private:
    // TODO: Implement wrapper for existing CUDA detection
};

std::unique_ptr<AprilTagAlgorithm> AprilTagAlgorithmFactory::create(AlgorithmType type) {
    switch (type) {
        case CPU:
            return std::make_unique<CpuAprilTagAlgorithm>();
            
        case CUDA_GPU:
            return std::make_unique<CudaAprilTagAlgorithm>();
            
        case FAST_APRILTAG:
#ifdef HAVE_CUDA_APRILTAG
            return std::make_unique<FastAprilTagAlgorithm>();
#else
            return nullptr;  // CUDA not available
#endif
            
        default:
            return nullptr;
    }
}

std::string AprilTagAlgorithmFactory::getName(AlgorithmType type) {
    switch (type) {
        case CPU:
            return "OpenCV CPU (AprilTag)";
        case CUDA_GPU:
            return "CUDA GPU (AprilTag)";
        case FAST_APRILTAG:
            return "Fast AprilTag";
        default:
            return "Unknown";
    }
}

