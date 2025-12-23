#ifndef APRILTAG_ALGORITHM_H
#define APRILTAG_ALGORITHM_H

#include "apriltag/common/zarray.h"
#include "apriltag/apriltag.h"
#include <opencv2/opencv.hpp>
#include <string>

/**
 * Base class for all AprilTag detection algorithms.
 * 
 * This abstract interface allows the GUI to use different detection
 * algorithms without needing to know their implementation details.
 */
class AprilTagAlgorithm {
public:
    virtual ~AprilTagAlgorithm() = default;
    
    /**
     * Initialize the algorithm with the given frame dimensions.
     * Called once when starting detection.
     * 
     * @param width Frame width in pixels
     * @param height Frame height in pixels
     * @return true if initialization successful, false otherwise
     */
    virtual bool initialize(int width, int height) = 0;
    
    /**
     * Process a single grayscale frame and return detections.
     * Called for each frame in the detection loop.
     * 
     * The caller is responsible for destroying the returned detections array
     * and all apriltag_detection_t pointers within it.
     * 
     * @param gray_frame Input grayscale frame (CV_8UC1), must be continuous
     * @param mirror Whether to apply horizontal mirroring
     * @return zarray_t* containing apriltag_detection_t* pointers, or nullptr on error
     */
    virtual zarray_t* processFrame(const cv::Mat& gray_frame, bool mirror) = 0;
    
    /**
     * Cleanup resources. Called when stopping detection.
     */
    virtual void cleanup() = 0;
    
    /**
     * Get the display name of this algorithm.
     */
    virtual std::string getName() const = 0;
    
    /**
     * Check if this algorithm requires CUDA/GPU support.
     */
    virtual bool requiresCuda() const = 0;
};

#endif // APRILTAG_ALGORITHM_H

