#ifndef APRILTAG_ALGORITHM_FACTORY_H
#define APRILTAG_ALGORITHM_FACTORY_H

#include "apriltag_algorithm.h"
#include <memory>

/**
 * Factory for creating AprilTag algorithm instances.
 */
class AprilTagAlgorithmFactory {
public:
    enum AlgorithmType {
        CPU = 0,              // OpenCV CPU (AprilTag)
        CUDA_GPU = 1,         // CUDA GPU (AprilTag) - existing implementation
        FAST_APRILTAG = 2     // Fast AprilTag - from video_visualize_fixed.cu
    };
    
    /**
     * Create an algorithm instance of the specified type.
     * 
     * @param type Algorithm type to create
     * @return Unique pointer to algorithm instance, or nullptr if creation failed
     */
    static std::unique_ptr<AprilTagAlgorithm> create(AlgorithmType type);
    
    /**
     * Get the display name for an algorithm type.
     */
    static std::string getName(AlgorithmType type);
};

#endif // APRILTAG_ALGORITHM_FACTORY_H

