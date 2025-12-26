#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <cmath>

#ifdef HAVE_CUDA_APRILTAG
#include "../src/apriltags_cuda/src/apriltag_gpu.h"
#include "../src/apriltags_cuda/src/apriltag_utils.h"
#endif

extern "C" {
#include "apriltag.h"
#include "tag36h11.h"
#include "common/zarray.h"
}

using namespace cv;
using namespace std;

// Check if detection coordinates are within image bounds
bool isValidDetection(apriltag_detection_t* det, int width, int height) {
    if (det->c[0] < 0 || det->c[0] >= width || det->c[1] < 0 || det->c[1] >= height) {
        return false;
    }
    for (int i = 0; i < 4; i++) {
        if (det->p[i][0] < 0 || det->p[i][0] >= width || 
            det->p[i][1] < 0 || det->p[i][1] >= height) {
            return false;
        }
    }
    return true;
}

// Fast AprilTag filtering (current implementation)
vector<apriltag_detection_t*> filterDuplicatesFast(const zarray_t* detections, int width, int height, double min_distance) {
    vector<apriltag_detection_t*> filtered;
    vector<apriltag_detection_t*> valid_dets;
    
    // First, collect valid detections
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
    
    // Sort by decision_margin (best first)
    sort(valid_dets.begin(), valid_dets.end(),
         [](apriltag_detection_t* a, apriltag_detection_t* b) {
             return a->decision_margin > b->decision_margin;
         });
    
    // Filter duplicates: for detections with same ID, only remove if within min_distance
    for (auto* det : valid_dets) {
        bool is_duplicate = false;
        
        for (auto* existing : filtered) {
            if (existing->id == det->id && existing->family == det->family) {
                double dx = det->c[0] - existing->c[0];
                double dy = det->c[1] - existing->c[1];
                double distance = sqrt(dx * dx + dy * dy);
                
                if (distance < min_distance) {
                    is_duplicate = true;
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

// CPU AprilTag filtering (for comparison)
vector<apriltag_detection_t*> filterDuplicatesCPU(const zarray_t* detections, int width, int height, double min_distance) {
    vector<apriltag_detection_t*> filtered;
    vector<apriltag_detection_t*> valid_dets;
    
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
    
    sort(valid_dets.begin(), valid_dets.end(),
         [](apriltag_detection_t* a, apriltag_detection_t* b) {
             return a->decision_margin > b->decision_margin;
         });
    
    // CPU algorithm doesn't filter by ID - it uses reconcile_detections which checks overlap
    // For this test, we'll just return all valid detections
    return valid_dets;
}

void printDetection(apriltag_detection_t* det, const string& prefix) {
    cout << prefix << "ID: " << det->id 
         << ", Decision Margin: " << fixed << setprecision(2) << det->decision_margin
         << ", Hamming: " << det->hamming
         << ", Center: (" << det->c[0] << ", " << det->c[1] << ")"
         << ", Corners: ";
    for (int i = 0; i < 4; i++) {
        cout << "(" << det->p[i][0] << "," << det->p[i][1] << ") ";
    }
    cout << endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <image_path>" << endl;
        return 1;
    }
    
    string image_path = argv[1];
    Mat frame = imread(image_path, IMREAD_GRAYSCALE);
    
    if (frame.empty()) {
        cerr << "Failed to load image: " << image_path << endl;
        return 1;
    }
    
    cout << "\n========================================" << endl;
    cout << "ANALYZING FRAME: " << image_path << endl;
    cout << "Image size: " << frame.cols << "x" << frame.rows << endl;
    cout << "========================================\n" << endl;
    
    // Ensure frame is contiguous
    if (!frame.isContinuous()) {
        frame = frame.clone();
    }
    
    // Initialize CPU detector
    apriltag_family_t* tf = tag36h11_create();
    apriltag_detector_t* td = apriltag_detector_create();
    apriltag_detector_add_family(td, tf);
    
    td->quad_decimate = 1.0;
    td->quad_sigma = 0.0;
    td->refine_edges = 1;
    td->decode_sharpening = 0.25;
    td->nthreads = 4;
    td->wp = workerpool_create(4);
    
    // Create image_u8_t for CPU detector
    image_u8_t im = {
        .width = frame.cols,
        .height = frame.rows,
        .stride = frame.cols,
        .buf = frame.data
    };
    
    // Run CPU detection
    cout << "=== CPU APRILTAG DETECTION ===" << endl;
    zarray_t* cpu_detections = apriltag_detector_detect(td, &im);
    cout << "CPU detections (before filtering): " << zarray_size(cpu_detections) << endl;
    
    for (int i = 0; i < zarray_size(cpu_detections); i++) {
        apriltag_detection_t* det;
        zarray_get(cpu_detections, i, &det);
        printDetection(det, "  CPU ");
    }
    
    vector<apriltag_detection_t*> cpu_filtered = filterDuplicatesCPU(cpu_detections, frame.cols, frame.rows, 50.0);
    cout << "\nCPU detections (after filtering): " << cpu_filtered.size() << endl;
    for (auto* det : cpu_filtered) {
        printDetection(det, "  CPU Filtered ");
    }
    
#ifdef HAVE_CUDA_APRILTAG
    // Initialize GPU detector
    cout << "\n=== FAST APRILTAG DETECTION ===" << endl;
    
    apriltag_detector_t* td_gpu = apriltag_detector_create();
    apriltag_detector_add_family(td_gpu, tf);
    td_gpu->quad_decimate = 2.0;
    td_gpu->quad_sigma = 0.0;
    td_gpu->refine_edges = 1;
    td_gpu->decode_sharpening = 0.25;
    td_gpu->nthreads = 1;
    td_gpu->wp = workerpool_create(1);
    
    frc971::apriltag::CameraMatrix cam{905.495617, 609.916016, 907.909470, 352.682645};
    frc971::apriltag::DistCoeffs dist{0.0, 0.0, 0.0, 0.0, 0.0};
    
    frc971::apriltag::GpuDetector gpu_detector(frame.cols, frame.rows, td_gpu, cam, dist);
    
    // Run GPU detection
    gpu_detector.DetectGpuOnly(frame.data);
    vector<frc971::apriltag::QuadCorners> quads = gpu_detector.FitQuads();
    
    cout << "Fast detections (quads found): " << quads.size() << endl;
    
    // Decode tags
    zarray_t* gpu_detections = zarray_create(sizeof(apriltag_detection_t*));
    zarray_t* poly0 = g2d_polygon_create_zeros(4);
    zarray_t* poly1 = g2d_polygon_create_zeros(4);
    
    vector<uint8_t> gray_buf(frame.cols * frame.rows);
    gpu_detector.CopyGrayHostTo(gray_buf);
    
    frc971::apriltag::DecodeTagsFromQuads(quads, gray_buf.data(), frame.cols, frame.rows,
                                          td_gpu, cam, dist, gpu_detections, poly0, poly1);
    
    // Scale coordinates (decimation = 2.0)
    for (int i = 0; i < zarray_size(gpu_detections); i++) {
        apriltag_detection_t* det;
        zarray_get(gpu_detections, i, &det);
        double decimate = 2.0;
        for (int j = 0; j < 4; j++) {
            det->p[j][0] = (det->p[j][0] - 0.5) * decimate + 0.5;
            det->p[j][1] = (det->p[j][1] - 0.5) * decimate + 0.5;
        }
        det->c[0] = (det->c[0] - 0.5) * decimate + 0.5;
        det->c[1] = (det->c[1] - 0.5) * decimate + 0.5;
    }
    
    cout << "Fast detections (before filtering): " << zarray_size(gpu_detections) << endl;
    
    for (int i = 0; i < zarray_size(gpu_detections); i++) {
        apriltag_detection_t* det;
        zarray_get(gpu_detections, i, &det);
        printDetection(det, "  Fast ");
    }
    
    vector<apriltag_detection_t*> fast_filtered = filterDuplicatesFast(gpu_detections, frame.cols, frame.rows, 50.0);
    cout << "\nFast detections (after filtering): " << fast_filtered.size() << endl;
    for (auto* det : fast_filtered) {
        printDetection(det, "  Fast Filtered ");
    }
    
    // Calculate distances between detections with same ID
    cout << "\n=== DISTANCE ANALYSIS ===" << endl;
    for (size_t i = 0; i < fast_filtered.size(); i++) {
        for (size_t j = i + 1; j < fast_filtered.size(); j++) {
            if (fast_filtered[i]->id == fast_filtered[j]->id) {
                double dx = fast_filtered[i]->c[0] - fast_filtered[j]->c[0];
                double dy = fast_filtered[i]->c[1] - fast_filtered[j]->c[1];
                double distance = sqrt(dx * dx + dy * dy);
                cout << "Distance between tag ID " << fast_filtered[i]->id 
                     << " detections: " << fixed << setprecision(2) << distance << " pixels" << endl;
            }
        }
    }
    
    zarray_destroy(poly0);
    zarray_destroy(poly1);
    zarray_destroy(gpu_detections);
    apriltag_detector_destroy(td_gpu);
#endif
    
    // Cleanup
    zarray_destroy(cpu_detections);
    apriltag_detector_destroy(td);
    tag36h11_destroy(tf);
    
    cout << "\n========================================" << endl;
    cout << "ANALYSIS COMPLETE" << endl;
    cout << "========================================\n" << endl;
    
    return 0;
}

