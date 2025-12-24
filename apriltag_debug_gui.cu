// AprilTag Detection Debugging GUI Tool
// Displays all debugging stages side by side for two input images
// Allows comparison of different preprocessing and detection stages

#include <opencv2/opencv.hpp>
#include <apriltag/apriltag.h>
#include <apriltag/tag36h11.h>
#include "apriltag_algorithm.h"
#include "apriltag_algorithm_factory.h"
#ifdef HAVE_CUDA_APRILTAG
#include "fast_apriltag_algorithm.h"
#include "cpu_apriltag_algorithm.h"
#endif

// CUDA GPU detector - include directly (file will be compiled as CUDA)
#ifdef HAVE_CUDA_APRILTAG
#include "../src/apriltags_cuda/src/apriltag_gpu.h"
// Forward declare DecodeTagsFromQuads (defined in apriltag_detect.cu)
namespace frc971::apriltag {
void DecodeTagsFromQuads(const std::vector<QuadCorners> &quad_corners,
                         const uint8_t *gray_buf, int width, int height,
                         apriltag_detector_t *td,
                         const CameraMatrix &camera_matrix,
                         const DistCoeffs &distortion_coefficients,
                         zarray_t *detections,
                         zarray_t *poly0,
                         zarray_t *poly1);
}
namespace apriltag_gpu = frc971::apriltag;
#include "g2d.h"  // For g2d_polygon_create_zeros and g2d_polygon_destroy
#endif
#include <apriltag/common/image_u8.h>
#include <apriltag/common/zarray.h>
#include <apriltag/common/workerpool.h>
#include <apriltag/common/g2d.h>

#include <QApplication>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QComboBox>
#include <QGroupBox>
#include <QFileDialog>
#include <QScrollArea>
#include <QMessageBox>
#include <QImage>
#include <QPixmap>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QDebug>
#include <iostream>
#include <cerrno>
#include <cstring>
#include <cstdio>
#include <unistd.h>
#include <fcntl.h>
#include <cstdio>
#include <unistd.h>
#include <fcntl.h>

using std::cerr;
using std::endl;
#include <QSlider>
#include <QSpinBox>
#include <QCheckBox>
#include <QTextEdit>
#include <QSplitter>
#include <QTabWidget>
#include <QTimer>
#include <QSlider>
#include <QSpinBox>
#include <QFormLayout>
#include <QDir>
#include <QDateTime>
#include <QFileInfo>
#include <QLineEdit>
#include <QRadioButton>
#include <QFile>
#include <QTextStream>
#include <QIODevice>

// MindVision SDK (optional)
#ifdef HAVE_MINDVISION_SDK
extern "C" {
#include "CameraApi.h"
}
#endif

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>

using namespace cv;
using namespace std;

// Camera mode structures
struct Mode {
    int width;
    int height;
    double fps;
    string label;
};

struct MVMode {
    int width;
    int height;
    int frame_speed_index;
    string label;
};

// Tag36h11 bit positions (1-indexed, from tag36h11.c)
static const int TAG36H11_BIT_X[36] = {
    1, 2, 3, 4, 5, 2, 3, 4, 3, 6, 6, 6, 6, 6, 5, 5, 5, 4, 6, 5, 4, 3, 2, 5, 4, 3, 4, 1, 1, 1, 1, 1, 2, 2, 2, 3
};
static const int TAG36H11_BIT_Y[36] = {
    1, 1, 1, 1, 1, 2, 2, 2, 3, 1, 2, 3, 4, 5, 2, 3, 4, 3, 6, 6, 6, 6, 6, 5, 5, 5, 4, 6, 5, 4, 3, 2, 5, 4, 3, 4
};

class AprilTagDebugGUI : public QWidget {
    Q_OBJECT

private:
    // All member variables declared early (nvcc requirement)
    // AprilTag detectors
    apriltag_family_t* tf_;
    apriltag_detector_t* td_;
    
    // Images
    Mat image1_, image2_;
    string image1_path_, image2_path_;
    
    // Camera
    VideoCapture cameraCap_;
#ifdef HAVE_MINDVISION_SDK
    CameraHandle mvHandle_;
#else
    void* mvHandle_;
#endif
    bool useMindVision_;
    bool cameraOpen_;
    int selectedCameraIndex_;
    vector<string> cameraList_;
    vector<bool> isMindVision_;
    vector<Mode> v4l2_modes_;
    vector<MVMode> mv_modes_;
    Mat currentFrame_;
    mutex frameMutex_;
    string suggestedCaptureFilename_;
    QTimer *previewTimer_;
    QTimer *calibrationPreviewTimer_;
    
    // Algorithm processing
    bool algorithmRunning_;
    VideoCapture algorithmCamera_;
    bool algorithmUseMindVision_;
    std::unique_ptr<AprilTagAlgorithm> currentAlgorithm_;  // For Fast AprilTag and future algorithms
    std::unique_ptr<AprilTagAlgorithm> captureAlgorithm_;  // For Capture tab preview
#ifdef HAVE_MINDVISION_SDK
    CameraHandle algorithmMvHandle_;
#else
    void* algorithmMvHandle_;
#endif
    std::thread *captureThread_;
    std::thread *processThread_;
    std::thread *detectionThread_;
    std::thread *displayThread_;
    Mat capturedFrame_;
    Mat processedFrame_;
    Mat detectedFrame_;
    std::mutex capturedFrameMutex_;
    std::mutex processedFrameMutex_;
    std::mutex detectedFrameMutex_;
    std::condition_variable capturedFrameReady_;
    std::condition_variable processedFrameReady_;
    std::condition_variable detectedFrameReady_;
    
    // Timing statistics
    double captureTime_;
    double processTime_;
    double detectionTime_;
    double displayTime_;
    double totalTime_;
    int frameCount_;
    double captureFPS_;
    double detectionFPS_;
    double displayFPS_;
    int captureFrameCount_;
    int detectionFrameCount_;
    int displayFrameCount_;
    chrono::high_resolution_clock::time_point captureFPSStart_;
    chrono::high_resolution_clock::time_point detectionFPSStart_;
    chrono::high_resolution_clock::time_point displayFPSStart_;
    
    // Latest detection data
    struct DetectionData {
        int id;
        double decision_margin;
        int hamming;
        Point2f corners[4];
        Point2f center;
    };
    vector<DetectionData> latestDetections_;
    Mat latestDetectedFrame_;
    mutex latestDetectionsMutex_;
    
    // Fisheye undistortion
    Mat fisheye_map1_, fisheye_map2_;
    Mat fisheye_K_, fisheye_D_;
    bool fisheye_undistort_enabled_;
    bool fisheye_calibration_loaded_;
    Size fisheye_image_size_;
    QLabel *fisheyeStatusIndicator_;
    
    // UI elements
    QComboBox *algorithmCombo_;
    QComboBox *algorithmCameraCombo_;
    QPushButton *algorithmStartBtn_;
    QPushButton *algorithmStopBtn_;
    QCheckBox *algorithmMirrorCheckbox_;
    QLabel *algorithmDisplayLabel_;
    QTextEdit *algorithmTimingText_;
    QLabel *algorithmFPSLabel_;
    QTextEdit *algorithmQualityText_;
    QTextEdit *algorithmPoseText_;
    QTextEdit *algorithmDetailedTimingText_;  // For Fast AprilTag detailed timing analysis

public:
    AprilTagDebugGUI(QWidget *parent = nullptr) : QWidget(parent),
        tf_(nullptr),
        td_(nullptr),
        previewTimer_(nullptr),
        calibrationPreviewTimer_(nullptr)
    {
        setupUI();
        
        // Load fisheye calibration on startup
        QString calib_path = "/home/nav/9202/Hiru/Apriltag/calibration_data/camera_params.yaml";
        loadFisheyeCalibration(calib_path);
        
        // Initialize AprilTag detector
        tf_ = tag36h11_create();
        td_ = apriltag_detector_create();
        apriltag_detector_add_family(td_, tf_);
        td_->quad_decimate = 1.0;
        td_->quad_sigma = 0.0;
        td_->refine_edges = 1;
        td_->decode_sharpening = 0.25;
        td_->nthreads = 4;
        td_->wp = workerpool_create(4);
    }

    ~AprilTagDebugGUI() {
        stopAlgorithm();  // Ensure threads are stopped
        apriltag_detector_destroy(td_);
        tag36h11_destroy(tf_);
    }

private slots:
    // Initialize Fast AprilTag algorithm after camera stabilizes (called after 2 second delay)
    void initializeFastAprilTagDetector() {
#ifdef HAVE_CUDA_APRILTAG
        if (!currentAlgorithm_ || !algorithmRunning_) {
            return;  // Algorithm not created or algorithm stopped
        }
        
        // Check if already initialized (processFrame will check this)
        // We'll initialize here with actual frame dimensions
        
        qDebug() << "Starting delayed Fast AprilTag algorithm initialization (2 seconds after camera start)...";
        
        // Get frame dimensions from camera (now it should be stable)
        int width = 1280, height = 1024;
        Mat testFrame;
        bool dimensionsFound = false;
        
        if (algorithmUseMindVision_) {
#ifdef HAVE_MINDVISION_SDK
            tSdkFrameHead frameHead;
            BYTE *pbyBuffer;
            if (CameraGetImageBuffer(algorithmMvHandle_, &frameHead, &pbyBuffer, 1000) == CAMERA_STATUS_SUCCESS) {
                width = frameHead.iWidth;
                height = frameHead.iHeight;
                dimensionsFound = true;
                CameraReleaseImageBuffer(algorithmMvHandle_, pbyBuffer);
                qDebug() << "MindVision camera dimensions:" << width << "x" << height;
            }
#endif
        } else if (algorithmCamera_.isOpened()) {
            algorithmCamera_ >> testFrame;
            if (!testFrame.empty()) {
                width = testFrame.cols;
                height = testFrame.rows;
                dimensionsFound = true;
                qDebug() << "V4L2 camera dimensions:" << width << "x" << height;
            }
        }
        
        // Validate dimensions
        if (width <= 0 || height <= 0 || width > 10000 || height > 10000) {
            qDebug() << "Error: Invalid frame dimensions:" << width << "x" << height;
            return;
        }
        
        // Ensure dimensions are even (required for quad_decimate = 2.0)
        if (width % 2 != 0) width = (width / 2) * 2;
        if (height % 2 != 0) height = (height / 2) * 2;
        
        if (!currentAlgorithm_->initialize(width, height)) {
            qDebug() << "Error: Failed to initialize Fast AprilTag algorithm with dimensions:" << width << "x" << height;
            currentAlgorithm_.reset();
        } else {
            qDebug() << "Fast AprilTag algorithm initialized successfully with dimensions:" << width << "x" << height;
        }
#endif
    }
    
    void loadImage1() {
        QString filename = QFileDialog::getOpenFileName(this, "Load Image 1", "", "Images (*.png *.jpg *.jpeg *.bmp)");
        if (!filename.isEmpty()) {
            image1_ = imread(filename.toStdString(), IMREAD_GRAYSCALE);
            if (!image1_.empty()) {
                image1_path_ = filename.toStdString();
                processImages();
            } else {
                QMessageBox::warning(this, "Error", "Failed to load image 1");
            }
        }
    }

    void loadImage2() {
        QString filename = QFileDialog::getOpenFileName(this, "Load Image 2", "", "Images (*.png *.jpg *.jpeg *.bmp)");
        if (!filename.isEmpty()) {
            image2_ = imread(filename.toStdString(), IMREAD_GRAYSCALE);
            if (!image2_.empty()) {
                image2_path_ = filename.toStdString();
                processImages();
            } else {
                QMessageBox::warning(this, "Error", "Failed to load image 2");
            }
        }
    }

    void stageChanged() {
        processImages();
    }

private slots:
    void setPixmapFromThread(const QPixmap &pixmap) {
        if (!pixmap.isNull() && algorithmDisplayLabel_) {
            algorithmDisplayLabel_->setPixmap(pixmap);
        }
    }
    
    void updateAlgorithmTimingSlot(double fps) {
        updateAlgorithmTiming(fps);
    }
    
    void updateAlgorithmMetricsSlot() {
        updateAlgorithmQualityAndPose();
    }
    
    void updateDetailedTimingSlot(const QString &timingText) {
        if (algorithmDetailedTimingText_) {
            algorithmDetailedTimingText_->setPlainText(timingText);
        }
    }
    
    void onAlgorithmCameraChanged(int index) {
        // Set mirror checkbox to checked by default for MindVision cameras
        if (algorithmMirrorCheckbox_ && index >= 0 && index < (int)isMindVision_.size()) {
            bool isMindVision = isMindVision_[index];
            algorithmMirrorCheckbox_->setChecked(isMindVision);
        }
    }
    
    void onCaptureAlgorithmChanged(int index) {
        // Initialize or cleanup algorithm based on selection
        if (cameraOpen_) {
            initializeCaptureAlgorithm();
        }
    }
    
    void savePatternVisualizations() {
        std::cerr << "=== savePatternVisualizations() CALLED ===" << std::endl;
        std::cerr.flush();
        qDebug() << "=== savePatternVisualizations() CALLED ===";
        qDebug() << "Button clicked, entering function...";
        
        std::lock_guard<std::mutex> lock(storedPatternsMutex_);
        std::cerr << "storedPatterns_.size() = " << storedPatterns_.size() << std::endl;
        std::cerr.flush();
        qDebug() << "savePatternVisualizations: storedPatterns_.size() =" << storedPatterns_.size();
        
        if (storedPatterns_.empty()) {
            QMessageBox::information(this, "Save Patterns", "No patterns to save. Please detect tags first.");
            return;
        }
        
        // Default save directory: output/pattern_visualizations/ (relative to workspace root, not build dir)
        QString workspaceRoot = QDir::currentPath();
        // If we're in build directory, go up to workspace root
        if (workspaceRoot.endsWith("/build") || workspaceRoot.endsWith("/Tools/build")) {
            workspaceRoot = QFileInfo(workspaceRoot).absolutePath();
            if (workspaceRoot.endsWith("/Tools")) {
                workspaceRoot = QFileInfo(workspaceRoot).absolutePath();
            }
        }
        QString defaultDir = workspaceRoot + "/output/pattern_visualizations";
        qDebug() << "Workspace root:" << workspaceRoot;
        qDebug() << "Saving to directory:" << defaultDir;
        
        // Convert to absolute path
        QDir defaultDirObj(defaultDir);
        QString dir = defaultDirObj.absolutePath();
        
        // Create directory if it doesn't exist
        bool dirCreated = QDir().mkpath(dir);
        qDebug() << "Directory creation result:" << dirCreated;
        
        if (!dirCreated) {
            QMessageBox::warning(this, "Error", QString("Failed to create directory: %1").arg(dir));
            return;
        }
        
        // Verify directory exists and is writable
        QDir selectedDir(dir);
        if (!selectedDir.exists()) {
            QMessageBox::warning(this, "Error", QString("Directory does not exist: %1").arg(dir));
            return;
        }
        if (!QFileInfo(dir).isWritable()) {
            QMessageBox::warning(this, "Error", QString("Directory is not writable: %1").arg(dir));
            return;
        }
        
        std::cerr << "Saving patterns to: " << std::string(dir.toLocal8Bit().constData()) << std::endl;
        std::cerr.flush();
        
        // Generate timestamp for filenames
        QDateTime now = QDateTime::currentDateTime();
        QString timestamp = now.toString("yyyyMMdd_hhmmss");
        
        int savedCount = 0;
        for (size_t i = 0; i < storedPatterns_.size(); i++) {
            const StoredPatternData& sp = storedPatterns_[i];
            
            // Create visualization image: warped image on left, pattern on right
            int tagSize = 36;
            int displaySize = 200;  // Size for each part (warped + pattern)
            int spacing = 10;
            int padding = 20;
            int totalWidth = displaySize * 2 + spacing + padding * 2;
            int totalHeight = displaySize + padding * 2 + 40;  // Extra space for header
            
            Mat vis = Mat::ones(totalHeight, totalWidth, CV_8UC3) * 240;
            
            // Header
            stringstream header;
            header << "Tag ID: " << sp.tag_id;
            putText(vis, header.str(), Point(padding, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 2);
            
            // Draw warped image (left side)
            if (!sp.warped_image.empty()) {
                Mat warped_resized;
                cv::resize(sp.warped_image, warped_resized, Size(displaySize, displaySize), 0, 0, INTER_NEAREST);
                Mat warped_bgr;
                if (warped_resized.channels() == 1) {
                    cvtColor(warped_resized, warped_bgr, COLOR_GRAY2BGR);
                } else {
                    warped_bgr = warped_resized;
                }
                warped_bgr.copyTo(vis(Rect(padding, padding + 40, displaySize, displaySize)));
                rectangle(vis, Rect(padding, padding + 40, displaySize, displaySize), Scalar(0, 0, 255), 2);
                putText(vis, "Warped", Point(padding, padding + 35), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
            }
            
            // Draw digitized pattern (right side) - 8x8 grid with border
            // Tag36h11: 8x8 cells total, 1-cell black border, 6x6 data region
            // Scale to 75% to fit width better
            int patternX = padding + displaySize + spacing;
            int cellSize = (int)((displaySize / 8) * 0.75);  // 8x8 grid, 75% of original size
            if (sp.pattern.size() == 6 && sp.pattern[0].size() == 6) {
                // Draw border cells (all black)
                // Top row (row 0)
                for (int c = 0; c < 8; c++) {
                    int y_pos = padding + 40;
                    int x_pos = patternX + c * cellSize;
                    Rect cell(x_pos, y_pos, cellSize, cellSize);
                    rectangle(vis, cell, Scalar(0, 0, 0), -1);  // Black border
                    rectangle(vis, cell, Scalar(128, 128, 128), 1);
                }
                // Bottom row (row 7)
                for (int c = 0; c < 8; c++) {
                    int y_pos = padding + 40 + 7 * cellSize;
                    int x_pos = patternX + c * cellSize;
                    Rect cell(x_pos, y_pos, cellSize, cellSize);
                    rectangle(vis, cell, Scalar(0, 0, 0), -1);  // Black border
                    rectangle(vis, cell, Scalar(128, 128, 128), 1);
                }
                // Left column (col 0, rows 1-6)
                for (int r = 1; r < 7; r++) {
                    int y_pos = padding + 40 + r * cellSize;
                    int x_pos = patternX;
                    Rect cell(x_pos, y_pos, cellSize, cellSize);
                    rectangle(vis, cell, Scalar(0, 0, 0), -1);  // Black border
                    rectangle(vis, cell, Scalar(128, 128, 128), 1);
                }
                // Right column (col 7, rows 1-6)
                for (int r = 1; r < 7; r++) {
                    int y_pos = padding + 40 + r * cellSize;
                    int x_pos = patternX + 7 * cellSize;
                    Rect cell(x_pos, y_pos, cellSize, cellSize);
                    rectangle(vis, cell, Scalar(0, 0, 0), -1);  // Black border
                    rectangle(vis, cell, Scalar(128, 128, 128), 1);
                }
                
                // Draw 6x6 data pattern in center (rows 1-6, columns 1-6)
                for (int r = 0; r < 6; r++) {
                    for (int c = 0; c < 6; c++) {
                        int val = sp.pattern[r][c];
                        bool is_black = val < 128;
                        
                        Scalar color = is_black ? Scalar(0, 0, 0) : Scalar(255, 255, 255);
                        // Map 6x6 pattern (r,c) to 8x8 grid position (r+1, c+1)
                        int y_pos = padding + 40 + (r + 1) * cellSize;
                        int x_pos = patternX + (c + 1) * cellSize;
                        Rect cell(x_pos, y_pos, cellSize, cellSize);
                        rectangle(vis, cell, color, -1);
                        rectangle(vis, cell, Scalar(128, 128, 128), 1);
                    }
                }
                rectangle(vis, Rect(patternX, padding + 40, displaySize, displaySize), Scalar(255, 0, 0), 2);
                putText(vis, "Digitized (8x8: border + data)", Point(patternX, padding + 35), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
            }
            
            // Save image
            QString filename = QString("%1/tag_%2_%3.png").arg(dir).arg(sp.tag_id).arg(timestamp);
            string filename_std = filename.toStdString();
            std::cerr << "Attempting to save pattern to: " << filename_std << std::endl;
            std::cerr.flush();
            qDebug() << "Attempting to save pattern to:" << filename;
            
            bool saved = imwrite(filename_std, vis);
            if (saved) {
                savedCount++;
                std::cerr << "Successfully saved: " << filename_std << std::endl;
                std::cerr.flush();
                qDebug() << "Successfully saved:" << filename;
            } else {
                std::cerr << "FAILED to save: " << filename_std << std::endl;
                std::cerr.flush();
                qDebug() << "Failed to save:" << filename;
            }
        }
        
        std::cerr << "Save complete. savedCount=" << savedCount << std::endl;
        std::cerr.flush();
        qDebug() << "Saved" << savedCount << "pattern visualization(s) to:" << dir;
        // Don't show message box - just log to console to avoid hanging the GUI
    }
    
    void startAlgorithm() {
        cerr << "\n\n=== startAlgorithm() CALLED ===" << endl;
        cerr.flush();
        
        // If already running, stop first (clean restart)
        if (algorithmRunning_) {
            cerr << "Algorithm already running, stopping first..." << endl;
            stopAlgorithm();
            // Give threads a moment to clean up
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // Require camera selection
        int cameraIndex = algorithmCameraCombo_->currentIndex();
        if (cameraIndex < 0 || cameraIndex >= (int)cameraList_.size()) {
            QMessageBox::warning(this, "Error", "Please select a camera");
            return;
        }
        
        algorithmRunning_ = true;
        frameCount_ = 0;
        captureTime_ = processTime_ = detectionTime_ = displayTime_ = totalTime_ = 0.0;
        captureFPS_ = detectionFPS_ = displayFPS_ = 0.0;
        captureFrameCount_ = detectionFrameCount_ = displayFrameCount_ = 0;
        
        // Open camera
        algorithmUseMindVision_ = isMindVision_[cameraIndex];
        
        // Set default settings for MindVision camera: fisheye correction enabled and mirror enabled
        if (algorithmUseMindVision_ && cameraIndex >= 0) {
            if (fisheye_calibration_loaded_ && fisheyeStatusIndicator_) {
                fisheye_undistort_enabled_ = true;
                fisheyeStatusIndicator_->setText("Fisheye Correction: APPLIED");
                fisheyeStatusIndicator_->setStyleSheet("background-color: #90EE90; padding: 5px; border: 1px solid #006400;");
            }
            // Ensure mirror checkbox is checked for MindVision cameras
            if (algorithmMirrorCheckbox_) {
                algorithmMirrorCheckbox_->setChecked(true);
            }
        }
        
        bool cameraOpened = false;
        
        if (algorithmUseMindVision_) {
#ifdef HAVE_MINDVISION_SDK
            // Check if camera is already open in Capture tab - if so, close it first
            // Only close if previewTimer_ exists (it might not exist during GUI initialization)
            if (cameraOpen_ && useMindVision_ && mvHandle_ != 0 && previewTimer_) {
                QMessageBox::information(this, "Info", "Closing camera in Capture tab to use in Algorithms tab");
                closeCamera();
            }
            
            // Open MindVision camera - use same logic as Capture tab
            CameraSdkStatus status = CameraSdkInit(1);
            if (status != CAMERA_STATUS_SUCCESS) {
                QMessageBox::warning(this, "Error", QString("Failed to initialize MindVision SDK (status: %1)").arg(status));
                algorithmRunning_ = false;
                return;
            }
            
            tSdkCameraDevInfo list[16];
            INT count = 16;
            status = CameraEnumerateDevice(list, &count);
            if (status != CAMERA_STATUS_SUCCESS || count == 0) {
                QMessageBox::warning(this, "Error", QString("No MindVision cameras found (status: %1, count: %2)").arg(status).arg(count));
                algorithmRunning_ = false;
                return;
            }
            
            // Find the correct camera index (skip V4L2 cameras)
            int mvIndex = 0;
            for (int i = 0; i < cameraIndex; i++) {
                if (isMindVision_[i]) mvIndex++;
            }
            
            if (mvIndex >= count) {
                QMessageBox::warning(this, "Error", QString("Camera index out of range (mvIndex: %1, available: %2)").arg(mvIndex).arg(count));
                algorithmRunning_ = false;
                return;
            }
            
            status = CameraInit(&list[mvIndex], -1, -1, &algorithmMvHandle_);
            if (status != CAMERA_STATUS_SUCCESS) {
                QString errorMsg = QString("Failed to open MindVision camera (status: %1)\n\n").arg(status);
                if (status == -18) {
                    errorMsg += "Error -18: Camera is already in use.\n";
                    errorMsg += "Please close the camera in the Capture tab first, or close any other application using the camera.";
                } else {
                    errorMsg += "Please check:\n";
                    errorMsg += "1. Camera is connected\n";
                    errorMsg += "2. No other application is using the camera\n";
                    errorMsg += "3. Camera permissions are correct";
                }
                QMessageBox::warning(this, "Error", errorMsg);
                algorithmRunning_ = false;
                return;
            }
            
            tSdkCameraCapbility cap;
            CameraGetCapability(algorithmMvHandle_, &cap);
            CameraSetIspOutFormat(algorithmMvHandle_, CAMERA_MEDIA_TYPE_MONO8);
            
            // Disable auto exposure
            BOOL ae_state = FALSE;
            CameraGetAeState(algorithmMvHandle_, &ae_state);
            if (ae_state) {
                CameraSetAeState(algorithmMvHandle_, FALSE);
            }
            
            CameraPlay(algorithmMvHandle_);
            cameraOpened = true;
            
            // Load and apply saved camera settings
            loadCameraSettingsForAlgorithm();
#endif
        } else if (cameraIndex >= 0) {
            // Open V4L2 camera - extract device number from camera name
            // Camera name format: "V4L2 Camera X" where X is the device number
            string cameraName = cameraList_[cameraIndex];
            size_t pos = cameraName.find_last_of(" ");
            if (pos != string::npos) {
                string deviceNumStr = cameraName.substr(pos + 1);
                try {
                    int deviceNum = stoi(deviceNumStr);
                    algorithmCamera_.open(deviceNum);
                    if (!algorithmCamera_.isOpened()) {
                        // Try opening with CAP_V4L2 explicitly
                        algorithmCamera_.open(deviceNum, CAP_V4L2);
                    }
                    if (algorithmCamera_.isOpened()) {
                        algorithmCamera_.set(CAP_PROP_FRAME_WIDTH, 1280);
                        algorithmCamera_.set(CAP_PROP_FRAME_HEIGHT, 1024);
                        cameraOpened = true;
                        
                        // Load and apply saved camera settings
                        loadCameraSettingsForAlgorithm();
                    } else {
                        qDebug() << "Failed to open V4L2 camera" << deviceNum << "- camera may be in use or not available";
                    }
                } catch (const std::exception& e) {
                    qDebug() << "Error parsing camera device number:" << e.what();
                    cameraOpened = false;
                }
            } else {
                qDebug() << "Invalid camera name format:" << cameraName.c_str();
                cameraOpened = false;
            }
        }
        
        if (!cameraOpened) {
            QString errorMsg = "Failed to open camera";
            if (algorithmUseMindVision_) {
                errorMsg += "\n\nPossible causes:\n";
                errorMsg += "- Camera is already in use (close Capture tab or other applications)\n";
                errorMsg += "- Camera is not connected\n";
                errorMsg += "- Camera driver issue";
            } else {
                errorMsg += "\n\nPossible causes:\n";
                errorMsg += "- Camera is already in use\n";
                errorMsg += "- Camera device not found\n";
                errorMsg += "- Insufficient permissions (try: sudo chmod 666 /dev/video*)";
            }
            QMessageBox::warning(this, "Error", errorMsg);
            algorithmRunning_ = false;
            return;
        }
        
        // Check which algorithm is selected
        int algorithmIndex = algorithmCombo_->currentIndex();
        
        // Initialize algorithm class for CPU (index 0) or Fast AprilTag (index 1)
        if (algorithmIndex == 0) {
            // CPU algorithm
            AprilTagAlgorithmFactory::AlgorithmType algoType = AprilTagAlgorithmFactory::CPU;
            currentAlgorithm_ = AprilTagAlgorithmFactory::create(algoType);
            if (!currentAlgorithm_) {
                QMessageBox::warning(this, "Error", "Failed to create CPU algorithm.");
                algorithmRunning_ = false;
                return;
            }
            
            // Get frame dimensions from camera
            int width = 1280, height = 1024;
            if (algorithmUseMindVision_) {
#ifdef HAVE_MINDVISION_SDK
                width = 1280;
                height = 1024;
#endif
            } else if (algorithmCamera_.isOpened()) {
                width = static_cast<int>(algorithmCamera_.get(CAP_PROP_FRAME_WIDTH));
                height = static_cast<int>(algorithmCamera_.get(CAP_PROP_FRAME_HEIGHT));
            }
            
            // Initialize immediately (CPU doesn't need delay)
            if (!currentAlgorithm_->initialize(width, height)) {
                QMessageBox::warning(this, "Error", QString("Failed to initialize CPU algorithm with dimensions %1x%2").arg(width).arg(height));
                currentAlgorithm_.reset();
                algorithmRunning_ = false;
                return;
            }
            qDebug() << "CPU algorithm initialized successfully with dimensions:" << width << "x" << height;
        } else if (algorithmIndex == 1) {
#ifdef HAVE_CUDA_APRILTAG
            AprilTagAlgorithmFactory::AlgorithmType algoType = AprilTagAlgorithmFactory::FAST_APRILTAG;
            currentAlgorithm_ = AprilTagAlgorithmFactory::create(algoType);
            if (!currentAlgorithm_) {
                QMessageBox::warning(this, "Error", "Failed to create Fast AprilTag algorithm. CUDA support required.");
                algorithmRunning_ = false;
                return;
            }
            
            // Get frame dimensions from camera
            int width = 1280, height = 1024;  // Default, will be updated when camera opens
            // Try to get dimensions from camera
            if (algorithmUseMindVision_) {
#ifdef HAVE_MINDVISION_SDK
                // Dimensions will be set when camera opens
                width = 1280;
                height = 1024;
#endif
            } else if (algorithmCamera_.isOpened()) {
                width = static_cast<int>(algorithmCamera_.get(CAP_PROP_FRAME_WIDTH));
                height = static_cast<int>(algorithmCamera_.get(CAP_PROP_FRAME_HEIGHT));
            }
            
            // Delay initialization until camera stabilizes
            // Store dimensions for later initialization
            // Will be initialized in initializeFastAprilTagDetector() slot after 2 seconds
            qDebug() << "Fast AprilTag algorithm will initialize after 2 seconds with dimensions:" << width << "x" << height;
#else
            QMessageBox::warning(this, "Error", "Fast AprilTag requires CUDA support. Please compile with CUDA.");
            algorithmRunning_ = false;
            return;
#endif
        } else {
            QMessageBox::warning(this, "Error", QString("Unknown algorithm index: %1").arg(algorithmIndex));
            algorithmRunning_ = false;
            return;
        }
        
        // Start threads with error handling
        cerr << "=== Starting threads ===" << endl;
        try {
            cerr << "Creating capture thread..." << endl;
            captureThread_ = new std::thread(&AprilTagDebugGUI::captureThreadFunction, this);
            cerr << "Capture thread created successfully" << endl;
            
            cerr << "Creating process thread..." << endl;
            processThread_ = new std::thread(&AprilTagDebugGUI::processThreadFunction, this);
            cerr << "Process thread created successfully" << endl;
            
            cerr << "Creating detection thread..." << endl;
            detectionThread_ = new std::thread(&AprilTagDebugGUI::detectionThreadFunction, this);
            cerr << "Detection thread created successfully" << endl;
            
            cerr << "Creating display thread..." << endl;
            displayThread_ = new std::thread(&AprilTagDebugGUI::displayThreadFunction, this);
            cerr << "Display thread created successfully" << endl;
            
            cerr << "All threads started successfully!" << endl;
        } catch (const std::exception& e) {
            cerr << "ERROR: Exception starting threads: " << e.what() << endl;
            QMessageBox::critical(this, "Thread Error", 
                QString("Failed to start threads:\n%1").arg(e.what()));
            // Cleanup if threads failed
            stopAlgorithm();
            return;
        } catch (...) {
            cerr << "ERROR: Unknown exception starting threads" << endl;
            QMessageBox::critical(this, "Thread Error", "Unknown error starting threads");
            stopAlgorithm();
            return;
        }
        
        algorithmStartBtn_->setEnabled(false);
        algorithmStopBtn_->setEnabled(true);
        
        // Schedule delayed initialization after 2 seconds for Fast AprilTag
#ifdef HAVE_CUDA_APRILTAG
        if (algorithmIndex == 1 && currentAlgorithm_) {
            // Fast AprilTag - delay for camera stabilization
            QTimer::singleShot(2000, this, &AprilTagDebugGUI::initializeFastAprilTagDetector);
            qDebug() << "Scheduled Fast AprilTag detector initialization in 2 seconds";
        }
#endif
    }
    
    void stopAlgorithm() {
        if (!algorithmRunning_) return;
        
        algorithmRunning_ = false;
        
        // Wake up all threads first (don't cleanup algorithm yet - threads might still use it)
        capturedFrameReady_.notify_all();
        processedFrameReady_.notify_all();
        detectedFrameReady_.notify_all();
        
        // Wait for threads to finish (they will destroy any pending detections first)
        if (captureThread_ && captureThread_->joinable()) {
            captureThread_->join();
            delete captureThread_;
            captureThread_ = nullptr;
        }
        if (processThread_ && processThread_->joinable()) {
            processThread_->join();
            delete processThread_;
            processThread_ = nullptr;
        }
        if (detectionThread_ && detectionThread_->joinable()) {
            detectionThread_->join();
            delete detectionThread_;
            detectionThread_ = nullptr;
        }
        if (displayThread_ && displayThread_->joinable()) {
            displayThread_->join();
            delete displayThread_;
            displayThread_ = nullptr;
        }
        
        // NOW cleanup algorithm class (after all threads have finished and destroyed their detections)
        if (currentAlgorithm_) {
            currentAlgorithm_->cleanup();
            currentAlgorithm_.reset();
        }
        
        // Close camera
        if (algorithmUseMindVision_) {
#ifdef HAVE_MINDVISION_SDK
            if (algorithmMvHandle_ != 0) {
                CameraStop(algorithmMvHandle_);
                CameraUnInit(algorithmMvHandle_);
                algorithmMvHandle_ = 0;
            }
#endif
        } else {
            if (algorithmCamera_.isOpened()) {
                algorithmCamera_.release();
            }
        }
        
        // Clean up stored detections
        {
            std::unique_lock<std::mutex> lock(latestDetectionsMutex_);
            latestDetections_.clear();
            latestDetectedFrame_ = Mat();
        }
        
        algorithmStartBtn_->setEnabled(true);
        algorithmStopBtn_->setEnabled(false);
        algorithmDisplayLabel_->setText("Stopped");
    }

private:
    // Thread functions for algorithm processing pipeline
    void captureThreadFunction() {
        cerr << "=== Capture thread STARTED ===" << endl;
        cerr.flush();
        
        while (algorithmRunning_) {
            auto start = chrono::high_resolution_clock::now();
            Mat frame;
            
            if (algorithmUseMindVision_) {
#ifdef HAVE_MINDVISION_SDK
                tSdkFrameHead frameHead;
                BYTE *pbyBuffer;
                if (CameraGetImageBuffer(algorithmMvHandle_, &frameHead, &pbyBuffer, 1000) == CAMERA_STATUS_SUCCESS) {
                    Mat temp(frameHead.iHeight, frameHead.iWidth, CV_8UC1, pbyBuffer);
                    frame = temp.clone();
                    CameraReleaseImageBuffer(algorithmMvHandle_, pbyBuffer);
                }
#endif
            } else {
                if (algorithmCamera_.isOpened()) {
                    algorithmCamera_ >> frame;
                }
            }
            
            auto end = chrono::high_resolution_clock::now();
            captureTime_ = chrono::duration<double, milli>(end - start).count();
            captureFrameCount_++;
            
            // Update capture FPS every 30 frames
            if (captureFrameCount_ % 30 == 0) {
                auto now = chrono::high_resolution_clock::now();
                double elapsed = chrono::duration<double>(now - captureFPSStart_).count();
                if (elapsed > 0) {
                    captureFPS_ = 30.0 / elapsed;
                    captureFPSStart_ = now;
                }
            }
            
            if (!frame.empty() && algorithmRunning_) {
                // Don't apply mirroring here - it's done algorithm-specific in detection thread
                // (CUDA: after GPU detection on gray_host/quads, CPU: before detection on gray frame)
                std::unique_lock<std::mutex> lock(capturedFrameMutex_);
                capturedFrame_ = frame.clone();
                lock.unlock();
                capturedFrameReady_.notify_one();
            }
            
            // Small delay to prevent CPU spinning
            this_thread::sleep_for(chrono::milliseconds(1));
        }
    }
    
    void processThreadFunction() {
        while (algorithmRunning_) {
            std::unique_lock<std::mutex> lock(capturedFrameMutex_);
            capturedFrameReady_.wait(lock, [this] { return !capturedFrame_.empty() || !algorithmRunning_; });
            
            if (!algorithmRunning_) break;
            
            Mat frame = capturedFrame_.clone();
            capturedFrame_ = Mat();  // Clear
            lock.unlock();
            
            if (frame.empty()) continue;
            
            auto start = chrono::high_resolution_clock::now();
            
            // Video processing: Apply fisheye correction if enabled
            Mat processed = frame.clone();
            if (fisheye_undistort_enabled_ && fisheye_calibration_loaded_) {
                processed = undistortFrame(frame);
            }
            
            auto end = chrono::high_resolution_clock::now();
            processTime_ = chrono::duration<double, milli>(end - start).count();
            
            if (algorithmRunning_) {
                std::unique_lock<std::mutex> lock2(processedFrameMutex_);
                processedFrame_ = processed.clone();
                lock2.unlock();
                processedFrameReady_.notify_one();
            }
        }
    }
    
    void detectionThreadFunction() {
        detectionFPSStart_ = chrono::high_resolution_clock::now();
        
        // Overhead timing accumulators (for Fast AprilTag only)
        static double overhead_frame_clone_ms = 0.0;
        static double overhead_grayscale_convert_ms = 0.0;
        static double overhead_store_data_ms = 0.0;
        static double overhead_draw_ms = 0.0;
        static double overhead_frame_copy_ms = 0.0;
        static int overhead_frame_count = 0;
        
        while (algorithmRunning_) {
            std::unique_lock<std::mutex> lock(processedFrameMutex_);
            processedFrameReady_.wait(lock, [this] { return !processedFrame_.empty() || !algorithmRunning_; });
            
            if (!algorithmRunning_) break;
            
            // Measure frame cloning time
            auto clone_start = chrono::high_resolution_clock::now();
            Mat frame = processedFrame_.clone();
            processedFrame_ = Mat();  // Clear
            lock.unlock();
            auto clone_end = chrono::high_resolution_clock::now();
            
            if (frame.empty()) continue;
            
            auto start = chrono::high_resolution_clock::now();
            
            // Measure grayscale conversion time
            auto convert_start = chrono::high_resolution_clock::now();
            Mat gray;
            if (frame.type() != CV_8UC1) {
                if (frame.channels() == 2) {
                    cvtColor(frame, gray, COLOR_YUV2GRAY_YUY2);
                } else if (frame.channels() == 3) {
                    cvtColor(frame, gray, COLOR_BGR2GRAY);
                } else {
                    gray = frame.clone();
                }
            } else {
                // Already grayscale - must be contiguous for Fast AprilTag
                // Always clone to ensure we have our own copy with guaranteed lifetime
                gray = frame.clone();
            }
            auto convert_end = chrono::high_resolution_clock::now();
            
            zarray_t *detections = nullptr;
            
            // Use algorithm class if available (Fast AprilTag)
            if (currentAlgorithm_) {
                bool mirror = algorithmMirrorCheckbox_ && algorithmMirrorCheckbox_->isChecked();
                detections = currentAlgorithm_->processFrame(gray, mirror);
                
                // Accumulate overhead timing for Fast AprilTag
                overhead_frame_clone_ms += chrono::duration<double, milli>(clone_end - clone_start).count();
                overhead_grayscale_convert_ms += chrono::duration<double, milli>(convert_end - convert_start).count();
                overhead_frame_count++;
                
                if (!detections) {
                    detections = zarray_create(sizeof(apriltag_detection_t*));
                }
                
                // Update detailed timing analysis in GUI periodically (every 10 frames for smooth updates)
                static int timing_update_counter = 0;
                if (++timing_update_counter >= 10) {
                    timing_update_counter = 0;
                    std::ostringstream oss;
                    bool algo_found = false;
                    double algo_total_ms = 0.0;
                    
                    // Try to cast to CpuAprilTagAlgorithm first
                    CpuAprilTagAlgorithm* cpu_algo = dynamic_cast<CpuAprilTagAlgorithm*>(currentAlgorithm_.get());
                    if (cpu_algo) {
                        algo_found = true;
                        std::string timing_report = cpu_algo->getTimingReport();
                        oss << timing_report;
                        algo_total_ms = cpu_algo->getLastFrameTiming().total_ms;
                    }
#ifdef HAVE_CUDA_APRILTAG
                    // Try to cast to FastAprilTagAlgorithm
                    else {
                        FastAprilTagAlgorithm* fast_algo = dynamic_cast<FastAprilTagAlgorithm*>(currentAlgorithm_.get());
                        if (fast_algo) {
                            algo_found = true;
                            std::string timing_report = fast_algo->getTimingReport();
                            oss << timing_report;
                            algo_total_ms = fast_algo->getLastFrameTiming().total_ms;
                        }
                    }
#endif
                    
                    if (algo_found) {
                        // Add detailed detection thread overhead info
                        oss << "\n--- Detection Thread Overhead (Detailed) ---\n";
                        oss << std::fixed << std::setprecision(3);
                        if (overhead_frame_count > 0) {
                            double avg_clone = overhead_frame_clone_ms / overhead_frame_count;
                            double avg_convert = overhead_grayscale_convert_ms / overhead_frame_count;
                            double avg_store = overhead_store_data_ms / overhead_frame_count;
                            double avg_draw = overhead_draw_ms / overhead_frame_count;
                            double avg_copy = overhead_frame_copy_ms / overhead_frame_count;
                            double total_overhead = avg_clone + avg_convert + avg_store + avg_draw + avg_copy;
                            
                            oss << "  Frame Clone:           " << std::setw(8) << avg_clone << " ms\n";
                            oss << "  Grayscale Convert:     " << std::setw(8) << avg_convert << " ms\n";
                            oss << "  Store Detection Data:  " << std::setw(8) << avg_store << " ms\n";
                            oss << "  Draw Detections:       " << std::setw(8) << avg_draw << " ms\n";
                            oss << "  Frame Copy (output):   " << std::setw(8) << avg_copy << " ms\n";
                            oss << "  ---------------------------------\n";
                            oss << "  Total Overhead:        " << std::setw(8) << total_overhead << " ms\n";
                            oss << "  Algorithm Time:        " << std::setw(8) << algo_total_ms << " ms\n";
                            oss << "  Detection Thread Time: " << std::setw(8) << detectionTime_ << " ms\n";
                            
                            double detection_time_ms = detectionTime_;
                            if (detection_time_ms > 0.0) {
                                double overhead_percent = 100.0 * total_overhead / detection_time_ms;
                                oss << "  Overhead %:            " << std::setw(8) << overhead_percent << "%\n";
                            }
                        }
                        
                        // Update GUI in main thread (thread-safe)
                        QMetaObject::invokeMethod(this, "updateDetailedTimingSlot", Qt::QueuedConnection, 
                                                Q_ARG(QString, QString::fromStdString(oss.str())));
                        
                        // Reset counters
                        overhead_frame_clone_ms = 0.0;
                        overhead_grayscale_convert_ms = 0.0;
                        overhead_store_data_ms = 0.0;
                        overhead_draw_ms = 0.0;
                        overhead_frame_copy_ms = 0.0;
                        overhead_frame_count = 0;
                    }
                }
            }
            detectionFrameCount_++;
            
            // Update detection FPS every 30 frames
            if (detectionFrameCount_ % 30 == 0) {
                auto now = chrono::high_resolution_clock::now();
                double elapsed = chrono::duration<double>(now - detectionFPSStart_).count();
                if (elapsed > 0) {
                    detectionFPS_ = 30.0 / elapsed;
                    detectionFPSStart_ = now;
                }
            }
            
            // Store detections for quality/pose analysis (store only needed data)
            // Do this BEFORE destroying detections
            {
                std::unique_lock<std::mutex> lock(latestDetectionsMutex_);
                latestDetections_.clear();
                
                // Extract and store detection data
                if (detections) {  // Safety check
                    for (int i = 0; i < zarray_size(detections); i++) {
                        apriltag_detection_t *det;
                        zarray_get(detections, i, &det);
                        
                        if (!det) continue;  // Safety check
                        
                        DetectionData det_data;
                        det_data.id = det->id;
                        det_data.decision_margin = det->decision_margin;
                        det_data.hamming = det->hamming;
                        det_data.corners[0] = Point2f(det->p[0][0], det->p[0][1]);
                        det_data.corners[1] = Point2f(det->p[1][0], det->p[1][1]);
                        det_data.corners[2] = Point2f(det->p[2][0], det->p[2][1]);
                        det_data.corners[3] = Point2f(det->p[3][0], det->p[3][1]);
                        det_data.center = Point2f(det->c[0], det->c[1]);
                        
                        latestDetections_.push_back(det_data);
                    }
                }
            }
            
            // Draw detections on frame
            // Apply mirror to display frame if mirroring was used (so coordinates match)
            auto draw_start = chrono::high_resolution_clock::now();
            Mat detected;
            bool use_mirror_display = (algorithmMirrorCheckbox_ && algorithmMirrorCheckbox_->isChecked());
            if (use_mirror_display) {
                Mat temp_frame = frame.clone();
                if (temp_frame.channels() == 1) {
                    cvtColor(temp_frame, detected, COLOR_GRAY2BGR);
                } else {
                    detected = temp_frame.clone();
                }
                flip(detected, detected, 1);  // Mirror display to match coordinates
            } else {
                detected = frame.clone();
                if (frame.channels() == 1) {
                    cvtColor(frame, detected, COLOR_GRAY2BGR);
                }
            }
            
            int num_detections_drawn = 0;
            if (detections) {  // Safety check
                for (int i = 0; i < zarray_size(detections); i++) {
                    apriltag_detection_t *det;
                    zarray_get(detections, i, &det);
                    
                    if (!det) continue;  // Safety check
                    
                    // Validate coordinates are within frame bounds
                    bool valid = true;
                    for (int j = 0; j < 4; j++) {
                        if (det->p[j][0] < 0 || det->p[j][0] >= detected.cols ||
                            det->p[j][1] < 0 || det->p[j][1] >= detected.rows) {
                            valid = false;
                            break;
                        }
                    }
                    if (det->c[0] < 0 || det->c[0] >= detected.cols ||
                        det->c[1] < 0 || det->c[1] >= detected.rows) {
                        valid = false;
                    }
                    
                    if (valid) {
                        // Draw quad
                        line(detected, Point((int)det->p[0][0], (int)det->p[0][1]), 
                             Point((int)det->p[1][0], (int)det->p[1][1]), Scalar(0, 255, 0), 2);
                        line(detected, Point((int)det->p[1][0], (int)det->p[1][1]), 
                             Point((int)det->p[2][0], (int)det->p[2][1]), Scalar(0, 255, 0), 2);
                        line(detected, Point((int)det->p[2][0], (int)det->p[2][1]), 
                             Point((int)det->p[3][0], (int)det->p[3][1]), Scalar(0, 255, 0), 2);
                        line(detected, Point((int)det->p[3][0], (int)det->p[3][1]), 
                             Point((int)det->p[0][0], (int)det->p[0][1]), Scalar(0, 255, 0), 2);
                        
                        // Draw ID
                        putText(detected, to_string(det->id), Point((int)det->c[0], (int)det->c[1]), 
                               FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
                        num_detections_drawn++;
                    } else {
                        // Log invalid coordinates for debugging
                        qDebug() << "Detection" << i << "has invalid coordinates - frame:" 
                                 << detected.cols << "x" << detected.rows
                                 << "corners:" << det->p[0][0] << "," << det->p[0][1] << "...";
                    }
                    
                    // Destroy detections (all algorithms own their detections)
                    if (currentAlgorithm_) {
                        apriltag_detection_destroy(det);
                    }
                }
            }
            auto draw_end = chrono::high_resolution_clock::now();
            
            // Accumulate overhead timing for Fast AprilTag
            if (currentAlgorithm_) {
                overhead_draw_ms += chrono::duration<double, milli>(draw_end - draw_start).count();
            }
            
            // Destroy detections array (all algorithms own their detections)
            if (detections) {
                zarray_destroy(detections);
            }
            
            // Measure frame copying time
            auto copy_start = chrono::high_resolution_clock::now();
            // Store latest frame for pose/quality display
            {
                std::unique_lock<std::mutex> lock(latestDetectionsMutex_);
                latestDetectedFrame_ = detected.clone();
            }
            
            if (algorithmRunning_) {
                std::unique_lock<std::mutex> lock2(detectedFrameMutex_);
                detectedFrame_ = detected.clone();
                lock2.unlock();
                detectedFrameReady_.notify_one();
            }
            auto copy_end = chrono::high_resolution_clock::now();
            
            // Accumulate overhead timing for Fast AprilTag
            if (currentAlgorithm_) {
                overhead_frame_copy_ms += chrono::duration<double, milli>(copy_end - copy_start).count();
            }
            
            auto end = chrono::high_resolution_clock::now();
            detectionTime_ = chrono::duration<double, milli>(end - start).count();
        }
    }
    
    void displayThreadFunction() {
        displayFPSStart_ = chrono::high_resolution_clock::now();
        int display_count = 0;
        while (algorithmRunning_) {
            std::unique_lock<std::mutex> lock(detectedFrameMutex_);
            detectedFrameReady_.wait(lock, [this] { return !detectedFrame_.empty() || !algorithmRunning_; });
            
            if (!algorithmRunning_) break;
            
            Mat frame = detectedFrame_.clone();
            detectedFrame_ = Mat();  // Clear
            lock.unlock();
            
            if (frame.empty()) {
                // Debug: Uncomment to debug empty frames
                // if (display_count % 100 == 0) {
                //     qDebug() << "Display thread: Received empty frame";
                // }
                continue;
            }
            
            display_count++;
            // Debug: Uncomment to debug frame display
            // if (display_count % 30 == 0) {
            //     qDebug() << "Display thread: Displaying frame" << display_count << "size:" << frame.cols << "x" << frame.rows;
            // }
            
            auto start = chrono::high_resolution_clock::now();
            
            // Convert to QPixmap and display
            Mat display;
            if (frame.channels() == 1) {
                cvtColor(frame, display, COLOR_GRAY2RGB);
            } else {
                cvtColor(frame, display, COLOR_BGR2RGB);
            }
            
            QImage qimg(display.data, display.cols, display.rows, display.step, QImage::Format_RGB888);
            QPixmap pixmap = QPixmap::fromImage(qimg.copy()).scaled(
                algorithmDisplayLabel_->width(), algorithmDisplayLabel_->height(),
                Qt::KeepAspectRatio, Qt::SmoothTransformation);
            
            // Update UI in main thread using slot (thread-safe)
            QMetaObject::invokeMethod(this, "setPixmapFromThread", Qt::QueuedConnection, Q_ARG(QPixmap, pixmap));
            
            // Update timing statistics
            auto end = chrono::high_resolution_clock::now();
            displayTime_ = chrono::duration<double, milli>(end - start).count();
            displayFrameCount_++;
            
            // Update display FPS every 30 frames
            if (displayFrameCount_ % 30 == 0) {
                auto now = chrono::high_resolution_clock::now();
                double elapsed = chrono::duration<double>(now - displayFPSStart_).count();
                if (elapsed > 0) {
                    displayFPS_ = 30.0 / elapsed;
                    displayFPSStart_ = now;
                }
            }
            
            frameCount_++;
            totalTime_ = captureTime_ + processTime_ + detectionTime_ + displayTime_;
            double fps = 1000.0 / totalTime_;
            
            // Update all displays - use QMetaObject::invokeMethod for thread-safe UI update
            QMetaObject::invokeMethod(this, "updateAlgorithmTimingSlot", Qt::QueuedConnection, Q_ARG(double, fps));
            QMetaObject::invokeMethod(this, "updateAlgorithmMetricsSlot", Qt::QueuedConnection);
        }
    }
    
    void updateAlgorithmTiming(double fps) {
        if (!algorithmFPSLabel_) return;
        
        // Display all three FPS values in the label: Capture, Detection, Display
        algorithmFPSLabel_->setText(QString("FPS: Capture: %1, Detection: %2, Display: %3")
            .arg(captureFPS_, 0, 'f', 1)
            .arg(detectionFPS_, 0, 'f', 1)
            .arg(displayFPS_, 0, 'f', 1));
    }
    
    void updateAlgorithmQualityAndPose() {
        std::unique_lock<std::mutex> lock(latestDetectionsMutex_);
        
        // Update quality metrics
        if (algorithmQualityText_) {
            QString qualityText;
            
            if (latestDetections_.empty()) {
                qualityText += "No detections";
            } else {
                qualityText += QString("Tags detected: %1\n\n").arg(latestDetections_.size());
                for (size_t i = 0; i < latestDetections_.size() && i < 5; i++) {  // Show max 5 tags
                    const DetectionData& det = latestDetections_[i];
                    qualityText += QString("Tag ID: %1\n").arg(det.id);
                    qualityText += QString("  Decision Margin: %1\n").arg(det.decision_margin, 0, 'f', 2);
                    qualityText += QString("  Hamming: %1\n\n").arg(det.hamming);
                }
                if (latestDetections_.size() > 5) {
                    qualityText += QString("... and %1 more\n").arg(latestDetections_.size() - 5);
                }
            }
            algorithmQualityText_->setPlainText(qualityText);
        }
        
        // Update pose estimation
        if (algorithmPoseText_ && fisheye_calibration_loaded_) {
            QString poseText;
            if (latestDetections_.empty()) {
                poseText = "No pose data\n(No detections)";
            } else {
                // Use first detection for pose (or can show multiple)
                const DetectionData& det = latestDetections_[0];
                
                // Tag size (in meters) - default 0.1m, can be made configurable
                double tagSize = 0.1;  // 10cm tag
                
                // 3D object points (tag corners in tag coordinate system)
                // Tag36h11: corners at (-tagSize/2, -tagSize/2, 0), (tagSize/2, -tagSize/2, 0), etc.
                vector<Point3f> objectPoints;
                objectPoints.push_back(Point3f(-tagSize/2, -tagSize/2, 0));
                objectPoints.push_back(Point3f( tagSize/2, -tagSize/2, 0));
                objectPoints.push_back(Point3f( tagSize/2,  tagSize/2, 0));
                objectPoints.push_back(Point3f(-tagSize/2,  tagSize/2, 0));
                
                // 2D image points (detected corners)
                vector<Point2f> imagePoints;
                imagePoints.push_back(det.corners[0]);
                imagePoints.push_back(det.corners[1]);
                imagePoints.push_back(det.corners[2]);
                imagePoints.push_back(det.corners[3]);
                
                // Solve PnP
                Mat rvec, tvec;
                Mat cameraMatrix = fisheye_K_;
                Mat distCoeffs = fisheye_D_;
                
                // For fisheye, use fisheye::solvePnP if available, otherwise regular solvePnP
                bool success = solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, SOLVEPNP_IPPE);
                
                if (success) {
                    // Convert rotation vector to rotation matrix, then to Euler angles
                    Mat rmat;
                    Rodrigues(rvec, rmat);
                    
                    // Extract translation (in meters)
                    double tx = tvec.at<double>(0);
                    double ty = tvec.at<double>(1);
                    double tz = tvec.at<double>(2);
                    
                    // Extract Euler angles (ZYX convention)
                    double sy = sqrt(rmat.at<double>(0,0) * rmat.at<double>(0,0) + rmat.at<double>(1,0) * rmat.at<double>(1,0));
                    bool singular = sy < 1e-6;
                    double rx, ry, rz;
                    if (!singular) {
                        rx = atan2(rmat.at<double>(2,1), rmat.at<double>(2,2));
                        ry = atan2(-rmat.at<double>(2,0), sy);
                        rz = atan2(rmat.at<double>(1,0), rmat.at<double>(0,0));
                    } else {
                        rx = atan2(-rmat.at<double>(1,2), rmat.at<double>(1,1));
                        ry = atan2(-rmat.at<double>(2,0), sy);
                        rz = 0;
                    }
                    
                    // Convert to degrees
                    rx = rx * 180.0 / CV_PI;
                    ry = ry * 180.0 / CV_PI;
                    rz = rz * 180.0 / CV_PI;
                    
                    poseText = QString("Tag ID: %1\n\n").arg(det.id);
                    poseText += "Translation (m):\n";
                    poseText += QString("  X: %1\n").arg(tx, 0, 'f', 3);
                    poseText += QString("  Y: %1\n").arg(ty, 0, 'f', 3);
                    poseText += QString("  Z: %1\n\n").arg(tz, 0, 'f', 3);
                    poseText += "Rotation (deg):\n";
                    poseText += QString("  Roll (X): %1\n").arg(rx, 0, 'f', 1);
                    poseText += QString("  Pitch (Y): %1\n").arg(ry, 0, 'f', 1);
                    poseText += QString("  Yaw (Z): %1\n").arg(rz, 0, 'f', 1);
                } else {
                    poseText = QString("Tag ID: %1\n").arg(det.id);
                    poseText += "Pose estimation failed";
                }
            }
            algorithmPoseText_->setPlainText(poseText);
        } else if (algorithmPoseText_) {
            algorithmPoseText_->setPlainText("No pose data\n(Calibration required)");
        }
    }
    
    void setupUI() {
        QVBoxLayout *mainLayout = new QVBoxLayout(this);
        
        // Create tab widget
        tabWidget_ = new QTabWidget(this);
        
        // ========== PROCESSING TAB ==========
        QWidget *processingTab = new QWidget(this);
        QVBoxLayout *processingLayout = new QVBoxLayout(processingTab);
        
        // Control panel
        QHBoxLayout *controlLayout = new QHBoxLayout();
        
        QPushButton *loadBtn1 = new QPushButton("Load Image 1", this);
        QPushButton *loadBtn2 = new QPushButton("Load Image 2", this);
        connect(loadBtn1, &QPushButton::clicked, this, &AprilTagDebugGUI::loadImage1);
        connect(loadBtn2, &QPushButton::clicked, this, &AprilTagDebugGUI::loadImage2);
        
        controlLayout->addWidget(loadBtn1);
        controlLayout->addWidget(loadBtn2);
        controlLayout->addStretch();
        
        processingLayout->addLayout(controlLayout);

        // Stage selection
        QGroupBox *stageGroup = new QGroupBox("Stage Selection", this);
        QHBoxLayout *stageLayout = new QHBoxLayout();
        
        // Preprocessing stage
        QLabel *preprocessLabel = new QLabel("Preprocessing:", this);
        preprocessCombo_ = new QComboBox(this);
        preprocessCombo_->addItems({
            "Original", "Histogram Equalization", "CLAHE (clip=2.0)", 
            "CLAHE (clip=3.0)", "CLAHE (clip=4.0)", "Gamma 1.2", 
            "Gamma 1.5", "Gamma 2.0", "Contrast Enhancement"
        });
        connect(preprocessCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), 
                this, &AprilTagDebugGUI::stageChanged);
        
        // Edge detection stage
        QLabel *edgeLabel = new QLabel("Edge Detection:", this);
        edgeCombo_ = new QComboBox(this);
        edgeCombo_->addItems({
            "None", "Canny (50,150)", "Canny (75,200)", "Canny (100,200)",
            "Sobel", "Laplacian", "Adaptive Threshold"
        });
        connect(edgeCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), 
                this, &AprilTagDebugGUI::stageChanged);
        
        // Detection stage
        QLabel *detectionLabel = new QLabel("Detection:", this);
        detectionCombo_ = new QComboBox(this);
        detectionCombo_->addItems({
            "Original", "With Detection", "Contours Only", "Quadrilaterals Only",
            "Convex Quads Only", "Tag-Sized Quads"
        });
        connect(detectionCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), 
                this, &AprilTagDebugGUI::stageChanged);
        
        // Advanced visualization stage
        QLabel *advancedLabel = new QLabel("Advanced:", this);
        advancedCombo_ = new QComboBox(this);
        advancedCombo_->addItems({
            "None", "Corner Refinement", "Warped Tags", "Pattern Extraction", "Hamming Decode"
        });
        connect(advancedCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), 
                this, &AprilTagDebugGUI::stageChanged);
        
        // Quad selection (for Warped Tags and later stages) - independent for each image
        QLabel *quadLabel1 = new QLabel("Quad (Img1):", this);
        quadCombo1_ = new QComboBox(this);
        quadCombo1_->setEnabled(false);  // Disabled until a quad stage is selected
        connect(quadCombo1_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &AprilTagDebugGUI::stageChanged);
        
        QLabel *quadLabel2 = new QLabel("Quad (Img2):", this);
        quadCombo2_ = new QComboBox(this);
        quadCombo2_->setEnabled(false);  // Disabled until a quad stage is selected
        connect(quadCombo2_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &AprilTagDebugGUI::stageChanged);
        
        connect(advancedCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), [this](int index) {
            // Enable quad selection for Warped Tags (2), Pattern Extraction (3), and Hamming Decode (4)
            bool enable = (index >= 2 && index <= 4);
            if (quadCombo1_) quadCombo1_->setEnabled(enable);
            if (quadCombo2_) quadCombo2_->setEnabled(enable);
            if (enable) {
                // Update quad list when switching to these modes
                stageChanged();
            }
        });
        
        stageLayout->addWidget(preprocessLabel);
        stageLayout->addWidget(preprocessCombo_);
        stageLayout->addWidget(edgeLabel);
        stageLayout->addWidget(edgeCombo_);
        stageLayout->addWidget(detectionLabel);
        stageLayout->addWidget(detectionCombo_);
        stageLayout->addWidget(advancedLabel);
        stageLayout->addWidget(advancedCombo_);
        stageLayout->addWidget(quadLabel1);
        stageLayout->addWidget(quadCombo1_);
        stageLayout->addWidget(quadLabel2);
        stageLayout->addWidget(quadCombo2_);
        
        // Mirror options (independent for each image)
        QLabel *mirrorLabel = new QLabel("Mirror:", this);
        mirrorCheckbox1_ = new QCheckBox("Image 1", this);
        mirrorCheckbox2_ = new QCheckBox("Image 2", this);
        connect(mirrorCheckbox1_, &QCheckBox::toggled, this, &AprilTagDebugGUI::stageChanged);
        connect(mirrorCheckbox2_, &QCheckBox::toggled, this, &AprilTagDebugGUI::stageChanged);
        stageLayout->addWidget(mirrorLabel);
        stageLayout->addWidget(mirrorCheckbox1_);
        stageLayout->addWidget(mirrorCheckbox2_);
        stageLayout->addStretch();
        
        stageGroup->setLayout(stageLayout);
        mainLayout->addWidget(stageGroup);

        // Create splitter for image and info panel
        QSplitter *mainSplitter = new QSplitter(Qt::Horizontal, this);
        
        // Image display area (side by side)
        QWidget *imageWidget = new QWidget(this);
        QHBoxLayout *imageLayout = new QHBoxLayout(imageWidget);
        
        // Image 1
        QGroupBox *img1Group = new QGroupBox("Image 1", this);
        QVBoxLayout *img1Layout = new QVBoxLayout();
        label1_ = new QLabel(this);
        label1_->setMinimumSize(640, 480);
        label1_->setAlignment(Qt::AlignCenter);
        label1_->setStyleSheet("border: 1px solid black; background-color: #f0f0f0;");
        label1_->setText("Load Image 1");
        img1Layout->addWidget(label1_);
        
        // Quality metrics box for Image 1
        QGroupBox *quality1Group = new QGroupBox("Quality Metrics", this);
        QVBoxLayout *quality1Layout = new QVBoxLayout();
        qualityText1_ = new QTextEdit(this);
        qualityText1_->setReadOnly(true);
        qualityText1_->setMaximumHeight(150);
        qualityText1_->setFont(QFont("Courier", 8));
        qualityText1_->setPlainText("Load image and select visualization mode to see quality metrics.");
        quality1Layout->addWidget(qualityText1_);
        quality1Group->setLayout(quality1Layout);
        img1Layout->addWidget(quality1Group);
        
        img1Group->setLayout(img1Layout);
        
        // Image 2
        QGroupBox *img2Group = new QGroupBox("Image 2", this);
        QVBoxLayout *img2Layout = new QVBoxLayout();
        label2_ = new QLabel(this);
        label2_->setMinimumSize(640, 480);
        label2_->setAlignment(Qt::AlignCenter);
        label2_->setStyleSheet("border: 1px solid black; background-color: #f0f0f0;");
        label2_->setText("Load Image 2");
        img2Layout->addWidget(label2_);
        
        // Quality metrics box for Image 2
        QGroupBox *quality2Group = new QGroupBox("Quality Metrics", this);
        QVBoxLayout *quality2Layout = new QVBoxLayout();
        qualityText2_ = new QTextEdit(this);
        qualityText2_->setReadOnly(true);
        qualityText2_->setMaximumHeight(150);
        qualityText2_->setFont(QFont("Courier", 8));
        qualityText2_->setPlainText("Load image and select visualization mode to see quality metrics.");
        quality2Layout->addWidget(qualityText2_);
        quality2Group->setLayout(quality2Layout);
        img2Layout->addWidget(quality2Group);
        
        img2Group->setLayout(img2Layout);
        
        imageLayout->addWidget(img1Group);
        imageLayout->addWidget(img2Group);
        imageLayout->setContentsMargins(0, 0, 0, 0);
        imageWidget->setLayout(imageLayout);
        mainSplitter->addWidget(imageWidget);
        
        // Information panel for extracted codes and steps
        QGroupBox *infoGroup = new QGroupBox("Extracted Codes & Decoding Steps", this);
        QVBoxLayout *infoLayout = new QVBoxLayout();
        infoText_ = new QTextEdit(this);
        infoText_->setReadOnly(true);
        infoText_->setMaximumWidth(400);
        infoText_->setFont(QFont("Courier", 9));
        infoText_->setPlainText("Select an Advanced visualization mode to see extracted codes and decoding steps here.");
        infoLayout->addWidget(infoText_);
        infoGroup->setLayout(infoLayout);
        mainSplitter->addWidget(infoGroup);
        
        // Set splitter sizes (70% for images, 30% for info)
        mainSplitter->setSizes({700, 300});
        processingLayout->addWidget(mainSplitter);
        processingTab->setLayout(processingLayout);
        tabWidget_->addTab(processingTab, "Processing");
        
        // ========== CAPTURE TAB ==========
        QWidget *captureTab = new QWidget(this);
        QVBoxLayout *captureLayout = new QVBoxLayout(captureTab);
        setupCaptureTab(captureLayout);
        tabWidget_->addTab(captureTab, "Capture");
        
        // ========== FISHEYE CORRECTION TAB ==========
        setupFisheyeTab();
        
        // ========== ALGORITHMS TAB ==========
        QWidget *algorithmsTab = new QWidget(this);
        QVBoxLayout *algorithmsLayout = new QVBoxLayout(algorithmsTab);
        setupAlgorithmsTab(algorithmsLayout);
        tabWidget_->addTab(algorithmsTab, "Algorithms");
        
        // Fisheye correction status indicator (at the top, outside tabs)
        QGroupBox *fisheyeStatusGroup = new QGroupBox("Fisheye Correction Status", this);
        QHBoxLayout *fisheyeStatusLayout = new QHBoxLayout();
        
        fisheyeStatusIndicator_ = new QLabel("Status: Not Applied", this);
        fisheyeStatusIndicator_->setStyleSheet(
            "QLabel { "
            "background-color: #ffcccc; "
            "color: #000000; "
            "padding: 8px; "
            "border: 2px solid #ff0000; "
            "border-radius: 5px; "
            "font-weight: bold; "
            "font-size: 12pt; "
            "}");
        fisheyeStatusLabel_->setAlignment(Qt::AlignCenter);
        
        fisheyeStatusLayout->addWidget(fisheyeStatusIndicator_);
        fisheyeStatusGroup->setLayout(fisheyeStatusLayout);
        mainLayout->addWidget(fisheyeStatusGroup);
        
        mainLayout->addWidget(tabWidget_);

        setWindowTitle("AprilTag Detection Debugging Tool");
        resize(1400, 700);
    }

    Mat preprocessImage(const Mat &img, int method) {
        Mat result;
        
        switch (method) {
            case 0: // Original
                result = img.clone();
                break;
            case 1: // Histogram Equalization
                equalizeHist(img, result);
                break;
            case 2: // CLAHE clip=2.0
                {
                    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
                    clahe->apply(img, result);
                }
                break;
            case 3: // CLAHE clip=3.0
                {
                    Ptr<CLAHE> clahe = createCLAHE(3.0, Size(8, 8));
                    clahe->apply(img, result);
                }
                break;
            case 4: // CLAHE clip=4.0
                {
                    Ptr<CLAHE> clahe = createCLAHE(4.0, Size(8, 8));
                    clahe->apply(img, result);
                }
                break;
            case 5: // Gamma 1.2
                {
                    double gamma = 1.2;
                    double inv_gamma = 1.0 / gamma;
                    Mat table(1, 256, CV_8U);
                    uchar *p = table.ptr();
                    for (int i = 0; i < 256; i++) {
                        p[i] = saturate_cast<uchar>(pow(i / 255.0, inv_gamma) * 255.0);
                    }
                    LUT(img, table, result);
                }
                break;
            case 6: // Gamma 1.5
                {
                    double gamma = 1.5;
                    double inv_gamma = 1.0 / gamma;
                    Mat table(1, 256, CV_8U);
                    uchar *p = table.ptr();
                    for (int i = 0; i < 256; i++) {
                        p[i] = saturate_cast<uchar>(pow(i / 255.0, inv_gamma) * 255.0);
                    }
                    LUT(img, table, result);
                }
                break;
            case 7: // Gamma 2.0
                {
                    double gamma = 2.0;
                    double inv_gamma = 1.0 / gamma;
                    Mat table(1, 256, CV_8U);
                    uchar *p = table.ptr();
                    for (int i = 0; i < 256; i++) {
                        p[i] = saturate_cast<uchar>(pow(i / 255.0, inv_gamma) * 255.0);
                    }
                    LUT(img, table, result);
                }
                break;
            case 8: // Contrast Enhancement
                img.convertTo(result, -1, 2.0, 50);
                break;
            default:
                result = img.clone();
        }
        
        return result;
    }

    Mat applyEdgeDetection(const Mat &img, int method) {
        Mat result;
        
        switch (method) {
            case 0: // None
                result = img.clone();
                break;
            case 1: // Canny (50, 150)
                Canny(img, result, 50, 150);
                break;
            case 2: // Canny (75, 200)
                Canny(img, result, 75, 200);
                break;
            case 3: // Canny (100, 200)
                Canny(img, result, 100, 200);
                break;
            case 4: // Sobel
                {
                    Mat sobel_x, sobel_y, sobel_combined;
                    Sobel(img, sobel_x, CV_16S, 1, 0, 3);
                    Sobel(img, sobel_y, CV_16S, 0, 1, 3);
                    convertScaleAbs(sobel_x, sobel_x);
                    convertScaleAbs(sobel_y, sobel_y);
                    addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0, result);
                }
                break;
            case 5: // Laplacian
                {
                    Mat laplacian;
                    Laplacian(img, laplacian, CV_16S, 3);
                    convertScaleAbs(laplacian, result);
                }
                break;
            case 6: // Adaptive Threshold
                adaptiveThreshold(img, result, 255, ADAPTIVE_THRESH_GAUSSIAN_C, 
                                THRESH_BINARY, 11, 2);
                break;
            default:
                result = img.clone();
        }
        
        return result;
    }

    Mat drawDetections(const Mat &img, int method) {
        Mat result = img.clone();
        
        if (method == 0) { // Original
            Mat color_result;
            if (result.channels() == 1) {
                cvtColor(result, color_result, COLOR_GRAY2BGR);
            } else {
                color_result = result.clone();
            }
            return color_result;
        }
        
        // Convert to image_u8_t for detection
        if (!result.isContinuous()) {
            result = result.clone();
        }
        
        image_u8_t im = {
            .width = result.cols,
            .height = result.rows,
            .stride = result.cols,
            .buf = result.data
        };
        
        zarray_t* detections = apriltag_detector_detect(td_, &im);
        
        // Convert to color for drawing
        Mat color_result;
        cvtColor(result, color_result, COLOR_GRAY2BGR);
        
        if (method == 1) { // With Detection
            // Draw all contours
            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;
            findContours(result, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
            
            // Draw AprilTag detections
            for (int i = 0; i < zarray_size(detections); i++) {
                apriltag_detection_t* det;
                zarray_get(detections, i, &det);
                
                // Draw tag outline
                line(color_result, Point(det->p[0][0], det->p[0][1]),
                     Point(det->p[1][0], det->p[1][1]), Scalar(0, 255, 0), 2);
                line(color_result, Point(det->p[1][0], det->p[1][1]),
                     Point(det->p[2][0], det->p[2][1]), Scalar(0, 255, 0), 2);
                line(color_result, Point(det->p[2][0], det->p[2][1]),
                     Point(det->p[3][0], det->p[3][1]), Scalar(0, 255, 0), 2);
                line(color_result, Point(det->p[3][0], det->p[3][1]),
                     Point(det->p[0][0], det->p[0][1]), Scalar(0, 255, 0), 2);
                
                // Draw center and ID
                circle(color_result, Point(det->c[0], det->c[1]), 5, Scalar(0, 0, 255), -1);
                putText(color_result, to_string(det->id), 
                       Point(det->c[0] + 10, det->c[1]),
                       FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
            }
            
            // Draw contours
            for (size_t i = 0; i < contours.size(); i++) {
                drawContours(color_result, contours, i, Scalar(255, 0, 0), 1);
            }
        } else if (method == 2) { // Contours Only
            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;
            findContours(result, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
            for (size_t i = 0; i < contours.size(); i++) {
                drawContours(color_result, contours, i, Scalar(255, 0, 0), 2);
            }
        } else if (method == 3) { // Quadrilaterals Only
            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;
            findContours(result, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
            
            for (size_t i = 0; i < contours.size(); i++) {
                vector<Point> approx;
                approxPolyDP(contours[i], approx, arcLength(contours[i], true) * 0.02, true);
                
                if (approx.size() == 4 && isContourConvex(approx)) {
                    // Check if it's tag-sized
                    double area = contourArea(approx);
                    if (area > 100 && area < 50000) {
                        drawContours(color_result, vector<vector<Point>>{approx}, 0, Scalar(0, 255, 0), 2);
                    }
                }
            }
        } else if (method == 4) { // Convex Quads Only
            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;
            findContours(result, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
            
            for (size_t i = 0; i < contours.size(); i++) {
                vector<Point> approx;
                approxPolyDP(contours[i], approx, arcLength(contours[i], true) * 0.02, true);
                
                if (approx.size() == 4 && isContourConvex(approx)) {
                    drawContours(color_result, vector<vector<Point>>{approx}, 0, Scalar(255, 255, 0), 2);
                }
            }
        } else if (method == 5) { // Tag-Sized Quads
            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;
            findContours(result, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
            
            for (size_t i = 0; i < contours.size(); i++) {
                double area = contourArea(contours[i]);
                if (area > 100 && area < 50000) {
                    vector<Point> approx;
                    approxPolyDP(contours[i], approx, arcLength(contours[i], true) * 0.02, true);
                    
                    if (approx.size() == 4 && isContourConvex(approx)) {
                        drawContours(color_result, vector<vector<Point>>{approx}, 0, Scalar(0, 255, 255), 2);
                    }
                }
            }
        }
        
        // Clean up detections
        for (int i = 0; i < zarray_size(detections); i++) {
            apriltag_detection_t* det;
            zarray_get(detections, i, &det);
            apriltag_detection_destroy(det);
        }
        zarray_destroy(detections);
        
        return color_result;
    }
    
    // Helper function to refine corners
    void refineCorners(const Mat& gray, vector<Point2f>& corners, int winSize = 5, int maxIter = 30) {
        if (corners.size() != 4) return;
        TermCriteria criteria(TermCriteria::EPS + TermCriteria::COUNT, maxIter, 0.001);
        cornerSubPix(gray, corners, Size(winSize, winSize), Size(-1, -1), criteria);
    }
    
    // Extract quadrilaterals from edge-detected image
    vector<vector<Point2f>> extractQuadrilaterals(const Mat& edges, const Mat& original) {
        vector<vector<Point2f>> quads;
        
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(edges, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        
        double tag_min_area = 500;
        double tag_max_area = 50000;
        
        for (size_t i = 0; i < contours.size(); i++) {
            double area = contourArea(contours[i]);
            if (area >= tag_min_area && area <= tag_max_area) {
                vector<Point> approx;
                double epsilon = 0.02 * arcLength(contours[i], true);
                approxPolyDP(contours[i], approx, epsilon, true);
                if (approx.size() == 4 && isContourConvex(approx)) {
                    vector<Point2f> quad;
                    for (int j = 0; j < 4; j++) {
                        quad.push_back(Point2f(approx[j].x, approx[j].y));
                    }
                    quads.push_back(quad);
                }
            }
        }
        
        return quads;
    }
    
    // Extract 6x6 pattern from warped tag image
    // Tag36h11 is 8x8 cells: 1-cell black border on all sides, 6x6 data region in center
    // The extracted 6x6 pattern contains ONLY data cells (no border)
    vector<vector<int>> extractPattern(const Mat& warped, int tagSize = 36, int borderSize = 4) {
        // Calculate border size: Tag36h11 is 8x8 cells, so border = tagSize / 8
        // Use provided borderSize if it makes sense, otherwise calculate
        if (borderSize <= 0) {
            borderSize = tagSize / 8;  // Tag36h11: 8 cells total, 1 cell border
        }
        
        // Add a small margin to avoid sampling too close to the border
        // This helps avoid edge cases where border pixels leak into data region
        int borderMargin = max(1, borderSize / 4);  // Add ~25% margin, minimum 1 pixel
        int effectiveBorderSize = borderSize + borderMargin;
        
        // Debug: Sample border pixels to verify border detection
        static int debug_count = 0;
        if (debug_count < 5) {
            qDebug() << "=== Pattern Extraction Debug #" << debug_count << "===";
            qDebug() << "tagSize:" << tagSize << "borderSize:" << borderSize << "effectiveBorderSize:" << effectiveBorderSize;
            // Sample border pixels (should be black/dark)
            vector<pair<string, Point>> border_samples = {
                {"Top-left", Point(borderSize/2, borderSize/2)},
                {"Top-center", Point(tagSize/2, borderSize/2)},
                {"Top-right", Point(tagSize - borderSize/2, borderSize/2)},
                {"Left-center", Point(borderSize/2, tagSize/2)},
                {"Right-center", Point(tagSize - borderSize/2, tagSize/2)},
                {"Bottom-left", Point(borderSize/2, tagSize - borderSize/2)},
                {"Bottom-center", Point(tagSize/2, tagSize - borderSize/2)},
                {"Bottom-right", Point(tagSize - borderSize/2, tagSize - borderSize/2)}
            };
            for (auto& sample : border_samples) {
                int x = min(max(0, sample.second.x), tagSize - 1);
                int y = min(max(0, sample.second.y), tagSize - 1);
                int val = (int)warped.at<uchar>(y, x);
                qDebug() << "Border" << sample.first.c_str() << "(" << x << "," << y << "):" << val << (val < 128 ? "BLACK" : "WHITE");
            }
            debug_count++;
        }
        
        // Use effectiveBorderSize to ensure we're well inside the data region
        int dataSize = tagSize - 2 * effectiveBorderSize;
        int cellSize = dataSize / 6;
        
        qDebug() << "dataSize:" << dataSize << "cellSize:" << cellSize << "borderMargin:" << borderMargin;
        
        vector<vector<int>> pattern(6, vector<int>(6));
        
        // Extract 6x6 data region (rows 1-6, columns 1-6 from 8x8 tag)
        // This excludes the border, so all 6x6 cells are data bits
        // Use effectiveBorderSize to sample further from the border edge
        for (int row = 0; row < 6; row++) {
            for (int col = 0; col < 6; col++) {
                // Sample center of each data cell, starting from effectiveBorderSize
                int x_center = effectiveBorderSize + col * cellSize + cellSize / 2;
                int y_center = effectiveBorderSize + row * cellSize + cellSize / 2;
                
                x_center = min(max(0, x_center), tagSize - 1);
                y_center = min(max(0, y_center), tagSize - 1);
                
                pattern[row][col] = (int)warped.at<uchar>(y_center, x_center);
            }
        }
        
        return pattern;
    }
    
    Mat drawAdvancedVisualization(const Mat &img, int method, int image_index = 1) {
        Mat result = img.clone();
        Mat color_result;
        cvtColor(result, color_result, COLOR_GRAY2BGR);
        
        switch (method) {
            case 1: { // Corner Refinement
                // Need edge-detected image for quad extraction
                Mat edges_for_quads;
                if (image_index == 1 && !image1_.empty()) {
                    Mat img1 = image1_.clone();
                    bool use_mirror = mirrorCheckbox1_ ? mirrorCheckbox1_->isChecked() : false;
                    if (use_mirror) flip(img1, img1, 1);
                    Mat processed1 = preprocessImage(img1, preprocessCombo_->currentIndex());
                    edges_for_quads = applyEdgeDetection(processed1, edgeCombo_->currentIndex());
                } else if (image_index == 2 && !image2_.empty()) {
                    Mat img2 = image2_.clone();
                    bool use_mirror = mirrorCheckbox2_ ? mirrorCheckbox2_->isChecked() : false;
                    if (use_mirror) flip(img2, img2, 1);
                    Mat processed2 = preprocessImage(img2, preprocessCombo_->currentIndex());
                    edges_for_quads = applyEdgeDetection(processed2, edgeCombo_->currentIndex());
                } else {
                    edges_for_quads = result.clone();
                }
                
                vector<vector<Point2f>> quads = extractQuadrilaterals(edges_for_quads, result);
                
                if (quads.empty()) {
                    putText(color_result, "No quadrilaterals found", Point(10, 30),
                           FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
                    if (infoText_) {
                        infoText_->setPlainText("No quadrilaterals found for corner refinement.");
                    }
                    QTextEdit* target_text = (image_index == 1) ? qualityText1_ : qualityText2_;
                    if (target_text) {
                        target_text->setPlainText("CORNER REFINEMENT\nNo quads found.\nCheck edge detection.");
                    }
                    return color_result;
                }
                
                // Refine corners
                vector<vector<Point2f>> refined_quads;
                for (size_t i = 0; i < quads.size(); i++) {
                    vector<Point2f> refined = quads[i];
                    refineCorners(result, refined);
                    refined_quads.push_back(refined);
                    
                    // Draw original quad (green, thin)
                    for (int j = 0; j < 4; j++) {
                        int next = (j + 1) % 4;
                        line(color_result, quads[i][j], quads[i][next], Scalar(0, 255, 0), 1);
                        circle(color_result, quads[i][j], 3, Scalar(0, 255, 0), -1);
                        // Draw corner number
                        putText(color_result, to_string(j), quads[i][j] + Point2f(5, -5),
                               FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
                    }
                    
                    // Draw refined quad (red, thick)
                    for (int j = 0; j < 4; j++) {
                        int next = (j + 1) % 4;
                        line(color_result, refined_quads[i][j], refined_quads[i][next], Scalar(0, 0, 255), 2);
                        circle(color_result, refined_quads[i][j], 5, Scalar(0, 0, 255), -1);
                    }
                    
                    // Draw movement vectors (yellow lines) and collect metrics
                    vector<double> corner_movements;
                    for (int j = 0; j < 4; j++) {
                        line(color_result, quads[i][j], refined_quads[i][j], Scalar(0, 255, 255), 1);
                        double dx = refined_quads[i][j].x - quads[i][j].x;
                        double dy = refined_quads[i][j].y - quads[i][j].y;
                        double dist = sqrt(dx*dx + dy*dy);
                        corner_movements.push_back(dist);
                        if (dist > 0.5) {  // Only show if moved significantly
                            stringstream ss;
                            ss << fixed << setprecision(1) << dist;
                            Point2f mid = (quads[i][j] + refined_quads[i][j]) * 0.5;
                            putText(color_result, ss.str(), mid, FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 255), 1);
                        }
                    }
                    
                    // Draw quad number
                    Point2f center = (refined_quads[i][0] + refined_quads[i][1] + 
                                     refined_quads[i][2] + refined_quads[i][3]) / 4;
                    putText(color_result, "Quad " + to_string(i), center + Point2f(10, -10),
                           FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 0), 2);
                    
                    // Store corner movements for info panel
                    if (i == 0) {  // Store first quad's movements for display
                        double avg_movement = 0;
                        double max_movement = 0;
                        for (double d : corner_movements) {
                            avg_movement += d;
                            if (d > max_movement) max_movement = d;
                        }
                        avg_movement /= 4.0;
                        updateCornerRefinementInfo(avg_movement, max_movement, image_index);
                    }
                }
                
                // Add legend
                putText(color_result, "Green: Original corners", Point(10, 30),
                       FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
                putText(color_result, "Red: Refined corners", Point(10, 60),
                       FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
                putText(color_result, "Yellow: Movement vectors", Point(10, 90),
                       FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
            }
            break;
            
            case 2: { // Warped Tags
                // Need edge-detected image for quad extraction
                Mat edges_for_quads;
                Mat original_for_warp;
                if (image_index == 1 && !image1_.empty()) {
                    Mat img1 = image1_.clone();
                    bool use_mirror = mirrorCheckbox1_ ? mirrorCheckbox1_->isChecked() : false;
                    if (use_mirror) flip(img1, img1, 1);
                    Mat processed1 = preprocessImage(img1, preprocessCombo_->currentIndex());
                    edges_for_quads = applyEdgeDetection(processed1, edgeCombo_->currentIndex());
                    original_for_warp = processed1.clone();
                } else if (image_index == 2 && !image2_.empty()) {
                    Mat img2 = image2_.clone();
                    bool use_mirror = mirrorCheckbox2_ ? mirrorCheckbox2_->isChecked() : false;
                    if (use_mirror) flip(img2, img2, 1);
                    Mat processed2 = preprocessImage(img2, preprocessCombo_->currentIndex());
                    edges_for_quads = applyEdgeDetection(processed2, edgeCombo_->currentIndex());
                    original_for_warp = processed2.clone();
                } else {
                    edges_for_quads = result.clone();
                    original_for_warp = result.clone();
                }
                
                vector<vector<Point2f>> quads = extractQuadrilaterals(edges_for_quads, original_for_warp);
                
                if (quads.empty()) {
                    putText(color_result, "No quadrilaterals found", Point(10, 30),
                           FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
                    // Update quad combo to empty (block signals to prevent recursion)
                    QComboBox* target_quad_combo = (image_index == 1) ? quadCombo1_ : quadCombo2_;
                    if (target_quad_combo) {
                        target_quad_combo->blockSignals(true);
                        target_quad_combo->clear();
                        target_quad_combo->blockSignals(false);
                    }
                    QTextEdit* target_text = (image_index == 1) ? qualityText1_ : qualityText2_;
                    if (target_text) {
                        target_text->setPlainText("WARPED TAGS\nNo quads found.\nCheck edge detection.");
                    }
                    return color_result;
                }
                
                // Update quad combo box with available quads (block signals to prevent recursion)
                QComboBox* target_quad_combo = (image_index == 1) ? quadCombo1_ : quadCombo2_;
                if (target_quad_combo) {
                    int current_selection = target_quad_combo->currentIndex();
                    target_quad_combo->blockSignals(true);  // Prevent signal emission during update
                    target_quad_combo->clear();
                    for (size_t i = 0; i < quads.size(); i++) {
                        target_quad_combo->addItem(QString("Quad %1").arg(i));
                    }
                    // Restore selection if valid
                    if (current_selection >= 0 && current_selection < (int)quads.size()) {
                        target_quad_combo->setCurrentIndex(current_selection);
                    } else if (quads.size() > 0) {
                        target_quad_combo->setCurrentIndex(0);
                    }
                    target_quad_combo->blockSignals(false);  // Re-enable signals
                }
                
                // Get selected quad index
                int selected_quad = target_quad_combo ? target_quad_combo->currentIndex() : 0;
                if (selected_quad < 0 || selected_quad >= (int)quads.size()) {
                    selected_quad = 0;
                }
                
                // Refine and warp selected quad
                int tagSize = 36;
                int scale = 15;  // Scale factor for better visibility
                int displaySize = tagSize * scale;
                int padding = 40;
                
                Mat composite = Mat::ones(displaySize + padding * 2, displaySize + padding * 2, CV_8UC1) * 128;
                
                vector<Point2f> refined = quads[selected_quad];
                refineCorners(original_for_warp, refined);
                
                vector<Point2f> dstQuad;
                dstQuad.push_back(Point2f(0, 0));
                dstQuad.push_back(Point2f(tagSize - 1, 0));
                dstQuad.push_back(Point2f(tagSize - 1, tagSize - 1));
                dstQuad.push_back(Point2f(0, tagSize - 1));
                
                Mat H = getPerspectiveTransform(refined, dstQuad);
                Mat warped;
                warpPerspective(original_for_warp, warped, H, Size(tagSize, tagSize));
                
                // Scale up
                Mat warped_large;
                cv::resize(warped, warped_large, Size(displaySize, displaySize), 0, 0, INTER_NEAREST);
                
                // Copy to composite (centered)
                warped_large.copyTo(composite(Rect(padding, padding, displaySize, displaySize)));
                
                // Scale to fit display
                cv::resize(composite, color_result, Size(result.cols, result.rows), 0, 0, INTER_LINEAR);
                cvtColor(color_result, color_result, COLOR_GRAY2BGR);
                
                // Add labels
                stringstream label;
                label << "Warped Tag - Quad " << selected_quad << " (36x36, scaled 15x)";
                putText(color_result, label.str(), Point(10, 30),
                       FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
                putText(color_result, "Total Quads: " + to_string(quads.size()), Point(10, 60),
                       FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
            }
            break;
            
            case 3: { // Pattern Extraction
                // Need edge-detected image for quad extraction
                Mat edges_for_quads;
                Mat original_for_warp;
                if (image_index == 1 && !image1_.empty()) {
                    Mat img1 = image1_.clone();
                    bool use_mirror = mirrorCheckbox1_ ? mirrorCheckbox1_->isChecked() : false;
                    if (use_mirror) flip(img1, img1, 1);
                    Mat processed1 = preprocessImage(img1, preprocessCombo_->currentIndex());
                    edges_for_quads = applyEdgeDetection(processed1, edgeCombo_->currentIndex());
                    original_for_warp = processed1.clone();
                } else if (image_index == 2 && !image2_.empty()) {
                    Mat img2 = image2_.clone();
                    bool use_mirror = mirrorCheckbox2_ ? mirrorCheckbox2_->isChecked() : false;
                    if (use_mirror) flip(img2, img2, 1);
                    Mat processed2 = preprocessImage(img2, preprocessCombo_->currentIndex());
                    edges_for_quads = applyEdgeDetection(processed2, edgeCombo_->currentIndex());
                    original_for_warp = processed2.clone();
                } else {
                    edges_for_quads = result.clone();
                    original_for_warp = result.clone();
                }
                
                vector<vector<Point2f>> quads = extractQuadrilaterals(edges_for_quads, original_for_warp);
                
                if (quads.empty()) {
                    putText(color_result, "No quadrilaterals found", Point(10, 30),
                           FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
                    // Update quad combo to empty (block signals to prevent recursion)
                    QComboBox* target_quad_combo = (image_index == 1) ? quadCombo1_ : quadCombo2_;
                    if (target_quad_combo) {
                        target_quad_combo->blockSignals(true);
                        target_quad_combo->clear();
                        target_quad_combo->blockSignals(false);
                    }
                    QTextEdit* target_text = (image_index == 1) ? qualityText1_ : qualityText2_;
                    if (target_text) {
                        target_text->setPlainText("PATTERN EXTRACTION\nNo quads found.\nCheck edge detection.");
                    }
                    return color_result;
                }
                
                // Update quad combo box with available quads (block signals to prevent recursion)
                QComboBox* target_quad_combo = (image_index == 1) ? quadCombo1_ : quadCombo2_;
                if (target_quad_combo) {
                    int current_selection = target_quad_combo->currentIndex();
                    target_quad_combo->blockSignals(true);  // Prevent signal emission during update
                    target_quad_combo->clear();
                    for (size_t i = 0; i < quads.size(); i++) {
                        target_quad_combo->addItem(QString("Quad %1").arg(i));
                    }
                    // Restore selection if valid
                    if (current_selection >= 0 && current_selection < (int)quads.size()) {
                        target_quad_combo->setCurrentIndex(current_selection);
                    } else if (quads.size() > 0) {
                        target_quad_combo->setCurrentIndex(0);
                    }
                    target_quad_combo->blockSignals(false);  // Re-enable signals
                }
                
                // Get selected quad index
                int selected_quad = target_quad_combo ? target_quad_combo->currentIndex() : 0;
                if (selected_quad < 0 || selected_quad >= (int)quads.size()) {
                    selected_quad = 0;
                }
                
                // Process selected quad
                vector<Point2f> refined = quads[selected_quad];
                refineCorners(original_for_warp, refined);
                
                int tagSize = 36;
                vector<Point2f> dstQuad;
                dstQuad.push_back(Point2f(0, 0));
                dstQuad.push_back(Point2f(tagSize - 1, 0));
                dstQuad.push_back(Point2f(tagSize - 1, tagSize - 1));
                dstQuad.push_back(Point2f(0, tagSize - 1));
                
                Mat H = getPerspectiveTransform(refined, dstQuad);
                Mat warped;
                warpPerspective(original_for_warp, warped, H, Size(tagSize, tagSize));
                
                // Extract pattern
                vector<vector<int>> pattern = extractPattern(warped, tagSize);
                
                // Extract 36-bit code and update info panel
                updatePatternExtractionInfo(pattern, warped, tagSize, selected_quad, image_index);
                
                // Visualize pattern with detailed information
                int cell_size = 100;
                int padding = 60;
                int grid_size = 6 * cell_size;
                int header_height = 80;
                Mat pattern_vis = Mat::ones(grid_size + padding * 2 + header_height, 
                                           grid_size + padding * 2, CV_8UC3) * 240;
                
                // Calculate statistics
                int black_count = 0;
                int white_count = 0;
                int total = 0;
                for (int row = 0; row < 6; row++) {
                    for (int col = 0; col < 6; col++) {
                        if (pattern[row][col] < 128) black_count++;
                        else white_count++;
                        total += pattern[row][col];
                    }
                }
                
                // Header with statistics
                stringstream header;
                header << "6x6 Pattern - Mean: " << (total / 36) << "  Black: " << black_count << "  White: " << white_count;
                putText(pattern_vis, header.str(), Point(padding, 40),
                       FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 0), 2);
                
                // Draw grid with pattern
                for (int row = 0; row < 6; row++) {
                    for (int col = 0; col < 6; col++) {
                        int val = pattern[row][col];
                        bool is_black = val < 128;
                        Scalar color = is_black ? Scalar(0, 0, 0) : Scalar(255, 255, 255);
                        int y_pos = header_height + padding + row * cell_size;
                        int x_pos = padding + col * cell_size;
                        Rect cell(x_pos, y_pos, cell_size, cell_size);
                        rectangle(pattern_vis, cell, color, -1);
                        rectangle(pattern_vis, cell, Scalar(128, 128, 128), 2);
                        
                        // Draw pixel value
                        stringstream ss;
                        ss << val;
                        putText(pattern_vis, ss.str(), Point(x_pos + 10, y_pos + 25),
                               FONT_HERSHEY_SIMPLEX, 0.6, is_black ? Scalar(255, 255, 255) : Scalar(0, 0, 0), 2);
                        
                        // Draw bit value (0 or 1)
                        string bit_str = is_black ? "1" : "0";
                        putText(pattern_vis, bit_str, Point(x_pos + 10, y_pos + 50),
                               FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
                        
                        // Draw coordinates
                        stringstream coord;
                        coord << "(" << row << "," << col << ")";
                        putText(pattern_vis, coord.str(), Point(x_pos + 5, y_pos + cell_size - 10),
                               FONT_HERSHEY_SIMPLEX, 0.4, Scalar(128, 0, 128), 1);
                    }
                }
                
                // Draw border indication (first/last row and column should be black in real tags)
                // Highlight border cells
                for (int i = 0; i < 6; i++) {
                    // Top row
                    Rect cell(padding + i * cell_size, header_height + padding, cell_size, cell_size);
                    rectangle(pattern_vis, cell, Scalar(255, 0, 0), 3);
                    // Bottom row
                    cell = Rect(padding + i * cell_size, header_height + padding + 5 * cell_size, cell_size, cell_size);
                    rectangle(pattern_vis, cell, Scalar(255, 0, 0), 3);
                    // Left column
                    cell = Rect(padding, header_height + padding + i * cell_size, cell_size, cell_size);
                    rectangle(pattern_vis, cell, Scalar(255, 0, 0), 3);
                    // Right column
                    cell = Rect(padding + 5 * cell_size, header_height + padding + i * cell_size, cell_size, cell_size);
                    rectangle(pattern_vis, cell, Scalar(255, 0, 0), 3);
                }
                
                // Scale to fit display
                cv::resize(pattern_vis, color_result, Size(result.cols, result.rows), 0, 0, INTER_LINEAR);
            }
            break;
            
            case 4: { // Hamming Decode - Enhanced visualization
                // Need original processed image for detection (not edge-detected)
                Mat original_for_detection;
                if (image_index == 1 && !image1_.empty()) {
                    Mat img1 = image1_.clone();
                    bool use_mirror = mirrorCheckbox1_ ? mirrorCheckbox1_->isChecked() : false;
                    if (use_mirror) flip(img1, img1, 1);
                    original_for_detection = preprocessImage(img1, preprocessCombo_->currentIndex());
                } else if (image_index == 2 && !image2_.empty()) {
                    Mat img2 = image2_.clone();
                    bool use_mirror = mirrorCheckbox2_ ? mirrorCheckbox2_->isChecked() : false;
                    if (use_mirror) flip(img2, img2, 1);
                    original_for_detection = preprocessImage(img2, preprocessCombo_->currentIndex());
                } else {
                    original_for_detection = result.clone();
                }
                
                // Draw detected tags with detailed information
                if (!original_for_detection.isContinuous()) {
                    original_for_detection = original_for_detection.clone();
                }
                
                image_u8_t im = {
                    .width = original_for_detection.cols,
                    .height = original_for_detection.rows,
                    .stride = original_for_detection.cols,
                    .buf = original_for_detection.data
                };
                
                zarray_t* detections = apriltag_detector_detect(td_, &im);
                
                // Convert original_for_detection to color for drawing
                cvtColor(original_for_detection, color_result, COLOR_GRAY2BGR);
                
                // Update quad combo box with detected tags (block signals to prevent recursion)
                QComboBox* target_quad_combo = (image_index == 1) ? quadCombo1_ : quadCombo2_;
                if (target_quad_combo) {
                    int current_selection = target_quad_combo->currentIndex();
                    target_quad_combo->blockSignals(true);  // Prevent signal emission during update
                    target_quad_combo->clear();
                    for (int i = 0; i < zarray_size(detections); i++) {
                        apriltag_detection_t* det;
                        zarray_get(detections, i, &det);
                        target_quad_combo->addItem(QString("Tag %1 (ID:%2)").arg(i).arg(det->id));
                    }
                    // Restore selection if valid
                    if (current_selection >= 0 && current_selection < zarray_size(detections)) {
                        target_quad_combo->setCurrentIndex(current_selection);
                    } else if (zarray_size(detections) > 0) {
                        target_quad_combo->setCurrentIndex(0);
                    }
                    target_quad_combo->blockSignals(false);  // Re-enable signals
                }
                
                // Get selected tag index
                int selected_tag = target_quad_combo ? target_quad_combo->currentIndex() : 0;
                if (selected_tag < 0 || selected_tag >= zarray_size(detections)) {
                    selected_tag = 0;
                }
                
                // Update info panel with decoding information
                updateHammingDecodeInfo(detections, selected_tag, image_index);
                
                // Draw background info box
                int num_dets = zarray_size(detections);
                rectangle(color_result, Point(10, 10), Point(400, 60 + num_dets * 30), Scalar(0, 0, 0), -1);
                rectangle(color_result, Point(10, 10), Point(400, 60 + num_dets * 30), Scalar(255, 255, 255), 2);
                putText(color_result, "Detected Tags: " + to_string(num_dets), Point(20, 40),
                       FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
                
                // Draw detections with detailed info
                for (int i = 0; i < zarray_size(detections); i++) {
                    apriltag_detection_t* det;
                    zarray_get(detections, i, &det);
                    
                    // Choose color based on decision margin (green = high confidence)
                    Scalar tag_color = (det->decision_margin > 60) ? Scalar(0, 255, 0) : 
                                      (det->decision_margin > 30) ? Scalar(0, 255, 255) : Scalar(0, 165, 255);
                    
                    // Draw tag outline (thick)
                    line(color_result, Point(det->p[0][0], det->p[0][1]),
                         Point(det->p[1][0], det->p[1][1]), tag_color, 3);
                    line(color_result, Point(det->p[1][0], det->p[1][1]),
                         Point(det->p[2][0], det->p[2][1]), tag_color, 3);
                    line(color_result, Point(det->p[2][0], det->p[2][1]),
                         Point(det->p[3][0], det->p[3][1]), tag_color, 3);
                    line(color_result, Point(det->p[3][0], det->p[3][1]),
                         Point(det->p[0][0], det->p[0][1]), tag_color, 3);
                    
                    // Draw corners with numbers
                    for (int j = 0; j < 4; j++) {
                        circle(color_result, Point(det->p[j][0], det->p[j][1]), 8, tag_color, -1);
                        circle(color_result, Point(det->p[j][0], det->p[j][1]), 8, Scalar(255, 255, 255), 1);
                        putText(color_result, to_string(j), Point(det->p[j][0] - 5, det->p[j][1] + 5),
                               FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
                    }
                    
                    // Draw center
                    circle(color_result, Point(det->c[0], det->c[1]), 10, Scalar(0, 0, 255), -1);
                    circle(color_result, Point(det->c[0], det->c[1]), 10, Scalar(255, 255, 255), 2);
                    
                    // Draw detailed label near tag
                    stringstream label;
                    label << "ID:" << det->id << " M:" << fixed << setprecision(1) << det->decision_margin;
                    int baseline = 0;
                    Size text_size = getTextSize(label.str(), FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline);
                    rectangle(color_result, Point(det->c[0] + 15, det->c[1] - text_size.height - 5),
                             Point(det->c[0] + 15 + text_size.width, det->c[1] + 5), Scalar(0, 0, 0), -1);
                    putText(color_result, label.str(), Point(det->c[0] + 15, det->c[1]),
                           FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 0), 2);
                    
                    // Draw in info box
                    stringstream info;
                    info << "Tag " << i << ": ID=" << det->id << " Margin=" << fixed << setprecision(1) << det->decision_margin;
                    putText(color_result, info.str(), Point(20, 70 + i * 30),
                           FONT_HERSHEY_SIMPLEX, 0.6, tag_color, 2);
                }
                
                // Clean up
                for (int i = 0; i < zarray_size(detections); i++) {
                    apriltag_detection_t* det;
                    zarray_get(detections, i, &det);
                    apriltag_detection_destroy(det);
                }
                zarray_destroy(detections);
            }
            break;
            
            default:
                cvtColor(result, color_result, COLOR_GRAY2BGR);
        }
        
        return color_result;
    }
    
    uint64_t extractCodeFromPattern(const vector<vector<int>>& pattern) {
        uint64_t code = 0;
        for (int i = 0; i < 36; i++) {
            // Convert 1-indexed to 0-indexed
            int x = TAG36H11_BIT_X[i] - 1;
            int y = TAG36H11_BIT_Y[i] - 1;
            
            // Extract bit (0 = white/bright, 1 = black/dark)
            int val = pattern[y][x];
            int bit = (val < 128) ? 1 : 0;
            
            code |= ((uint64_t)bit << i);
        }
        return code;
    }
    
    void updatePatternExtractionInfo(const vector<vector<int>>& pattern, const Mat& warped, int tagSize, int quad_index = 0, int image_index = 1) {
        if (!infoText_) return;
        
        stringstream ss;
        ss << "=== PATTERN EXTRACTION & QUALITY METRICS ===\n";
        ss << "Quad Number: " << quad_index << "\n\n";
        
        // Pattern statistics
        int black_count = 0;
        int white_count = 0;
        int total = 0;
        int black_sum = 0;
        int white_sum = 0;
        for (int row = 0; row < 6; row++) {
            for (int col = 0; col < 6; col++) {
                int val = pattern[row][col];
                total += val;
                if (val < 128) {
                    black_count++;
                    black_sum += val;
                } else {
                    white_count++;
                    white_sum += val;
                }
            }
        }
        int mean_pixel = (total / 36);
        int black_mean = black_count > 0 ? (black_sum / black_count) : 0;
        int white_mean = white_count > 0 ? (white_sum / white_count) : 255;
        int contrast = white_mean - black_mean;
        
        ss << "=== QUALITY METRICS ===\n\n";
        
        // Border Black Ratio (most important for detection quality)
        int border_black = 0;
        int border_total = 0;
        // Top 2 rows (border)
        for (int row = 0; row < 2; row++) {
            for (int col = 0; col < 6; col++) {
                border_total++;
                if (pattern[row][col] < 128) border_black++;
            }
        }
        // Bottom 2 rows (border)
        for (int row = 4; row < 6; row++) {
            for (int col = 0; col < 6; col++) {
                border_total++;
                if (pattern[row][col] < 128) border_black++;
            }
        }
        // Left/right borders (middle rows)
        for (int row = 2; row < 4; row++) {
            border_total += 2;
            if (pattern[row][0] < 128) border_black++;
            if (pattern[row][5] < 128) border_black++;
        }
        double border_ratio = border_total > 0 ? ((double)border_black / border_total) : 0.0;
        
        ss << "1. BORDER BLACK RATIO: " << fixed << setprecision(1) << (border_ratio * 100) << "%\n";
        if (border_ratio >= 0.90) {
            ss << "   ✓ EXCELLENT (should be ~100%)\n";
        } else if (border_ratio >= 0.80) {
            ss << "   ⚠ ACCEPTABLE (border has some issues)\n";
        } else {
            ss << "   ✗ POOR (broken border - detection issues!)\n";
        }
        ss << "   (" << border_black << "/" << border_total << " border cells)\n\n";
        
        ss << "2. PATTERN CONTRAST: " << contrast << "\n";
        if (contrast >= 100) {
            ss << "   ✓ EXCELLENT (clear distinction)\n";
        } else if (contrast >= 60) {
            ss << "   ⚠ GOOD (adequate contrast)\n";
        } else {
            ss << "   ✗ POOR (low contrast - may cause issues)\n";
        }
        ss << "   Black mean: " << black_mean << ", White mean: " << white_mean << "\n\n";
        
        ss << "3. PATTERN STATISTICS:\n";
        ss << "   Mean pixel value: " << mean_pixel << "\n";
        ss << "   Black pixels: " << black_count << " (mean: " << black_mean << ")\n";
        ss << "   White pixels: " << white_count << " (mean: " << white_mean << ")\n\n";
        
        // Display 6x6 pattern
        ss << "6x6 Pattern (0=white, 1=black):\n";
        ss << "    0 1 2 3 4 5\n";
        ss << "  ┌─────────────┐\n";
        for (int row = 0; row < 6; row++) {
            ss << row << " │";
            for (int col = 0; col < 6; col++) {
                int val = pattern[row][col];
                ss << ((val < 128) ? "1" : "0") << " ";
            }
            ss << "│\n";
        }
        ss << "  └─────────────┘\n\n";
        
        // Display pixel values
        ss << "Pixel Values:\n";
        for (int row = 0; row < 6; row++) {
            for (int col = 0; col < 6; col++) {
                ss << setw(4) << pattern[row][col];
            }
            ss << "\n";
        }
        ss << "\n";
        
        // Calculate warped tag quality metrics
        Scalar mean_val, stddev_val;
        meanStdDev(warped, mean_val, stddev_val);
        double warped_contrast = stddev_val[0];  // Standard deviation indicates contrast
        
        ss << "4. WARPED TAG QUALITY:\n";
        ss << "   Contrast (std dev): " << fixed << setprecision(2) << warped_contrast << "\n";
        if (warped_contrast >= 40) {
            ss << "   ✓ EXCELLENT (sharp, clear image)\n";
        } else if (warped_contrast >= 25) {
            ss << "   ⚠ GOOD (adequate sharpness)\n";
        } else {
            ss << "   ✗ POOR (blurry or low contrast)\n";
        }
        ss << "   Mean brightness: " << fixed << setprecision(1) << mean_val[0] << "\n\n";
        
        // Extract 36-bit code
        uint64_t code = extractCodeFromPattern(pattern);
        ss << "=== EXTRACTED CODE ===\n";
        ss << "Decimal: " << code << "\n";
        ss << "Hex: 0x" << hex << setfill('0') << setw(9) << code << dec << "\n";
        ss << "Binary: ";
        for (int i = 35; i >= 0; i--) {
            ss << ((code >> i) & 1);
            if (i % 9 == 0 && i > 0) ss << " ";
        }
        ss << "\n\n";
        
        // Show bit extraction order (abbreviated)
        ss << "=== BIT EXTRACTION (first 12 bits) ===\n";
        ss << "Bit  Pos    Value  Pixel\n";
        for (int i = 0; i < min(12, 36); i++) {
            int x = TAG36H11_BIT_X[i] - 1;
            int y = TAG36H11_BIT_Y[i] - 1;
            int val = pattern[y][x];
            int bit = (val < 128) ? 1 : 0;
            ss << setw(2) << i << "  (" << TAG36H11_BIT_X[i] << "," << TAG36H11_BIT_Y[i] << ")   " << bit;
            ss << "     " << val << "\n";
        }
        if (36 > 12) {
            ss << "... (" << (36 - 12) << " more bits)\n";
        }
        
        infoText_->setPlainText(QString::fromStdString(ss.str()));
        
        // Also update quality metrics box for pattern extraction
        QTextEdit* target_text = (image_index == 1) ? qualityText1_ : qualityText2_;
        if (target_text) {
            stringstream q_ss;
            q_ss << "PATTERN EXTRACTION QUALITY\n";
            q_ss << "Border ratio: " << fixed << setprecision(1) << (border_ratio * 100) << "%\n";
            if (border_ratio >= 0.90) {
                q_ss << "Status: ✓ EXCELLENT\n";
            } else if (border_ratio >= 0.80) {
                q_ss << "Status: ⚠ ACCEPTABLE\n";
            } else {
                q_ss << "Status: ✗ POOR\n";
            }
            q_ss << "Contrast: " << contrast << "\n";
            if (contrast >= 100) {
                q_ss << "Status: ✓ EXCELLENT\n";
            } else if (contrast >= 60) {
                q_ss << "Status: ⚠ GOOD\n";
            } else {
                q_ss << "Status: ✗ POOR\n";
            }
            q_ss << "Warped contrast: " << fixed << setprecision(1) << warped_contrast << "\n";
            if (warped_contrast >= 40) {
                q_ss << "Status: ✓ EXCELLENT\n";
            } else if (warped_contrast >= 25) {
                q_ss << "Status: ⚠ GOOD\n";
            } else {
                q_ss << "Status: ✗ POOR\n";
            }
            target_text->setPlainText(QString::fromStdString(q_ss.str()));
        }
    }
    
    void updateCornerRefinementInfo(double avg_movement, double max_movement, int image_index = 1) {
        if (!infoText_) return;
        
        stringstream ss;
        ss << "=== CORNER REFINEMENT & QUALITY ===\n\n";
        
        ss << "=== QUALITY METRICS ===\n\n";
        ss << "1. CORNER REFINEMENT ACCURACY:\n";
        ss << "   Average movement: " << fixed << setprecision(2) << avg_movement << " pixels\n";
        ss << "   Maximum movement: " << fixed << setprecision(2) << max_movement << " pixels\n";
        if (avg_movement < 1.0 && max_movement < 2.0) {
            ss << "   ✓ EXCELLENT (very stable corners)\n";
        } else if (avg_movement < 2.0 && max_movement < 5.0) {
            ss << "   ⚠ GOOD (acceptable stability)\n";
        } else {
            ss << "   ✗ POOR (unstable corners - may indicate blur/poor lighting)\n";
        }
        ss << "\n";
        
        ss << "Process:\n";
        ss << "1. Detect quadrilaterals from contours\n";
        ss << "2. Apply cornerSubPix() for sub-pixel refinement\n";
        ss << "3. Visualize original vs refined positions\n\n";
        ss << "Interpretation:\n";
        ss << "• Small movements (< 1-2 px) = stable, accurate corners\n";
        ss << "• Large movements (> 5 px) = unstable, may indicate:\n";
        ss << "  - Image blur\n";
        ss << "  - Poor lighting/contrast\n";
        ss << "  - Low resolution tag\n";
        ss << "  - Motion blur\n";
        
        infoText_->setPlainText(QString::fromStdString(ss.str()));
        
        // Also update quality metrics box
        QTextEdit* target_text = (image_index == 1) ? qualityText1_ : qualityText2_;
        if (target_text) {
            stringstream q_ss;
            q_ss << "CORNER REFINEMENT QUALITY\n";
            q_ss << "Avg movement: " << fixed << setprecision(2) << avg_movement << " px\n";
            q_ss << "Max movement: " << fixed << setprecision(2) << max_movement << " px\n";
            if (avg_movement < 1.0 && max_movement < 2.0) {
                q_ss << "Status: ✓ EXCELLENT\n";
            } else if (avg_movement < 2.0 && max_movement < 5.0) {
                q_ss << "Status: ⚠ GOOD\n";
            } else {
                q_ss << "Status: ✗ POOR\n";
            }
            target_text->setPlainText(QString::fromStdString(q_ss.str()));
        }
    }
    
    void updateHammingDecodeInfo(zarray_t* detections, int selected_tag = 0, int image_index = 1) {
        if (!infoText_) return;
        
        stringstream ss;
        ss << "=== HAMMING CODE DECODING & QUALITY ===\n\n";
        ss << "Detected Tags: " << zarray_size(detections) << "\n";
        ss << "Selected Tag: " << selected_tag << "\n\n";
        
        if (zarray_size(detections) == 0) {
            ss << "No tags detected.\n\n";
            ss << "Possible reasons:\n";
            ss << "  • Pattern doesn't match any valid codeword\n";
            ss << "  • Too many bit errors (> 5 for Tag36h11)\n";
            ss << "  • Hamming distance exceeds correction limit\n";
            ss << "  • Broken border (border black ratio < 80%)\n";
            ss << "  • Low contrast pattern\n";
            infoText_->setPlainText(QString::fromStdString(ss.str()));
            return;
        }
        
        // Show selected tag in detail with quality metrics
        if (selected_tag >= 0 && selected_tag < zarray_size(detections)) {
            apriltag_detection_t* det;
            zarray_get(detections, selected_tag, &det);
            
            ss << "=== SELECTED TAG " << selected_tag << " ===\n\n";
            
            ss << "=== QUALITY METRICS ===\n\n";
            
            ss << "1. DECISION MARGIN: " << fixed << setprecision(2) << det->decision_margin << "\n";
            if (det->decision_margin > 60) {
                ss << "   ✓ EXCELLENT (high confidence, very reliable)\n";
            } else if (det->decision_margin > 30) {
                ss << "   ⚠ GOOD (medium confidence, reliable)\n";
            } else {
                ss << "   ✗ POOR (low confidence, may be unreliable)\n";
            }
            ss << "   Range: 0-255 (higher = better)\n";
            ss << "   Measures: Average difference between bit intensity\n";
            ss << "             and decision threshold\n\n";
            
            ss << "2. HAMMING DISTANCE: " << static_cast<int>(det->hamming) << " errors\n";
            if (det->hamming == 0) {
                ss << "   ✓ PERFECT (no errors)\n";
            } else if (det->hamming <= 5) {
                ss << "   ✓ GOOD (correctable errors for Tag36h11)\n";
            } else {
                ss << "   ✗ POOR (uncorrectable - exceeds 5 error limit)\n";
            }
            ss << "   Tag36h11 can correct up to 5 bit errors\n\n";
            
            ss << "=== TAG INFORMATION ===\n";
            ss << "ID: " << det->id << "\n";
            ss << "Center: (" << fixed << setprecision(2) << det->c[0] << ", " << det->c[1] << ")\n";
            ss << "Corners:\n";
            for (int j = 0; j < 4; j++) {
                ss << "  " << j << ": (" << fixed << setprecision(2) 
                   << det->p[j][0] << ", " << det->p[j][1] << ")\n";
            }
            ss << "\n";
        }
        
        // Show summary of all tags
        if (zarray_size(detections) > 1) {
            ss << "=== ALL TAGS ===\n";
            for (int i = 0; i < zarray_size(detections); i++) {
                apriltag_detection_t* det;
                zarray_get(detections, i, &det);
                ss << "Tag " << i << ": ID=" << det->id 
                   << ", Margin=" << fixed << setprecision(1) << det->decision_margin;
                if (i == selected_tag) ss << " [SELECTED]";
                ss << "\n";
            }
            ss << "\n";
        }
        
        ss << "=== DECODING PROCESS ===\n";
        ss << "1. Extract 6x6 pattern from warped tag\n";
        ss << "2. Sample 36 bits using Tag36h11 bit positions\n";
        ss << "3. Try all 4 rotations (0°, 90°, 180°, 270°)\n";
        ss << "4. Calculate Hamming distance to valid codewords\n";
        ss << "5. Correct up to 5 bit errors (Tag36h11 capability)\n";
        ss << "6. Return tag ID if within error correction limit\n\n";
        
        ss << "=== QUALITY ASSESSMENT ===\n";
        ss << "Best overall quality indicator: Decision Margin\n";
        ss << "• > 60: Excellent - very reliable detection\n";
        ss << "• 30-60: Good - reliable for most applications\n";
        ss << "• < 30: Poor - may have false positives/negatives\n";
        
        infoText_->setPlainText(QString::fromStdString(ss.str()));
    }
    
    void updateQualityMetrics(const Mat& img, int advancedMode, int image_index) {
        QTextEdit* target_text = (image_index == 1) ? qualityText1_ : qualityText2_;
        if (!target_text) return;
        
        if (advancedMode == 0) {
            // Basic detection metrics
            image_u8_t im = {
                .width = img.cols,
                .height = img.rows,
                .stride = img.cols,
                .buf = const_cast<uint8_t*>(img.data)
            };
            
            zarray_t* detections = apriltag_detector_detect(td_, &im);
            int num_dets = zarray_size(detections);
            
            stringstream ss;
            ss << "BASIC DETECTION\n";
            ss << "Tags detected: " << num_dets << "\n";
            if (num_dets > 0) {
                ss << "Status: ✓ TAGS FOUND\n";
            } else {
                ss << "Status: ✗ NO TAGS\n";
            }
            
            // Clean up
            for (int i = 0; i < zarray_size(detections); i++) {
                apriltag_detection_t* det;
                zarray_get(detections, i, &det);
                apriltag_detection_destroy(det);
            }
            zarray_destroy(detections);
            
            target_text->setPlainText(QString::fromStdString(ss.str()));
        } else if (advancedMode == 2) {
            // Warped Tags - show basic quality info
            stringstream ss;
            ss << "WARPED TAGS\n";
            ss << "Check visual quality:\n";
            ss << "• Square shape (36x36)\n";
            ss << "• Sharp edges\n";
            ss << "• Good contrast\n";
            target_text->setPlainText(QString::fromStdString(ss.str()));
        }
    }
    
    void updateInfoPanel(int advancedMode) {
        if (!infoText_) return;
        
        if (advancedMode == 0) {
            infoText_->setPlainText("Select an Advanced visualization mode to see extracted codes, decoding steps, and quality metrics here.");
        } else if (advancedMode == 1) {
            // Corner refinement info will be updated when processing happens
            infoText_->setPlainText("=== CORNER REFINEMENT ===\n\n"
                                   "Processing... Quality metrics will appear here.\n\n"
                                   "This stage refines quadrilateral corners using sub-pixel accuracy.\n\n"
                                   "Green = Original corners\n"
                                   "Red = Refined corners\n"
                                   "Yellow lines = Movement vectors");
        } else if (advancedMode == 2) {
            stringstream ss;
            ss << "=== WARPED TAGS & QUALITY ===\n\n";
            ss << "=== QUALITY METRICS ===\n\n";
            ss << "1. WARP QUALITY:\n";
            ss << "   • Check if warped tag is square (36x36)\n";
            ss << "   • Look for clean, sharp edges\n";
            ss << "   • Distorted/blurry = poor quality\n\n";
            ss << "Process:\n";
            ss << "1. Compute homography matrix from quad to square\n";
            ss << "2. Warp tag to 36x36 pixel square\n";
            ss << "3. Normalize orientation for pattern extraction\n\n";
            ss << "Good quality: Clean, square, sharp warped image\n";
            ss << "Poor quality: Distorted, blurry, or non-square\n";
            infoText_->setPlainText(QString::fromStdString(ss.str()));
        }
    }

    QPixmap matToPixmap(const Mat &mat) {
        if (mat.empty()) {
            return QPixmap();
        }
        
        Mat display;
        if (mat.channels() == 1) {
            cvtColor(mat, display, COLOR_GRAY2RGB);
        } else {
            cvtColor(mat, display, COLOR_BGR2RGB);
        }
        
        QImage qimg(display.data, display.cols, display.rows, display.step, QImage::Format_RGB888);
        return QPixmap::fromImage(qimg).scaled(640, 480, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    }

    void processImages() {
        if (image1_.empty() && image2_.empty()) {
            return;
        }
        
        // Process Image 1
        if (!image1_.empty()) {
            Mat img1 = image1_.clone();
            
            // Apply mirror if needed (before preprocessing for better results)
            bool use_mirror = mirrorCheckbox1_ ? mirrorCheckbox1_->isChecked() : false;
            if (use_mirror) {
                flip(img1, img1, 1);
            }
            
            Mat processed1 = preprocessImage(img1, preprocessCombo_->currentIndex());
            Mat edged1 = applyEdgeDetection(processed1, edgeCombo_->currentIndex());
            
            bool use_advanced = advancedCombo_ && advancedCombo_->currentIndex() > 0;
            Mat final1;
            if (use_advanced) {
                int advMode = advancedCombo_->currentIndex();
                final1 = drawAdvancedVisualization(edged1, advMode, 1);  // Pass image_index = 1
                updateInfoPanel(advMode);
                updateQualityMetrics(edged1, advMode, 1);  // Update quality metrics for image 1
            } else {
                final1 = drawDetections(edged1, detectionCombo_->currentIndex());
                if (infoText_) {
                    infoText_->setPlainText("Select an Advanced visualization mode to see extracted codes and decoding steps here.");
                }
                updateQualityMetrics(edged1, 0, 1);  // Update quality metrics for image 1 (basic detection)
            }
            label1_->setPixmap(matToPixmap(final1));
        } else {
            label1_->setText("Load Image 1");
        }
        
        // Process Image 2
        if (!image2_.empty()) {
            Mat img2 = image2_.clone();
            
            // Apply mirror if needed
            bool use_mirror = mirrorCheckbox2_ ? mirrorCheckbox2_->isChecked() : false;
            if (use_mirror) {
                flip(img2, img2, 1);
            }
            
            Mat processed2 = preprocessImage(img2, preprocessCombo_->currentIndex());
            Mat edged2 = applyEdgeDetection(processed2, edgeCombo_->currentIndex());
            
            bool use_advanced = advancedCombo_ && advancedCombo_->currentIndex() > 0;
            Mat final2;
            if (use_advanced) {
                int advMode = advancedCombo_->currentIndex();
                final2 = drawAdvancedVisualization(edged2, advMode, 2);  // Pass image index 2
                updateQualityMetrics(edged2, advMode, 2);  // Update quality metrics for image 2
            } else {
                final2 = drawDetections(edged2, detectionCombo_->currentIndex());
                updateQualityMetrics(edged2, 0, 2);  // No advanced mode, but still show basic metrics
            }
            label2_->setPixmap(matToPixmap(final2));
        } else {
            label2_->setText("Load Image 2");
        }
    }
    
    void setupCaptureTab(QVBoxLayout *layout) {
        // Main horizontal split layout (left: controls, right: preview/pattern)
        QHBoxLayout *mainSplitLayout = new QHBoxLayout();
        
        // ========== LEFT HALF: Controls ==========
        QVBoxLayout *leftControlsLayout = new QVBoxLayout();
        
        // Camera selection
        QGroupBox *cameraGroup = new QGroupBox("Camera Selection", this);
        QHBoxLayout *cameraLayout = new QHBoxLayout();
        QLabel *cameraLabel = new QLabel("Camera:", this);
        cameraCombo_ = new QComboBox(this);
        connect(cameraCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), 
                this, &AprilTagDebugGUI::openCamera);
        cameraLayout->addWidget(cameraLabel);
        cameraLayout->addWidget(cameraCombo_);
        cameraLayout->addStretch();
        cameraGroup->setLayout(cameraLayout);
        leftControlsLayout->addWidget(cameraGroup);
        
        // Resolution/FPS selection
        QGroupBox *modeGroup = new QGroupBox("Resolution & FPS", this);
        QHBoxLayout *modeLayout = new QHBoxLayout();
        QLabel *modeLabel = new QLabel("Mode:", this);
        modeCombo_ = new QComboBox(this);
        modeCombo_->setEnabled(false);  // Enabled when camera opens
        connect(modeCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), 
                this, &AprilTagDebugGUI::onModeChanged);
        modeLayout->addWidget(modeLabel);
        modeLayout->addWidget(modeCombo_);
        modeLayout->addStretch();
        modeGroup->setLayout(modeLayout);
        leftControlsLayout->addWidget(modeGroup);
        
        // Camera settings
        QGroupBox *settingsGroup = new QGroupBox("Camera Settings", this);
        QFormLayout *settingsLayout = new QFormLayout();
        
        // Exposure
        exposureSlider_ = new QSlider(Qt::Horizontal, this);
        exposureSlider_->setRange(0, 100);
        exposureSlider_->setValue(50);
        exposureSpin_ = new QSpinBox(this);
        exposureSpin_->setRange(0, 100);
        exposureSpin_->setValue(50);
        connect(exposureSlider_, &QSlider::valueChanged, exposureSpin_, &QSpinBox::setValue);
        connect(exposureSpin_, QOverload<int>::of(&QSpinBox::valueChanged), 
                exposureSlider_, &QSlider::setValue);
        connect(exposureSlider_, &QSlider::valueChanged, this, &AprilTagDebugGUI::updateCameraSettings);
        QHBoxLayout *exposureLayout = new QHBoxLayout();
        exposureLayout->addWidget(exposureSlider_);
        exposureLayout->addWidget(exposureSpin_);
        settingsLayout->addRow("Exposure:", exposureLayout);
        
        // Gain
        gainSlider_ = new QSlider(Qt::Horizontal, this);
        gainSlider_->setRange(0, 100);
        gainSlider_->setValue(50);
        gainSpin_ = new QSpinBox(this);
        gainSpin_->setRange(0, 100);
        gainSpin_->setValue(50);
        connect(gainSlider_, &QSlider::valueChanged, gainSpin_, &QSpinBox::setValue);
        connect(gainSpin_, QOverload<int>::of(&QSpinBox::valueChanged), 
                gainSlider_, &QSlider::setValue);
        connect(gainSlider_, &QSlider::valueChanged, this, &AprilTagDebugGUI::updateCameraSettings);
        QHBoxLayout *gainLayout = new QHBoxLayout();
        gainLayout->addWidget(gainSlider_);
        gainLayout->addWidget(gainSpin_);
        settingsLayout->addRow("Gain:", gainLayout);
        
        // Brightness
        brightnessSlider_ = new QSlider(Qt::Horizontal, this);
        brightnessSlider_->setRange(0, 255);
        brightnessSlider_->setValue(128);
        brightnessSpin_ = new QSpinBox(this);
        brightnessSpin_->setRange(0, 255);
        brightnessSpin_->setValue(128);
        connect(brightnessSlider_, &QSlider::valueChanged, brightnessSpin_, &QSpinBox::setValue);
        connect(brightnessSpin_, QOverload<int>::of(&QSpinBox::valueChanged), 
                brightnessSlider_, &QSlider::setValue);
        connect(brightnessSlider_, &QSlider::valueChanged, this, &AprilTagDebugGUI::updateCameraSettings);
        QHBoxLayout *brightnessLayout = new QHBoxLayout();
        brightnessLayout->addWidget(brightnessSlider_);
        brightnessLayout->addWidget(brightnessSpin_);
        settingsLayout->addRow("Brightness:", brightnessLayout);
        
        // Contrast (MindVision and V4L2)
        contrastSlider_ = new QSlider(Qt::Horizontal, this);
        contrastSlider_->setRange(0, 100);
        contrastSlider_->setValue(50);
        contrastSpin_ = new QSpinBox(this);
        contrastSpin_->setRange(0, 100);
        contrastSpin_->setValue(50);
        connect(contrastSlider_, &QSlider::valueChanged, contrastSpin_, &QSpinBox::setValue);
        connect(contrastSpin_, QOverload<int>::of(&QSpinBox::valueChanged), 
                contrastSlider_, &QSlider::setValue);
        connect(contrastSlider_, &QSlider::valueChanged, this, &AprilTagDebugGUI::updateCameraSettings);
        QHBoxLayout *contrastLayout = new QHBoxLayout();
        contrastLayout->addWidget(contrastSlider_);
        contrastLayout->addWidget(contrastSpin_);
        settingsLayout->addRow("Contrast:", contrastLayout);
        
        // Saturation (MindVision and V4L2)
        saturationSlider_ = new QSlider(Qt::Horizontal, this);
        saturationSlider_->setRange(0, 100);
        saturationSlider_->setValue(50);
        saturationSpin_ = new QSpinBox(this);
        saturationSpin_->setRange(0, 100);
        saturationSpin_->setValue(50);
        connect(saturationSlider_, &QSlider::valueChanged, saturationSpin_, &QSpinBox::setValue);
        connect(saturationSpin_, QOverload<int>::of(&QSpinBox::valueChanged), 
                saturationSlider_, &QSlider::setValue);
        connect(saturationSlider_, &QSlider::valueChanged, this, &AprilTagDebugGUI::updateCameraSettings);
        QHBoxLayout *saturationLayout = new QHBoxLayout();
        saturationLayout->addWidget(saturationSlider_);
        saturationLayout->addWidget(saturationSpin_);
        settingsLayout->addRow("Saturation:", saturationLayout);
        
        // Sharpness (MindVision and V4L2)
        sharpnessSlider_ = new QSlider(Qt::Horizontal, this);
        sharpnessSlider_->setRange(0, 100);
        sharpnessSlider_->setValue(50);
        sharpnessSpin_ = new QSpinBox(this);
        sharpnessSpin_->setRange(0, 100);
        sharpnessSpin_->setValue(50);
        connect(sharpnessSlider_, &QSlider::valueChanged, sharpnessSpin_, &QSpinBox::setValue);
        connect(sharpnessSpin_, QOverload<int>::of(&QSpinBox::valueChanged), 
                sharpnessSlider_, &QSlider::setValue);
        connect(sharpnessSlider_, &QSlider::valueChanged, this, &AprilTagDebugGUI::updateCameraSettings);
        QHBoxLayout *sharpnessLayout = new QHBoxLayout();
        sharpnessLayout->addWidget(sharpnessSlider_);
        sharpnessLayout->addWidget(sharpnessSpin_);
        settingsLayout->addRow("Sharpness:", sharpnessLayout);
        
        settingsGroup->setLayout(settingsLayout);
        leftControlsLayout->addWidget(settingsGroup);
        
        // Algorithm selection for preview
        QGroupBox *algorithmGroup = new QGroupBox("AprilTag Detection (Preview)", this);
        QVBoxLayout *algorithmLayout = new QVBoxLayout();
        
        QHBoxLayout *algorithmSelectLayout = new QHBoxLayout();
        QLabel *algorithmLabel = new QLabel("Algorithm:", this);
        captureAlgorithmCombo_ = new QComboBox(this);
        captureAlgorithmCombo_->addItem("None");
        captureAlgorithmCombo_->addItem("OpenCV CPU (AprilTag)");
#ifdef HAVE_CUDA_APRILTAG
        captureAlgorithmCombo_->addItem("Fast AprilTag");
#endif
        connect(captureAlgorithmCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
                this, &AprilTagDebugGUI::onCaptureAlgorithmChanged);
        algorithmSelectLayout->addWidget(algorithmLabel);
        algorithmSelectLayout->addWidget(captureAlgorithmCombo_);
        algorithmSelectLayout->addStretch();
        algorithmLayout->addLayout(algorithmSelectLayout);
        
        captureMirrorCheckbox_ = new QCheckBox("Mirror (Horizontal Flip)", this);
        algorithmLayout->addWidget(captureMirrorCheckbox_);
        
        algorithmGroup->setLayout(algorithmLayout);
        leftControlsLayout->addWidget(algorithmGroup);
        
        // Controls on left side
        loadImageBtn_ = new QPushButton("Load Image", this);
        connect(loadImageBtn_, &QPushButton::clicked, this, &AprilTagDebugGUI::loadImage1);
        leftControlsLayout->addWidget(loadImageBtn_);
        
        captureBtn_ = new QPushButton("Capture Frame", this);
        captureBtn_->setEnabled(false);
        connect(captureBtn_, &QPushButton::clicked, this, &AprilTagDebugGUI::captureFrame);
        leftControlsLayout->addWidget(captureBtn_);
        
        saveSettingsBtn_ = new QPushButton("Save Camera Settings", this);
        saveSettingsBtn_->setEnabled(false);
        connect(saveSettingsBtn_, &QPushButton::clicked, this, &AprilTagDebugGUI::saveCameraSettings);
        leftControlsLayout->addWidget(saveSettingsBtn_);
        
        // Video preview on left side (bottom)
        previewLabel_ = new QLabel(this);
        previewLabel_->setMinimumSize(640, 480);
        previewLabel_->setAlignment(Qt::AlignCenter);
        previewLabel_->setStyleSheet("border: 1px solid black; background-color: #f0f0f0;");
        previewLabel_->setText("No camera selected");
        previewLabel_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        leftControlsLayout->addWidget(previewLabel_);
        
        // ========== RIGHT HALF: Pattern Visualization Only ==========
        QVBoxLayout *rightPreviewLayout = new QVBoxLayout();
        
        // Pattern visualization
        QGroupBox *patternGroup = new QGroupBox("Pattern Visualization", this);
        QVBoxLayout *patternGroupLayout = new QVBoxLayout();
        capturePatternLabel_ = new QLabel("No detection", this);
        capturePatternLabel_->setMinimumSize(400, 400);
        capturePatternLabel_->setAlignment(Qt::AlignCenter);
        capturePatternLabel_->setStyleSheet("border: 1px solid black; background-color: #f0f0f0;");
        capturePatternLabel_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        patternGroupLayout->addWidget(capturePatternLabel_);
        
        // Save patterns button
        savePatternsBtn_ = new QPushButton("Save Pattern Visualizations", this);
        savePatternsBtn_->setEnabled(false);
        connect(savePatternsBtn_, &QPushButton::clicked, this, &AprilTagDebugGUI::savePatternVisualizations);
        patternGroupLayout->addWidget(savePatternsBtn_);
        
        patternGroup->setLayout(patternGroupLayout);
        rightPreviewLayout->addWidget(patternGroup);
        
        // Pattern info text
        capturePatternInfoText_ = new QTextEdit(this);
        capturePatternInfoText_->setReadOnly(true);
        capturePatternInfoText_->setFont(QFont("Courier", 9));
        capturePatternInfoText_->setPlainText("Pattern and Hamming code will appear here when tags are detected.");
        capturePatternInfoText_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        rightPreviewLayout->addWidget(capturePatternInfoText_);
        
        // Add both halves to main split layout with equal stretch
        QWidget *leftContainer = new QWidget(this);
        leftContainer->setLayout(leftControlsLayout);
        mainSplitLayout->addWidget(leftContainer, 1);  // Stretch factor 1 (50%)
        
        QWidget *rightContainer = new QWidget(this);
        rightContainer->setLayout(rightPreviewLayout);
        mainSplitLayout->addWidget(rightContainer, 1);  // Stretch factor 1 (50%)
        
        layout->addLayout(mainSplitLayout);
        
        // Setup preview timer FIRST (before enumerateCameras, which might trigger signals)
        previewTimer_ = new QTimer(this);
        connect(previewTimer_, &QTimer::timeout, this, &AprilTagDebugGUI::updatePreview);
        
        // Initialize camera enumeration (this may trigger signals, so previewTimer_ must exist)
        cameraCombo_->blockSignals(true);  // Block signals during enumeration
        enumerateCameras();
        cameraCombo_->blockSignals(false);  // Re-enable signals
        
        // Calibration preview timer (separate timer for calibration view)
        calibrationPreviewTimer_ = new QTimer(this);
        connect(calibrationPreviewTimer_, &QTimer::timeout, this, &AprilTagDebugGUI::updateCalibrationPreview);
    }
    
    void setupFisheyeTab() {
        // This will be called from setupUI after tabWidget_ is created
        QWidget *fisheyeTab = new QWidget(this);
        QVBoxLayout *fisheyeLayout = new QVBoxLayout(fisheyeTab);
        setupFisheyeTabContent(fisheyeLayout);
        tabWidget_->addTab(fisheyeTab, "Fisheye Correction");
    }
    
    void setupAlgorithmsTab(QVBoxLayout *layout) {
        // Algorithm selection
        QGroupBox *algorithmGroup = new QGroupBox("Algorithm Selection", this);
        QHBoxLayout *algorithmLayout = new QHBoxLayout();
        
        QLabel *algorithmLabel = new QLabel("Algorithm:", this);
        algorithmCombo_ = new QComboBox(this);
        algorithmCombo_->addItem("OpenCV CPU (AprilTag)");
        algorithmCombo_->addItem("Fast AprilTag");
        algorithmLayout->addWidget(algorithmLabel);
        algorithmLayout->addWidget(algorithmCombo_);
        algorithmLayout->addStretch();
        algorithmGroup->setLayout(algorithmLayout);
        layout->addWidget(algorithmGroup);
        
        // Camera selection (for algorithm processing)
        QGroupBox *cameraGroup = new QGroupBox("Camera Selection", this);
        QHBoxLayout *cameraLayout = new QHBoxLayout();
        
        QLabel *cameraLabel = new QLabel("Camera:", this);
        algorithmCameraCombo_ = new QComboBox(this);
        // Populate from already-enumerated cameras (Capture tab enumerates them first)
        // Use the same cameraList_ that was populated in setupCaptureTab()
        for (size_t i = 0; i < cameraList_.size(); i++) {
            algorithmCameraCombo_->addItem(QString::fromStdString(cameraList_[i]));
        }
        
        cameraLayout->addWidget(cameraLabel);
        cameraLayout->addWidget(algorithmCameraCombo_);
        cameraLayout->addStretch();
        cameraGroup->setLayout(cameraLayout);
        layout->addWidget(cameraGroup);
        
        // Control buttons
        QHBoxLayout *buttonLayout = new QHBoxLayout();
        algorithmStartBtn_ = new QPushButton("Start", this);
        algorithmStopBtn_ = new QPushButton("Stop", this);
        algorithmStopBtn_->setEnabled(false);
        connect(algorithmStartBtn_, &QPushButton::clicked, this, &AprilTagDebugGUI::startAlgorithm);
        connect(algorithmStopBtn_, &QPushButton::clicked, this, &AprilTagDebugGUI::stopAlgorithm);
        buttonLayout->addWidget(algorithmStartBtn_);
        buttonLayout->addWidget(algorithmStopBtn_);
        
        // Mirror checkbox (must be created before connecting signals)
        algorithmMirrorCheckbox_ = new QCheckBox("Mirror (Horizontal Flip)", this);
        buttonLayout->addWidget(algorithmMirrorCheckbox_);
        
        // Connect camera selection change to set mirror checkbox for MindVision cameras
        // (after checkbox is created)
        connect(algorithmCameraCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
                this, &AprilTagDebugGUI::onAlgorithmCameraChanged);
        
        // Set initial mirror checkbox state if a MindVision camera is selected
        if (algorithmCameraCombo_->count() > 0) {
            onAlgorithmCameraChanged(algorithmCameraCombo_->currentIndex());
        }
        
        buttonLayout->addStretch();
        layout->addLayout(buttonLayout);
        
        // Top row: Video preview (left) + Algorithm Timing Analysis (right, full height)
        QHBoxLayout *previewAndTimingLayout = new QHBoxLayout();
        
        // Left side: Video preview with Quality Metrics and Pose Estimation below
        QVBoxLayout *leftSideLayout = new QVBoxLayout();
        
        // Video preview
        QGroupBox *displayGroup = new QGroupBox("Live Preview", this);
        QVBoxLayout *displayLayout = new QVBoxLayout();
        algorithmDisplayLabel_ = new QLabel("No video", this);
        algorithmDisplayLabel_->setMinimumSize(480, 360);  // Reduced from 640x480
        algorithmDisplayLabel_->setAlignment(Qt::AlignCenter);
        algorithmDisplayLabel_->setStyleSheet("border: 1px solid gray; background-color: black; color: white;");
        displayLayout->addWidget(algorithmDisplayLabel_);
        displayGroup->setLayout(displayLayout);
        leftSideLayout->addWidget(displayGroup);
        
        // Quality Metrics and Pose Estimation boxes below video (same style as Algorithm Timing Analysis)
        QHBoxLayout *metricsLayout = new QHBoxLayout();
        
        // Quality Metrics box
        QGroupBox *qualityGroup = new QGroupBox("Quality Metrics", this);
        QVBoxLayout *qualityLayout = new QVBoxLayout();
        algorithmQualityText_ = new QTextEdit(this);
        algorithmQualityText_->setReadOnly(true);
        algorithmQualityText_->setFont(QFont("Courier", 9));  // Monospace font for alignment (same as timing analysis)
        algorithmQualityText_->setPlainText("No detections");
        qualityLayout->addWidget(algorithmQualityText_, 1);  // Expand to fill space (same as timing analysis)
        qualityGroup->setLayout(qualityLayout);
        metricsLayout->addWidget(qualityGroup);
        
        // Pose box
        QGroupBox *poseGroup = new QGroupBox("Pose Estimation", this);
        QVBoxLayout *poseLayout = new QVBoxLayout();
        algorithmPoseText_ = new QTextEdit(this);
        algorithmPoseText_->setReadOnly(true);
        algorithmPoseText_->setFont(QFont("Courier", 9));  // Monospace font for alignment (same as timing analysis)
        algorithmPoseText_->setPlainText("No pose data");
        poseLayout->addWidget(algorithmPoseText_, 1);  // Expand to fill space (same as timing analysis)
        poseGroup->setLayout(poseLayout);
        metricsLayout->addWidget(poseGroup);
        
        leftSideLayout->addLayout(metricsLayout);
        previewAndTimingLayout->addLayout(leftSideLayout, 2);  // Takes 2 parts of space
        
        // Right side: Algorithm Timing Analysis (full height, goes all the way down)
        QGroupBox *detailedTimingGroup = new QGroupBox("Algorithm Timing Analysis", this);
        QVBoxLayout *detailedTimingLayout = new QVBoxLayout();
        
        // FPS counters at the top
        algorithmFPSLabel_ = new QLabel("FPS: Capture: 0.0, Detection: 0.0, Display: 0.0", this);
        algorithmFPSLabel_->setStyleSheet("font-size: 14pt; font-weight: bold; color: #0066cc;");
        detailedTimingLayout->addWidget(algorithmFPSLabel_);
        
        // Detailed timing text (expands to fill available space)
        algorithmDetailedTimingText_ = new QTextEdit(this);
        algorithmDetailedTimingText_->setReadOnly(true);
        algorithmDetailedTimingText_->setFont(QFont("Courier", 9));  // Monospace font for alignment
        algorithmDetailedTimingText_->setPlainText(
            "Timing analysis will appear here when an algorithm is running.\n\n"
            "Supported algorithms:\n"
            "- CPU AprilTag: Shows grayscale conversion, mirror, and detection timing\n"
            "- Fast AprilTag: Shows detailed GPU/CPU breakdown including:\n"
            "  - DetectGpuOnly\n"
            "  - FitQuads\n"
            "  - Mirror operations\n"
            "  - CopyGrayHostTo\n"
            "  - DecodeTags\n"
            "  - ScaleCoords\n"
            "  - FilterDuplicates\n\n"
            "Updated every 10 frames."
        );
        detailedTimingLayout->addWidget(algorithmDetailedTimingText_, 1);  // Expand to fill space
        detailedTimingGroup->setLayout(detailedTimingLayout);
        previewAndTimingLayout->addWidget(detailedTimingGroup, 1);  // Takes 1 part of space
        
        layout->addLayout(previewAndTimingLayout);
    }
    
    void setupFisheyeTabContent(QVBoxLayout *layout) {
        // Calibration file path
        QGroupBox *calibGroup = new QGroupBox("Calibration File", this);
        QFormLayout *calibLayout = new QFormLayout();
        
        QLabel *calibPathLabel = new QLabel("Path:", this);
        calibPathEdit_ = new QLineEdit("/home/nav/9202/Hiru/Apriltag/calibration_data/camera_params.yaml", this);
        QPushButton *browseBtn = new QPushButton("Browse...", this);
        QPushButton *loadBtn = new QPushButton("Load Calibration", this);
        connect(browseBtn, &QPushButton::clicked, this, &AprilTagDebugGUI::browseCalibrationFile);
        connect(loadBtn, &QPushButton::clicked, this, &AprilTagDebugGUI::loadCalibrationFromUI);
        
        QHBoxLayout *pathLayout = new QHBoxLayout();
        pathLayout->addWidget(calibPathEdit_);
        pathLayout->addWidget(browseBtn);
        pathLayout->addWidget(loadBtn);
        calibLayout->addRow(calibPathLabel, pathLayout);
        calibGroup->setLayout(calibLayout);
        layout->addWidget(calibGroup);
        
        // Status label
        QLabel *calibStatusLabel = new QLabel("Calibration: Not loaded", this);
        layout->addWidget(calibStatusLabel);
        
        // Store reference to update it
        fisheyeStatusLabel_ = calibStatusLabel;
        
        // Test image selection
        QGroupBox *testGroup = new QGroupBox("Preview Correction", this);
        QVBoxLayout *testLayout = new QVBoxLayout();
        
        QPushButton *loadTestImageBtn = new QPushButton("Load Test Image", this);
        connect(loadTestImageBtn, &QPushButton::clicked, this, &AprilTagDebugGUI::loadTestImageForFisheye);
        testLayout->addWidget(loadTestImageBtn);
        
        // Side-by-side preview
        QHBoxLayout *previewLayout = new QHBoxLayout();
        fisheyeOriginalLabel_ = new QLabel("Original", this);
        fisheyeOriginalLabel_->setMinimumSize(320, 240);
        fisheyeOriginalLabel_->setAlignment(Qt::AlignCenter);
        fisheyeOriginalLabel_->setStyleSheet("border: 1px solid black; background-color: #f0f0f0;");
        fisheyeOriginalLabel_->setText("Load an image to preview");
        
        fisheyeCorrectedLabel_ = new QLabel("Corrected", this);
        fisheyeCorrectedLabel_->setMinimumSize(320, 240);
        fisheyeCorrectedLabel_->setAlignment(Qt::AlignCenter);
        fisheyeCorrectedLabel_->setStyleSheet("border: 1px solid black; background-color: #f0f0f0;");
        fisheyeCorrectedLabel_->setText("Load an image to preview");
        
        previewLayout->addWidget(fisheyeOriginalLabel_);
        previewLayout->addWidget(fisheyeCorrectedLabel_);
        testLayout->addLayout(previewLayout);
        
        // Selection radio buttons
        QHBoxLayout *selectionLayout = new QHBoxLayout();
        selectionLayout->addWidget(new QLabel("Use for captures:", this));
        fisheyeUseOriginalRadio_ = new QRadioButton("Original (No Correction)", this);
        fisheyeUseCorrectedRadio_ = new QRadioButton("Corrected (With Fisheye Correction)", this);
        fisheyeUseCorrectedRadio_->setChecked(true);  // Default to corrected
        connect(fisheyeUseOriginalRadio_, &QRadioButton::toggled, this, &AprilTagDebugGUI::onFisheyeSelectionChanged);
        connect(fisheyeUseCorrectedRadio_, &QRadioButton::toggled, this, &AprilTagDebugGUI::onFisheyeSelectionChanged);
        selectionLayout->addWidget(fisheyeUseOriginalRadio_);
        selectionLayout->addWidget(fisheyeUseCorrectedRadio_);
        selectionLayout->addStretch();
        testLayout->addLayout(selectionLayout);
        
        testGroup->setLayout(testLayout);
        layout->addWidget(testGroup);
        
        // Calibration process section
        QGroupBox *calibProcessGroup = new QGroupBox("Calibration Process (6x6 Checkerboard - Auto Capture)", this);
        QVBoxLayout *calibProcessLayout = new QVBoxLayout();
        
        QLabel *calibInfoLabel = new QLabel(
            "Show the 6x6 checkerboard to the camera. The system will automatically\n"
            "capture images when the board is detected and stable. Each grid position\n"
            "will be highlighted in yellow once captured.", this);
        calibInfoLabel->setWordWrap(true);
        calibProcessLayout->addWidget(calibInfoLabel);
        
        QHBoxLayout *calibControlLayout = new QHBoxLayout();
        QPushButton *startCalibBtn = new QPushButton("Start Calibration", this);
        resetCalibBtn_ = new QPushButton("Reset", this);
        resetCalibBtn_->setEnabled(false);
        connect(startCalibBtn, &QPushButton::clicked, this, &AprilTagDebugGUI::startCalibration);
        connect(resetCalibBtn_, &QPushButton::clicked, this, &AprilTagDebugGUI::resetCalibration);
        calibControlLayout->addWidget(startCalibBtn);
        calibControlLayout->addWidget(resetCalibBtn_);
        calibControlLayout->addStretch();
        calibProcessLayout->addLayout(calibControlLayout);
        
        // Calibration preview (show current camera feed with checkerboard detection)
        calibrationPreviewLabel_ = new QLabel("Start calibration to see preview", this);
        calibrationPreviewLabel_->setMinimumSize(640, 480);
        calibrationPreviewLabel_->setAlignment(Qt::AlignCenter);
        calibrationPreviewLabel_->setStyleSheet("border: 1px solid black; background-color: #f0f0f0;");
        calibProcessLayout->addWidget(calibrationPreviewLabel_);
        
        // Status and progress
        calibrationStatusLabel_ = new QLabel("Status: Not started", this);
        calibrationProgressLabel_ = new QLabel("Captured: 0 images", this);
        calibProcessLayout->addWidget(calibrationStatusLabel_);
        calibProcessLayout->addWidget(calibrationProgressLabel_);
        
        // Save calibration button
        saveCalibBtn_ = new QPushButton("Save Calibration", this);
        saveCalibBtn_->setEnabled(false);
        connect(saveCalibBtn_, &QPushButton::clicked, this, &AprilTagDebugGUI::saveCalibration);
        calibProcessLayout->addWidget(saveCalibBtn_);
        
        calibProcessGroup->setLayout(calibProcessLayout);
        layout->addWidget(calibProcessGroup);
        
        layout->addStretch();
    }
    
    Mat drawGridOverlay(const Mat& img, const Scalar& color = Scalar(0, 255, 0), int gridSpacing = 50) {
        Mat overlay = img.clone();
        if (overlay.channels() == 1) {
            cvtColor(overlay, overlay, COLOR_GRAY2RGB);
        } else if (overlay.channels() == 3) {
            overlay = overlay.clone();
        }
        
        // Draw horizontal grid lines
        for (int y = gridSpacing; y < overlay.rows; y += gridSpacing) {
            line(overlay, Point(0, y), Point(overlay.cols, y), color, 1);
        }
        
        // Draw vertical grid lines
        for (int x = gridSpacing; x < overlay.cols; x += gridSpacing) {
            line(overlay, Point(x, 0), Point(x, overlay.rows), color, 1);
        }
        
        // Draw corner markers
        int markerSize = 20;
        // Top-left
        line(overlay, Point(0, 0), Point(markerSize, 0), color, 2);
        line(overlay, Point(0, 0), Point(0, markerSize), color, 2);
        // Top-right
        line(overlay, Point(overlay.cols - markerSize, 0), Point(overlay.cols, 0), color, 2);
        line(overlay, Point(overlay.cols, 0), Point(overlay.cols, markerSize), color, 2);
        // Bottom-left
        line(overlay, Point(0, overlay.rows - markerSize), Point(0, overlay.rows), color, 2);
        line(overlay, Point(0, overlay.rows), Point(markerSize, overlay.rows), color, 2);
        // Bottom-right
        line(overlay, Point(overlay.cols - markerSize, overlay.rows), Point(overlay.cols, overlay.rows), color, 2);
        line(overlay, Point(overlay.cols, overlay.rows - markerSize), Point(overlay.cols, overlay.rows), color, 2);
        
        // Draw center marker
        Point center(overlay.cols / 2, overlay.rows / 2);
        circle(overlay, center, 10, color, 2);
        line(overlay, Point(center.x - 15, center.y), Point(center.x + 15, center.y), color, 2);
        line(overlay, Point(center.x, center.y - 15), Point(center.x, center.y + 15), color, 2);
        
        return overlay;
    }
    
    void loadTestImageForFisheye() {
        QString filename = QFileDialog::getOpenFileName(this, "Load Test Image", "", "Images (*.png *.jpg *.jpeg *.bmp)");
        if (filename.isEmpty()) return;
        
        Mat testImage = imread(filename.toStdString(), IMREAD_GRAYSCALE);
        if (testImage.empty()) {
            QMessageBox::warning(this, "Error", "Failed to load test image");
            return;
        }
        
        // Create grid overlay for visualization
        Mat gridOriginal = drawGridOverlay(testImage, Scalar(0, 255, 0), std::max(30, testImage.cols / 20));
        
        // Display original with grid
        Mat displayOriginal;
        if (gridOriginal.channels() == 3) {
            displayOriginal = gridOriginal.clone();
        } else {
            cvtColor(gridOriginal, displayOriginal, COLOR_GRAY2RGB);
        }
        
        // Draw same reference lines on original to show distortion
        int width = displayOriginal.cols;
        int height = displayOriginal.rows;
        
        // Draw center horizontal line (will appear curved due to distortion)
        line(displayOriginal, Point(0, height / 2), Point(width, height / 2), Scalar(255, 0, 255), 2);
        
        // Draw center vertical line (will appear curved due to distortion)
        line(displayOriginal, Point(width / 2, 0), Point(width / 2, height), Scalar(255, 0, 255), 2);
        
        // Draw diagonal lines (will appear curved)
        line(displayOriginal, Point(0, 0), Point(width, height), Scalar(255, 255, 0), 1);
        line(displayOriginal, Point(width, 0), Point(0, height), Scalar(255, 255, 0), 1);
        
        // Draw corner-to-corner lines (will appear curved)
        int margin = 50;
        line(displayOriginal, Point(margin, margin), Point(width - margin, margin), Scalar(0, 255, 255), 1);
        line(displayOriginal, Point(margin, margin), Point(margin, height - margin), Scalar(0, 255, 255), 1);
        line(displayOriginal, Point(width - margin, height - margin), Point(width - margin, margin), Scalar(0, 255, 255), 1);
        line(displayOriginal, Point(width - margin, height - margin), Point(margin, height - margin), Scalar(0, 255, 255), 1);
        
        // Add label text
        putText(displayOriginal, "ORIGINAL (Distorted)", Point(10, 30), 
                FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
        
        QImage qimgOriginal(displayOriginal.data, displayOriginal.cols, displayOriginal.rows, displayOriginal.step, QImage::Format_RGB888);
        QPixmap pixmapOriginal = QPixmap::fromImage(qimgOriginal.copy()).scaled(320, 240, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        fisheyeOriginalLabel_->setPixmap(pixmapOriginal);
        
        // Display corrected (if calibration loaded)
        if (fisheye_calibration_loaded_ && !fisheye_K_.empty() && !fisheye_D_.empty()) {
            Mat corrected = undistortFrame(testImage);
            
            // Draw grid on corrected image to show how lines are straightened
            Mat gridCorrected = drawGridOverlay(corrected, Scalar(0, 255, 255), std::max(30, corrected.cols / 20));
            
            // Draw additional reference lines to visualize correction
            Mat displayCorrected;
            if (gridCorrected.channels() == 3) {
                displayCorrected = gridCorrected.clone();
            } else {
                cvtColor(gridCorrected, displayCorrected, COLOR_GRAY2RGB);
            }
            
            // Draw reference lines on corrected image - these should be perfectly straight
            int width = displayCorrected.cols;
            int height = displayCorrected.rows;
            
            // Draw center horizontal line (should be perfectly straight after correction)
            line(displayCorrected, Point(0, height / 2), Point(width, height / 2), Scalar(255, 0, 255), 2);
            
            // Draw center vertical line (should be perfectly straight after correction)
            line(displayCorrected, Point(width / 2, 0), Point(width / 2, height), Scalar(255, 0, 255), 2);
            
            // Draw horizontal lines at different heights (should be straight)
            line(displayCorrected, Point(0, height / 4), Point(width, height / 4), Scalar(255, 0, 255), 1);
            line(displayCorrected, Point(0, 3 * height / 4), Point(width, 3 * height / 4), Scalar(255, 0, 255), 1);
            
            // Draw vertical lines at different positions (should be straight)
            line(displayCorrected, Point(width / 4, 0), Point(width / 4, height), Scalar(255, 0, 255), 1);
            line(displayCorrected, Point(3 * width / 4, 0), Point(3 * width / 4, height), Scalar(255, 0, 255), 1);
            
            // Draw diagonal lines (should be straight)
            line(displayCorrected, Point(0, 0), Point(width, height), Scalar(255, 255, 0), 2);
            line(displayCorrected, Point(width, 0), Point(0, height), Scalar(255, 255, 0), 2);
            
            // Draw border rectangle (should have straight edges)
            int margin = 30;
            rectangle(displayCorrected, Rect(margin, margin, width - 2*margin, height - 2*margin), Scalar(0, 255, 255), 2);
            
            // Add label text
            putText(displayCorrected, "CORRECTED (Undistorted)", Point(10, 30), 
                    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 255), 2);
            
            QImage qimgCorrected(displayCorrected.data, displayCorrected.cols, displayCorrected.rows, displayCorrected.step, QImage::Format_RGB888);
            QPixmap pixmapCorrected = QPixmap::fromImage(qimgCorrected.copy()).scaled(320, 240, Qt::KeepAspectRatio, Qt::SmoothTransformation);
            fisheyeCorrectedLabel_->setPixmap(pixmapCorrected);
        } else {
            fisheyeCorrectedLabel_->setText("Calibration not loaded");
        }
    }
    
    void onFisheyeSelectionChanged() {
        // Update the enabled state based on selection
        fisheye_undistort_enabled_ = fisheyeUseCorrectedRadio_->isChecked();
        
        // Clear maps if disabled
        if (!fisheye_undistort_enabled_) {
            fisheye_map1_ = Mat();
            fisheye_map2_ = Mat();
        }
        
        // Update status indicator at top
        updateFisheyeStatusIndicator();
        
        qDebug() << "Fisheye correction for captures:" << (fisheye_undistort_enabled_ ? "ENABLED" : "DISABLED");
    }
    
    void updateFisheyeStatusIndicator() {
        if (!fisheyeStatusIndicator_) return;
        
        if (fisheye_undistort_enabled_ && fisheye_calibration_loaded_) {
            fisheyeStatusIndicator_->setText("✓ Fisheye Correction: APPLIED");
            fisheyeStatusIndicator_->setStyleSheet(
                "QLabel { "
                "background-color: #ccffcc; "
                "color: #000000; "
                "padding: 8px; "
                "border: 2px solid #00aa00; "
                "border-radius: 5px; "
                "font-weight: bold; "
                "font-size: 12pt; "
                "}");
        } else if (fisheye_calibration_loaded_) {
            fisheyeStatusIndicator_->setText("✗ Fisheye Correction: NOT APPLIED (Calibration loaded but disabled)");
            fisheyeStatusIndicator_->setStyleSheet(
                "QLabel { "
                "background-color: #ffffcc; "
                "color: #000000; "
                "padding: 8px; "
                "border: 2px solid #aaaa00; "
                "border-radius: 5px; "
                "font-weight: bold; "
                "font-size: 12pt; "
                "}");
        } else {
            fisheyeStatusIndicator_->setText("✗ Fisheye Correction: NOT APPLIED (No calibration loaded)");
            fisheyeStatusIndicator_->setStyleSheet(
                "QLabel { "
                "background-color: #ffcccc; "
                "color: #000000; "
                "padding: 8px; "
                "border: 2px solid #ff0000; "
                "border-radius: 5px; "
                "font-weight: bold; "
                "font-size: 12pt; "
                "}");
        }
    }
    
    void startCalibration() {
        if (!cameraOpen_) {
            QMessageBox::warning(this, "Error", "Please open a camera first");
            return;
        }
        
        calibrationInProgress_ = true;
        objectPoints_.clear();
        imagePoints_.clear();
        calibrationImages_.clear();
        gridCaptured_.clear();
        gridCaptured_.resize(36, false);  // 6x6 = 36 grid positions (but stop at 34)
        stableFrameCount_ = 0;
        lastStableFrame_ = Mat();
        
        resetCalibBtn_->setEnabled(true);
        saveCalibBtn_->setEnabled(false);
        calibrationStatusLabel_->setText("Status: Calibration in progress - show checkerboard to camera");
        calibrationProgressLabel_->setText("Captured: 0/34 grid positions");
        
        // Start calibration preview timer
        calibrationPreviewTimer_->start(33);  // ~30 FPS
    }
    
    int getGridPosition(const vector<Point2f>& corners, const Size& imageSize) {
        // Calculate center of checkerboard
        Point2f center(0, 0);
        for (const auto& pt : corners) {
            center += pt;
        }
        center.x /= corners.size();
        center.y /= corners.size();
        
        // Map center to 6x6 grid position based on actual image position
        // Normalize to [0, 1] range
        float normX = center.x / imageSize.width;
        float normY = center.y / imageSize.height;
        
        // Clamp to valid range
        normX = max(0.0f, min(1.0f, normX));
        normY = max(0.0f, min(1.0f, normY));
        
        // Map to grid coordinates (0-5)
        int gridX = static_cast<int>(normX * 6);
        int gridY = static_cast<int>(normY * 6);
        
        // Clamp to valid grid range
        gridX = max(0, min(5, gridX));
        gridY = max(0, min(5, gridY));
        
        return gridY * 6 + gridX;
    }
    
    void captureCalibrationImageAuto(const Mat& frame, const vector<Point2f>& corners) {
        if (!calibrationInProgress_) return;
        
        // Convert to grayscale if needed
        Mat gray;
        if (frame.channels() == 3) {
            cvtColor(frame, gray, COLOR_BGR2GRAY);
        } else {
            gray = frame.clone();
        }
        
        // Refine corners
        vector<Point2f> refinedCorners = corners;
        cornerSubPix(gray, refinedCorners, Size(11, 11), Size(-1, -1), 
            TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
        
        // Create object points (3D points in real world coordinates)
        vector<Point3f> obj;
        for (int i = 0; i < checkerboardSize_.height; i++) {
            for (int j = 0; j < checkerboardSize_.width; j++) {
                obj.push_back(Point3f(j, i, 0));
            }
        }
        
        // Get grid position based on actual frame size
        Size frameSize = frame.size();
        int gridPos = getGridPosition(refinedCorners, frameSize);
        
        // Check if this grid position is already captured
        if (gridPos >= 0 && gridPos < 36 && !gridCaptured_[gridPos]) {
            // Store points
            objectPoints_.push_back(obj);
            imagePoints_.push_back(refinedCorners);
            calibrationImages_.push_back(frame.clone());
            gridCaptured_[gridPos] = true;
            
            int capturedCount = 0;
            for (bool captured : gridCaptured_) {
                if (captured) capturedCount++;
            }
            
            calibrationProgressLabel_->setText(QString("Captured: %1/34 grid positions").arg(capturedCount));
            
            // Enable save button if we have enough images
            if (imagePoints_.size() >= 3) {
                saveCalibBtn_->setEnabled(true);
            }
            
            // Stop calibration if 34 tiles captured
            if (capturedCount >= 34) {
                calibrationInProgress_ = false;
                calibrationPreviewTimer_->stop();
                calibrationStatusLabel_->setText("Status: Calibration complete (34/34 positions captured)");
                QMessageBox::information(this, "Calibration Complete", 
                    "Successfully captured 34 grid positions!\n\nClick 'Save Calibration' to compute and save calibration parameters.");
            }
        }
    }
    
    void resetCalibration() {
        calibrationInProgress_ = false;
        objectPoints_.clear();
        imagePoints_.clear();
        calibrationImages_.clear();
        gridCaptured_.clear();
        gridCaptured_.resize(36, false);
        stableFrameCount_ = 0;
        lastStableFrame_ = Mat();
        
        resetCalibBtn_->setEnabled(false);
        saveCalibBtn_->setEnabled(false);
        calibrationStatusLabel_->setText("Status: Not started");
        calibrationProgressLabel_->setText("Captured: 0/34 grid positions");
        
        calibrationPreviewTimer_->stop();
        calibrationPreviewLabel_->setText("Start calibration to see preview");
    }
    
    void updateCalibrationPreview() {
        if (!calibrationInProgress_ || !cameraOpen_) {
            calibrationPreviewTimer_->stop();
            return;
        }
        
        Mat frame;
        {
            lock_guard<mutex> lock(frameMutex_);
            if (currentFrame_.empty()) return;
            frame = currentFrame_.clone();
        }
        
        if (frame.empty()) return;
        
        // Convert to grayscale if needed
        Mat gray;
        if (frame.channels() == 3) {
            cvtColor(frame, gray, COLOR_BGR2GRAY);
        } else {
            gray = frame.clone();
        }
        
        // Find checkerboard corners
        vector<Point2f> corners;
        bool found = findChessboardCorners(gray, checkerboardSize_, corners, 
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);
        
        Mat display;
        if (frame.channels() == 1) {
            cvtColor(frame, display, COLOR_GRAY2BGR);
        } else {
            display = frame.clone();
        }
        
        // Draw 6x6 grid overlay
        int gridRows = 6;
        int gridCols = 6;
        int cellWidth = display.cols / gridCols;
        int cellHeight = display.rows / gridRows;
        
        // Draw grid lines
        for (int i = 0; i <= gridRows; i++) {
            line(display, Point(0, i * cellHeight), Point(display.cols, i * cellHeight), Scalar(128, 128, 128), 1);
        }
        for (int j = 0; j <= gridCols; j++) {
            line(display, Point(j * cellWidth, 0), Point(j * cellWidth, display.rows), Scalar(128, 128, 128), 1);
        }
        
        // Highlight captured grid positions with yellow border and dot
        for (int i = 0; i < gridRows; i++) {
            for (int j = 0; j < gridCols; j++) {
                int gridPos = i * gridCols + j;
                if (gridPos < gridCaptured_.size() && gridCaptured_[gridPos]) {
                    Rect cellRect(j * cellWidth, i * cellHeight, cellWidth, cellHeight);
                    rectangle(display, cellRect, Scalar(0, 255, 255), 3);  // Yellow border only (thick)
                    
                    // Draw small dot in top-left corner
                    Point dotPos(j * cellWidth + 5, i * cellHeight + 5);
                    circle(display, dotPos, 4, Scalar(0, 255, 255), -1);  // Yellow filled circle
                }
            }
        }
        
        // Draw checkerboard corners if found
        if (found) {
            // Refine corners
            vector<Point2f> refinedCorners = corners;
            cornerSubPix(gray, refinedCorners, Size(11, 11), Size(-1, -1), 
                TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
            
            // Draw checkerboard corners with colored lines (this shows the detected pattern)
            drawChessboardCorners(display, checkerboardSize_, refinedCorners, found);
            
            // Calculate current grid position for visual feedback
            Size frameSize = display.size();
            int currentGridPos = getGridPosition(refinedCorners, frameSize);
            
            // Highlight the current grid position with a different color (cyan border)
            if (currentGridPos >= 0 && currentGridPos < 36) {
                int gridY = currentGridPos / 6;
                int gridX = currentGridPos % 6;
                Rect currentCellRect(gridX * cellWidth, gridY * cellHeight, cellWidth, cellHeight);
                rectangle(display, currentCellRect, Scalar(255, 255, 0), 2);  // Cyan border for current position
            }
            
            // Check stability (compare with last stable frame)
            bool isStable = false;
            if (lastStableFrame_.empty()) {
                isStable = true;
                stableFrameCount_ = 1;
            } else {
                // Simple stability check: compare corner positions
                double maxDiff = 0;
                for (size_t i = 0; i < refinedCorners.size() && i < corners.size(); i++) {
                    double diff = norm(refinedCorners[i] - corners[i]);
                    maxDiff = max(maxDiff, diff);
                }
                if (maxDiff < 5.0) {  // Corners haven't moved much
                    stableFrameCount_++;
                    if (stableFrameCount_ >= STABLE_THRESHOLD) {
                        isStable = true;
                    }
                } else {
                    stableFrameCount_ = 0;
                }
            }
            
            if (isStable && calibrationInProgress_) {
                lastStableFrame_ = frame.clone();
                // Auto-capture
                captureCalibrationImageAuto(frame, refinedCorners);
                
                int capturedCount = 0;
                for (bool captured : gridCaptured_) {
                    if (captured) capturedCount++;
                }
                
                if (capturedCount < 34) {
                    putText(display, "CHECKERBOARD DETECTED - AUTO CAPTURED", Point(10, 30), 
                        FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
                } else {
                    putText(display, "CALIBRATION COMPLETE (34/34)", Point(10, 30), 
                        FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
                }
            } else {
                putText(display, "CHECKERBOARD DETECTED - Stabilizing...", Point(10, 30), 
                    FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
            }
        } else {
            stableFrameCount_ = 0;
            lastStableFrame_ = Mat();
            putText(display, "Move checkerboard into view", Point(10, 30), 
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
        }
        
        // Convert to RGB for display
        Mat displayRGB;
        cvtColor(display, displayRGB, COLOR_BGR2RGB);
        
        // Show count
        int capturedCount = 0;
        for (bool captured : gridCaptured_) {
            if (captured) capturedCount++;
        }
        putText(displayRGB, QString("Captured: %1/34").arg(capturedCount).toStdString(), 
            Point(10, displayRGB.rows - 20), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 0), 2);
        
        // Show message if calibration is complete
        if (capturedCount >= 34) {
            putText(displayRGB, "CALIBRATION COMPLETE - Click Save", Point(10, 60), 
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
        }
        
        QImage qimg(displayRGB.data, displayRGB.cols, displayRGB.rows, displayRGB.step, QImage::Format_RGB888);
        QPixmap pixmap = QPixmap::fromImage(qimg.copy()).scaled(640, 480, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        calibrationPreviewLabel_->setPixmap(pixmap);
    }
    
    void saveCalibration() {
        if (imagePoints_.size() < 3) {
            QMessageBox::warning(this, "Error", "Need at least 3 images for calibration");
            return;
        }
        
        // Get output filename
        QString filename = QFileDialog::getSaveFileName(this, "Save Calibration File", 
            calibPathEdit_->text(), "YAML Files (*.yaml *.yml);;All Files (*)");
        if (filename.isEmpty()) return;
        
        // Perform fisheye calibration
        Mat K = Mat::eye(3, 3, CV_64F);
        Mat D;
        vector<Mat> rvecs, tvecs;
        
        Size imageSize = calibrationImages_[0].size();
        if (calibrationImages_[0].channels() == 1) {
            imageSize = Size(calibrationImages_[0].cols, calibrationImages_[0].rows);
        } else {
            imageSize = Size(calibrationImages_[0].cols, calibrationImages_[0].rows);
        }
        
        int flags = fisheye::CALIB_RECOMPUTE_EXTRINSIC | fisheye::CALIB_CHECK_COND | fisheye::CALIB_FIX_SKEW;
        double rms = fisheye::calibrate(objectPoints_, imagePoints_, imageSize, K, D, rvecs, tvecs, flags,
            TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-6));
        
        // Save to YAML file
        FileStorage fs(filename.toStdString(), FileStorage::WRITE);
        if (fs.isOpened()) {
            fs << "camera_matrix" << K;
            fs << "distortion_coefficients" << D;
            fs << "image_width" << imageSize.width;
            fs << "image_height" << imageSize.height;
            fs.release();
            
            QMessageBox::information(this, "Success", 
                QString("Calibration saved successfully!\n\nRMS Error: %1\nCalibration file: %2")
                .arg(rms, 0, 'f', 4).arg(filename));
            
            // Update path and load the new calibration
            calibPathEdit_->setText(filename);
            if (loadFisheyeCalibration(filename)) {
                if (fisheyeStatusLabel_) {
                    fisheyeStatusLabel_->setText(QString("Calibration: Loaded (Size: %1x%2)")
                        .arg(fisheye_image_size_.width).arg(fisheye_image_size_.height));
                }
            }
            
            // Stop calibration process
            resetCalibration();
        } else {
            QMessageBox::warning(this, "Error", "Failed to save calibration file");
        }
    }
    
    void browseCalibrationFile() {
        QString filename = QFileDialog::getOpenFileName(this, "Select Calibration File", 
            calibPathEdit_->text(), "YAML Files (*.yaml *.yml);;All Files (*)");
        if (!filename.isEmpty()) {
            calibPathEdit_->setText(filename);
        }
    }
    
    void loadCalibrationFromUI() {
        QString path = calibPathEdit_->text();
        if (loadFisheyeCalibration(path)) {
            if (fisheyeStatusLabel_) {
                fisheyeStatusLabel_->setText(QString("Calibration: Loaded (Size: %1x%2)")
                    .arg(fisheye_image_size_.width).arg(fisheye_image_size_.height));
            }
            // Enable radio buttons (if UI is already created)
            if (fisheyeUseCorrectedRadio_ && fisheyeUseOriginalRadio_) {
                fisheyeUseCorrectedRadio_->setEnabled(true);
                fisheyeUseOriginalRadio_->setEnabled(true);
            }
        } else {
            if (fisheyeStatusLabel_) {
                fisheyeStatusLabel_->setText("Calibration: Failed to load");
            }
            // Disable radio buttons (if UI is already created)
            if (fisheyeUseCorrectedRadio_ && fisheyeUseOriginalRadio_) {
                fisheyeUseCorrectedRadio_->setEnabled(false);
                fisheyeUseOriginalRadio_->setEnabled(false);
            }
        }
        
        // Update top status indicator
        updateFisheyeStatusIndicator();
    }
    
    bool loadFisheyeCalibration(const QString& calib_path) {
        QFileInfo file_info(calib_path);
        if (!file_info.exists()) {
            qDebug() << "Fisheye calibration file not found:" << calib_path;
            return false;
        }
        
        string path = calib_path.toStdString();
        FileStorage fs(path, FileStorage::READ);
        
        if (!fs.isOpened()) {
            qDebug() << "Failed to open fisheye calibration file:" << calib_path;
            return false;
        }
        
        fs["camera_matrix"] >> fisheye_K_;
        fs["distortion_coefficients"] >> fisheye_D_;
        
        int img_width = 0, img_height = 0;
        if (fs["image_width"].isInt()) {
            img_width = static_cast<int>(fs["image_width"]);
        }
        if (fs["image_height"].isInt()) {
            img_height = static_cast<int>(fs["image_height"]);
        }
        
        fs.release();
        
        if (fisheye_K_.empty() || fisheye_D_.empty()) {
            qDebug() << "Failed to read camera matrix or distortion coefficients";
            return false;
        }
        
        fisheye_image_size_ = Size(img_width, img_height);
        fisheye_calibration_loaded_ = true;
        
        // Enable corrected radio button if calibration loaded (if UI is already created)
        if (fisheyeUseCorrectedRadio_ && fisheyeUseOriginalRadio_) {
            fisheyeUseCorrectedRadio_->setEnabled(true);
            fisheyeUseOriginalRadio_->setEnabled(true);
            // Set default to corrected if calibration just loaded
            if (!fisheyeUseOriginalRadio_->isChecked() && !fisheyeUseCorrectedRadio_->isChecked()) {
                fisheyeUseCorrectedRadio_->setChecked(true);
                fisheye_undistort_enabled_ = true;
            }
        }
        
        qDebug() << "Fisheye calibration loaded successfully";
        qDebug() << "Camera matrix K rows:" << fisheye_K_.rows << "cols:" << fisheye_K_.cols;
        qDebug() << "Distortion coefficients D rows:" << fisheye_D_.rows << "cols:" << fisheye_D_.cols;
        qDebug() << "Image size:" << fisheye_image_size_.width << "x" << fisheye_image_size_.height;
        
        return true;
    }
    
    void initFisheyeUndistortMaps(const Size& image_size) {
        if (!fisheye_undistort_enabled_ || fisheye_K_.empty() || fisheye_D_.empty()) {
            return;
        }
        
        // Check if maps are already initialized for this size
        if (!fisheye_map1_.empty() && fisheye_map1_.size() == image_size) {
            return;
        }
        
        // Initialize undistortion maps
        fisheye::initUndistortRectifyMap(
            fisheye_K_,
            fisheye_D_,
            Mat::eye(3, 3, CV_32F),  // Identity matrix (no rotation)
            fisheye_K_,  // Use same camera matrix for output
            image_size,
            CV_16SC2,  // map1 type
            fisheye_map1_,
            fisheye_map2_
        );
        
        qDebug() << "Fisheye undistortion maps initialized for size:" << image_size.width << "x" << image_size.height;
    }
    
    Mat undistortFrame(const Mat& frame) {
        if (!fisheye_undistort_enabled_ || fisheye_K_.empty() || fisheye_D_.empty()) {
            return frame;  // Return original frame if undistortion not enabled
        }
        
        // Initialize maps if needed (first frame or size changed)
        Size frameSize = frame.size();
        if (fisheye_map1_.empty() || fisheye_map1_.size() != frameSize) {
            initFisheyeUndistortMaps(frameSize);
        }
        
        if (fisheye_map1_.empty() || fisheye_map2_.empty()) {
            return frame;  // Return original if maps failed to initialize
        }
        
        Mat undistorted;
        // Use INTER_CUBIC for better quality (preserves edges better than LINEAR)
        remap(frame, undistorted, fisheye_map1_, fisheye_map2_, INTER_CUBIC, BORDER_CONSTANT);
        
        return undistorted;
    }
    
    void enumerateCameras() {
        cameraCombo_->clear();
        cameraList_.clear();
        isMindVision_.clear();
        
        // Enumerate V4L2 cameras
        // Temporarily redirect stderr to suppress OpenCV warnings during enumeration
        fflush(stderr);
        int saved_stderr = dup(STDERR_FILENO);
        FILE *null_file = fopen("/dev/null", "w");
        if (null_file) {
            dup2(fileno(null_file), STDERR_FILENO);
        }
        
        for (int i = 0; i < 10; i++) {
            VideoCapture testCap(i);
            if (testCap.isOpened()) {
                cameraList_.push_back("V4L2 Camera " + to_string(i));
                isMindVision_.push_back(false);
                cameraCombo_->addItem(QString("V4L2 Camera %1").arg(i));
                testCap.release();
            }
        }
        
        // Restore stderr
        if (null_file) {
            fflush(stderr);
            dup2(saved_stderr, STDERR_FILENO);
            ::close(saved_stderr);
            fclose(null_file);
        }
        
        // Enumerate MindVision cameras
#ifdef HAVE_MINDVISION_SDK
        CameraSdkStatus status = CameraSdkInit(1);
        if (status == CAMERA_STATUS_SUCCESS) {
            tSdkCameraDevInfo list[16];
            INT count = 16;
            status = CameraEnumerateDevice(list, &count);
            if (status == CAMERA_STATUS_SUCCESS && count > 0) {
                for (int i = 0; i < count; i++) {
                    string name = list[i].acFriendlyName[0] ? list[i].acFriendlyName : list[i].acProductName;
                    cameraList_.push_back("MindVision: " + name);
                    isMindVision_.push_back(true);
                    cameraCombo_->addItem(QString("MindVision: %1").arg(QString::fromStdString(name)));
                }
            }
        }
#endif
    }
    
    void openCamera() {
        // Only close camera if it's already open (previewTimer_ might not exist yet during initialization)
        if (cameraOpen_ && previewTimer_ != nullptr) {
            closeCamera();
        }
        
        int index = cameraCombo_->currentIndex();
        if (index < 0 || index >= (int)cameraList_.size()) {
            return;
        }
        
        selectedCameraIndex_ = index;
        useMindVision_ = isMindVision_[index];
        
        if (useMindVision_) {
#ifdef HAVE_MINDVISION_SDK
            // Open MindVision camera
            CameraSdkStatus status = CameraSdkInit(1);
            if (status != CAMERA_STATUS_SUCCESS) {
                previewLabel_->setText("Failed to initialize MindVision SDK");
                return;
            }
            
            tSdkCameraDevInfo list[16];
            INT count = 16;
            status = CameraEnumerateDevice(list, &count);
            if (status != CAMERA_STATUS_SUCCESS || count == 0) {
                previewLabel_->setText("No MindVision cameras found");
                return;
            }
            
            // Find the correct camera index (skip V4L2 cameras)
            int mvIndex = 0;
            for (int i = 0; i < index; i++) {
                if (isMindVision_[i]) mvIndex++;
            }
            
            status = CameraInit(&list[mvIndex], -1, -1, &mvHandle_);
            if (status != CAMERA_STATUS_SUCCESS) {
                previewLabel_->setText("Failed to open MindVision camera");
                return;
            }
            
            tSdkCameraCapbility cap;
            CameraGetCapability(mvHandle_, &cap);
            CameraSetIspOutFormat(mvHandle_, CAMERA_MEDIA_TYPE_MONO8);
            
            // Disable auto exposure
            BOOL ae_state = FALSE;
            CameraGetAeState(mvHandle_, &ae_state);
            if (ae_state) {
                CameraSetAeState(mvHandle_, FALSE);
            }
            
            CameraPlay(mvHandle_);
            
            // Initialize MindVision modes
            mv_modes_.clear();
            mv_modes_.push_back(MVMode{1280, 1024, FRAME_SPEED_SUPER, "1280x1024 @211 FPS"});
            mv_modes_.push_back(MVMode{1280, 1024, FRAME_SPEED_HIGH, "1280x1024 @106 FPS"});
            mv_modes_.push_back(MVMode{480, 640, FRAME_SPEED_SUPER, "480x640 @790 FPS"});
            
            // Populate mode combo
            modeCombo_->blockSignals(true);
            modeCombo_->clear();
            for (const auto &m : mv_modes_) {
                modeCombo_->addItem(QString::fromStdString(m.label));
            }
            modeCombo_->setCurrentIndex(0);
            modeCombo_->setEnabled(true);
            modeCombo_->blockSignals(false);
            applyMVMode(0);
            
            // Initialize slider values from camera
            double current_exposure = 0.0;
            if (CameraGetExposureTime(mvHandle_, &current_exposure) == CAMERA_STATUS_SUCCESS) {
                double min_exposure = 1000.0;
                double max_exposure = 100000.0;
                int slider_value = static_cast<int>(((max_exposure - current_exposure) / (max_exposure - min_exposure)) * 100.0);
                slider_value = std::max(0, std::min(100, slider_value));
                exposureSlider_->blockSignals(true);
                exposureSlider_->setValue(slider_value);
                exposureSpin_->setValue(slider_value);
                exposureSlider_->blockSignals(false);
            }
            
            int current_gain_r = 0, current_gain_g = 0, current_gain_b = 0;
            if (CameraGetGain(mvHandle_, &current_gain_r, &current_gain_g, &current_gain_b) == CAMERA_STATUS_SUCCESS) {
                gainSlider_->blockSignals(true);
                gainSlider_->setValue(current_gain_r);
                gainSpin_->setValue(current_gain_r);
                gainSlider_->blockSignals(false);
            }
            
            INT current_analog_gain = 0;
            if (CameraGetAnalogGain(mvHandle_, &current_analog_gain) == CAMERA_STATUS_SUCCESS) {
                int brightness_value = (current_analog_gain * 255) / 100;
                brightnessSlider_->blockSignals(true);
                brightnessSlider_->setValue(brightness_value);
                brightnessSpin_->setValue(brightness_value);
                brightnessSlider_->blockSignals(false);
            }
            
            int current_contrast = 0;
            if (CameraGetContrast(mvHandle_, &current_contrast) == CAMERA_STATUS_SUCCESS) {
                contrastSlider_->blockSignals(true);
                contrastSlider_->setValue(current_contrast);
                contrastSpin_->setValue(current_contrast);
                contrastSlider_->blockSignals(false);
            }
            
            int current_saturation = 0;
            if (CameraGetSaturation(mvHandle_, &current_saturation) == CAMERA_STATUS_SUCCESS) {
                saturationSlider_->blockSignals(true);
                saturationSlider_->setValue(current_saturation);
                saturationSpin_->setValue(current_saturation);
                saturationSlider_->blockSignals(false);
            }
            
            int current_sharpness = 0;
            if (CameraGetSharpness(mvHandle_, &current_sharpness) == CAMERA_STATUS_SUCCESS) {
                sharpnessSlider_->blockSignals(true);
                sharpnessSlider_->setValue(current_sharpness);
                sharpnessSpin_->setValue(current_sharpness);
                sharpnessSlider_->blockSignals(false);
            }
            
            cameraOpen_ = true;
#else
            previewLabel_->setText("MindVision SDK not available");
            return;
#endif
        } else {
            // Open V4L2 camera
            // Find the actual V4L2 index
            int v4l2Index = 0;
            for (int i = 0; i < index; i++) {
                if (!isMindVision_[i]) v4l2Index++;
            }
            
            cameraCap_.open(v4l2Index);
            if (!cameraCap_.isOpened()) {
                previewLabel_->setText(QString("Failed to open V4L2 camera %1").arg(v4l2Index));
                return;
            }
            
            // Initialize V4L2 modes
            v4l2_modes_.clear();
            v4l2_modes_.push_back(Mode{640, 480, 30.0, "640x480 @30 FPS"});
            v4l2_modes_.push_back(Mode{1280, 720, 30.0, "1280x720 @30 FPS"});
            v4l2_modes_.push_back(Mode{1920, 1080, 30.0, "1920x1080 @30 FPS"});
            v4l2_modes_.push_back(Mode{640, 480, 60.0, "640x480 @60 FPS"});
            
            // Populate mode combo
            modeCombo_->blockSignals(true);
            modeCombo_->clear();
            for (const auto &m : v4l2_modes_) {
                modeCombo_->addItem(QString::fromStdString(m.label));
            }
            modeCombo_->setCurrentIndex(0);
            modeCombo_->setEnabled(true);
            modeCombo_->blockSignals(false);
            applyMode(0);
            
            cameraOpen_ = true;
        }
        
        captureBtn_->setEnabled(true);
        saveSettingsBtn_->setEnabled(true);
        previewTimer_->start(33);  // ~30 FPS
        
        // Initialize algorithm if one is selected
        initializeCaptureAlgorithm();
        
        // Set mirror checkbox default based on camera type
        if (captureMirrorCheckbox_) {
            captureMirrorCheckbox_->setChecked(useMindVision_);
        }
        
        // Load camera settings from config file
        loadCameraSettings();
        
        // Generate initial capture filename
        suggestedCaptureFilename_ = generateCaptureFilename().toStdString();
    }
    
    void closeCamera() {
        // Safety check: previewTimer_ might not be initialized yet during GUI startup
        if (previewTimer_ != nullptr) {
            previewTimer_->stop();
        }
        
        if (useMindVision_ && mvHandle_ != 0) {
#ifdef HAVE_MINDVISION_SDK
            CameraStop(mvHandle_);
            CameraUnInit(mvHandle_);
            mvHandle_ = 0;
#endif
        } else if (cameraCap_.isOpened()) {
            cameraCap_.release();
        }
        
        cameraOpen_ = false;
        captureBtn_->setEnabled(false);
        if (saveSettingsBtn_) saveSettingsBtn_->setEnabled(false);
        modeCombo_->setEnabled(false);
        modeCombo_->clear();
        previewLabel_->setText("Camera closed");
        
        // Clean up capture algorithm
        if (captureAlgorithm_) {
            captureAlgorithm_->cleanup();
            captureAlgorithm_.reset();
        }
    }
    
    void initializeCaptureAlgorithm() {
        // Clean up existing algorithm
        if (captureAlgorithm_) {
            captureAlgorithm_->cleanup();
            captureAlgorithm_.reset();
        }
        
        if (!captureAlgorithmCombo_ || captureAlgorithmCombo_->currentIndex() <= 0) {
            return;  // No algorithm selected or "None" selected
        }
        
        int algorithmIndex = captureAlgorithmCombo_->currentIndex();
        
        // Create algorithm instance
        if (algorithmIndex == 1) {
            // CPU algorithm
            AprilTagAlgorithmFactory::AlgorithmType algoType = AprilTagAlgorithmFactory::CPU;
            captureAlgorithm_ = AprilTagAlgorithmFactory::create(algoType);
            if (captureAlgorithm_) {
                // Get frame dimensions
                int width = 1280, height = 1024;
                Mat testFrame;
                if (useMindVision_ && mvHandle_ != 0) {
#ifdef HAVE_MINDVISION_SDK
                    tSdkFrameHead frameHead;
                    BYTE *pbyBuffer;
                    if (CameraGetImageBuffer(mvHandle_, &frameHead, &pbyBuffer, 1000) == CAMERA_STATUS_SUCCESS) {
                        width = frameHead.iWidth;
                        height = frameHead.iHeight;
                        CameraReleaseImageBuffer(mvHandle_, pbyBuffer);
                    }
#endif
                } else if (cameraCap_.isOpened()) {
                    cameraCap_ >> testFrame;
                    if (!testFrame.empty()) {
                        width = testFrame.cols;
                        height = testFrame.rows;
                    }
                }
                captureAlgorithm_->initialize(width, height);
            }
        } else if (algorithmIndex == 2) {
#ifdef HAVE_CUDA_APRILTAG
            // Fast AprilTag
            AprilTagAlgorithmFactory::AlgorithmType algoType = AprilTagAlgorithmFactory::FAST_APRILTAG;
            captureAlgorithm_ = AprilTagAlgorithmFactory::create(algoType);
            if (captureAlgorithm_) {
                // Get frame dimensions
                int width = 1280, height = 1024;
                Mat testFrame;
                if (useMindVision_ && mvHandle_ != 0) {
#ifdef HAVE_MINDVISION_SDK
                    tSdkFrameHead frameHead;
                    BYTE *pbyBuffer;
                    if (CameraGetImageBuffer(mvHandle_, &frameHead, &pbyBuffer, 1000) == CAMERA_STATUS_SUCCESS) {
                        width = frameHead.iWidth;
                        height = frameHead.iHeight;
                        CameraReleaseImageBuffer(mvHandle_, pbyBuffer);
                    }
#endif
                } else if (cameraCap_.isOpened()) {
                    cameraCap_ >> testFrame;
                    if (!testFrame.empty()) {
                        width = testFrame.cols;
                        height = testFrame.rows;
                    }
                }
                captureAlgorithm_->initialize(width, height);
            }
#endif
        }
    }
    
    void updateCameraSettings() {
        if (!cameraOpen_) return;
        
        if (useMindVision_ && mvHandle_ != 0) {
#ifdef HAVE_MINDVISION_SDK
            // Update MindVision camera settings
            // Exposure: Map slider (0-100) to exposure time (100000 to 1000 microseconds)
            double min_exposure = 1000.0;
            double max_exposure = 100000.0;
            double exposure = max_exposure - (exposureSlider_->value() / 100.0) * (max_exposure - min_exposure);
            CameraSetExposureTime(mvHandle_, exposure);
            
            // Gain (RGB gains, same value for all channels)
            int gain = gainSlider_->value();
            CameraSetGain(mvHandle_, gain, gain, gain);
            
            // Brightness: Map slider (0-255) to analog gain (0-100)
            INT analogGain = (brightnessSlider_->value() * 100) / 255;
            CameraSetAnalogGain(mvHandle_, analogGain);
            
            // Contrast
            CameraSetContrast(mvHandle_, contrastSlider_->value());
            
            // Saturation
            CameraSetSaturation(mvHandle_, saturationSlider_->value());
            
            // Sharpness
            CameraSetSharpness(mvHandle_, sharpnessSlider_->value());
#endif
        } else if (cameraCap_.isOpened()) {
            // Update V4L2 camera settings
            cameraCap_.set(CAP_PROP_EXPOSURE, exposureSlider_->value());
            cameraCap_.set(CAP_PROP_GAIN, gainSlider_->value());
            cameraCap_.set(CAP_PROP_BRIGHTNESS, brightnessSlider_->value());
            cameraCap_.set(CAP_PROP_CONTRAST, contrastSlider_->value());
            cameraCap_.set(CAP_PROP_SATURATION, saturationSlider_->value());
            cameraCap_.set(CAP_PROP_SHARPNESS, sharpnessSlider_->value());
        }
    }
    
    void onModeChanged(int mode_index) {
        if (!cameraOpen_ || mode_index < 0) return;
        
        if (useMindVision_ && mvHandle_ != 0) {
            applyMVMode(mode_index);
        } else if (cameraCap_.isOpened()) {
            applyMode(mode_index);
        }
    }
    
    void applyMode(int mode_index) {
        if (mode_index < 0 || mode_index >= static_cast<int>(v4l2_modes_.size())) return;
        if (!cameraCap_.isOpened()) return;
        
        const Mode &m = v4l2_modes_[mode_index];
        cameraCap_.set(CAP_PROP_FRAME_WIDTH, m.width);
        cameraCap_.set(CAP_PROP_FRAME_HEIGHT, m.height);
        cameraCap_.set(CAP_PROP_FPS, m.fps);
    }
    
    void applyMVMode(int mode_index) {
#ifdef HAVE_MINDVISION_SDK
        if (mode_index < 0 || mode_index >= static_cast<int>(mv_modes_.size())) return;
        if (mvHandle_ == 0) return;
        
        const MVMode &m = mv_modes_[mode_index];
        CameraSetImageResolutionEx(mvHandle_, 0xff, 0, 0, 0, 0, m.width, m.height, 0, 0);
        CameraSetFrameSpeed(mvHandle_, m.frame_speed_index);
#endif
    }
    
    void updatePreview() {
        if (!cameraOpen_) return;
        
        Mat frame;
        if (useMindVision_ && mvHandle_ != 0) {
#ifdef HAVE_MINDVISION_SDK
            BYTE *pbyBuffer = nullptr;
            tSdkFrameHead sFrameInfo;
            CameraSdkStatus status = CameraGetImageBuffer(mvHandle_, &sFrameInfo, &pbyBuffer, 100);
            if (status == CAMERA_STATUS_SUCCESS && pbyBuffer) {
                int width = sFrameInfo.iWidth;
                int height = sFrameInfo.iHeight;
                frame = Mat(height, width, CV_8UC1, pbyBuffer).clone();
                CameraReleaseImageBuffer(mvHandle_, pbyBuffer);
            } else {
                return;
            }
#else
            return;
#endif
        } else if (cameraCap_.isOpened()) {
            cameraCap_ >> frame;
            if (frame.empty()) return;
        } else {
            return;
        }
        
        if (!frame.empty()) {
            Mat frame_for_display = frame;
            {
                lock_guard<mutex> lock(frameMutex_);
                // Apply fisheye undistortion if enabled (based on user selection)
                if (fisheye_undistort_enabled_ && fisheye_calibration_loaded_) {
                    frame_for_display = undistortFrame(frame);
                }
                // Store the frame (undistorted if enabled, original otherwise)
                currentFrame_ = frame_for_display.clone();
            }
            
            // Run detection if algorithm is selected (before converting to display format)
            zarray_t *detections = nullptr;
            bool mirror = captureMirrorCheckbox_ && captureMirrorCheckbox_->isChecked();
            if (captureAlgorithm_ && captureAlgorithmCombo_ && captureAlgorithmCombo_->currentIndex() > 0 && !frame_for_display.empty()) {
                // Validate frame before processing
                if (frame_for_display.data == nullptr || frame_for_display.rows <= 0 || frame_for_display.cols <= 0) {
                    qDebug() << "Invalid frame for detection: data=" << (void*)frame_for_display.data 
                             << "rows=" << frame_for_display.rows << "cols=" << frame_for_display.cols;
                } else {
                    try {
                        // Always clone the frame first to ensure it's independent
                        // Then convert to grayscale if needed (algorithms expect grayscale)
                        Mat frame_for_detection;
                        if (frame_for_display.channels() == 3) {
                            cvtColor(frame_for_display, frame_for_detection, COLOR_BGR2GRAY);
                        } else if (frame_for_display.channels() == 1) {
                            frame_for_detection = frame_for_display.clone();
                        } else {
                            qDebug() << "Unsupported frame format: channels=" << frame_for_display.channels();
                            frame_for_detection = Mat();
                        }
                        
                        // Always ensure the frame is continuous and independent
                        if (!frame_for_detection.empty() && !frame_for_detection.isContinuous()) {
                            frame_for_detection = frame_for_detection.clone();
                        }
                        
                        if (frame_for_detection.empty()) {
                            qDebug() << "Failed to convert frame to grayscale for detection";
                        } else {
                            // Validate frame before processing
                            if (frame_for_detection.data == nullptr || frame_for_detection.rows <= 0 || frame_for_detection.cols <= 0) {
                                qDebug() << "Invalid frame_for_detection: data=" << (void*)frame_for_detection.data 
                                         << "rows=" << frame_for_detection.rows << "cols=" << frame_for_detection.cols;
                            } else {
                                // Verify algorithm is still valid
                                if (!captureAlgorithm_) {
                                    qDebug() << "captureAlgorithm_ is null!";
                                } else {
                                    qDebug() << "Calling processFrame for algorithm:" << QString::fromStdString(captureAlgorithm_->getName());
                                    qDebug() << "Frame info: rows=" << frame_for_detection.rows << "cols=" << frame_for_detection.cols 
                                             << "channels=" << frame_for_detection.channels() << "data=" << (void*)frame_for_detection.data
                                             << "mirror=" << mirror;
                                    qDebug() << "About to call processFrame...";
                                    std::cerr << "About to call processFrame..." << std::endl;
                                    std::cerr.flush();
                                    
                                    // Verify algorithm object is valid before calling
                                    if (!captureAlgorithm_) {
                                        qDebug() << "ERROR: captureAlgorithm_ is null!";
                                        std::cerr << "ERROR: captureAlgorithm_ is null!" << std::endl;
                                        std::cerr.flush();
                                    } else {
                                        std::cerr << "captureAlgorithm_ pointer is valid: " << (void*)captureAlgorithm_.get() << std::endl;
                                        std::cerr.flush();
                                        
                                        // Try to call with exception handling
                                        try {
                                            std::cerr << "Attempting to call processFrame..." << std::endl;
                                            std::cerr.flush();
                                            detections = captureAlgorithm_->processFrame(frame_for_detection, mirror);
                                            std::cerr << "processFrame call succeeded" << std::endl;
                                            std::cerr.flush();
                                        } catch (const std::exception& e) {
                                            qDebug() << "Exception caught in processFrame call:" << e.what();
                                            std::cerr << "Exception caught in processFrame call: " << e.what() << std::endl;
                                            std::cerr.flush();
                                            detections = nullptr;
                                        } catch (...) {
                                            qDebug() << "Unknown exception caught in processFrame call";
                                            std::cerr << "Unknown exception caught in processFrame call" << std::endl;
                                            std::cerr.flush();
                                            detections = nullptr;
                                        }
                                    }
                                    qDebug() << "processFrame call completed";
                                    std::cerr << "processFrame call completed" << std::endl;
                                    qDebug() << "processFrame returned, detections pointer:" << (void*)detections;
                                    // Validate returned detections pointer
                                    if (detections != nullptr) {
                                        // Try to get size to validate the array is valid
                                        try {
                                            int size = zarray_size(detections);
                                            qDebug() << "Detections array size:" << size;
                                            if (size < 0) {
                                                qDebug() << "Invalid detections size:" << size;
                                                // Don't destroy here - let the cleanup code handle it
                                                detections = nullptr;
                                            }
                                        } catch (const std::exception& e) {
                                            qDebug() << "Exception getting detections size:" << e.what();
                                            // Don't destroy here - detections might be invalid
                                            detections = nullptr;
                                        } catch (...) {
                                            qDebug() << "Unknown exception getting detections size";
                                            // Don't destroy here - detections might be invalid
                                            detections = nullptr;
                                        }
                                    } else {
                                        qDebug() << "processFrame returned nullptr detections";
                                    }
                                }
                                }
                            }
                        }
                    } catch (const std::exception& e) {
                        qDebug() << "Exception in processFrame:" << e.what();
                        detections = nullptr;
                    } catch (...) {
                        qDebug() << "Unknown exception in processFrame";
                        detections = nullptr;
                    }
                }
            }
            
            Mat display;
            if (frame_for_display.channels() == 1) {
                cvtColor(frame_for_display, display, COLOR_GRAY2RGB);
            } else {
                cvtColor(frame_for_display, display, COLOR_BGR2RGB);
            }
            
            // Apply mirror to display frame if mirroring was used (so coordinates match)
            // This matches the behavior in the Algorithms tab
            if (mirror) {
                flip(display, display, 1);  // Mirror display to match detection coordinates
            }
            
            // Draw detections on display and extract patterns from all detections
            // Save all detection data before destroying them
            struct DetectionData {
                int id;
                double decision_margin;
                int hamming;
                vector<Point2f> corners;
            };
            vector<DetectionData> all_detections_data;
            
            if (detections) {
                int num_detections = 0;
                try {
                    num_detections = zarray_size(detections);
                } catch (...) {
                    qDebug() << "Error getting detections array size";
                    num_detections = 0;
                }
                
                if (num_detections > 0) {
                    // Save all detection data before processing
                    for (int i = 0; i < num_detections; i++) {
                        apriltag_detection_t* det = nullptr;
                        try {
                            zarray_get(detections, i, &det);
                        } catch (...) {
                            qDebug() << "Error getting detection" << i;
                            continue;
                        }
                        if (det) {
                            try {
                                DetectionData data;
                                // Validate detection structure before accessing
                                if (det->p == nullptr || det->c == nullptr) {
                                    qDebug() << "Invalid detection structure at index" << i;
                                    continue;
                                }
                                data.id = det->id;
                                data.decision_margin = det->decision_margin;
                                data.hamming = det->hamming;
                                // Save corners (will un-mirror later if needed)
                                // Check for NaN or invalid values
                                bool valid_corners = true;
                                for (int j = 0; j < 4; j++) {
                                    if (!isfinite(det->p[j][0]) || !isfinite(det->p[j][1])) {
                                        valid_corners = false;
                                        break;
                                    }
                                }
                                if (!valid_corners) {
                                    qDebug() << "Invalid corner coordinates in detection" << i;
                                    continue;
                                }
                                data.corners.push_back(Point2f(det->p[0][0], det->p[0][1]));
                                data.corners.push_back(Point2f(det->p[1][0], det->p[1][1]));
                                data.corners.push_back(Point2f(det->p[2][0], det->p[2][1]));
                                data.corners.push_back(Point2f(det->p[3][0], det->p[3][1]));
                                all_detections_data.push_back(data);
                            } catch (const std::exception& e) {
                                qDebug() << "Error copying detection data" << i << ":" << e.what();
                                continue;
                            } catch (...) {
                                qDebug() << "Error copying detection data" << i;
                                continue;
                            }
                        }
                    }
                }
                
                // Draw detections on display
                if (num_detections > 0 && detections) {
                    for (int i = 0; i < num_detections; i++) {
                        apriltag_detection_t *det = nullptr;
                        try {
                            zarray_get(detections, i, &det);
                        } catch (...) {
                            qDebug() << "Error getting detection for drawing" << i;
                            continue;
                        }
                        
                        if (!det) continue;
                        
                        // Validate detection structure
                        if (det->p == nullptr || det->c == nullptr) {
                            qDebug() << "Invalid detection structure for drawing at index" << i;
                            continue;
                        }
                        
                        // Validate coordinates are within frame bounds and are finite
                        bool valid = true;
                        for (int j = 0; j < 4; j++) {
                            if (!isfinite(det->p[j][0]) || !isfinite(det->p[j][1])) {
                                valid = false;
                                break;
                            }
                            if (det->p[j][0] < 0 || det->p[j][0] >= display.cols ||
                                det->p[j][1] < 0 || det->p[j][1] >= display.rows) {
                                valid = false;
                                break;
                            }
                        }
                        if (!isfinite(det->c[0]) || !isfinite(det->c[1])) {
                            valid = false;
                        } else if (det->c[0] < 0 || det->c[0] >= display.cols ||
                                   det->c[1] < 0 || det->c[1] >= display.rows) {
                            valid = false;
                        }
                        
                        if (valid) {
                            // Draw quad (green lines)
                            line(display, Point((int)det->p[0][0], (int)det->p[0][1]), 
                                 Point((int)det->p[1][0], (int)det->p[1][1]), Scalar(0, 255, 0), 2);
                            line(display, Point((int)det->p[1][0], (int)det->p[1][1]), 
                                 Point((int)det->p[2][0], (int)det->p[2][1]), Scalar(0, 255, 0), 2);
                            line(display, Point((int)det->p[2][0], (int)det->p[2][1]), 
                                 Point((int)det->p[3][0], (int)det->p[3][1]), Scalar(0, 255, 0), 2);
                            line(display, Point((int)det->p[3][0], (int)det->p[3][1]), 
                                 Point((int)det->p[0][0], (int)det->p[0][1]), Scalar(0, 255, 0), 2);
                            
                            // Draw ID
                            putText(display, to_string(det->id), Point((int)det->c[0], (int)det->c[1]), 
                                   FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
                        }
                        
                        // Destroy individual detections (both algorithm and non-algorithm create new detections)
                        apriltag_detection_destroy(det);
                    }
                }
            }
            
            // Clean up detections array
            // FastAprilTagAlgorithm creates a NEW zarray_t* each call that must be destroyed
            // For direct CPU detection, we also need to destroy the array
            if (detections) {
                try {
                    // Destroy individual detections first (if not already destroyed above)
                    // Actually, we already destroyed them in the loop above, so just destroy the array
                    zarray_destroy(detections);
                } catch (...) {
                    qDebug() << "Error destroying detections array";
                }
                detections = nullptr;
            }
            
            // Extract and display patterns from all detections
            // Skip pattern extraction if no detections or if frame is invalid
            if (!all_detections_data.empty() && capturePatternLabel_ && capturePatternInfoText_ && 
                !frame_for_display.empty() && frame_for_display.data != nullptr &&
                frame_for_display.rows > 0 && frame_for_display.cols > 0) {
                // Convert frame_for_display to grayscale if needed (use original before mirroring)
                Mat gray_for_pattern = frame_for_display.clone();
                if (gray_for_pattern.empty()) {
                    // Frame clone failed, skip pattern extraction
                } else {
                    if (gray_for_pattern.channels() == 3) {
                        cvtColor(frame_for_display, gray_for_pattern, COLOR_BGR2GRAY);
                    }
                    
                    int num_tags = all_detections_data.size();
                    
                    // Extract patterns first (before calculating grid, since some may fail)
                    struct PatternData {
                        size_t detection_idx;  // Index into all_detections_data
                        vector<vector<int>> pattern;
                        Mat warped_image;  // The warped tag image before digitization
                    };
                    vector<PatternData> all_patterns;
                for (size_t tag_idx = 0; tag_idx < all_detections_data.size(); tag_idx++) {
                    const DetectionData& data = all_detections_data[tag_idx];
                    
                    // Safety check: ensure corners are valid
                    if (data.corners.size() != 4) {
                        continue; // Skip if corners are invalid
                    }
                    
                    // Extract pattern from detection
                    vector<Point2f> corners = data.corners;
                    if (mirror) {
                        // Un-mirror the coordinates (flip x coordinates)
                        int frame_width = gray_for_pattern.cols;
                        for (size_t i = 0; i < corners.size(); i++) {
                            corners[i].x = frame_width - 1 - corners[i].x;
                        }
                        // Swap corners to maintain correct orientation after un-mirroring
                        Point2f temp = corners[0];
                        corners[0] = corners[1];
                        corners[1] = temp;
                        temp = corners[2];
                        corners[2] = corners[3];
                        corners[3] = temp;
                    }
                    
                    // Validate corners before processing
                    bool corners_valid = true;
                    for (size_t i = 0; i < corners.size(); i++) {
                        if (corners[i].x < 0 || corners[i].x >= gray_for_pattern.cols ||
                            corners[i].y < 0 || corners[i].y >= gray_for_pattern.rows) {
                            corners_valid = false;
                            break;
                        }
                    }
                    if (!corners_valid || corners.size() != 4) {
                        continue; // Skip this detection if corners are invalid
                    }
                    
                    // Refine corners
                    try {
                        refineCorners(gray_for_pattern, corners);
                    } catch (...) {
                        continue; // Skip if corner refinement fails
                    }
                    
                    // Warp tag to square
                    int tagSize = 36;
                    vector<Point2f> dstQuad;
                    dstQuad.push_back(Point2f(0, 0));
                    dstQuad.push_back(Point2f(tagSize - 1, 0));
                    dstQuad.push_back(Point2f(tagSize - 1, tagSize - 1));
                    dstQuad.push_back(Point2f(0, tagSize - 1));
                    
                    Mat H;
                    Mat warped;
                    try {
                        H = getPerspectiveTransform(corners, dstQuad);
                        if (H.empty()) {
                            continue; // Skip if transform matrix is invalid
                        }
                        warpPerspective(gray_for_pattern, warped, H, Size(tagSize, tagSize));
                        if (warped.empty()) {
                            continue; // Skip if warping failed
                        }
                    } catch (...) {
                        continue; // Skip if transformation fails
                    }
                    
                    // Extract pattern
                    vector<vector<int>> pattern = extractPattern(warped, tagSize);
                    PatternData pdata;
                    pdata.detection_idx = tag_idx;
                    pdata.pattern = pattern;
                    pdata.warped_image = warped.clone();  // Store the warped image before digitization
                    all_patterns.push_back(pdata);
                }
                
                // Calculate grid layout based on successfully extracted patterns
                int num_patterns = all_patterns.size();
                if (num_patterns > 0) {
                    // Calculate grid layout: try to make it roughly square
                    int cols = (int)ceil(sqrt(num_patterns));
                    int rows = (int)ceil((double)num_patterns / cols);
                    
                    // Determine cell size based on number of patterns (smaller when multiple)
                    // Display 8x8 grid: 6x6 data + 1-cell black border on all sides
                    // Scale to 75% to fit width better
                    int base_cell_size = (num_patterns == 1) ? 50 : 30;
                    int cell_size = (int)(base_cell_size * 0.75);  // 75% of original size
                    if (cell_size < 1) cell_size = 1;  // Ensure minimum size
                    int padding = (num_patterns == 1) ? 30 : 15;
                    int grid_size = 8 * cell_size;  // 8x8 grid (6x6 data + border)
                    int header_height = (num_patterns == 1) ? 40 : 25;
                    int spacing = 10; // Space between multiple patterns
                    
                    // Calculate total visualization size
                    // Each pattern cell includes: warped image (left) + digitized pattern (right)
                    int warped_image_size = grid_size;  // Same size as pattern grid
                    int cell_total_width = warped_image_size + spacing + grid_size;  // Warped image + spacing + pattern
                    int total_width = cols * (cell_total_width + padding * 2) + (cols - 1) * spacing;
                    int total_height = rows * (grid_size + padding * 2 + header_height) + (rows - 1) * spacing;
                    
                    // Ensure valid dimensions before creating Mat
                    Mat pattern_vis;
                    if (total_width > 0 && total_height > 0 && grid_size > 0 && cell_size > 0) {
                        pattern_vis = Mat::ones(total_height, total_width, CV_8UC3) * 240;
                    
                    // Draw all successfully extracted patterns
                    for (size_t pattern_idx = 0; pattern_idx < all_patterns.size(); pattern_idx++) {
                        const PatternData& pdata = all_patterns[pattern_idx];
                        // Safety check: ensure detection index is valid
                        if (pdata.detection_idx >= all_detections_data.size()) {
                            continue; // Skip if detection index is out of bounds
                        }
                        const DetectionData& data = all_detections_data[pdata.detection_idx];
                        const vector<vector<int>>& pattern = pdata.pattern;
                        
                        // Safety check: ensure pattern is valid (6x6)
                        if (pattern.size() != 6) {
                            continue; // Skip if pattern has wrong number of rows
                        }
                        // Check all rows have correct size
                        bool pattern_valid = true;
                        for (size_t i = 0; i < pattern.size(); i++) {
                            if (pattern[i].size() != 6) {
                                pattern_valid = false;
                                break;
                            }
                        }
                        if (!pattern_valid) {
                            continue; // Skip if pattern rows have wrong size
                        }
                        
                        // Calculate position in grid
                        int col = pattern_idx % cols;
                        int row = pattern_idx / cols;
                        int cell_total_width = warped_image_size + spacing + grid_size;
                        int x_offset = col * (cell_total_width + padding * 2) + (col * spacing);
                        int y_offset = row * (grid_size + padding * 2 + header_height) + (row * spacing);
                        
                        // Draw header
                        stringstream header;
                        header << "ID:" << data.id;
                        putText(pattern_vis, header.str(), Point(x_offset + padding, y_offset + 20),
                               FONT_HERSHEY_SIMPLEX, (num_patterns == 1) ? 0.5 : 0.4, Scalar(0, 0, 0), 1);
                        
                        // Draw warped image (before digitization) on the left
                        if (!pdata.warped_image.empty() && pdata.warped_image.rows > 0 && pdata.warped_image.cols > 0) {
                            Mat warped_resized;
                            cv::resize(pdata.warped_image, warped_resized, Size(warped_image_size, warped_image_size), 0, 0, INTER_NEAREST);
                            
                            // Convert to BGR for display
                            Mat warped_bgr;
                            if (warped_resized.channels() == 1) {
                                cvtColor(warped_resized, warped_bgr, COLOR_GRAY2BGR);
                            } else {
                                warped_bgr = warped_resized;
                            }
                            
                            // DEBUG: Draw border extraction region on warped image
                            int debug_tagSize = pdata.warped_image.rows;  // Original warped image size
                            int debug_borderSize = (debug_tagSize <= 0) ? 4 : debug_tagSize / 8;
                            int debug_borderMargin = max(1, debug_borderSize / 4);
                            int debug_effectiveBorderSize = debug_borderSize + debug_borderMargin;
                            double scale = (double)warped_image_size / debug_tagSize;
                            
                            // Draw actual border boundary (where border ends) in green
                            int actual_border_x1 = (int)(debug_borderSize * scale);
                            int actual_border_y1 = (int)(debug_borderSize * scale);
                            int actual_border_x2 = (int)((debug_tagSize - debug_borderSize) * scale);
                            int actual_border_y2 = (int)((debug_tagSize - debug_borderSize) * scale);
                            // Top border line (green - actual border)
                            line(warped_bgr, Point(0, actual_border_y1), Point(warped_image_size - 1, actual_border_y1), Scalar(0, 255, 0), 1);
                            // Bottom border line (green)
                            line(warped_bgr, Point(0, actual_border_y2), Point(warped_image_size - 1, actual_border_y2), Scalar(0, 255, 0), 1);
                            // Left border line (green)
                            line(warped_bgr, Point(actual_border_x1, 0), Point(actual_border_x1, warped_image_size - 1), Scalar(0, 255, 0), 1);
                            // Right border line (green)
                            line(warped_bgr, Point(actual_border_x2, 0), Point(actual_border_x2, warped_image_size - 1), Scalar(0, 255, 0), 1);
                            
                            // Draw effective extraction boundary (where we start sampling) in red
                            int effective_border_x1 = (int)(debug_effectiveBorderSize * scale);
                            int effective_border_y1 = (int)(debug_effectiveBorderSize * scale);
                            int effective_border_x2 = (int)((debug_tagSize - debug_effectiveBorderSize) * scale);
                            int effective_border_y2 = (int)((debug_tagSize - debug_effectiveBorderSize) * scale);
                            // Top extraction line (red - where we start sampling)
                            line(warped_bgr, Point(0, effective_border_y1), Point(warped_image_size - 1, effective_border_y1), Scalar(0, 0, 255), 1);
                            // Bottom extraction line (red)
                            line(warped_bgr, Point(0, effective_border_y2), Point(warped_image_size - 1, effective_border_y2), Scalar(0, 0, 255), 1);
                            // Left extraction line (red)
                            line(warped_bgr, Point(effective_border_x1, 0), Point(effective_border_x1, warped_image_size - 1), Scalar(0, 0, 255), 1);
                            // Right extraction line (red)
                            line(warped_bgr, Point(effective_border_x2, 0), Point(effective_border_x2, warped_image_size - 1), Scalar(0, 0, 255), 1);
                            
                            // Copy warped image to visualization
                            Rect warped_rect(x_offset + padding, y_offset + header_height + padding, 
                                           warped_image_size, warped_image_size);
                            warped_bgr.copyTo(pattern_vis(warped_rect));
                            
                            // Draw border around warped image
                            rectangle(pattern_vis, warped_rect, Scalar(0, 0, 255), 2);
                            
                            // Label for warped image
                            putText(pattern_vis, "Warped (green=border, red=extraction)", Point(x_offset + padding, y_offset + header_height + padding - 5),
                                   FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 0, 255), 1);
                        }
                        
                        // Pattern grid position (to the right of warped image)
                        int pattern_x_offset = x_offset + padding + warped_image_size + spacing;
                        
                        // Draw 8x8 grid with pattern (6x6 data + 1-cell black border)
                        // Tag36h11 structure: 8x8 cells total, 1-cell black border, 6x6 data region
                        
                        // Bounds check: ensure pattern grid fits within pattern_vis
                        int pattern_grid_right = pattern_x_offset + grid_size;
                        int pattern_grid_bottom = y_offset + header_height + padding + grid_size;
                        if (pattern_grid_right > pattern_vis.cols || pattern_grid_bottom > pattern_vis.rows) {
                            qDebug() << "Warning: Pattern grid extends beyond visualization bounds. Skipping pattern visualization.";
                            continue; // Skip this pattern to avoid out-of-bounds access
                        }
                        
                        // First, draw border cells (all black)
                        // Top row (row 0)
                        for (int c = 0; c < 8; c++) {
                            int y_pos = y_offset + header_height + padding;
                            int x_pos = pattern_x_offset + c * cell_size;
                            // Defensive bounds check
                            if (x_pos + cell_size > pattern_vis.cols || y_pos + cell_size > pattern_vis.rows) break;
                            Rect cell(x_pos, y_pos, cell_size, cell_size);
                            rectangle(pattern_vis, cell, Scalar(0, 0, 0), -1);  // Black border
                            rectangle(pattern_vis, cell, Scalar(128, 128, 128), 1);
                        }
                        // Bottom row (row 7)
                        for (int c = 0; c < 8; c++) {
                            int y_pos = y_offset + header_height + padding + 7 * cell_size;
                            int x_pos = pattern_x_offset + c * cell_size;
                            // Defensive bounds check
                            if (x_pos + cell_size > pattern_vis.cols || y_pos + cell_size > pattern_vis.rows) break;
                            Rect cell(x_pos, y_pos, cell_size, cell_size);
                            rectangle(pattern_vis, cell, Scalar(0, 0, 0), -1);  // Black border
                            rectangle(pattern_vis, cell, Scalar(128, 128, 128), 1);
                        }
                        // Left column (col 0, rows 1-6)
                        for (int r = 1; r < 7; r++) {
                            int y_pos = y_offset + header_height + padding + r * cell_size;
                            int x_pos = pattern_x_offset;
                            // Defensive bounds check
                            if (x_pos + cell_size > pattern_vis.cols || y_pos + cell_size > pattern_vis.rows) break;
                            Rect cell(x_pos, y_pos, cell_size, cell_size);
                            rectangle(pattern_vis, cell, Scalar(0, 0, 0), -1);  // Black border
                            rectangle(pattern_vis, cell, Scalar(128, 128, 128), 1);
                        }
                        // Right column (col 7, rows 1-6)
                        for (int r = 1; r < 7; r++) {
                            int y_pos = y_offset + header_height + padding + r * cell_size;
                            int x_pos = pattern_x_offset + 7 * cell_size;
                            // Defensive bounds check
                            if (x_pos + cell_size > pattern_vis.cols || y_pos + cell_size > pattern_vis.rows) break;
                            Rect cell(x_pos, y_pos, cell_size, cell_size);
                            rectangle(pattern_vis, cell, Scalar(0, 0, 0), -1);  // Black border
                            rectangle(pattern_vis, cell, Scalar(128, 128, 128), 1);
                        }
                        
                        // Now draw the 6x6 data pattern in the center (rows 1-6, columns 1-6)
                        // Defensive bounds checking (pattern should already be validated above)
                        for (int r = 0; r < 6 && r < (int)pattern.size(); r++) {
                            if (r >= (int)pattern.size() || pattern[r].size() != 6) continue;
                            for (int c = 0; c < 6 && c < (int)pattern[r].size(); c++) {
                                int val = pattern[r][c];
                                bool is_black = val < 128;
                                
                                Scalar color = is_black ? Scalar(0, 0, 0) : Scalar(255, 255, 255);
                                // Map 6x6 pattern (r,c) to 8x8 grid position (r+1, c+1)
                                int y_pos = y_offset + header_height + padding + (r + 1) * cell_size;
                                int x_pos = pattern_x_offset + (c + 1) * cell_size;
                                // Defensive bounds check
                                if (x_pos + cell_size > pattern_vis.cols || y_pos + cell_size > pattern_vis.rows) continue;
                                Rect cell(x_pos, y_pos, cell_size, cell_size);
                                rectangle(pattern_vis, cell, color, -1);
                                rectangle(pattern_vis, cell, Scalar(128, 128, 128), 1);
                                
                                // Draw bit value (only if cell is large enough)
                                if (cell_size >= 25) {
                                    string bit_str = is_black ? "1" : "0";
                                    int font_scale = (num_patterns == 1) ? 0.6 : 0.4;
                                    Scalar text_color = is_black ? Scalar(255, 255, 255) : Scalar(0, 0, 0);
                                    putText(pattern_vis, bit_str, Point(x_pos + cell_size/4, y_pos + cell_size/2),
                                           FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1);
                                }
                            }
                        }
                        
                        // Label for digitized pattern
                        putText(pattern_vis, "Digitized (8x8: border + 6x6 data)", Point(pattern_x_offset, y_offset + header_height + padding - 5),
                               FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 0, 255), 1);
                    }
                    
                        // Display pattern visualization
                        if (!pattern_vis.empty() && pattern_vis.cols > 0 && pattern_vis.rows > 0 && pattern_vis.data != nullptr) {
                            try {
                                // Ensure Mat is continuous (QImage requires contiguous data)
                                Mat pattern_vis_cont;
                                if (!pattern_vis.isContinuous()) {
                                    pattern_vis_cont = pattern_vis.clone();
                                } else {
                                    pattern_vis_cont = pattern_vis;
                                }
                                
                                // Verify the Mat is valid before creating QImage
                                if (pattern_vis_cont.data != nullptr && pattern_vis_cont.cols > 0 && pattern_vis_cont.rows > 0) {
                                    QImage pattern_qimg(pattern_vis_cont.data, pattern_vis_cont.cols, pattern_vis_cont.rows, 
                                                       pattern_vis_cont.step, QImage::Format_RGB888);
                                    if (!pattern_qimg.isNull()) {
                                        QPixmap pattern_pixmap = QPixmap::fromImage(pattern_qimg.copy());
                                        if (!pattern_pixmap.isNull() && capturePatternLabel_) {
                                            capturePatternLabel_->setPixmap(pattern_pixmap);
                                        }
                                    } else {
                                        qDebug() << "Failed to create QImage from pattern visualization";
                                    }
                                } else {
                                    qDebug() << "Pattern visualization Mat is invalid for QImage conversion";
                                }
                            } catch (const std::exception& e) {
                                qDebug() << "Exception converting pattern visualization to QImage/QPixmap:" << e.what();
                            } catch (...) {
                                qDebug() << "Unknown error converting pattern visualization to QImage/QPixmap";
                            }
                        }
                    } else {
                        qDebug() << "Skipping pattern visualization due to invalid dimensions - width:" << total_width 
                                 << "height:" << total_height << "grid_size:" << grid_size << "cell_size:" << cell_size;
                    }
                    
                    // Store patterns for saving
                    {
                        std::lock_guard<std::mutex> lock(storedPatternsMutex_);
                        storedPatterns_.clear();
                        qDebug() << "Storing" << all_patterns.size() << "patterns for saving";
                        for (size_t i = 0; i < all_patterns.size(); i++) {
                            const PatternData& pdata = all_patterns[i];
                            if (pdata.detection_idx < all_detections_data.size()) {
                                const DetectionData& data = all_detections_data[pdata.detection_idx];
                                StoredPatternData sp;
                                sp.tag_id = data.id;
                                sp.warped_image = pdata.warped_image.clone();
                                sp.pattern = pdata.pattern;
                                storedPatterns_.push_back(sp);
                                qDebug() << "  - Stored pattern for tag ID:" << sp.tag_id << "warped empty:" << sp.warped_image.empty() << "pattern size:" << sp.pattern.size();
                            }
                        }
                        qDebug() << "Total stored patterns:" << storedPatterns_.size();
                    }
                    
                    // Enable save button
                    if (savePatternsBtn_) {
                        savePatternsBtn_->setEnabled(true);
                    }
                } else {
                    // No patterns successfully extracted - clear the display
                    if (capturePatternLabel_) {
                        capturePatternLabel_->clear();
                        capturePatternLabel_->setText("Pattern extraction failed for all detections");
                    }
                    // Disable save button and clear stored patterns
                    if (savePatternsBtn_) {
                        savePatternsBtn_->setEnabled(false);
                    }
                    {
                        std::lock_guard<std::mutex> lock(storedPatternsMutex_);
                        storedPatterns_.clear();
                    }
                }
                
                // Build info text: show patterns first, then hamming codes
                stringstream info_ss;
                info_ss << "=== DETECTED TAGS: " << num_tags << " ===\n";
                info_ss << "=== SUCCESSFULLY EXTRACTED PATTERNS: " << all_patterns.size() << " ===\n\n";
                
                for (size_t pattern_idx = 0; pattern_idx < all_patterns.size(); pattern_idx++) {
                    const PatternData& pdata = all_patterns[pattern_idx];
                    // Safety check: ensure detection index is valid
                    if (pdata.detection_idx >= all_detections_data.size()) {
                        continue; // Skip if detection index is out of bounds
                    }
                    const DetectionData& data = all_detections_data[pdata.detection_idx];
                    const vector<vector<int>>& pattern = pdata.pattern;
                    
                    // Safety check: ensure pattern is valid (6x6)
                    if (pattern.size() != 6) {
                        continue; // Skip if pattern has wrong number of rows
                    }
                    // Check all rows have correct size
                    bool pattern_valid = true;
                    for (size_t i = 0; i < pattern.size(); i++) {
                        if (pattern[i].size() != 6) {
                            pattern_valid = false;
                            break;
                        }
                    }
                    if (!pattern_valid) {
                        continue; // Skip if pattern rows have wrong size
                    }
                    
                    info_ss << "--- Tag " << (pattern_idx + 1) << " (ID: " << data.id << ") ---\n";
                    
                    // Show 6x6 Pattern (data cells only, no border)
                    // Tag36h11 structure: 8x8 cells total, 1-cell black border, 6x6 data region
                    info_ss << "6x6 Data Pattern (border excluded):\n";
                    info_ss << "    0 1 2 3 4 5\n";
                    info_ss << "  ┌─────────────┐\n";
                    // Defensive bounds checking (pattern should already be validated above)
                    for (int row = 0; row < 6 && row < (int)pattern.size(); row++) {
                        if (row >= (int)pattern.size() || pattern[row].size() != 6) continue;
                        info_ss << row << " │";
                        for (int col = 0; col < 6 && col < (int)pattern[row].size(); col++) {
                            int val = pattern[row][col];
                            info_ss << ((val < 128) ? "1" : "0") << " ";
                        }
                        info_ss << "│\n";
                    }
                    info_ss << "  └─────────────┘\n";
                    info_ss << "Note: This 6x6 pattern contains ONLY data cells.\n";
                    info_ss << "      The 1-cell black border is excluded (Tag36h11: 8x8 total, 6x6 data).\n\n";
                    
                    // Then show hamming code info
                    uint64_t code = extractCodeFromPattern(pattern);
                    info_ss << "Hamming Code:\n";
                    info_ss << "  Decision Margin: " << fixed << setprecision(2) << data.decision_margin << "\n";
                    info_ss << "  Hamming: " << data.hamming << "\n";
                    info_ss << "  Decimal: " << code << "\n";
                    info_ss << "  Hex: 0x" << hex << setfill('0') << setw(9) << code << dec << "\n";
                    info_ss << "  Binary: ";
                    for (int i = 35; i >= 0; i--) {
                        info_ss << ((code >> i) & 1);
                        if (i % 9 == 0 && i > 0) info_ss << " ";
                    }
                    info_ss << "\n\n";
                    }
                    
                    capturePatternInfoText_->setPlainText(QString::fromStdString(info_ss.str()));
                } // End of if (!gray_for_pattern.empty())
            } else if (capturePatternLabel_ && capturePatternInfoText_) {
                // No detections
                capturePatternLabel_->clear();
                capturePatternLabel_->setText("No detection");
                capturePatternInfoText_->setPlainText("No tags detected. Pattern will appear here when tags are found.");
                // Disable save button and clear stored patterns
                if (savePatternsBtn_) {
                    savePatternsBtn_->setEnabled(false);
                }
                {
                    std::lock_guard<std::mutex> lock(storedPatternsMutex_);
                    storedPatterns_.clear();
                }
            }
            
            // Convert to QImage safely (copy data to avoid dangling pointer)
            QImage qimg;
            if (display.channels() == 3) {
                qimg = QImage(display.data, display.cols, display.rows, display.step, QImage::Format_RGB888).copy();
            } else {
                qimg = QImage(display.data, display.cols, display.rows, display.step, QImage::Format_Grayscale8).copy();
            }
            QPixmap pixmap = QPixmap::fromImage(qimg).scaled(640, 480, Qt::KeepAspectRatio, Qt::SmoothTransformation);
            previewLabel_->setPixmap(pixmap);
        }
    }
    
    void captureFrame() {
        if (!cameraOpen_) return;
        
        Mat frame;
        {
            lock_guard<mutex> lock(frameMutex_);
            if (currentFrame_.empty()) return;
            frame = currentFrame_.clone();
        }
        
        if (frame.empty()) return;
        
        // Generate suggested filename with camera name and timestamp
        QString suggestedFilename = generateCaptureFilename();
        
        // Show save dialog with suggested filename (user can edit it)
        QString filename = QFileDialog::getSaveFileName(
            this, 
            "Save Captured Frame", 
            suggestedFilename,
            "Image Files (*.png *.jpg *.jpeg *.bmp);;PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)");
        
        if (!filename.isEmpty()) {
            // Ensure the directory exists
            QFileInfo fileInfo(filename);
            QDir dir = fileInfo.absoluteDir();
            if (!dir.exists()) {
                dir.mkpath(".");
            }
            
            // Save frame (preserve original format)
            bool saved = imwrite(filename.toStdString(), frame);
            if (!saved) {
                QMessageBox::warning(this, "Error", QString("Failed to save frame to %1").arg(filename));
                return;
            }
            
            QMessageBox::information(this, "Capture", QString("Frame saved to %1").arg(filename));
            
            // Load it as Image 1 for processing (convert to grayscale for processing)
            if (frame.channels() == 1) {
                // Already grayscale
                image1_ = frame.clone();
            } else {
                // Convert to grayscale
                cvtColor(frame, image1_, COLOR_BGR2GRAY);
            }
            image1_path_ = filename.toStdString();
            
            // Process the image
            processImages();
        }
    }

    // Tab widget
    QTabWidget *tabWidget_;
    
    // UI elements (Processing tab)
    QPushButton *loadBtn1_, *loadBtn2_;
    QComboBox *preprocessCombo_, *edgeCombo_, *detectionCombo_, *advancedCombo_;
    QComboBox *quadCombo1_, *quadCombo2_;  // Independent quad selection for each image
    QCheckBox *mirrorCheckbox1_, *mirrorCheckbox2_;  // Independent mirror for each image
    QLabel *label1_, *label2_;
    QTextEdit *infoText_;
    QTextEdit *qualityText1_, *qualityText2_;  // Quality metrics for each image
    
    // UI elements (Capture tab)
    QComboBox *cameraCombo_;
    QComboBox *modeCombo_;  // Resolution/FPS selection
    QComboBox *captureAlgorithmCombo_;  // Algorithm selection for Capture tab
    QCheckBox *captureMirrorCheckbox_;  // Mirror checkbox for Capture tab
    QLabel *previewLabel_;
    QLabel *capturePatternLabel_;  // Pattern visualization for Capture tab
    QTextEdit *capturePatternInfoText_;  // Pattern info text for Capture tab
    QPushButton *savePatternsBtn_;  // Button to save pattern visualizations
    QPushButton *loadImageBtn_;
    QPushButton *captureBtn_;
    QPushButton *saveSettingsBtn_;
    
    // Stored pattern data for saving
    struct StoredPatternData {
        int tag_id;
        Mat warped_image;
        vector<vector<int>> pattern;
    };
    vector<StoredPatternData> storedPatterns_;
    mutex storedPatternsMutex_;
    // previewTimer_ and calibrationPreviewTimer_ already declared at line 159-160
    
    // UI elements (Fisheye tab)
    QLineEdit *calibPathEdit_;
    QLabel *fisheyeStatusLabel_;  // Status label in Fisheye tab
    // fisheyeStatusIndicator_ already declared at line 231
    QPushButton *loadTestImageBtn_;
    QLabel *fisheyeOriginalLabel_;
    QLabel *fisheyeCorrectedLabel_;
    QRadioButton *fisheyeUseOriginalRadio_;
    QRadioButton *fisheyeUseCorrectedRadio_;
    
    // UI elements (Algorithms tab) - duplicates removed, already declared at line 233-246
    
    // Calibration process UI
    QPushButton *resetCalibBtn_;
    QLabel *calibrationPreviewLabel_;
    QLabel *calibrationStatusLabel_;
    QLabel *calibrationProgressLabel_;
    QPushButton *saveCalibBtn_;
    
    // Calibration data
    bool calibrationInProgress_ = false;
    vector<vector<Point3f>> objectPoints_;  // 3D points in real world space
    vector<vector<Point2f>> imagePoints_;   // 2D points in image plane
    Size checkerboardSize_ = Size(6, 6);     // Inner corners (6x6)
    vector<Mat> calibrationImages_;
    vector<bool> gridCaptured_;             // Track which grid positions are captured (6x6 = 36 positions)
    Mat lastStableFrame_;                    // Last frame with stable checkerboard detection
    int stableFrameCount_ = 0;               // Count of consecutive stable detections
    static const int STABLE_THRESHOLD = 10;  // Frames needed for stable detection
    
    // Camera settings
    QSlider *exposureSlider_;
    QSlider *gainSlider_;
    QSlider *brightnessSlider_;
    QSlider *contrastSlider_;
    QSlider *saturationSlider_;
    QSlider *sharpnessSlider_;
    QSpinBox *exposureSpin_;
    QSpinBox *gainSpin_;
    QSpinBox *brightnessSpin_;
    QSpinBox *contrastSpin_;
    QSpinBox *saturationSpin_;
    QSpinBox *sharpnessSpin_;
    
    // All member variables declared above - duplicates removed (UI elements declared later in file at line 4635+)
    
    // Helper function to generate capture filename
    QString generateCaptureFilename() {
        // Create input directory if it doesn't exist
        QDir inputDir("input");
        if (!inputDir.exists()) {
            inputDir.mkpath(".");
        }
        
        // Get camera name
        QString cameraName = "UnknownCamera";
        if (selectedCameraIndex_ >= 0 && selectedCameraIndex_ < static_cast<int>(cameraList_.size())) {
            QString fullName = QString::fromStdString(cameraList_[selectedCameraIndex_]);
            // Clean up camera name (remove special characters, spaces)
            // Replace non-alphanumeric characters with underscore
            QString cleanName;
            for (int i = 0; i < fullName.length() && cleanName.length() < 30; i++) {
                QChar c = fullName[i];
                if (c.isLetterOrNumber() || c == '_') {
                    cleanName.append(c);
                } else {
                    cleanName.append('_');
                }
            }
            cameraName = cleanName.isEmpty() ? "Camera" : cleanName;
        }
        
        // Generate timestamp: YYYYMMDD_HHMMSS
        QDateTime now = QDateTime::currentDateTime();
        QString timestamp = now.toString("yyyyMMdd_HHmmss");
        
        // Generate filename: input/CameraName_YYYYMMDD_HHMMSS.png
        QString filename = QString("input/%1_%2.png").arg(cameraName).arg(timestamp);
        
        return filename;
    }
    
    void saveCameraSettings() {
        if (!cameraOpen_ || cameraList_.empty() || selectedCameraIndex_ < 0) return;
        
        QString configPath = "camera_settings.txt";
        QFile file(configPath);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QMessageBox::warning(this, "Error", "Failed to save camera settings to " + configPath);
            return;
        }
        
        QTextStream out(&file);
        out << "# Camera Settings Configuration\n";
        out << "# Format: camera_name=setting_name=value\n";
        out << "\n";
        
        QString cameraName = QString::fromStdString(cameraList_[selectedCameraIndex_]);
        out << "[Camera]\n";
        out << "name=" << cameraName << "\n";
        out << "type=" << (useMindVision_ ? "MindVision" : "V4L2") << "\n";
        out << "\n";
        
        out << "[Settings]\n";
        out << "exposure=" << exposureSlider_->value() << "\n";
        out << "gain=" << gainSlider_->value() << "\n";
        out << "brightness=" << brightnessSlider_->value() << "\n";
        out << "contrast=" << contrastSlider_->value() << "\n";
        out << "saturation=" << saturationSlider_->value() << "\n";
        out << "sharpness=" << sharpnessSlider_->value() << "\n";
        if (modeCombo_->currentIndex() >= 0) {
            out << "mode_index=" << modeCombo_->currentIndex() << "\n";
        }
        
        file.close();
        QMessageBox::information(this, "Success", "Camera settings saved to " + configPath);
    }
    
    void loadCameraSettings() {
        QString configPath = "camera_settings.txt";
        QFile file(configPath);
        if (!file.exists() || !file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            // Config file doesn't exist, use default values
            return;
        }
        
        QTextStream in(&file);
        QString cameraName = QString::fromStdString(cameraList_[selectedCameraIndex_]);
        QString currentType = useMindVision_ ? "MindVision" : "V4L2";
        
        QString section;
        bool settingsSection = false;
        bool cameraMatches = false;
        
        while (!in.atEnd()) {
            QString line = in.readLine().trimmed();
            
            // Skip comments and empty lines
            if (line.startsWith("#") || line.isEmpty()) continue;
            
            // Check for section headers
            if (line.startsWith("[") && line.endsWith("]")) {
                section = line.mid(1, line.length() - 2);
                settingsSection = (section == "Settings");
                if (section == "Camera") {
                    cameraMatches = false;
                }
                continue;
            }
            
            // Parse Camera section
            if (section == "Camera") {
                if (line.startsWith("name=")) {
                    QString savedName = line.mid(5).trimmed();
                    if (savedName == cameraName) {
                        cameraMatches = true;
                    }
                } else if (line.startsWith("type=")) {
                    QString savedType = line.mid(5).trimmed();
                    if (savedType != currentType) {
                        cameraMatches = false;  // Type mismatch
                    }
                }
            }
            
            // Parse Settings section (only if camera matches)
            if (settingsSection && cameraMatches) {
                if (line.contains("=")) {
                    QStringList parts = line.split("=");
                    if (parts.size() == 2) {
                        QString key = parts[0].trimmed();
                        QString value = parts[1].trimmed();
                        bool ok;
                        int intValue = value.toInt(&ok);
                        
                        if (!ok) continue;
                        
                        // Block signals while setting values to prevent immediate camera updates
                        if (key == "exposure") {
                            exposureSlider_->blockSignals(true);
                            exposureSlider_->setValue(intValue);
                            exposureSpin_->setValue(intValue);
                            exposureSlider_->blockSignals(false);
                        } else if (key == "gain") {
                            gainSlider_->blockSignals(true);
                            gainSlider_->setValue(intValue);
                            gainSpin_->setValue(intValue);
                            gainSlider_->blockSignals(false);
                        } else if (key == "brightness") {
                            brightnessSlider_->blockSignals(true);
                            brightnessSlider_->setValue(intValue);
                            brightnessSpin_->setValue(intValue);
                            brightnessSlider_->blockSignals(false);
                        } else if (key == "contrast") {
                            contrastSlider_->blockSignals(true);
                            contrastSlider_->setValue(intValue);
                            contrastSpin_->setValue(intValue);
                            contrastSlider_->blockSignals(false);
                        } else if (key == "saturation") {
                            saturationSlider_->blockSignals(true);
                            saturationSlider_->setValue(intValue);
                            saturationSpin_->setValue(intValue);
                            saturationSlider_->blockSignals(false);
                        } else if (key == "sharpness") {
                            sharpnessSlider_->blockSignals(true);
                            sharpnessSlider_->setValue(intValue);
                            sharpnessSpin_->setValue(intValue);
                            sharpnessSlider_->blockSignals(false);
                        } else if (key == "mode_index") {
                            if (intValue >= 0 && intValue < modeCombo_->count()) {
                                modeCombo_->blockSignals(true);
                                modeCombo_->setCurrentIndex(intValue);
                                modeCombo_->blockSignals(false);
                                onModeChanged(intValue);
                            }
                        }
                    }
                }
            }
        }
        
        file.close();
        
        // Apply settings to camera after loading
        if (cameraMatches) {
            updateCameraSettings();
        }
    }
    
    void loadCameraSettingsForAlgorithm() {
        QString configPath = "camera_settings.txt";
        QFile file(configPath);
        if (!file.exists() || !file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            // Config file doesn't exist, use default values
            return;
        }
        
        QTextStream in(&file);
        int cameraIndex = algorithmCameraCombo_->currentIndex();
        if (cameraIndex < 0 || cameraIndex >= (int)cameraList_.size()) {
            file.close();
            return;
        }
        
        QString cameraName = QString::fromStdString(cameraList_[cameraIndex]);
        QString currentType = algorithmUseMindVision_ ? "MindVision" : "V4L2";
        
        QString section;
        bool settingsSection = false;
        bool cameraMatches = false;
        int savedExposure = -1, savedGain = -1, savedBrightness = -1;
        int savedContrast = -1, savedSaturation = -1, savedSharpness = -1;
        int savedModeIndex = -1;
        
        while (!in.atEnd()) {
            QString line = in.readLine().trimmed();
            
            // Skip comments and empty lines
            if (line.startsWith("#") || line.isEmpty()) continue;
            
            // Check for section headers
            if (line.startsWith("[") && line.endsWith("]")) {
                section = line.mid(1, line.length() - 2);
                settingsSection = (section == "Settings");
                if (section == "Camera") {
                    cameraMatches = false;
                }
                continue;
            }
            
            // Parse Camera section
            if (section == "Camera") {
                if (line.startsWith("name=")) {
                    QString savedName = line.mid(5).trimmed();
                    if (savedName == cameraName) {
                        cameraMatches = true;
                    }
                } else if (line.startsWith("type=")) {
                    QString savedType = line.mid(5).trimmed();
                    if (savedType != currentType) {
                        cameraMatches = false;  // Type mismatch
                    }
                }
            }
            
            // Parse Settings section (only if camera matches)
            if (settingsSection && cameraMatches) {
                if (line.contains("=")) {
                    QStringList parts = line.split("=");
                    if (parts.size() == 2) {
                        QString key = parts[0].trimmed();
                        QString value = parts[1].trimmed();
                        bool ok;
                        int intValue = value.toInt(&ok);
                        
                        if (!ok) continue;
                        
                        if (key == "exposure") {
                            savedExposure = intValue;
                        } else if (key == "gain") {
                            savedGain = intValue;
                        } else if (key == "brightness") {
                            savedBrightness = intValue;
                        } else if (key == "contrast") {
                            savedContrast = intValue;
                        } else if (key == "saturation") {
                            savedSaturation = intValue;
                        } else if (key == "sharpness") {
                            savedSharpness = intValue;
                        } else if (key == "mode_index") {
                            savedModeIndex = intValue;
                        }
                    }
                }
            }
        }
        
        file.close();
        
        // Apply settings directly to camera if they were loaded
        if (cameraMatches) {
            if (algorithmUseMindVision_) {
#ifdef HAVE_MINDVISION_SDK
                if (algorithmMvHandle_ != 0) {
                    // Apply exposure
                    if (savedExposure >= 0) {
                        double min_exposure = 1000.0;
                        double max_exposure = 100000.0;
                        double exposure = max_exposure - (savedExposure / 100.0) * (max_exposure - min_exposure);
                        CameraSetExposureTime(algorithmMvHandle_, exposure);
                    }
                    
                    // Apply gain
                    if (savedGain >= 0) {
                        CameraSetGain(algorithmMvHandle_, savedGain, savedGain, savedGain);
                    }
                    
                    // Apply brightness (analog gain)
                    if (savedBrightness >= 0) {
                        INT analogGain = (savedBrightness * 100) / 255;
                        CameraSetAnalogGain(algorithmMvHandle_, analogGain);
                    }
                    
                    // Apply contrast
                    if (savedContrast >= 0) {
                        CameraSetContrast(algorithmMvHandle_, savedContrast);
                    }
                    
                    // Apply saturation
                    if (savedSaturation >= 0) {
                        CameraSetSaturation(algorithmMvHandle_, savedSaturation);
                    }
                    
                    // Apply sharpness
                    if (savedSharpness >= 0) {
                        CameraSetSharpness(algorithmMvHandle_, savedSharpness);
                    }
                }
#endif
            } else {
                // Apply V4L2 settings
                if (algorithmCamera_.isOpened()) {
                    if (savedExposure >= 0) {
                        algorithmCamera_.set(CAP_PROP_EXPOSURE, savedExposure);
                    }
                    if (savedGain >= 0) {
                        algorithmCamera_.set(CAP_PROP_GAIN, savedGain);
                    }
                    if (savedBrightness >= 0) {
                        algorithmCamera_.set(CAP_PROP_BRIGHTNESS, savedBrightness);
                    }
                    if (savedContrast >= 0) {
                        algorithmCamera_.set(CAP_PROP_CONTRAST, savedContrast);
                    }
                    if (savedSaturation >= 0) {
                        algorithmCamera_.set(CAP_PROP_SATURATION, savedSaturation);
                    }
                    if (savedSharpness >= 0) {
                        algorithmCamera_.set(CAP_PROP_SHARPNESS, savedSharpness);
                    }
                }
            }
        }
    }
};

#include "apriltag_debug_gui.moc"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    
    AprilTagDebugGUI window;
    window.show();
    
    return app.exec();
}

