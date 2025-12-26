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
#include <QCoreApplication>
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
#include <cstdlib>  // For setenv

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
#include <QProcess>
#include <QRegularExpression>
#include <QSet>

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
#include <map>

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

// Camera setting definition structure
enum SettingType {
    SETTING_INT,
    SETTING_BOOL,
    SETTING_MENU,
    SETTING_DOUBLE
};

struct CameraSetting {
    string name;           // Setting name (e.g., "exposure", "brightness")
    string display_name;   // Display name (e.g., "Exposure", "Brightness")
    SettingType type;      // Type of setting
    int min_value;         // Minimum value
    int max_value;         // Maximum value
    int default_value;     // Default value
    int current_value;     // Current value
    string v4l2_id;        // V4L2 control ID (for Arducam) or empty for MindVision
    vector<string> menu_options;  // For menu type settings
};

struct CameraSettingsProfile {
    string camera_name;    // Camera identifier
    string camera_type;    // "MindVision" or "Arducam" or "V4L2"
    vector<CameraSetting> settings;  // Available settings for this camera
    map<string, int> saved_values;   // Saved values keyed by setting name
    map<string, double> algorithm_settings;  // Algorithm tuning settings (doubles) keyed by setting name
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
    
    // Camera settings profiles (per camera)
    map<string, CameraSettingsProfile> cameraSettingsProfiles_;
    CameraSettingsProfile currentCameraSettings_;
    
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
    QCheckBox *algorithmDebugCheckbox_;  // Debug mode: show quads before filtering
    QLabel *algorithmDisplayLabel_;
    QTextEdit *algorithmTimingText_;
    QLabel *algorithmFPSLabel_;
    QTextEdit *algorithmQualityText_;
    QTextEdit *algorithmPoseText_;
    QTextEdit *algorithmDetailedTimingText_;  // For Fast AprilTag detailed timing analysis
    
    // Stored pattern data for saving (declared early for nvcc)
    struct StoredPatternData {
        int tag_id;
        Mat warped_image;
        vector<vector<int>> pattern;
    };
    vector<StoredPatternData> storedPatterns_;
    mutex storedPatternsMutex_;
    
    // UI elements (Processing tab)
    QTabWidget *tabWidget_;
    QComboBox *preprocessCombo_;
    QComboBox *edgeCombo_;
    QComboBox *detectionCombo_;
    QComboBox *advancedCombo_;
    QComboBox *quadCombo1_;
    QComboBox *quadCombo2_;
    QCheckBox *mirrorCheckbox1_;
    QCheckBox *mirrorCheckbox2_;
    QLabel *label1_;
    QLabel *label2_;
    QTextEdit *qualityText1_;
    QTextEdit *qualityText2_;
    QTextEdit *infoText_;
    
    // UI elements (Capture tab)
    QComboBox *captureAlgorithmCombo_;
    QCheckBox *captureMirrorCheckbox_;
    QLabel *capturePreviewLabel_;
    QPushButton *loadImageBtn_;
    QPushButton *captureBtn_;
    QPushButton *saveSettingsBtn_;
    QLabel *capturePatternLabel_;
    QPushButton *savePatternsBtn_;
    QTextEdit *capturePatternInfoText_;
    
    // UI elements (Camera settings)
    QComboBox *modeCombo_;
    QComboBox *cameraCombo_;
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
    
    // Dynamic camera controls (for V4L2/Arducam - stores all controls)
    map<string, QWidget*> dynamicControlWidgets_;  // Maps setting name to widget (QSlider, QCheckBox, QComboBox)
    map<string, QSlider*> dynamicSliders_;  // Maps setting name to QSlider
    map<string, QSpinBox*> dynamicSpinBoxes_;  // Maps setting name to QSpinBox
    map<string, QCheckBox*> dynamicCheckBoxes_;  // Maps setting name to QCheckBox
    map<string, QComboBox*> dynamicComboBoxes_;  // Maps setting name to QComboBox
    QHBoxLayout *dynamicSettingsLayout_;  // Layout for dynamic controls
    QScrollArea *dynamicSettingsScrollArea_;  // Scroll area for dynamic controls
    QWidget *dynamicSettingsWidget_;  // Widget containing dynamic controls
    
    // UI elements (Fisheye tab)
    QLineEdit *calibPathEdit_;
    QLabel *fisheyeStatusLabel_;  // Status label in Fisheye tab
    
    // Settings tab UI elements
    QCheckBox *settingsLoadCalibrationOnStart_;
    QLineEdit *settingsCalibrationPath_;
    QCheckBox *settingsEnableFisheyeForMindVision_;
    QPushButton *settingsSaveBtn_;
    QPushButton *settingsLoadBtn_;
    
    // Settings tab camera settings UI elements
    QComboBox *settingsCameraCombo_;
    QSlider *settingsExposureSlider_;
    QSlider *settingsGainSlider_;
    
    // Algorithm tuning settings UI elements
    // Preprocessing
    QDoubleSpinBox *preprocessClaheClipSpin_;
    QDoubleSpinBox *preprocessGammaSpin_;
    QDoubleSpinBox *preprocessContrastSpin_;
    QCheckBox *preprocessHistEqCheck_;
    
    // Edge Detection
    QSpinBox *cannyLowSpin_;
    QSpinBox *cannyHighSpin_;
    QSpinBox *adaptiveThreshBlockSpin_;
    QSpinBox *adaptiveThreshConstantSpin_;
    
    // Detection Parameters
    QDoubleSpinBox *quadDecimateSpin_;
    QDoubleSpinBox *quadSigmaSpin_;
    QCheckBox *refineEdgesCheck_;
    QDoubleSpinBox *decodeSharpeningSpin_;
    QSpinBox *nthreadsSpin_;
    
    // Quad Threshold Parameters
    QSpinBox *minClusterPixelsSpin_;
    QDoubleSpinBox *maxLineFitMseSpin_;
    QDoubleSpinBox *criticalAngleSpin_;
    QSpinBox *minWhiteBlackDiffSpin_;
    
    // Advanced Parameters
    QSpinBox *cornerRefineWinSizeSpin_;
    QSpinBox *cornerRefineMaxIterSpin_;
    QSpinBox *patternBorderSizeSpin_;
    QSpinBox *tagMinAreaSpin_;
    QSpinBox *tagMaxAreaSpin_;
    QSlider *settingsBrightnessSlider_;
    QSlider *settingsContrastSlider_;
    QSlider *settingsSaturationSlider_;
    QSlider *settingsSharpnessSlider_;
    QSpinBox *settingsExposureSpin_;
    QSpinBox *settingsGainSpin_;
    QSpinBox *settingsBrightnessSpin_;
    QSpinBox *settingsContrastSpin_;
    QSpinBox *settingsSaturationSpin_;
    QSpinBox *settingsSharpnessSpin_;
    QComboBox *settingsModeCombo_;
    
    // Initialize settings pointers to nullptr in constructor
    QPushButton *loadTestImageBtn_;
    QLabel *fisheyeOriginalLabel_;
    QLabel *fisheyeCorrectedLabel_;
    QRadioButton *fisheyeUseOriginalRadio_;
    QRadioButton *fisheyeUseCorrectedRadio_;
    
    // UI elements (Calibration tab)
    QPushButton *resetCalibBtn_;
    QLabel *calibrationPreviewLabel_;
    QLabel *calibrationStatusLabel_;
    QLabel *calibrationProgressLabel_;
    QPushButton *saveCalibBtn_;
    
    // UI elements (Preview)
    QLabel *previewLabel_;
    
    // Calibration data
    bool calibrationInProgress_;
    vector<vector<Point3f>> objectPoints_;  // 3D points in real world space
    vector<vector<Point2f>> imagePoints_;   // 2D points in image plane
    Size checkerboardSize_;     // Inner corners (6x6)
    vector<Mat> calibrationImages_;
    vector<bool> gridCaptured_;             // Track which grid positions are captured (6x6 = 36 positions)
    Mat lastStableFrame_;                    // Last frame with stable checkerboard detection
    int stableFrameCount_;               // Count of consecutive stable detections
    static const int STABLE_THRESHOLD = 10;  // Frames needed for stable detection

public:
    AprilTagDebugGUI(QWidget *parent = nullptr) : QWidget(parent),
        tf_(nullptr),
        td_(nullptr),
        previewTimer_(nullptr),
        calibrationPreviewTimer_(nullptr),
        settingsLoadCalibrationOnStart_(nullptr),
        settingsCalibrationPath_(nullptr),
        settingsEnableFisheyeForMindVision_(nullptr),
        settingsSaveBtn_(nullptr),
        settingsLoadBtn_(nullptr),
        settingsCameraCombo_(nullptr),
        settingsExposureSlider_(nullptr),
        settingsGainSlider_(nullptr),
        settingsBrightnessSlider_(nullptr),
        settingsContrastSlider_(nullptr),
        settingsSaturationSlider_(nullptr),
        settingsSharpnessSlider_(nullptr),
        settingsExposureSpin_(nullptr),
        settingsGainSpin_(nullptr),
        settingsBrightnessSpin_(nullptr),
        settingsContrastSpin_(nullptr),
        settingsSaturationSpin_(nullptr),
        settingsSharpnessSpin_(nullptr),
        settingsModeCombo_(nullptr),
        dynamicSettingsLayout_(nullptr),
        dynamicSettingsScrollArea_(nullptr),
        dynamicSettingsWidget_(nullptr)
    {
        setupUI();
        
        // Load and apply settings after UI is fully set up
        if (settingsLoadCalibrationOnStart_ && settingsCalibrationPath_) {
            loadSettingsFromFile();
            applySettings();
        } else {
            // Fallback to default path if settings UI not ready
            QString calib_path = "/home/nav/9202/Hiru/Apriltag/calibration_data/camera_params.yaml";
            if (loadFisheyeCalibration(calib_path)) {
                qDebug() << "Fisheye calibration loaded successfully on startup (default path)";
                if (fisheyeStatusLabel_) {
                    fisheyeStatusLabel_->setText(QString("Calibration: Loaded (Size: %1x%2)")
                        .arg(fisheye_image_size_.width)
                        .arg(fisheye_image_size_.height));
                }
                updateFisheyeStatusIndicator();
            }
        }
        
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
        
        // Switch to Capture tab on startup (first tab, index 0)
        QTimer::singleShot(100, this, [this]() {
            if (tabWidget_) {
                tabWidget_->setCurrentIndex(0); // Capture tab is first (index 0)
            }
            // Camera combo defaults to "None" (set in setupCaptureTab)
        });
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
            // Immediately apply current algorithm settings to ensure consistency
            applyAlgorithmSettingsToAlgorithm(currentAlgorithm_.get());
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
            
            // Enable fisheye correction for MindVision cameras if setting is enabled (if calibration is loaded)
            // Check if settings UI is initialized before accessing it
            bool enableFisheyeForMV = true;  // Default to true
            if (settingsEnableFisheyeForMindVision_) {
                enableFisheyeForMV = settingsEnableFisheyeForMindVision_->isChecked();
            }
            if (isMindVision && enableFisheyeForMV && fisheye_calibration_loaded_ && fisheyeStatusIndicator_) {
                fisheye_undistort_enabled_ = true;
                fisheyeStatusIndicator_->setText("Fisheye Correction: APPLIED");
                fisheyeStatusIndicator_->setStyleSheet("background-color: #90EE90; padding: 5px; border: 1px solid #006400;");
            }
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
        
        // Convert to absolute path
        QDir defaultDirObj(defaultDir);
        QString dir = defaultDirObj.absolutePath();
        
        // Create directory if it doesn't exist
        bool dirCreated = QDir().mkpath(dir);
        
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
        
        
        // Generate timestamp for filenames
        QDateTime now = QDateTime::currentDateTime();
        QString timestamp = now.toString("yyyyMMdd_hhmmss");
        
        int savedCount = 0;
        for (size_t i = 0; i < storedPatterns_.size(); i++) {
            const StoredPatternData& sp = storedPatterns_[i];
            
            // Use same dimensions as display: warped (left) + gray (middle) + digitized (right)
            // Match the display layout exactly
            int padding = 20;
            int spacing = 15;
            int header_height = 30;
            
            // Calculate box size to match display (same calculation as in display code)
            int estimated_width = 1200;
            int available_width_per_row = estimated_width - 2 * padding;
            int box_size = (available_width_per_row - 2 * spacing) / 3;
            if (box_size < 50) box_size = 50;
            if (box_size > 200) box_size = 200;
            
            int warped_image_size = box_size;
            int gray_box_size = box_size;
            int grid_size = box_size;
            int cell_size = box_size / 8;  // 8x8 grid
            
            int cell_total_width = warped_image_size + spacing + gray_box_size + spacing + grid_size;
            int total_width = cell_total_width + padding * 2;
            int total_height = grid_size + padding * 2 + header_height;
            
            Mat vis = Mat::ones(total_height, total_width, CV_8UC3) * 240;
            
            // Header
            stringstream header;
            header << "ID:" << sp.tag_id;
            putText(vis, header.str(), Point(padding, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
            
            int x_offset = padding;
            int y_offset = header_height + padding;
            
            // 1. Draw warped image (left side)
            if (!sp.warped_image.empty()) {
                Mat warped_resized;
                cv::resize(sp.warped_image, warped_resized, Size(warped_image_size, warped_image_size), 0, 0, INTER_NEAREST);
                Mat warped_bgr;
                if (warped_resized.channels() == 1) {
                    cvtColor(warped_resized, warped_bgr, COLOR_GRAY2BGR);
                } else {
                    warped_bgr = warped_resized;
                }
                
                // Draw border extraction lines (same as display)
                int debug_tagSize = sp.warped_image.rows;
                int debug_borderSize = (debug_tagSize <= 0) ? 4 : debug_tagSize / 8;
                int debug_borderMargin = max(1, debug_borderSize / 4);
                int debug_effectiveBorderSize = debug_borderSize + debug_borderMargin;
                double scale = (double)warped_image_size / debug_tagSize;
                
                // Draw actual border boundary (green)
                int actual_border_x1 = (int)(debug_borderSize * scale);
                int actual_border_y1 = (int)(debug_borderSize * scale);
                int actual_border_x2 = (int)((debug_tagSize - debug_borderSize) * scale);
                int actual_border_y2 = (int)((debug_tagSize - debug_borderSize) * scale);
                line(warped_bgr, Point(0, actual_border_y1), Point(warped_image_size - 1, actual_border_y1), Scalar(0, 255, 0), 1);
                line(warped_bgr, Point(0, actual_border_y2), Point(warped_image_size - 1, actual_border_y2), Scalar(0, 255, 0), 1);
                line(warped_bgr, Point(actual_border_x1, 0), Point(actual_border_x1, warped_image_size - 1), Scalar(0, 255, 0), 1);
                line(warped_bgr, Point(actual_border_x2, 0), Point(actual_border_x2, warped_image_size - 1), Scalar(0, 255, 0), 1);
                
                // Draw effective extraction boundary (red)
                int effective_border_x1 = (int)(debug_effectiveBorderSize * scale);
                int effective_border_y1 = (int)(debug_effectiveBorderSize * scale);
                int effective_border_x2 = (int)((debug_tagSize - debug_effectiveBorderSize) * scale);
                int effective_border_y2 = (int)((debug_tagSize - debug_effectiveBorderSize) * scale);
                line(warped_bgr, Point(0, effective_border_y1), Point(warped_image_size - 1, effective_border_y1), Scalar(0, 0, 255), 1);
                line(warped_bgr, Point(0, effective_border_y2), Point(warped_image_size - 1, effective_border_y2), Scalar(0, 0, 255), 1);
                line(warped_bgr, Point(effective_border_x1, 0), Point(effective_border_x1, warped_image_size - 1), Scalar(0, 0, 255), 1);
                line(warped_bgr, Point(effective_border_x2, 0), Point(effective_border_x2, warped_image_size - 1), Scalar(0, 0, 255), 1);
                
                warped_bgr.copyTo(vis(Rect(x_offset, y_offset, warped_image_size, warped_image_size)));
                rectangle(vis, Rect(x_offset, y_offset, warped_image_size, warped_image_size), Scalar(0, 0, 255), 2);
                putText(vis, "Warped (green=border, red=extraction)", Point(x_offset, y_offset - 5),
                       FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 0, 255), 1);
            }
            
            // 2. Draw gray color box (middle) - 8x8 grid with actual gray values
            int gray_x_offset = x_offset + warped_image_size + spacing;
            if (sp.pattern.size() == 6 && sp.pattern[0].size() == 6) {
                // Draw border cells (all black)
                // Top row (row 0)
                for (int c = 0; c < 8; c++) {
                    int y_pos = y_offset;
                    int x_pos = gray_x_offset + c * cell_size;
                    Rect cell(x_pos, y_pos, cell_size, cell_size);
                    rectangle(vis, cell, Scalar(0, 0, 0), -1);
                    rectangle(vis, cell, Scalar(128, 128, 128), 1);
                }
                // Bottom row (row 7)
                for (int c = 0; c < 8; c++) {
                    int y_pos = y_offset + 7 * cell_size;
                    int x_pos = gray_x_offset + c * cell_size;
                    Rect cell(x_pos, y_pos, cell_size, cell_size);
                    rectangle(vis, cell, Scalar(0, 0, 0), -1);
                    rectangle(vis, cell, Scalar(128, 128, 128), 1);
                }
                // Left column (col 0, rows 1-6)
                for (int r = 1; r < 7; r++) {
                    int y_pos = y_offset + r * cell_size;
                    int x_pos = gray_x_offset;
                    Rect cell(x_pos, y_pos, cell_size, cell_size);
                    rectangle(vis, cell, Scalar(0, 0, 0), -1);
                    rectangle(vis, cell, Scalar(128, 128, 128), 1);
                }
                // Right column (col 7, rows 1-6)
                for (int r = 1; r < 7; r++) {
                    int y_pos = y_offset + r * cell_size;
                    int x_pos = gray_x_offset + 7 * cell_size;
                    Rect cell(x_pos, y_pos, cell_size, cell_size);
                    rectangle(vis, cell, Scalar(0, 0, 0), -1);
                    rectangle(vis, cell, Scalar(128, 128, 128), 1);
                }
                
                // Draw 6x6 data pattern with actual gray values
                for (int r = 0; r < 6; r++) {
                    for (int c = 0; c < 6; c++) {
                        int val = sp.pattern[r][c];
                        Scalar gray_color(val, val, val);
                        int y_pos = y_offset + (r + 1) * cell_size;
                        int x_pos = gray_x_offset + (c + 1) * cell_size;
                        Rect gray_cell(x_pos, y_pos, cell_size, cell_size);
                        rectangle(vis, gray_cell, gray_color, -1);
                        rectangle(vis, gray_cell, Scalar(128, 128, 128), 1);
                    }
                }
                putText(vis, "Gray Values (8x8: border + 6x6 data)", Point(gray_x_offset, y_offset - 5),
                       FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 0, 255), 1);
            }
            
            // 3. Draw digitized pattern (right side) - 8x8 grid with black/white
            int pattern_x_offset = gray_x_offset + gray_box_size + spacing;
            if (sp.pattern.size() == 6 && sp.pattern[0].size() == 6) {
                // Draw border cells (all black)
                // Top row (row 0)
                for (int c = 0; c < 8; c++) {
                    int y_pos = y_offset;
                    int x_pos = pattern_x_offset + c * cell_size;
                    Rect cell(x_pos, y_pos, cell_size, cell_size);
                    rectangle(vis, cell, Scalar(0, 0, 0), -1);
                    rectangle(vis, cell, Scalar(128, 128, 128), 1);
                }
                // Bottom row (row 7)
                for (int c = 0; c < 8; c++) {
                    int y_pos = y_offset + 7 * cell_size;
                    int x_pos = pattern_x_offset + c * cell_size;
                    Rect cell(x_pos, y_pos, cell_size, cell_size);
                    rectangle(vis, cell, Scalar(0, 0, 0), -1);
                    rectangle(vis, cell, Scalar(128, 128, 128), 1);
                }
                // Left column (col 0, rows 1-6)
                for (int r = 1; r < 7; r++) {
                    int y_pos = y_offset + r * cell_size;
                    int x_pos = pattern_x_offset;
                    Rect cell(x_pos, y_pos, cell_size, cell_size);
                    rectangle(vis, cell, Scalar(0, 0, 0), -1);
                    rectangle(vis, cell, Scalar(128, 128, 128), 1);
                }
                // Right column (col 7, rows 1-6)
                for (int r = 1; r < 7; r++) {
                    int y_pos = y_offset + r * cell_size;
                    int x_pos = pattern_x_offset + 7 * cell_size;
                    Rect cell(x_pos, y_pos, cell_size, cell_size);
                    rectangle(vis, cell, Scalar(0, 0, 0), -1);
                    rectangle(vis, cell, Scalar(128, 128, 128), 1);
                }
                
                // Draw 6x6 data pattern (black/white)
                for (int r = 0; r < 6; r++) {
                    for (int c = 0; c < 6; c++) {
                        int val = sp.pattern[r][c];
                        bool is_black = val < 128;
                        Scalar color = is_black ? Scalar(0, 0, 0) : Scalar(255, 255, 255);
                        int y_pos = y_offset + (r + 1) * cell_size;
                        int x_pos = pattern_x_offset + (c + 1) * cell_size;
                        Rect cell(x_pos, y_pos, cell_size, cell_size);
                        rectangle(vis, cell, color, -1);
                        rectangle(vis, cell, Scalar(128, 128, 128), 1);
                    }
                }
                putText(vis, "Digitized (8x8: border + 6x6 data)", Point(pattern_x_offset, y_offset - 5),
                       FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 0, 255), 1);
            }
            
            // Save image with same dimensions as display
            QString filename = QString("%1/tag_%2_%3.png").arg(dir).arg(sp.tag_id).arg(timestamp);
            string filename_std = filename.toStdString();
            std::cerr << "Attempting to save pattern to: " << filename_std << std::endl;
            std::cerr.flush();
            qDebug() << "Attempting to save pattern to:" << filename;
            
            bool saved = imwrite(filename_std, vis);
            if (saved) {
                savedCount++;
            }
        }
        
        qDebug() << "Saved" << savedCount << "pattern visualization(s) to:" << dir;
        // Don't show message box - just log to console to avoid hanging the GUI
    }
    
    void startAlgorithm() {
        // If already running, stop first (clean restart)
        if (algorithmRunning_) {
            stopAlgorithm();
            // Give threads a moment to clean up
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // Check if a camera is selected and open (shared camera from top-level controls)
        if (!cameraOpen_ || selectedCameraIndex_ < 0) {
            QMessageBox::warning(this, "Error", "Please select and open a camera using the camera controls at the top of the application");
            return;
        }
        
        algorithmRunning_ = true;
        frameCount_ = 0;
        captureTime_ = processTime_ = detectionTime_ = displayTime_ = totalTime_ = 0.0;
        captureFPS_ = detectionFPS_ = displayFPS_ = 0.0;
        captureFrameCount_ = detectionFrameCount_ = displayFrameCount_ = 0;
        captureFPSStart_ = chrono::high_resolution_clock::now();
        detectionFPSStart_ = chrono::high_resolution_clock::now();
        displayFPSStart_ = chrono::high_resolution_clock::now();
        
        // Use the shared camera (already open from top-level controls)
        algorithmUseMindVision_ = useMindVision_;
        
        // Set default settings for MindVision camera: fisheye correction enabled and mirror enabled
        if (algorithmUseMindVision_) {
            // Enable fisheye correction for MindVision cameras if setting is enabled (if calibration is loaded)
            bool enableFisheyeForMV = true;  // Default to true
            if (settingsEnableFisheyeForMindVision_) {
                enableFisheyeForMV = settingsEnableFisheyeForMindVision_->isChecked();
            }
            if (enableFisheyeForMV && fisheye_calibration_loaded_ && fisheyeStatusIndicator_) {
                fisheye_undistort_enabled_ = true;
                fisheyeStatusIndicator_->setText("Fisheye Correction: APPLIED");
                fisheyeStatusIndicator_->setStyleSheet("background-color: #90EE90; padding: 5px; border: 1px solid #006400;");
            }
            // Ensure mirror checkbox is checked for MindVision cameras
            if (algorithmMirrorCheckbox_) {
                algorithmMirrorCheckbox_->setChecked(true);
            }
        }
        
        // Camera is already open from top-level controls, use shared camera instance
        // For MindVision: use mvHandle_ (shared)
        // For V4L2: use cameraCap_ (shared)
        
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
            // Immediately apply current algorithm settings to ensure consistency with other algorithms
            applyAlgorithmSettingsToAlgorithm(currentAlgorithm_.get());
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
        try {
            captureThread_ = new std::thread(&AprilTagDebugGUI::captureThreadFunction, this);
            processThread_ = new std::thread(&AprilTagDebugGUI::processThreadFunction, this);
            cerr << "Process thread created successfully" << endl;
            
            cerr << "Creating detection thread..." << endl;
            detectionThread_ = new std::thread(&AprilTagDebugGUI::detectionThreadFunction, this);
            cerr << "Detection thread created successfully" << endl;
            
            cerr << "Creating display thread..." << endl;
            // DEBUG MODE: Skip display thread to avoid Qt/OpenGL interference
            // displayThread_ = new std::thread(&AprilTagDebugGUI::displayThreadFunction, this);
            displayThread_ = nullptr;
            cerr << "Display thread DISABLED (debug mode - no Qt/OpenGL rendering)" << endl;
            
            cerr << "All threads started successfully!" << endl;
        } catch (const std::exception& e) {
            QMessageBox::critical(this, "Thread Error", 
                QString("Failed to start threads:\n%1").arg(e.what()));
            // Cleanup if threads failed
            stopAlgorithm();
            return;
        } catch (...) {
            QMessageBox::critical(this, "Thread Error", "Unknown error starting threads");
            stopAlgorithm();
            return;
        }
        
        algorithmStartBtn_->setEnabled(false);
        algorithmStopBtn_->setEnabled(true);
        
        // NOTE: Fast AprilTag detector initialization is now done lazily in processFrame()
        // when called from the detection thread. This ensures CUDA context is created in
        // the same thread that uses it, preventing thread-local CUDA context issues.
        // We no longer initialize from the GUI thread via QTimer::singleShot.
#ifdef HAVE_CUDA_APRILTAG
        if (algorithmIndex == 1 && currentAlgorithm_) {
            qDebug() << "Fast AprilTag algorithm created - will initialize lazily in detection thread on first processFrame() call";
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
        
        // Don't close camera - it's shared with other tabs and managed by top-level controls
        // Camera will be closed when user closes it from the top-level camera controls
        
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
        while (algorithmRunning_) {
            auto start = chrono::high_resolution_clock::now();
            Mat frame;
            
            if (algorithmUseMindVision_) {
#ifdef HAVE_MINDVISION_SDK
                tSdkFrameHead frameHead;
                BYTE *pbyBuffer;
                if (CameraGetImageBuffer(mvHandle_, &frameHead, &pbyBuffer, 1000) == CAMERA_STATUS_SUCCESS) {
                    Mat temp(frameHead.iHeight, frameHead.iWidth, CV_8UC1, pbyBuffer);
                    frame = temp.clone();
                    CameraReleaseImageBuffer(mvHandle_, pbyBuffer);
                }
#endif
            } else {
                if (cameraCap_.isOpened()) {
                    cameraCap_ >> frame;
                    // Convert color to grayscale immediately for faster processing (if color camera)
                    if (frame.channels() == 3) {
                        Mat gray;
                        cvtColor(frame, gray, COLOR_BGR2GRAY);
                        frame = gray;  // Use grayscale for detection (faster)
                    }
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
            
            // Log timing
            auto end = chrono::high_resolution_clock::now();
            detectionTime_ = chrono::duration<double, milli>(end - start).count();
            
            // Store detections for quality/pose analysis and draw on frame for preview
            Mat displayFrame;
            {
                std::unique_lock<std::mutex> lock(latestDetectionsMutex_);
                latestDetections_.clear();
                
                // Use the original frame (not gray) for display - same as capture tab
                // Convert to BGR/RGB for display
                if (frame.channels() == 1) {
                    cvtColor(frame, displayFrame, COLOR_GRAY2BGR);
                } else {
                    displayFrame = frame.clone();
                }
                
                bool mirror = algorithmMirrorCheckbox_ && algorithmMirrorCheckbox_->isChecked();
                
                // Apply mirror to display frame if mirroring was used (so coordinates match)
                // This matches the behavior in the Capture tab
                if (mirror) {
                    flip(displayFrame, displayFrame, 1);  // Horizontal flip
                }
                
                // Debug: Draw numbered quads before filtering if debug mode is enabled
                bool debug_enabled = algorithmDebugCheckbox_ && algorithmDebugCheckbox_->isChecked();
                if (debug_enabled && currentAlgorithm_) {
#ifdef HAVE_CUDA_APRILTAG
                    FastAprilTagAlgorithm* fast_algo = dynamic_cast<FastAprilTagAlgorithm*>(currentAlgorithm_.get());
                    if (fast_algo) {
                        std::vector<frc971::apriltag::QuadCorners> quads = fast_algo->getLastFrameQuads();
                        std::cerr << "[DEBUG] Drawing " << quads.size() << " quads on video display" << std::endl;
                        
                        // Draw each quad with a number
                        for (size_t i = 0; i < quads.size(); i++) {
                            const auto& quad = quads[i];
                            Scalar quad_color(255, 255, 0);  // Yellow for quads
                            int thickness = 2;
                            
                            // Draw quad outline
                            for (int j = 0; j < 4; j++) {
                                int next = (j + 1) % 4;
                                Point2i p1(static_cast<int>(quad.corners[j][0]), static_cast<int>(quad.corners[j][1]));
                                Point2i p2(static_cast<int>(quad.corners[next][0]), static_cast<int>(quad.corners[next][1]));
                                
                                // Clamp to frame bounds
                                p1.x = std::max(0, std::min(displayFrame.cols - 1, p1.x));
                                p1.y = std::max(0, std::min(displayFrame.rows - 1, p1.y));
                                p2.x = std::max(0, std::min(displayFrame.cols - 1, p2.x));
                                p2.y = std::max(0, std::min(displayFrame.rows - 1, p2.y));
                                
                                line(displayFrame, p1, p2, quad_color, thickness);
                            }
                            
                            // Calculate center for number label
                            Point2f center(0, 0);
                            for (int j = 0; j < 4; j++) {
                                center.x += quad.corners[j][0];
                                center.y += quad.corners[j][1];
                            }
                            center.x /= 4.0f;
                            center.y /= 4.0f;
                            
                            Point2i center_int(static_cast<int>(center.x), static_cast<int>(center.y));
                            center_int.x = std::max(0, std::min(displayFrame.cols - 1, center_int.x));
                            center_int.y = std::max(0, std::min(displayFrame.rows - 1, center_int.y));
                            
                            // Draw quad number
                            putText(displayFrame, std::to_string(i), center_int,
                                   FONT_HERSHEY_SIMPLEX, 0.6, quad_color, 2);
                        }
                    }
#endif
                }
                
                if (detections) {
                    for (int i = 0; i < zarray_size(detections); i++) {
                        apriltag_detection_t *det;
                        zarray_get(detections, i, &det);
                        if (!det) continue;
                        
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
                        
                        // Verify coordinates are within frame bounds
                        int frame_width = displayFrame.cols;
                        int frame_height = displayFrame.rows;
                        
                        // Draw detection on frame (coordinates should match gray frame dimensions)
                        Point2i p0((int)det->p[0][0], (int)det->p[0][1]);
                        Point2i p1((int)det->p[1][0], (int)det->p[1][1]);
                        Point2i p2((int)det->p[2][0], (int)det->p[2][1]);
                        Point2i p3((int)det->p[3][0], (int)det->p[3][1]);
                        Point2i center((int)det->c[0], (int)det->c[1]);
                        
                        // Clamp coordinates to frame bounds (safety check)
                        p0.x = std::max(0, std::min(frame_width - 1, p0.x));
                        p0.y = std::max(0, std::min(frame_height - 1, p0.y));
                        p1.x = std::max(0, std::min(frame_width - 1, p1.x));
                        p1.y = std::max(0, std::min(frame_height - 1, p1.y));
                        p2.x = std::max(0, std::min(frame_width - 1, p2.x));
                        p2.y = std::max(0, std::min(frame_height - 1, p2.y));
                        p3.x = std::max(0, std::min(frame_width - 1, p3.x));
                        p3.y = std::max(0, std::min(frame_height - 1, p3.y));
                        center.x = std::max(0, std::min(frame_width - 1, center.x));
                        center.y = std::max(0, std::min(frame_height - 1, center.y));
                        
                        line(displayFrame, p0, p1, Scalar(0, 255, 0), 2);
                        line(displayFrame, p1, p2, Scalar(0, 255, 0), 2);
                        line(displayFrame, p2, p3, Scalar(0, 255, 0), 2);
                        line(displayFrame, p3, p0, Scalar(0, 255, 0), 2);
                        
                        // Draw tag ID
                        stringstream ss;
                        ss << det->id;
                        putText(displayFrame, ss.str(), Point2i(center.x - 10, center.y - 10),
                                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
                    }
                }
            }
            
            // Update preview display (thread-safe via Qt signal)
            auto display_start = chrono::high_resolution_clock::now();
            if (!displayFrame.empty() && algorithmDisplayLabel_) {
                // Convert Mat to QPixmap
                Mat rgbFrame;
                if (displayFrame.channels() == 3) {
                    cvtColor(displayFrame, rgbFrame, COLOR_BGR2RGB);
                } else {
                    rgbFrame = displayFrame;
                }
                
                QImage qimg(rgbFrame.data, rgbFrame.cols, rgbFrame.rows, rgbFrame.step, QImage::Format_RGB888);
                QPixmap pixmap = QPixmap::fromImage(qimg);
                
                // Scale to fit label while maintaining aspect ratio
                if (!pixmap.isNull() && algorithmDisplayLabel_) {
                    QSize labelSize = algorithmDisplayLabel_->size();
                    if (labelSize.width() > 0 && labelSize.height() > 0) {
                        pixmap = pixmap.scaled(labelSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);
                    }
                    // Update GUI in main thread (thread-safe)
                    QMetaObject::invokeMethod(this, "setPixmapFromThread", Qt::QueuedConnection,
                                            Q_ARG(QPixmap, pixmap));
                }
            }
            auto display_end = chrono::high_resolution_clock::now();
            displayTime_ = chrono::duration<double, milli>(display_end - display_start).count();
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
            
            // Update FPS display in GUI (all three threads: capture/read, detection/decode, display)
            QMetaObject::invokeMethod(this, "updateAlgorithmTimingSlot", Qt::QueuedConnection,
                                    Q_ARG(double, 0.0));  // FPS value not used, we read from member variables
            
            // Destroy detections array
            if (detections) {
                for (int i = 0; i < zarray_size(detections); i++) {
                    apriltag_detection_t *det;
                    zarray_get(detections, i, &det);
                    if (det && currentAlgorithm_) {
                        apriltag_detection_destroy(det);
                    }
                }
                zarray_destroy(detections);
            }
        }
    }
    
    void displayThreadFunction() {
        // DEBUG MODE: Display thread disabled - no Qt/OpenGL rendering
        // Just wait for algorithm to stop
        while (algorithmRunning_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    
    void updateAlgorithmTiming(double fps) {
        if (!algorithmFPSLabel_) return;
        
        // Display all three FPS values in the label: Read (capture), Decode (detection), Display
        algorithmFPSLabel_->setText(QString("FPS: Read: %1, Decode: %2, Display: %3")
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
        
        // ========== TOP-LEVEL CAMERA CONTROLS (OUTSIDE TABS) ==========
        QGroupBox *topCameraGroup = new QGroupBox("Camera Controls", this);
        QVBoxLayout *topCameraLayout = new QVBoxLayout();
        
        // First row: Camera selection and Resolution/FPS
        QHBoxLayout *cameraRowLayout = new QHBoxLayout();
        
        // Camera selection
        QLabel *cameraLabel = new QLabel("Camera:", this);
        cameraCombo_ = new QComboBox(this);
        connect(cameraCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), 
                this, &AprilTagDebugGUI::openCamera);
        cameraRowLayout->addWidget(cameraLabel);
        cameraRowLayout->addWidget(cameraCombo_);
        cameraRowLayout->addSpacing(20);
        
        // Resolution/FPS selection
        QLabel *modeLabel = new QLabel("Resolution & FPS:", this);
        modeCombo_ = new QComboBox(this);
        modeCombo_->setEnabled(false);  // Enabled when camera opens
        connect(modeCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), 
                this, &AprilTagDebugGUI::onModeChanged);
        cameraRowLayout->addWidget(modeLabel);
        cameraRowLayout->addWidget(modeCombo_);
        cameraRowLayout->addStretch();
        
        topCameraLayout->addLayout(cameraRowLayout);
        
        // Second row: Camera settings (Exposure, Gain, Brightness, Contrast, Saturation, Sharpness)
        QHBoxLayout *settingsRowLayout = new QHBoxLayout();
        
        // Exposure
        QLabel *exposureLabel = new QLabel("Exposure:", this);
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
        settingsRowLayout->addWidget(exposureLabel);
        settingsRowLayout->addWidget(exposureSlider_);
        settingsRowLayout->addWidget(exposureSpin_);
        settingsRowLayout->addSpacing(10);
        
        // Gain
        QLabel *gainLabel = new QLabel("Gain:", this);
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
        settingsRowLayout->addWidget(gainLabel);
        settingsRowLayout->addWidget(gainSlider_);
        settingsRowLayout->addWidget(gainSpin_);
        settingsRowLayout->addSpacing(10);
        
        // Brightness
        QLabel *brightnessLabel = new QLabel("Brightness:", this);
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
        settingsRowLayout->addWidget(brightnessLabel);
        settingsRowLayout->addWidget(brightnessSlider_);
        settingsRowLayout->addWidget(brightnessSpin_);
        settingsRowLayout->addSpacing(10);
        
        // Contrast
        QLabel *contrastLabel = new QLabel("Contrast:", this);
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
        settingsRowLayout->addWidget(contrastLabel);
        settingsRowLayout->addWidget(contrastSlider_);
        settingsRowLayout->addWidget(contrastSpin_);
        settingsRowLayout->addSpacing(10);
        
        // Saturation
        QLabel *saturationLabel = new QLabel("Saturation:", this);
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
        settingsRowLayout->addWidget(saturationLabel);
        settingsRowLayout->addWidget(saturationSlider_);
        settingsRowLayout->addWidget(saturationSpin_);
        settingsRowLayout->addSpacing(10);
        
        // Sharpness
        QLabel *sharpnessLabel = new QLabel("Sharpness:", this);
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
        settingsRowLayout->addWidget(sharpnessLabel);
        settingsRowLayout->addWidget(sharpnessSlider_);
        settingsRowLayout->addWidget(sharpnessSpin_);
        settingsRowLayout->addStretch();
        
        topCameraLayout->addLayout(settingsRowLayout);
        
        // Dynamic settings row (for V4L2/Arducam - will be populated when camera is selected)
        // Use a scrollable area so all controls are visible
        dynamicSettingsScrollArea_ = new QScrollArea(this);
        dynamicSettingsScrollArea_->setWidgetResizable(true);
        dynamicSettingsScrollArea_->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        dynamicSettingsScrollArea_->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        dynamicSettingsScrollArea_->setMinimumHeight(100);  // Minimum height for controls
        dynamicSettingsScrollArea_->setMaximumHeight(200);  // Allow expansion but limit height
        dynamicSettingsScrollArea_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        dynamicSettingsScrollArea_->setFrameShape(QFrame::Box);  // Add visible border
        dynamicSettingsScrollArea_->setFrameShadow(QFrame::Sunken);
        dynamicSettingsWidget_ = new QWidget(this);
        dynamicSettingsLayout_ = new QHBoxLayout(dynamicSettingsWidget_);
        dynamicSettingsLayout_->setContentsMargins(10, 10, 10, 10);
        dynamicSettingsLayout_->setSpacing(10);
        dynamicSettingsWidget_->setLayout(dynamicSettingsLayout_);
        dynamicSettingsWidget_->setMinimumHeight(80);  // Ensure minimum height
        dynamicSettingsScrollArea_->setWidget(dynamicSettingsWidget_);
        topCameraLayout->addWidget(dynamicSettingsScrollArea_);
        
        // Third row: Save settings button
        QHBoxLayout *buttonRowLayout = new QHBoxLayout();
        saveSettingsBtn_ = new QPushButton("Save Camera Settings", this);
        saveSettingsBtn_->setEnabled(false);
        connect(saveSettingsBtn_, &QPushButton::clicked, this, &AprilTagDebugGUI::saveCameraSettings);
        buttonRowLayout->addWidget(saveSettingsBtn_);
        buttonRowLayout->addStretch();
        topCameraLayout->addLayout(buttonRowLayout);
        
        topCameraGroup->setLayout(topCameraLayout);
        mainLayout->addWidget(topCameraGroup);
        
        // Initialize camera enumeration (before tabs are created)
        cameraCombo_->blockSignals(true);  // Block signals during enumeration
        enumerateCameras();
        // Set default to "None" (index 0)
        cameraCombo_->setCurrentIndex(0);
        cameraCombo_->blockSignals(false);  // Re-enable signals
        
        // Setup preview timer (needed for camera operations)
        previewTimer_ = new QTimer(this);
        connect(previewTimer_, &QTimer::timeout, this, &AprilTagDebugGUI::updatePreview);
        
        // Create tab widget
        tabWidget_ = new QTabWidget(this);
        
        // ========== PROCESSING TAB ==========
        QWidget *processingTab = new QWidget(this);
        QVBoxLayout *processingLayout = new QVBoxLayout(processingTab);
        
        // Stage selection (at the top)
        QGroupBox *stageGroup = new QGroupBox("Stage Selection", this);
        stageGroup->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
        QHBoxLayout *stageLayout = new QHBoxLayout();
        stageLayout->setContentsMargins(5, 5, 5, 5);
        stageLayout->setSpacing(5);
        
        // Preprocessing stage
        QLabel *preprocessLabel = new QLabel("Preprocessing:", this);
        preprocessCombo_ = new QComboBox(this);
        preprocessCombo_->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Fixed);
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
        edgeCombo_->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Fixed);
        edgeCombo_->addItems({
            "None", "Canny (50,150)", "Canny (75,200)", "Canny (100,200)",
            "Sobel", "Laplacian", "Adaptive Threshold"
        });
        connect(edgeCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), 
                this, &AprilTagDebugGUI::stageChanged);
        
        // Detection stage
        QLabel *detectionLabel = new QLabel("Detection:", this);
        detectionCombo_ = new QComboBox(this);
        detectionCombo_->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Fixed);
        detectionCombo_->addItems({
            "Original", "With Detection", "Contours Only", "Quadrilaterals Only",
            "Convex Quads Only", "Tag-Sized Quads"
        });
        connect(detectionCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), 
                this, &AprilTagDebugGUI::stageChanged);
        
        // Advanced visualization stage
        QLabel *advancedLabel = new QLabel("Advanced:", this);
        advancedCombo_ = new QComboBox(this);
        advancedCombo_->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Fixed);
        advancedCombo_->addItems({
            "None", "Corner Refinement", "Warped Tags", "Pattern Extraction", "Hamming Decode"
        });
        connect(advancedCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged), 
                this, &AprilTagDebugGUI::stageChanged);
        
        // Quad selection (for Warped Tags and later stages) - independent for each image
        QLabel *quadLabel1 = new QLabel("Quad (Img1):", this);
        quadCombo1_ = new QComboBox(this);
        quadCombo1_->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Fixed);
        quadCombo1_->setEnabled(false);  // Disabled until a quad stage is selected
        connect(quadCombo1_, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &AprilTagDebugGUI::stageChanged);
        
        QLabel *quadLabel2 = new QLabel("Quad (Img2):", this);
        quadCombo2_ = new QComboBox(this);
        quadCombo2_->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Fixed);
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
        stageGroup->setMaximumHeight(stageGroup->sizeHint().height() + 10);  // Limit height to fit content
        processingLayout->addWidget(stageGroup);
        
        // Control panel (after Stage Selection)
        QHBoxLayout *controlLayout = new QHBoxLayout();
        
        QPushButton *loadBtn1 = new QPushButton("Load Image 1", this);
        QPushButton *loadBtn2 = new QPushButton("Load Image 2", this);
        connect(loadBtn1, &QPushButton::clicked, this, &AprilTagDebugGUI::loadImage1);
        connect(loadBtn2, &QPushButton::clicked, this, &AprilTagDebugGUI::loadImage2);
        
        controlLayout->addWidget(loadBtn1);
        controlLayout->addWidget(loadBtn2);
        controlLayout->addStretch();
        
        processingLayout->addLayout(controlLayout);

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
        
        // ========== CAPTURE TAB (FIRST) ==========
        QWidget *captureTab = new QWidget(this);
        QVBoxLayout *captureLayout = new QVBoxLayout(captureTab);
        setupCaptureTab(captureLayout);
        tabWidget_->addTab(captureTab, "Capture");
        
        // ========== ALGORITHMS TAB (SECOND) ==========
        QWidget *algorithmsTab = new QWidget(this);
        QVBoxLayout *algorithmsLayout = new QVBoxLayout(algorithmsTab);
        setupAlgorithmsTab(algorithmsLayout);
        tabWidget_->addTab(algorithmsTab, "Algorithms");
        
        // ========== FISHEYE CORRECTION TAB (THIRD) ==========
        setupFisheyeTab();
        
        // ========== SETTINGS TAB (FOURTH) ==========
        setupSettingsTab();
        
        // ========== PROCESSING TAB (FIFTH) ==========
        tabWidget_->addTab(processingTab, "Processing");
        
        // Fisheye correction status indicator (below camera controls, outside tabs)
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
        fisheyeStatusIndicator_->setAlignment(Qt::AlignCenter);
        
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
                if (preprocessHistEqCheck_ && preprocessHistEqCheck_->isChecked()) {
                    equalizeHist(img, result);
                } else {
                    equalizeHist(img, result);
                }
                break;
            case 2: // CLAHE clip=2.0 (or use tunable value)
            case 3: // CLAHE clip=3.0 (or use tunable value)
            case 4: // CLAHE clip=4.0 (or use tunable value)
                {
                    double claheClip = 3.0;  // Default
                    if (preprocessClaheClipSpin_) {
                        claheClip = preprocessClaheClipSpin_->value();
                    } else {
                        // Fallback to method-based values
                        if (method == 2) claheClip = 2.0;
                        else if (method == 3) claheClip = 3.0;
                        else if (method == 4) claheClip = 4.0;
                    }
                    Ptr<CLAHE> clahe = createCLAHE(claheClip, Size(8, 8));
                    clahe->apply(img, result);
                }
                break;
            case 5: // Gamma 1.2 (or use tunable value)
            case 6: // Gamma 1.5 (or use tunable value)
            case 7: // Gamma 2.0 (or use tunable value)
                {
                    double gamma = 1.5;  // Default
                    if (preprocessGammaSpin_) {
                        gamma = preprocessGammaSpin_->value();
                    } else {
                        // Fallback to method-based values
                        if (method == 5) gamma = 1.2;
                        else if (method == 6) gamma = 1.5;
                        else if (method == 7) gamma = 2.0;
                    }
                    double inv_gamma = 1.0 / gamma;
                    Mat table(1, 256, CV_8U);
                    uchar *p = table.ptr();
                    for (int i = 0; i < 256; i++) {
                        p[i] = saturate_cast<uchar>(pow(i / 255.0, inv_gamma) * 255.0);
                    }
                    LUT(img, table, result);
                }
                break;
            case 8: // Contrast Enhancement (or use tunable value)
                {
                    double contrast = 2.0;  // Default
                    if (preprocessContrastSpin_) {
                        contrast = preprocessContrastSpin_->value();
                    }
                    img.convertTo(result, -1, contrast, 50);
                }
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
            case 1: // Canny (50, 150) - or use tunable values
            case 2: // Canny (75, 200) - or use tunable values
            case 3: // Canny (100, 200) - or use tunable values
                {
                    int cannyLow = 50;  // Default
                    int cannyHigh = 150;  // Default
                    if (cannyLowSpin_ && cannyHighSpin_) {
                        cannyLow = cannyLowSpin_->value();
                        cannyHigh = cannyHighSpin_->value();
                    } else {
                        // Fallback to method-based values
                        if (method == 1) { cannyLow = 50; cannyHigh = 150; }
                        else if (method == 2) { cannyLow = 75; cannyHigh = 200; }
                        else if (method == 3) { cannyLow = 100; cannyHigh = 200; }
                    }
                    Canny(img, result, cannyLow, cannyHigh);
                }
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
            case 6: // Adaptive Threshold - use tunable values
                {
                    int blockSize = 11;  // Default
                    int constant = 2;  // Default
                    if (adaptiveThreshBlockSpin_ && adaptiveThreshConstantSpin_) {
                        blockSize = adaptiveThreshBlockSpin_->value();
                        constant = adaptiveThreshConstantSpin_->value();
                    }
                    adaptiveThreshold(img, result, 255, ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    THRESH_BINARY, blockSize, constant);
                }
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
    
    // Helper function to refine corners - uses tunable parameters
    void refineCorners(const Mat& gray, vector<Point2f>& corners, int winSize = -1, int maxIter = -1) {
        if (corners.size() != 4) return;
        
        // Use tunable parameters if available, otherwise use defaults
        if (winSize < 0) {
            winSize = cornerRefineWinSizeSpin_ ? cornerRefineWinSizeSpin_->value() : 5;
        }
        if (maxIter < 0) {
            maxIter = cornerRefineMaxIterSpin_ ? cornerRefineMaxIterSpin_->value() : 30;
        }
        
        TermCriteria criteria(TermCriteria::EPS + TermCriteria::COUNT, maxIter, 0.001);
        cornerSubPix(gray, corners, Size(winSize, winSize), Size(-1, -1), criteria);
    }
    
    // Extract quadrilaterals from edge-detected image - uses tunable parameters
    vector<vector<Point2f>> extractQuadrilaterals(const Mat& edges, const Mat& original) {
        vector<vector<Point2f>> quads;
        
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(edges, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        
        // Use tunable parameters if available
        double tag_min_area = tagMinAreaSpin_ ? tagMinAreaSpin_->value() : 500;
        double tag_max_area = tagMaxAreaSpin_ ? tagMaxAreaSpin_->value() : 50000;
        
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
    vector<vector<int>> extractPattern(const Mat& warped, int tagSize = 36, int borderSize = -1) {
        // Use tunable parameter if available
        if (borderSize < 0) {
            borderSize = patternBorderSizeSpin_ ? patternBorderSizeSpin_->value() : 4;
        }
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
        
        // Safety check: ensure pattern is valid (6x6)
        if (pattern.size() != 6) {
            qDebug() << "extractCodeFromPattern: Invalid pattern size (rows):" << pattern.size();
            return 0;
        }
        
        for (int i = 0; i < 36; i++) {
            // Convert 1-indexed to 0-indexed
            int x = TAG36H11_BIT_X[i] - 1;
            int y = TAG36H11_BIT_Y[i] - 1;
            
            // Bounds checking
            if (y < 0 || y >= (int)pattern.size() || x < 0 || x >= (int)pattern[y].size()) {
                qDebug() << "extractCodeFromPattern: Out of bounds access at bit" << i << "x=" << x << "y=" << y 
                         << "pattern size:" << pattern.size() << "row" << y << "size:" << (y >= 0 && y < (int)pattern.size() ? pattern[y].size() : 0);
                continue; // Skip this bit
            }
            
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
        
        // Note: Camera selection, resolution/FPS, and settings are now at the top of the application
        // (outside tabs) so they are shared across all tabs
        
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
        
        // Note: Save Camera Settings button is now at the top of the application (shared across all tabs)
        
        // Video preview on left side
        previewLabel_ = new QLabel(this);
        previewLabel_->setMinimumSize(640, 480);
        previewLabel_->setAlignment(Qt::AlignCenter);
        previewLabel_->setStyleSheet("border: 1px solid black; background-color: #f0f0f0;");
        previewLabel_->setText("No camera selected");
        previewLabel_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        leftControlsLayout->addWidget(previewLabel_);
        
        // ========== ALGORITHM TUNING SETTINGS (Below Preview) ==========
        QGroupBox *algorithmTuningGroup = new QGroupBox("Algorithm Tuning (AprilTag Detection Performance)", this);
        QVBoxLayout *algorithmTuningLayout = new QVBoxLayout();
        
        QLabel *algorithmTuningInfo = new QLabel(
            "Tune preprocessing, edge detection, detection, and advanced parameters for optimal AprilTag detection performance.",
            this);
        algorithmTuningInfo->setWordWrap(true);
        algorithmTuningInfo->setStyleSheet("color: #666; padding: 5px; font-size: 9pt;");
        algorithmTuningLayout->addWidget(algorithmTuningInfo);
        
        // Buttons at the top (side by side)
        QHBoxLayout *buttonLayout = new QHBoxLayout();
        QPushButton *loadDefaultsBtn = new QPushButton("Load Default Settings", this);
        loadDefaultsBtn->setStyleSheet("font-weight: bold; padding: 5px;");
        connect(loadDefaultsBtn, &QPushButton::clicked, this, &AprilTagDebugGUI::loadDefaultAlgorithmSettings);
        
        QPushButton *applyAlgorithmSettingsBtn = new QPushButton("Apply Algorithm Settings", this);
        applyAlgorithmSettingsBtn->setStyleSheet("font-weight: bold; padding: 5px;");
        connect(applyAlgorithmSettingsBtn, &QPushButton::clicked, this, &AprilTagDebugGUI::applyAlgorithmSettings);
        
        buttonLayout->addWidget(loadDefaultsBtn);
        buttonLayout->addWidget(applyAlgorithmSettingsBtn);
        algorithmTuningLayout->addLayout(buttonLayout);
        
        // Use tabs for different parameter categories
        QTabWidget *tuningTabs = new QTabWidget(this);
        tuningTabs->setMaximumHeight(400);  // Limit height to keep it compact
        
        // === PREPROCESSING TAB ===
        QWidget *preprocessTab = new QWidget(this);
        QFormLayout *preprocessLayout = new QFormLayout(preprocessTab);
        
        preprocessHistEqCheck_ = new QCheckBox("Enable Histogram Equalization", this);
        preprocessHistEqCheck_->setChecked(false);
        preprocessLayout->addRow("Histogram Equalization:", preprocessHistEqCheck_);
        
        preprocessClaheClipSpin_ = new QDoubleSpinBox(this);
        preprocessClaheClipSpin_->setRange(1.0, 8.0);
        preprocessClaheClipSpin_->setValue(3.0);
        preprocessClaheClipSpin_->setSingleStep(0.5);
        preprocessClaheClipSpin_->setDecimals(1);
        preprocessLayout->addRow("CLAHE Clip Limit:", preprocessClaheClipSpin_);
        
        preprocessGammaSpin_ = new QDoubleSpinBox(this);
        preprocessGammaSpin_->setRange(0.5, 3.0);
        preprocessGammaSpin_->setValue(1.5);
        preprocessGammaSpin_->setSingleStep(0.1);
        preprocessGammaSpin_->setDecimals(1);
        preprocessLayout->addRow("Gamma Correction:", preprocessGammaSpin_);
        
        preprocessContrastSpin_ = new QDoubleSpinBox(this);
        preprocessContrastSpin_->setRange(0.5, 3.0);
        preprocessContrastSpin_->setValue(1.5);
        preprocessContrastSpin_->setSingleStep(0.1);
        preprocessContrastSpin_->setDecimals(1);
        preprocessLayout->addRow("Contrast Multiplier:", preprocessContrastSpin_);
        
        preprocessTab->setLayout(preprocessLayout);
        tuningTabs->addTab(preprocessTab, "Preprocessing");
        
        // === EDGE DETECTION TAB ===
        QWidget *edgeTab = new QWidget(this);
        QFormLayout *edgeLayout = new QFormLayout(edgeTab);
        
        cannyLowSpin_ = new QSpinBox(this);
        cannyLowSpin_->setRange(10, 200);
        cannyLowSpin_->setValue(50);
        edgeLayout->addRow("Canny Low Threshold:", cannyLowSpin_);
        
        cannyHighSpin_ = new QSpinBox(this);
        cannyHighSpin_->setRange(50, 300);
        cannyHighSpin_->setValue(150);
        edgeLayout->addRow("Canny High Threshold:", cannyHighSpin_);
        
        adaptiveThreshBlockSpin_ = new QSpinBox(this);
        adaptiveThreshBlockSpin_->setRange(3, 31);
        adaptiveThreshBlockSpin_->setValue(11);
        adaptiveThreshBlockSpin_->setSingleStep(2);
        edgeLayout->addRow("Adaptive Threshold Block Size:", adaptiveThreshBlockSpin_);
        
        adaptiveThreshConstantSpin_ = new QSpinBox(this);
        adaptiveThreshConstantSpin_->setRange(-10, 10);
        adaptiveThreshConstantSpin_->setValue(2);
        edgeLayout->addRow("Adaptive Threshold Constant:", adaptiveThreshConstantSpin_);
        
        edgeTab->setLayout(edgeLayout);
        tuningTabs->addTab(edgeTab, "Edge Detection");
        
        // === DETECTION PARAMETERS TAB ===
        QWidget *detectionTab = new QWidget(this);
        QFormLayout *detectionLayout = new QFormLayout(detectionTab);
        
        quadDecimateSpin_ = new QDoubleSpinBox(this);
        quadDecimateSpin_->setRange(1.0, 4.0);
        quadDecimateSpin_->setValue(2.0);
        quadDecimateSpin_->setSingleStep(0.5);
        quadDecimateSpin_->setDecimals(1);
        detectionLayout->addRow("Quad Decimate:", quadDecimateSpin_);
        
        quadSigmaSpin_ = new QDoubleSpinBox(this);
        quadSigmaSpin_->setRange(0.0, 2.0);
        quadSigmaSpin_->setValue(0.0);
        quadSigmaSpin_->setSingleStep(0.1);
        quadSigmaSpin_->setDecimals(1);
        detectionLayout->addRow("Quad Sigma (Gaussian Blur):", quadSigmaSpin_);
        
        refineEdgesCheck_ = new QCheckBox("Refine Edges", this);
        refineEdgesCheck_->setChecked(true);
        detectionLayout->addRow("Refine Edges:", refineEdgesCheck_);
        
        decodeSharpeningSpin_ = new QDoubleSpinBox(this);
        decodeSharpeningSpin_->setRange(0.0, 1.0);
        decodeSharpeningSpin_->setValue(0.25);
        decodeSharpeningSpin_->setSingleStep(0.05);
        decodeSharpeningSpin_->setDecimals(2);
        detectionLayout->addRow("Decode Sharpening:", decodeSharpeningSpin_);
        
        nthreadsSpin_ = new QSpinBox(this);
        nthreadsSpin_->setRange(1, 16);
        nthreadsSpin_->setValue(4);
        detectionLayout->addRow("Number of Threads:", nthreadsSpin_);
        
        detectionTab->setLayout(detectionLayout);
        tuningTabs->addTab(detectionTab, "Detection");
        
        // === QUAD THRESHOLD PARAMETERS TAB ===
        QWidget *quadThreshTab = new QWidget(this);
        QFormLayout *quadThreshLayout = new QFormLayout(quadThreshTab);
        
        minClusterPixelsSpin_ = new QSpinBox(this);
        minClusterPixelsSpin_->setRange(1, 20);
        minClusterPixelsSpin_->setValue(6);
        quadThreshLayout->addRow("Min Cluster Pixels:", minClusterPixelsSpin_);
        
        maxLineFitMseSpin_ = new QDoubleSpinBox(this);
        maxLineFitMseSpin_->setRange(1.0, 20.0);
        maxLineFitMseSpin_->setValue(8.0);
        maxLineFitMseSpin_->setSingleStep(0.5);
        maxLineFitMseSpin_->setDecimals(1);
        quadThreshLayout->addRow("Max Line Fit MSE:", maxLineFitMseSpin_);
        
        criticalAngleSpin_ = new QDoubleSpinBox(this);
        criticalAngleSpin_->setRange(1.0, 20.0);
        criticalAngleSpin_->setValue(7.0);
        criticalAngleSpin_->setSingleStep(0.5);
        criticalAngleSpin_->setDecimals(1);
        quadThreshLayout->addRow("Critical Angle (degrees):", criticalAngleSpin_);
        
        minWhiteBlackDiffSpin_ = new QSpinBox(this);
        minWhiteBlackDiffSpin_->setRange(1, 20);
        minWhiteBlackDiffSpin_->setValue(6);
        quadThreshLayout->addRow("Min White-Black Diff:", minWhiteBlackDiffSpin_);
        
        quadThreshTab->setLayout(quadThreshLayout);
        tuningTabs->addTab(quadThreshTab, "Quad Threshold");
        
        // === ADVANCED PARAMETERS TAB ===
        QWidget *advancedTab = new QWidget(this);
        QFormLayout *advancedLayout = new QFormLayout(advancedTab);
        
        cornerRefineWinSizeSpin_ = new QSpinBox(this);
        cornerRefineWinSizeSpin_->setRange(3, 21);
        cornerRefineWinSizeSpin_->setValue(5);
        cornerRefineWinSizeSpin_->setSingleStep(2);
        advancedLayout->addRow("Corner Refinement Window Size:", cornerRefineWinSizeSpin_);
        
        cornerRefineMaxIterSpin_ = new QSpinBox(this);
        cornerRefineMaxIterSpin_->setRange(10, 100);
        cornerRefineMaxIterSpin_->setValue(30);
        cornerRefineMaxIterSpin_->setSingleStep(5);
        advancedLayout->addRow("Corner Refinement Max Iterations:", cornerRefineMaxIterSpin_);
        
        patternBorderSizeSpin_ = new QSpinBox(this);
        patternBorderSizeSpin_->setRange(1, 10);
        patternBorderSizeSpin_->setValue(4);
        advancedLayout->addRow("Pattern Border Size:", patternBorderSizeSpin_);
        
        tagMinAreaSpin_ = new QSpinBox(this);
        tagMinAreaSpin_->setRange(100, 10000);
        tagMinAreaSpin_->setValue(500);
        tagMinAreaSpin_->setSingleStep(100);
        advancedLayout->addRow("Tag Min Area (pixels):", tagMinAreaSpin_);
        
        tagMaxAreaSpin_ = new QSpinBox(this);
        tagMaxAreaSpin_->setRange(1000, 100000);
        tagMaxAreaSpin_->setValue(50000);
        tagMaxAreaSpin_->setSingleStep(1000);
        advancedLayout->addRow("Tag Max Area (pixels):", tagMaxAreaSpin_);
        
        advancedTab->setLayout(advancedLayout);
        tuningTabs->addTab(advancedTab, "Advanced");
        
        algorithmTuningLayout->addWidget(tuningTabs);
        
        algorithmTuningGroup->setLayout(algorithmTuningLayout);
        leftControlsLayout->addWidget(algorithmTuningGroup);
        
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
        // Set default to "None" (index 0)
        cameraCombo_->setCurrentIndex(0);
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
    
    void setupSettingsTab() {
        QWidget *settingsTab = new QWidget(this);
        QVBoxLayout *settingsLayout = new QVBoxLayout(settingsTab);
        
        // Scroll area for settings
        QScrollArea *scrollArea = new QScrollArea(this);
        scrollArea->setWidgetResizable(true);
        scrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        scrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        
        QWidget *settingsContent = new QWidget(this);
        QVBoxLayout *contentLayout = new QVBoxLayout(settingsContent);
        
        // ========== FISHEYE SETTINGS ==========
        QGroupBox *fisheyeSettingsGroup = new QGroupBox("Fisheye Correction Settings", this);
        QVBoxLayout *fisheyeSettingsLayout = new QVBoxLayout();
        
        // Load calibration on startup
        settingsLoadCalibrationOnStart_ = new QCheckBox("Load calibration file on startup", this);
        settingsLoadCalibrationOnStart_->setChecked(true);  // Default enabled
        fisheyeSettingsLayout->addWidget(settingsLoadCalibrationOnStart_);
        
        // Calibration file path
        QHBoxLayout *calibPathLayout = new QHBoxLayout();
        QLabel *calibPathLabel = new QLabel("Calibration file path:", this);
        settingsCalibrationPath_ = new QLineEdit(this);
        settingsCalibrationPath_->setText("/home/nav/9202/Hiru/Apriltag/calibration_data/camera_params.yaml");
        QPushButton *browseCalibBtn = new QPushButton("Browse...", this);
        connect(browseCalibBtn, &QPushButton::clicked, this, [this]() {
            QString filename = QFileDialog::getOpenFileName(this, "Select Calibration File", 
                settingsCalibrationPath_->text(), "YAML Files (*.yaml *.yml);;All Files (*)");
            if (!filename.isEmpty()) {
                settingsCalibrationPath_->setText(filename);
            }
        });
        calibPathLayout->addWidget(calibPathLabel);
        calibPathLayout->addWidget(settingsCalibrationPath_);
        calibPathLayout->addWidget(browseCalibBtn);
        fisheyeSettingsLayout->addLayout(calibPathLayout);
        
        // Enable fisheye for MindVision cameras by default
        settingsEnableFisheyeForMindVision_ = new QCheckBox("Enable fisheye correction for MindVision cameras by default", this);
        settingsEnableFisheyeForMindVision_->setChecked(true);  // Default enabled
        fisheyeSettingsLayout->addWidget(settingsEnableFisheyeForMindVision_);
        
        fisheyeSettingsGroup->setLayout(fisheyeSettingsLayout);
        contentLayout->addWidget(fisheyeSettingsGroup);
        
        // ========== CAMERA SETTINGS ==========
        QGroupBox *cameraSettingsGroup = new QGroupBox("Camera Settings (Per Camera)", this);
        QVBoxLayout *cameraSettingsLayout = new QVBoxLayout();
        
        QLabel *cameraSettingsInfo = new QLabel(
            "Select a camera and edit its settings. Settings are saved per camera to camera_settings.txt",
            this);
        cameraSettingsInfo->setWordWrap(true);
        cameraSettingsInfo->setStyleSheet("color: #666; padding: 5px;");
        cameraSettingsLayout->addWidget(cameraSettingsInfo);
        
        // Camera selector with Save button
        QHBoxLayout *cameraSelectLayout = new QHBoxLayout();
        QLabel *cameraSelectLabel = new QLabel("Camera:", this);
        settingsCameraCombo_ = new QComboBox(this);
        // Populate with cameras (skip "None")
        for (size_t i = 1; i < cameraList_.size(); i++) {
            settingsCameraCombo_->addItem(QString::fromStdString(cameraList_[i]));
        }
        connect(settingsCameraCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
                this, &AprilTagDebugGUI::onSettingsCameraChanged);
        QPushButton *saveCameraSettingsBtn = new QPushButton("Save Camera Settings", this);
        connect(saveCameraSettingsBtn, &QPushButton::clicked, this, &AprilTagDebugGUI::saveCameraSettingsFromSettingsTab);
        cameraSelectLayout->addWidget(cameraSelectLabel);
        cameraSelectLayout->addWidget(settingsCameraCombo_);
        cameraSelectLayout->addWidget(saveCameraSettingsBtn);
        cameraSelectLayout->addStretch();
        cameraSettingsLayout->addLayout(cameraSelectLayout);
        
        // Camera settings form
        QFormLayout *settingsFormLayout = new QFormLayout();
        
        // Exposure
        settingsExposureSlider_ = new QSlider(Qt::Horizontal, this);
        settingsExposureSlider_->setRange(0, 100);
        settingsExposureSlider_->setValue(50);
        settingsExposureSpin_ = new QSpinBox(this);
        settingsExposureSpin_->setRange(0, 100);
        settingsExposureSpin_->setValue(50);
        connect(settingsExposureSlider_, &QSlider::valueChanged, settingsExposureSpin_, &QSpinBox::setValue);
        connect(settingsExposureSpin_, QOverload<int>::of(&QSpinBox::valueChanged),
                settingsExposureSlider_, &QSlider::setValue);
        QHBoxLayout *exposureLayout = new QHBoxLayout();
        exposureLayout->addWidget(settingsExposureSlider_);
        exposureLayout->addWidget(settingsExposureSpin_);
        settingsFormLayout->addRow("Exposure:", exposureLayout);
        
        // Gain
        settingsGainSlider_ = new QSlider(Qt::Horizontal, this);
        settingsGainSlider_->setRange(0, 100);
        settingsGainSlider_->setValue(50);
        settingsGainSpin_ = new QSpinBox(this);
        settingsGainSpin_->setRange(0, 100);
        settingsGainSpin_->setValue(50);
        connect(settingsGainSlider_, &QSlider::valueChanged, settingsGainSpin_, &QSpinBox::setValue);
        connect(settingsGainSpin_, QOverload<int>::of(&QSpinBox::valueChanged),
                settingsGainSlider_, &QSlider::setValue);
        QHBoxLayout *gainLayout = new QHBoxLayout();
        gainLayout->addWidget(settingsGainSlider_);
        gainLayout->addWidget(settingsGainSpin_);
        settingsFormLayout->addRow("Gain:", gainLayout);
        
        // Brightness
        settingsBrightnessSlider_ = new QSlider(Qt::Horizontal, this);
        settingsBrightnessSlider_->setRange(0, 100);
        settingsBrightnessSlider_->setValue(50);
        settingsBrightnessSpin_ = new QSpinBox(this);
        settingsBrightnessSpin_->setRange(0, 100);
        settingsBrightnessSpin_->setValue(50);
        connect(settingsBrightnessSlider_, &QSlider::valueChanged, settingsBrightnessSpin_, &QSpinBox::setValue);
        connect(settingsBrightnessSpin_, QOverload<int>::of(&QSpinBox::valueChanged),
                settingsBrightnessSlider_, &QSlider::setValue);
        QHBoxLayout *brightnessLayout = new QHBoxLayout();
        brightnessLayout->addWidget(settingsBrightnessSlider_);
        brightnessLayout->addWidget(settingsBrightnessSpin_);
        settingsFormLayout->addRow("Brightness:", brightnessLayout);
        
        // Contrast
        settingsContrastSlider_ = new QSlider(Qt::Horizontal, this);
        settingsContrastSlider_->setRange(0, 100);
        settingsContrastSlider_->setValue(50);
        settingsContrastSpin_ = new QSpinBox(this);
        settingsContrastSpin_->setRange(0, 100);
        settingsContrastSpin_->setValue(50);
        connect(settingsContrastSlider_, &QSlider::valueChanged, settingsContrastSpin_, &QSpinBox::setValue);
        connect(settingsContrastSpin_, QOverload<int>::of(&QSpinBox::valueChanged),
                settingsContrastSlider_, &QSlider::setValue);
        QHBoxLayout *contrastLayout = new QHBoxLayout();
        contrastLayout->addWidget(settingsContrastSlider_);
        contrastLayout->addWidget(settingsContrastSpin_);
        settingsFormLayout->addRow("Contrast:", contrastLayout);
        
        // Saturation
        settingsSaturationSlider_ = new QSlider(Qt::Horizontal, this);
        settingsSaturationSlider_->setRange(0, 100);
        settingsSaturationSlider_->setValue(50);
        settingsSaturationSpin_ = new QSpinBox(this);
        settingsSaturationSpin_->setRange(0, 100);
        settingsSaturationSpin_->setValue(50);
        connect(settingsSaturationSlider_, &QSlider::valueChanged, settingsSaturationSpin_, &QSpinBox::setValue);
        connect(settingsSaturationSpin_, QOverload<int>::of(&QSpinBox::valueChanged),
                settingsSaturationSlider_, &QSlider::setValue);
        QHBoxLayout *saturationLayout = new QHBoxLayout();
        saturationLayout->addWidget(settingsSaturationSlider_);
        saturationLayout->addWidget(settingsSaturationSpin_);
        settingsFormLayout->addRow("Saturation:", saturationLayout);
        
        // Sharpness
        settingsSharpnessSlider_ = new QSlider(Qt::Horizontal, this);
        settingsSharpnessSlider_->setRange(0, 100);
        settingsSharpnessSlider_->setValue(50);
        settingsSharpnessSpin_ = new QSpinBox(this);
        settingsSharpnessSpin_->setRange(0, 100);
        settingsSharpnessSpin_->setValue(50);
        connect(settingsSharpnessSlider_, &QSlider::valueChanged, settingsSharpnessSpin_, &QSpinBox::setValue);
        connect(settingsSharpnessSpin_, QOverload<int>::of(&QSpinBox::valueChanged),
                settingsSharpnessSlider_, &QSlider::setValue);
        QHBoxLayout *sharpnessLayout = new QHBoxLayout();
        sharpnessLayout->addWidget(settingsSharpnessSlider_);
        sharpnessLayout->addWidget(settingsSharpnessSpin_);
        settingsFormLayout->addRow("Sharpness:", sharpnessLayout);
        
        // Mode
        QLabel *modeLabel = new QLabel("Mode:", this);
        settingsModeCombo_ = new QComboBox(this);
        settingsFormLayout->addRow(modeLabel, settingsModeCombo_);
        
        cameraSettingsLayout->addLayout(settingsFormLayout);
        cameraSettingsGroup->setLayout(cameraSettingsLayout);
        contentLayout->addWidget(cameraSettingsGroup);
        
        contentLayout->addStretch();
        settingsContent->setLayout(contentLayout);
        scrollArea->setWidget(settingsContent);
        settingsLayout->addWidget(scrollArea);
        
        // Save/Load buttons at the bottom
        QHBoxLayout *buttonLayout = new QHBoxLayout();
        settingsSaveBtn_ = new QPushButton("Save Settings", this);
        settingsLoadBtn_ = new QPushButton("Load Settings", this);
        connect(settingsSaveBtn_, &QPushButton::clicked, this, &AprilTagDebugGUI::saveSettings);
        connect(settingsLoadBtn_, &QPushButton::clicked, this, &AprilTagDebugGUI::loadSettings);
        buttonLayout->addWidget(settingsLoadBtn_);
        buttonLayout->addWidget(settingsSaveBtn_);
        buttonLayout->addStretch();
        settingsLayout->addLayout(buttonLayout);
        
        settingsTab->setLayout(settingsLayout);
        tabWidget_->addTab(settingsTab, "Settings");
    }
    
    void loadSettingsFromFile() {
        QString configPath = QDir::homePath() + "/.apriltag_debug_gui_config.yaml";
        
        if (!QFile::exists(configPath)) {
            qDebug() << "Config file not found, using defaults:" << configPath;
            return;
        }
        
        FileStorage fs(configPath.toStdString(), FileStorage::READ);
        if (!fs.isOpened()) {
            qDebug() << "Failed to open config file:" << configPath;
            return;
        }
        
        // Load fisheye settings
        if (fs["fisheye"].isMap()) {
            FileNode fisheyeNode = fs["fisheye"];
            
            if (!fisheyeNode["load_on_startup"].empty()) {
                bool loadOnStartup = static_cast<int>(fisheyeNode["load_on_startup"]);
                if (settingsLoadCalibrationOnStart_) {
                    settingsLoadCalibrationOnStart_->setChecked(loadOnStartup != 0);
                }
            }
            
            if (!fisheyeNode["calibration_path"].empty()) {
                string calibPath;
                fisheyeNode["calibration_path"] >> calibPath;
                if (settingsCalibrationPath_) {
                    settingsCalibrationPath_->setText(QString::fromStdString(calibPath));
                }
            }
            
            if (!fisheyeNode["enable_for_mindvision"].empty()) {
                bool enableForMV = static_cast<int>(fisheyeNode["enable_for_mindvision"]);
                if (settingsEnableFisheyeForMindVision_) {
                    settingsEnableFisheyeForMindVision_->setChecked(enableForMV != 0);
                }
            }
        }
        
        fs.release();
        
        qDebug() << "Settings loaded from:" << configPath;
    }
    
    void saveSettings() {
        QString configPath = QDir::homePath() + "/.apriltag_debug_gui_config.yaml";
        QFileInfo fileInfo(configPath);
        QDir dir = fileInfo.absoluteDir();
        if (!dir.exists()) {
            dir.mkpath(".");
        }
        
        FileStorage fs(configPath.toStdString(), FileStorage::WRITE);
        if (!fs.isOpened()) {
            QMessageBox::warning(this, "Error", QString("Failed to open config file for writing: %1").arg(configPath));
            return;
        }
        
        // Save fisheye settings
        fs << "fisheye" << "{";
        fs << "load_on_startup" << settingsLoadCalibrationOnStart_->isChecked();
        fs << "calibration_path" << settingsCalibrationPath_->text().toStdString();
        fs << "enable_for_mindvision" << settingsEnableFisheyeForMindVision_->isChecked();
        fs << "}";
        
        // Save camera settings from Settings tab if a camera is selected
        if (settingsCameraCombo_ && settingsCameraCombo_->currentIndex() >= 0) {
            saveCameraSettingsFromSettingsTab();
        }
        
        // Algorithm settings are now saved per camera in camera_settings.txt
        // No need to save globally here
        
        fs.release();
        
        QMessageBox::information(this, "Settings", QString("Settings saved to:\n%1").arg(configPath));
        qDebug() << "Settings saved to:" << configPath;
    }
    
    void loadSettings() {
        loadSettingsFromFile();
        applySettings();
    }
    
    void applySettings() {
        // Apply fisheye settings
        if (settingsLoadCalibrationOnStart_ && settingsLoadCalibrationOnStart_->isChecked()) {
            if (settingsCalibrationPath_ && !settingsCalibrationPath_->text().isEmpty()) {
                QString calibPath = settingsCalibrationPath_->text();
                if (loadFisheyeCalibration(calibPath)) {
                    qDebug() << "Calibration loaded from settings:" << calibPath;
                    if (fisheyeStatusLabel_) {
                        fisheyeStatusLabel_->setText(QString("Calibration: Loaded (Size: %1x%2)")
                            .arg(fisheye_image_size_.width)
                            .arg(fisheye_image_size_.height));
                    }
                    updateFisheyeStatusIndicator();
                }
            }
        }
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
        
        // Note: Camera selection is now at the top of the application (shared across all tabs)
        // The algorithm will use the camera selected in the top-level camera controls
        
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
        
        // Debug checkbox
        algorithmDebugCheckbox_ = new QCheckBox("Debug Mode (Show Quads)", this);
        algorithmDebugCheckbox_->setToolTip("Enable to show numbered quads before filtering on video display");
        buttonLayout->addWidget(algorithmDebugCheckbox_);
        
        // Note: Camera selection is now at the top of the application (shared across all tabs)
        // Connect to top-level camera selection to update mirror checkbox when camera changes
        if (cameraCombo_) {
            connect(cameraCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
                    this, [this]() {
                        if (cameraOpen_ && useMindVision_ && algorithmMirrorCheckbox_) {
                            algorithmMirrorCheckbox_->setChecked(true);
                        }
                    });
        }
        
        // Set initial mirror checkbox state based on current camera
        if (cameraOpen_ && useMindVision_ && algorithmMirrorCheckbox_) {
            algorithmMirrorCheckbox_->setChecked(true);
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
        
        // Add "None" option as first item (default selection)
        cameraCombo_->addItem("None");
        cameraList_.push_back("None");
        isMindVision_.push_back(false);
        
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
                // Try to get camera name from sysfs
                string cameraName = "V4L2 Camera " + to_string(i);
                string sysfsPath = "/sys/class/video4linux/video" + to_string(i) + "/name";
                QFile nameFile(QString::fromStdString(sysfsPath));
                if (nameFile.exists() && nameFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
                    QTextStream in(&nameFile);
                    QString name = in.readLine().trimmed();
                    if (!name.isEmpty()) {
                        // Check if it's an Arducam camera
                        if (name.contains("Arducam", Qt::CaseInsensitive)) {
                            cameraName = "Arducam: " + name.toStdString() + " (/dev/video" + to_string(i) + ")";
                        } else {
                            cameraName = name.toStdString() + " (/dev/video" + to_string(i) + ")";
                        }
                    }
                    nameFile.close();
                }
                
                cameraList_.push_back(cameraName);
                isMindVision_.push_back(false);
                cameraCombo_->addItem(QString::fromStdString(cameraName));
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
        // Refresh camera list before opening to ensure we have current connected cameras
        int previousIndex = cameraCombo_->currentIndex();
        QString previousCameraName;
        if (previousIndex >= 0 && previousIndex < cameraCombo_->count()) {
            previousCameraName = cameraCombo_->currentText();
        }
        
        // Re-enumerate cameras to get current list
        cameraCombo_->blockSignals(true);  // Prevent signal during enumeration
        enumerateCameras();
        
        // Try to restore previous selection if camera still exists
        int newIndex = 0;
        if (!previousCameraName.isEmpty()) {
            for (int i = 0; i < cameraCombo_->count(); i++) {
                if (cameraCombo_->itemText(i) == previousCameraName) {
                    newIndex = i;
                    break;
                }
            }
        }
        cameraCombo_->setCurrentIndex(newIndex);
        cameraCombo_->blockSignals(false);
        
        // Only close camera if it's already open (previewTimer_ might not exist yet during initialization)
        if (cameraOpen_ && previewTimer_ != nullptr) {
            closeCamera();
        }
        
        int index = cameraCombo_->currentIndex();
        if (index < 0 || index >= (int)cameraList_.size()) {
            return;
        }
        
        // Handle "None" selection
        if (index == 0 && cameraList_[0] == "None") {
            previewLabel_->setText("No camera selected");
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
            
            // Check if camera is already opened by another application
            BOOL isOpened = FALSE;
            CameraIsOpened(&list[mvIndex], &isOpened);
            if (isOpened) {
                QString errorMsg = QString("Camera '%1' (SN: %2) is already in use by another application.\n\n"
                                          "Please close other applications using this camera and try again.")
                                          .arg(QString::fromLocal8Bit(list[mvIndex].acFriendlyName))
                                          .arg(QString::fromLocal8Bit(list[mvIndex].acSn));
                previewLabel_->setText(errorMsg);
                QMessageBox::warning(this, "Camera Already in Use", errorMsg);
                return;
            }
            
            status = CameraInit(&list[mvIndex], -1, -1, &mvHandle_);
            if (status != CAMERA_STATUS_SUCCESS) {
                QString errorMsg;
                QString cameraName = QString::fromLocal8Bit(list[mvIndex].acFriendlyName);
                QString cameraSN = QString::fromLocal8Bit(list[mvIndex].acSn);
                
                switch (status) {
                    case CAMERA_STATUS_DEVICE_IS_OPENED:
                        errorMsg = QString("Camera '%1' (SN: %2) is already in use.\n\n"
                                          "Error code: %3 (CAMERA_STATUS_DEVICE_IS_OPENED)\n\n"
                                          "Please close other applications using this camera and try again.")
                                          .arg(cameraName).arg(cameraSN).arg(status);
                        break;
                    case CAMERA_STATUS_ACCESS_DENY:
                        errorMsg = QString("Access denied to camera '%1' (SN: %2).\n\n"
                                          "Error code: %3 (CAMERA_STATUS_ACCESS_DENY)\n\n"
                                          "The camera may be locked by another process.")
                                          .arg(cameraName).arg(cameraSN).arg(status);
                        break;
                    case CAMERA_STATUS_NO_DEVICE_FOUND:
                        errorMsg = QString("Camera '%1' (SN: %2) not found.\n\n"
                                          "Error code: %3 (CAMERA_STATUS_NO_DEVICE_FOUND)")
                                          .arg(cameraName).arg(cameraSN).arg(status);
                        break;
                    case CAMERA_STATUS_COMM_ERROR:
                        errorMsg = QString("Communication error with camera '%1' (SN: %2).\n\n"
                                          "Error code: %3 (CAMERA_STATUS_COMM_ERROR)\n\n"
                                          "Please check the USB connection.")
                                          .arg(cameraName).arg(cameraSN).arg(status);
                        break;
                    default:
                        errorMsg = QString("Failed to open MindVision camera '%1' (SN: %2).\n\n"
                                          "Error code: %3\n\n"
                                          "Please check the camera connection and try again.")
                                          .arg(cameraName).arg(cameraSN).arg(status);
                        break;
                }
                
                previewLabel_->setText(errorMsg);
                QMessageBox::warning(this, "Camera Open Failed", errorMsg);
                qDebug() << "CameraInit failed with status:" << status << "for camera:" << cameraName;
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
            
            // Load camera settings profile for MindVision (with camera handle for capability query)
            string cameraName = cameraList_[index];
            // First, try to load from config file
            map<string, CameraSettingsProfile> allProfiles = loadAllCameraSettingsProfiles();
            if (allProfiles.find(cameraName) != allProfiles.end()) {
                currentCameraSettings_ = allProfiles[cameraName];
            } else {
                // Create new profile using actual camera capabilities
                currentCameraSettings_ = defineMindVisionSettings(cameraName, mvHandle_);
            }
            loadCameraSettingsProfileForCamera(cameraName, "MindVision");
            applySavedValuesToProfile();
            
            // Apply saved settings to camera BEFORE reading current values
            applyCameraSettingsFromProfile();
            
            // Load and apply algorithm settings for this camera
            loadAlgorithmSettingsForCamera();
            
            // Populate dynamic camera controls (will hide scroll area for MindVision)
            populateDynamicCameraControls();
            
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
            modeCombo_->setEnabled(true);
            modeCombo_->blockSignals(false);
            
            // Load saved mode from Settings tab (will be applied after settings are loaded)
            // Check if there's a saved mode_index in the profile
            int savedModeIndex = 0;
            if (currentCameraSettings_.saved_values.find("mode_index") != currentCameraSettings_.saved_values.end()) {
                savedModeIndex = currentCameraSettings_.saved_values.at("mode_index");
                if (savedModeIndex >= 0 && savedModeIndex < modeCombo_->count()) {
                    modeCombo_->setCurrentIndex(savedModeIndex);
                    applyMVMode(savedModeIndex);
                } else {
                    modeCombo_->setCurrentIndex(0);
                    applyMVMode(0);
                }
            } else {
                // Default to first mode if no saved setting
                modeCombo_->setCurrentIndex(0);
                applyMVMode(0);
            }
            
            // Note: Settings are now applied from saved profile via applyCameraSettingsFromProfile()
            // No need to read current values from camera as they will be overwritten by saved settings
            
            cameraOpen_ = true;
            
            // Enable fisheye correction for MindVision cameras if setting is enabled (if calibration is loaded)
            bool enableFisheyeForMV = true;  // Default to true
            if (settingsEnableFisheyeForMindVision_) {
                enableFisheyeForMV = settingsEnableFisheyeForMindVision_->isChecked();
            }
            if (enableFisheyeForMV && fisheye_calibration_loaded_ && fisheyeStatusIndicator_) {
                fisheye_undistort_enabled_ = true;
                fisheyeStatusIndicator_->setText("Fisheye Correction: APPLIED");
                fisheyeStatusIndicator_->setStyleSheet("background-color: #90EE90; padding: 5px; border: 1px solid #006400;");
            }
#else
            previewLabel_->setText("MindVision SDK not available");
            return;
#endif
        } else {
            // Open V4L2 camera
            // Extract device number from camera name (e.g., "Arducam: ... (/dev/video2)" or "V4L2 Camera 2")
            int v4l2Index = -1;
            string cameraNameStr = cameraList_[index];
            qDebug() << "=== EXTRACTING DEVICE NUMBER ===";
            qDebug() << "Camera name string:" << QString::fromStdString(cameraNameStr);
            qDebug() << "Camera name length:" << cameraNameStr.length();
            qDebug() << "Camera name bytes:" << cameraNameStr.c_str();
            
            // Try to extract from name like "Arducam: ... (/dev/videoX)" or "Name (/dev/videoX)"
            string searchPattern = "(/dev/video";
            size_t devPos = cameraNameStr.find(searchPattern);
            qDebug() << "Searching for pattern:" << QString::fromStdString(searchPattern);
            qDebug() << "Pattern found at position:" << (devPos != string::npos ? (int)devPos : -1);
            if (devPos != string::npos) {
                // Pattern "(/dev/video" is 11 characters: ( / d e v / v i d e o
                // So device number starts at devPos + 11 (right after "video")
                size_t start = devPos + 11;
                size_t end = cameraNameStr.find(")", start);
                qDebug() << "Found pattern, start position:" << (int)start << ", looking for closing ')' at position:" << (end != string::npos ? (int)end : -1);
                if (end != string::npos && end > start) {
                    string deviceNumStr = cameraNameStr.substr(start, end - start);
                    qDebug() << "Extracted device number string (length" << deviceNumStr.length() << "): '" << QString::fromStdString(deviceNumStr) << "'";
                    
                    // Trim any whitespace (shouldn't be needed, but just in case)
                    while (!deviceNumStr.empty() && (deviceNumStr[0] == ' ' || deviceNumStr[0] == '\t')) {
                        deviceNumStr = deviceNumStr.substr(1);
                    }
                    while (!deviceNumStr.empty() && (deviceNumStr.back() == ' ' || deviceNumStr.back() == '\t')) {
                        deviceNumStr.pop_back();
                    }
                    qDebug() << "Final device number string:" << QString::fromStdString(deviceNumStr);
                    if (!deviceNumStr.empty()) {
                        try {
                            v4l2Index = stoi(deviceNumStr);
                            qDebug() << "Successfully extracted device number" << v4l2Index << "from camera name";
                        } catch (const std::exception& e) {
                            qDebug() << "Failed to parse device number:" << e.what();
                            v4l2Index = -1;
                        } catch (...) {
                            qDebug() << "Failed to parse device number (unknown error)";
                            v4l2Index = -1;
                        }
                    } else {
                        qDebug() << "Device number string is empty after trimming";
                        v4l2Index = -1;
                    }
                } else {
                    qDebug() << "Could not find closing parenthesis after /dev/video or invalid range";
                }
            } else {
                qDebug() << "Could not find (/dev/video pattern in camera name";
            }
            
            // Fallback: if extraction failed, try to find device number from "V4L2 Camera X" format
            if (v4l2Index < 0) {
                size_t pos = cameraNameStr.find_last_of(" ");
                if (pos != string::npos) {
                    string deviceNumStr = cameraNameStr.substr(pos + 1);
                    try {
                        v4l2Index = stoi(deviceNumStr);
                        qDebug() << "Extracted device number" << v4l2Index << "from camera name (fallback):" << QString::fromStdString(cameraNameStr);
                    } catch (...) {
                        // If stoi fails, use counting method as last resort
                        v4l2Index = 0;
                        for (int i = 0; i < index; i++) {
                            if (!isMindVision_[i]) v4l2Index++;
                        }
                        qDebug() << "Using calculated device index" << v4l2Index << "as fallback";
                    }
                } else {
                    // Last resort: count non-MindVision cameras
                    v4l2Index = 0;
                    for (int i = 0; i < index; i++) {
                        if (!isMindVision_[i]) v4l2Index++;
                    }
                    qDebug() << "Using calculated device index" << v4l2Index << "as last resort";
                }
            }
            
            qDebug() << "Opening V4L2 camera with device index:" << v4l2Index;
            cameraCap_.open(v4l2Index);
            if (!cameraCap_.isOpened()) {
                // Try with CAP_V4L2 explicitly
                qDebug() << "Failed to open with default backend, trying CAP_V4L2";
                cameraCap_.open(v4l2Index, CAP_V4L2);
            }
            if (!cameraCap_.isOpened()) {
                previewLabel_->setText(QString("Failed to open V4L2 camera %1 (device /dev/video%1)").arg(v4l2Index));
                qDebug() << "Failed to open V4L2 camera" << v4l2Index << "for camera:" << QString::fromStdString(cameraNameStr);
                return;
            }
            qDebug() << "Successfully opened V4L2 camera" << v4l2Index;
            
            // Check if this is an Arducam camera
            bool isArducam = false;
            QString cameraNameQStr = QString::fromStdString(cameraList_[index]);
            string cameraName = cameraList_[index];
            if (cameraNameQStr.contains("Arducam", Qt::CaseInsensitive)) {
                isArducam = true;
                // Load camera settings profile for Arducam
                loadCameraSettingsProfileForCamera(cameraName, "Arducam", v4l2Index);
            } else {
                // Load camera settings profile for other V4L2 cameras
                loadCameraSettingsProfileForCamera(cameraName, "V4L2", v4l2Index);
            }
            applySavedValuesToProfile();
            
            // Apply saved settings to camera and populate UI
            applyCameraSettingsFromProfile();
            
            // Load and apply algorithm settings for this camera
            loadAlgorithmSettingsForCamera();
            
            // Populate dynamic camera controls in the scroll area
            populateDynamicCameraControls();
            
            // Initialize V4L2 modes
            v4l2_modes_.clear();
            
            if (isArducam) {
                // For Arducam: Read available resolutions from the camera
                QString devicePath = QString("/dev/video%1").arg(v4l2Index);
                qDebug() << "Querying Arducam resolutions from:" << devicePath;
                QProcess v4l2Process;
                v4l2Process.start("v4l2-ctl", QStringList() << "--device" << devicePath << "--list-formats-ext");
                v4l2Process.waitForFinished(3000);  // Wait up to 3 seconds
                
                qDebug() << "v4l2-ctl exit code:" << v4l2Process.exitCode();
                if (v4l2Process.exitCode() == 0) {
                    QString output = v4l2Process.readAllStandardOutput();
                    qDebug() << "v4l2-ctl output length:" << output.length();
                    QTextStream stream(&output);
                    QString line;
                    int currentWidth = 0, currentHeight = 0;
                    QSet<QString> seenResolutions;  // To avoid duplicates
                    int linesProcessed = 0;
                    
                    while (stream.readLineInto(&line)) {
                        linesProcessed++;
                        // Look for "Size: Discrete WxH" lines
                        QRegularExpression sizeRegex(R"(Size: Discrete (\d+)x(\d+))");
                        if (!sizeRegex.isValid()) {
                            qDebug() << "Invalid regex pattern for size matching:" << sizeRegex.errorString();
                            continue;
                        }
                        QRegularExpressionMatch match = sizeRegex.match(line);
                        if (match.hasMatch()) {
                            currentWidth = match.captured(1).toInt();
                            currentHeight = match.captured(2).toInt();
                            qDebug() << "Found resolution:" << currentWidth << "x" << currentHeight;
                        }
                        
                        // Look for "Interval: Discrete X.XXXs (YYY.YYY fps)" lines
                        QRegularExpression fpsRegex("Interval: Discrete [\\d.]+s \\(([\\d.]+) fps\\)");
                        if (!fpsRegex.isValid()) {
                            qDebug() << "Invalid regex pattern for FPS matching:" << fpsRegex.errorString();
                            continue;
                        }
                        QRegularExpressionMatch fpsMatch = fpsRegex.match(line);
                        if (fpsMatch.hasMatch() && currentWidth > 0 && currentHeight > 0) {
                            double fps = fpsMatch.captured(1).toDouble();
                            QString resolutionKey = QString("%1x%2").arg(currentWidth).arg(currentHeight);
                            
                            // Only add if we haven't seen this exact resolution yet
                            // Or if we have, prefer higher FPS
                            QString modeLabel = QString("%1x%2 @%3 FPS").arg(currentWidth).arg(currentHeight).arg(fps, 0, 'f', 1);
                            if (!seenResolutions.contains(resolutionKey) || 
                                (v4l2_modes_.size() > 0 && v4l2_modes_.back().width == currentWidth && 
                                 v4l2_modes_.back().height == currentHeight && v4l2_modes_.back().fps < fps)) {
                                // Remove previous entry for this resolution if it exists
                                if (seenResolutions.contains(resolutionKey)) {
                                    for (auto it = v4l2_modes_.begin(); it != v4l2_modes_.end(); ++it) {
                                        if (it->width == currentWidth && it->height == currentHeight) {
                                            v4l2_modes_.erase(it);
                                            break;
                                        }
                                    }
                                }
                                v4l2_modes_.push_back(Mode{currentWidth, currentHeight, fps, modeLabel.toStdString()});
                                seenResolutions.insert(resolutionKey);
                            }
                        }
                    }
                    
                    qDebug() << "Processed" << linesProcessed << "lines from v4l2-ctl output";
                    qDebug() << "Found" << v4l2_modes_.size() << "resolutions";
                    
                    // Sort by resolution (width*height) descending, then by FPS descending
                    std::sort(v4l2_modes_.begin(), v4l2_modes_.end(), 
                        [](const Mode& a, const Mode& b) {
                            int areaA = a.width * a.height;
                            int areaB = b.width * b.height;
                            if (areaA != areaB) return areaA > areaB;
                            return a.fps > b.fps;
                        });
                } else {
                    qDebug() << "v4l2-ctl process failed with exit code:" << v4l2Process.exitCode();
                    QString errorOutput = v4l2Process.readAllStandardError();
                    if (!errorOutput.isEmpty()) {
                        qDebug() << "v4l2-ctl error output:" << errorOutput;
                    }
                }
                
                // Fallback to default modes if v4l2-ctl failed or no modes found
                if (v4l2_modes_.empty()) {
                    qDebug() << "No resolutions found, using default Arducam modes";
                    v4l2_modes_.push_back(Mode{1920, 1200, 50.0, "1920x1200 @50 FPS"});
                    v4l2_modes_.push_back(Mode{1920, 1200, 30.0, "1920x1200 @30 FPS"});
                    v4l2_modes_.push_back(Mode{960, 600, 80.0, "960x600 @80 FPS"});
                    v4l2_modes_.push_back(Mode{960, 600, 60.0, "960x600 @60 FPS"});
                    v4l2_modes_.push_back(Mode{960, 600, 30.0, "960x600 @30 FPS"});
                }
            } else {
                // For other V4L2 cameras: Use default modes
            v4l2_modes_.push_back(Mode{640, 480, 30.0, "640x480 @30 FPS"});
            v4l2_modes_.push_back(Mode{1280, 720, 30.0, "1280x720 @30 FPS"});
            v4l2_modes_.push_back(Mode{1920, 1080, 30.0, "1920x1080 @30 FPS"});
            v4l2_modes_.push_back(Mode{640, 480, 60.0, "640x480 @60 FPS"});
            }
            
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
        if (saveSettingsBtn_) {
        saveSettingsBtn_->setEnabled(true);
        }
        previewTimer_->start(33);  // ~30 FPS
        
        // Initialize algorithm if one is selected
        initializeCaptureAlgorithm();
        
        // Set mirror checkbox default based on camera type
        if (captureMirrorCheckbox_) {
            captureMirrorCheckbox_->setChecked(useMindVision_);
        }
        
        // Load camera settings from config file (legacy format)
        loadCameraSettings();
        
        // Also apply settings from the new profile structure
        // This will be used to populate the Settings tab UI
        
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
        
        // Clear dynamic controls when camera is closed
        clearDynamicCameraControls();
        
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
                // Initialize Fast AprilTag immediately (matching standalone program pattern)
                // The standalone creates GpuDetector in main thread and uses it from same thread
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
            // Update V4L2 camera settings - standard controls
            cameraCap_.set(CAP_PROP_EXPOSURE, exposureSlider_->value());
            cameraCap_.set(CAP_PROP_GAIN, gainSlider_->value());
            cameraCap_.set(CAP_PROP_BRIGHTNESS, brightnessSlider_->value());
            cameraCap_.set(CAP_PROP_CONTRAST, contrastSlider_->value());
            cameraCap_.set(CAP_PROP_SATURATION, saturationSlider_->value());
            cameraCap_.set(CAP_PROP_SHARPNESS, sharpnessSlider_->value());
            
            // Update dynamic V4L2 controls using v4l2-ctl
            int v4l2Index = -1;
            for (int i = 0; i < selectedCameraIndex_; i++) {
                if (!isMindVision_[i]) v4l2Index++;
            }
            
            if (v4l2Index >= 0) {
                QString devicePath = QString("/dev/video%1").arg(v4l2Index);
                
                // Update all dynamic controls
                for (const auto& pair : dynamicSliders_) {
                    const string& settingName = pair.first;
                    QSlider* slider = pair.second;
                    
                    // Find the setting in the profile
                    for (const auto& setting : currentCameraSettings_.settings) {
                        if (setting.name == settingName && !setting.v4l2_id.empty()) {
                            QProcess v4l2Process;
                            v4l2Process.start("v4l2-ctl", QStringList() << "--device" << devicePath 
                                             << "--set-ctrl" << QString("%1=%2").arg(QString::fromStdString(setting.v4l2_id)).arg(slider->value()));
                            v4l2Process.waitForFinished(500);
                            break;
                        }
                    }
                }
                
                for (const auto& pair : dynamicCheckBoxes_) {
                    const string& settingName = pair.first;
                    QCheckBox* checkBox = pair.second;
                    
                    // Find the setting in the profile
                    for (const auto& setting : currentCameraSettings_.settings) {
                        if (setting.name == settingName && !setting.v4l2_id.empty()) {
                            QProcess v4l2Process;
                            v4l2Process.start("v4l2-ctl", QStringList() << "--device" << devicePath 
                                             << "--set-ctrl" << QString("%1=%2").arg(QString::fromStdString(setting.v4l2_id)).arg(checkBox->isChecked() ? 1 : 0));
                            v4l2Process.waitForFinished(500);
                            break;
                        }
                    }
                }
                
                for (const auto& pair : dynamicComboBoxes_) {
                    const string& settingName = pair.first;
                    QComboBox* comboBox = pair.second;
                    
                    // Find the setting in the profile
                    for (const auto& setting : currentCameraSettings_.settings) {
                        if (setting.name == settingName && !setting.v4l2_id.empty()) {
                            int value = setting.min_value + comboBox->currentIndex();
                            QProcess v4l2Process;
                            v4l2Process.start("v4l2-ctl", QStringList() << "--device" << devicePath 
                                             << "--set-ctrl" << QString("%1=%2").arg(QString::fromStdString(setting.v4l2_id)).arg(value));
                            v4l2Process.waitForFinished(500);
                            break;
                        }
                    }
                }
            }
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
                                    // Verify algorithm object is valid before calling
                                    if (!captureAlgorithm_) {
                                        detections = nullptr;
                                    } else {
                                        // Try to call with exception handling
                                        try {
                                            detections = captureAlgorithm_->processFrame(frame_for_detection, mirror);
                                            
                                        } catch (const std::exception& e) {
                                            qDebug() << "Exception caught in processFrame call:" << e.what();
                                            detections = nullptr;
                                        } catch (...) {
                                            qDebug() << "Unknown exception caught in processFrame call";
                                            detections = nullptr;
                                        }
                                    }
                                    // Validate returned detections pointer
                                    if (detections != nullptr) {
                                        // Try to get size to validate the array is valid
                                        try {
                                            int size = zarray_size(detections);
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
                
                // Calculate layout: one tag per row
                int num_patterns = all_patterns.size();
                if (num_patterns > 0) {
                    // One tag per row: each tag gets its own row
                    int cols = 1;
                    int rows = num_patterns;
                    
                    // Calculate available space and size to fit all 3 boxes
                    // Each pattern cell includes: warped image (left) + gray color box (middle) + digitized pattern (right)
                    // Estimate available width (will be adjusted based on actual label size)
                    int estimated_width = 1200;  // Estimate, will be adjusted
                    int padding = 20;
                    int spacing = 15; // Space between boxes
                    int header_height = 30;
                    
                    // Calculate box size to fit 3 boxes in available width
                    // total_width = 3 * box_size + 2 * spacing + 2 * padding
                    // box_size = (total_width - 2 * spacing - 2 * padding) / 3
                    int available_width_per_row = estimated_width - 2 * padding;
                    int box_size = (available_width_per_row - 2 * spacing) / 3;
                    if (box_size < 50) box_size = 50;  // Minimum size
                    if (box_size > 200) box_size = 200;  // Maximum size
                    
                    int cell_size = box_size / 8;  // 8x8 grid, so cell_size = box_size / 8
                    if (cell_size < 1) cell_size = 1;
                    
                    int grid_size = box_size;  // 8x8 grid
                    int warped_image_size = box_size;
                    int gray_box_size = box_size;  // Gray color box same size, 8x8
                    
                    int cell_total_width = warped_image_size + spacing + gray_box_size + spacing + grid_size;  // Warped + gray + pattern
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
                        
                        // Calculate position: one tag per row
                        int col = 0;
                        int row = pattern_idx;
                        int cell_total_width = warped_image_size + spacing + grid_size + spacing + gray_box_size;
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
                        
                        // Gray color box position (middle, between warped image and digitized pattern)
                        int gray_x_offset = x_offset + padding + warped_image_size + spacing;
                        
                        // Draw 8x8 gray color box (middle, showing actual gray values)
                        // Bounds check
                        int gray_box_right = gray_x_offset + gray_box_size;
                        int gray_box_bottom = y_offset + header_height + padding + gray_box_size;
                        if (gray_box_right <= pattern_vis.cols && gray_box_bottom <= pattern_vis.rows) {
                            // Draw 8x8 grid with actual gray color values (same structure as digitized pattern)
                            // First, draw border cells (all black)
                            // Top row (row 0)
                            for (int c = 0; c < 8; c++) {
                                int y_pos = y_offset + header_height + padding;
                                int x_pos = gray_x_offset + c * cell_size;
                                if (x_pos + cell_size > pattern_vis.cols || y_pos + cell_size > pattern_vis.rows) break;
                                Rect cell(x_pos, y_pos, cell_size, cell_size);
                                rectangle(pattern_vis, cell, Scalar(0, 0, 0), -1);  // Black border
                                rectangle(pattern_vis, cell, Scalar(128, 128, 128), 1);
                            }
                            // Bottom row (row 7)
                            for (int c = 0; c < 8; c++) {
                                int y_pos = y_offset + header_height + padding + 7 * cell_size;
                                int x_pos = gray_x_offset + c * cell_size;
                                if (x_pos + cell_size > pattern_vis.cols || y_pos + cell_size > pattern_vis.rows) break;
                                Rect cell(x_pos, y_pos, cell_size, cell_size);
                                rectangle(pattern_vis, cell, Scalar(0, 0, 0), -1);  // Black border
                                rectangle(pattern_vis, cell, Scalar(128, 128, 128), 1);
                            }
                            // Left column (col 0, rows 1-6)
                            for (int r = 1; r < 7; r++) {
                                int y_pos = y_offset + header_height + padding + r * cell_size;
                                int x_pos = gray_x_offset;
                                if (x_pos + cell_size > pattern_vis.cols || y_pos + cell_size > pattern_vis.rows) break;
                                Rect cell(x_pos, y_pos, cell_size, cell_size);
                                rectangle(pattern_vis, cell, Scalar(0, 0, 0), -1);  // Black border
                                rectangle(pattern_vis, cell, Scalar(128, 128, 128), 1);
                            }
                            // Right column (col 7, rows 1-6)
                            for (int r = 1; r < 7; r++) {
                                int y_pos = y_offset + header_height + padding + r * cell_size;
                                int x_pos = gray_x_offset + 7 * cell_size;
                                if (x_pos + cell_size > pattern_vis.cols || y_pos + cell_size > pattern_vis.rows) break;
                                Rect cell(x_pos, y_pos, cell_size, cell_size);
                                rectangle(pattern_vis, cell, Scalar(0, 0, 0), -1);  // Black border
                                rectangle(pattern_vis, cell, Scalar(128, 128, 128), 1);
                            }
                            
                            // Now draw the 6x6 data pattern in the center (rows 1-6, columns 1-6) with actual gray values
                            for (int r = 0; r < 6 && r < (int)pattern.size(); r++) {
                                if (r >= (int)pattern.size() || pattern[r].size() != 6) continue;
                                for (int c = 0; c < 6 && c < (int)pattern[r].size(); c++) {
                                    int val = pattern[r][c];
                                    // Use actual gray value
                                    Scalar gray_color(val, val, val);
                                    
                                    // Map 6x6 pattern (r,c) to 8x8 grid position (r+1, c+1)
                                    int y_pos = y_offset + header_height + padding + (r + 1) * cell_size;
                                    int x_pos = gray_x_offset + (c + 1) * cell_size;
                                    
                                    // Defensive bounds check
                                    if (x_pos + cell_size > pattern_vis.cols || y_pos + cell_size > pattern_vis.rows) continue;
                                    
                                    Rect gray_cell(x_pos, y_pos, cell_size, cell_size);
                                    rectangle(pattern_vis, gray_cell, gray_color, -1);
                                    rectangle(pattern_vis, gray_cell, Scalar(128, 128, 128), 1);
                                }
                            }
                            
                            // Label for gray color box
                            putText(pattern_vis, "Gray Values (8x8: border + 6x6 data)", Point(gray_x_offset, y_offset + header_height + padding - 5),
                                   FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 0, 255), 1);
                        }
                        
                        // Pattern grid position (to the right of gray color box)
                        int pattern_x_offset = gray_x_offset + gray_box_size + spacing;
                        
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
                                
                                // No text - just black and white squares
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
                        for (size_t i = 0; i < all_patterns.size(); i++) {
                            const PatternData& pdata = all_patterns[i];
                            if (pdata.detection_idx < all_detections_data.size()) {
                                const DetectionData& data = all_detections_data[pdata.detection_idx];
                                StoredPatternData sp;
                                sp.tag_id = data.id;
                                sp.warped_image = pdata.warped_image.clone();
                                sp.pattern = pdata.pattern;
                                storedPatterns_.push_back(sp);
                            }
                        }
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
                try {
                info_ss << "=== DETECTED TAGS: " << num_tags << " ===\n";
                info_ss << "=== SUCCESSFULLY EXTRACTED PATTERNS: " << all_patterns.size() << " ===\n\n";
                
                for (size_t pattern_idx = 0; pattern_idx < all_patterns.size(); pattern_idx++) {
                        try {
                    const PatternData& pdata = all_patterns[pattern_idx];
                    // Safety check: ensure detection index is valid
                    if (pdata.detection_idx >= all_detections_data.size()) {
                                qDebug() << "Skipping pattern" << pattern_idx << "due to invalid detection_idx:" << pdata.detection_idx << ">= size:" << all_detections_data.size();
                        continue; // Skip if detection index is out of bounds
                    }
                    const DetectionData& data = all_detections_data[pdata.detection_idx];
                    const vector<vector<int>>& pattern = pdata.pattern;
                    
                    // Safety check: ensure pattern is valid (6x6)
                    if (pattern.size() != 6) {
                                qDebug() << "Skipping pattern" << pattern_idx << "due to invalid pattern size (rows):" << pattern.size();
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
                                qDebug() << "Skipping pattern" << pattern_idx << "due to invalid pattern row sizes";
                        continue; // Skip if pattern rows have wrong size
                    }
                    
                    info_ss << "--- Tag " << (pattern_idx + 1) << " (ID: " << data.id << ") ---\n";
                    
                            // Show hamming code info (pattern visualization is shown in the image above)
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
                        } catch (const std::exception& e) {
                            qDebug() << "Exception processing pattern" << pattern_idx << ":" << e.what();
                            continue;
                        } catch (...) {
                            qDebug() << "Unknown exception processing pattern" << pattern_idx;
                            continue;
                        }
                    }
                } catch (const std::exception& e) {
                    qDebug() << "Exception building info text:" << e.what();
                    info_ss.str(""); // Clear the stream
                    info_ss << "Error building pattern info: " << e.what();
                } catch (...) {
                    qDebug() << "Unknown exception building info text";
                    info_ss.str(""); // Clear the stream
                    info_ss << "Error building pattern info";
                }
                
                // Safety check before setting text
                if (capturePatternInfoText_) {
                    try {
                    capturePatternInfoText_->setPlainText(QString::fromStdString(info_ss.str()));
                    } catch (const std::exception& e) {
                        qDebug() << "Error setting pattern info text:" << e.what();
                    } catch (...) {
                        qDebug() << "Unknown error setting pattern info text";
                    }
                }
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

    // All member variables declared above - duplicates removed (UI elements declared earlier in private section)
    
    // All member variables declared above - duplicates removed (calibration data and camera settings declared earlier in private section)
    
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
    
    // ========== Camera Settings Profile Functions ==========
    
    // Query Arducam settings from v4l2-ctl
    CameraSettingsProfile queryArducamSettings(int v4l2Index, const string& cameraName) {
        CameraSettingsProfile profile;
        profile.camera_name = cameraName;
        profile.camera_type = "Arducam";
        
        QString devicePath = QString("/dev/video%1").arg(v4l2Index);
        QProcess v4l2Process;
        v4l2Process.start("v4l2-ctl", QStringList() << "--device" << devicePath << "--list-ctrls");
        v4l2Process.waitForFinished(3000);
        
        if (v4l2Process.exitCode() == 0) {
            QString output = v4l2Process.readAllStandardOutput();
            QTextStream stream(&output);
            QString line;
            
            while (stream.readLineInto(&line)) {
                line = line.trimmed();
                if (line.isEmpty()) continue;
                
                // Skip section headers like "User Controls" and "Camera Controls"
                if (line.contains("Controls") && !line.contains("0x")) {
                    continue;
                }
                
                // Parse V4L2 control line with improved regex to handle all formats:
                // "brightness 0x00980900 (int)    : min=-64 max=64 step=1 default=0 value=64"
                // "auto_exposure 0x009a0901 (menu)   : min=0 max=3 default=0 value=0 (Auto Mode)"
                // "white_balance_automatic 0x0098090c (bool)   : default=1 value=1"
                // "exposure_time_absolute 0x009a0902 (int)    : min=5 max=660 step=1 default=5 value=5 flags=inactive"
                
                // More flexible regex that handles optional flags and menu text
                QRegularExpression ctrlRegex(R"((\w+)\s+0x[0-9a-fA-F]+\s+\((int|bool|menu)\)\s*:\s*(?:min=(-?\d+)\s+)?(?:max=(-?\d+)\s+)?(?:step=(\d+)\s+)?(?:default=(-?\d+)\s+)?value=(-?\d+)(?:\s+\([^)]+\))?(?:\s+flags=\w+)?)");
                QRegularExpressionMatch match = ctrlRegex.match(line);
                
                if (match.hasMatch()) {
                    CameraSetting setting;
                    setting.name = match.captured(1).toStdString();
                    
                    // Create display name (capitalize first letter, replace underscores with spaces)
                    QString displayName = QString::fromStdString(setting.name);
                    displayName = displayName.replace("_", " ");
                    if (!displayName.isEmpty()) {
                        displayName[0] = displayName[0].toUpper();
                    }
                    setting.display_name = displayName.toStdString();
                    setting.v4l2_id = match.captured(1).toStdString();
                    
                    QString typeStr = match.captured(2);
                    if (typeStr == "int") {
                        setting.type = SETTING_INT;
                    } else if (typeStr == "bool") {
                        setting.type = SETTING_BOOL;
                    } else if (typeStr == "menu") {
                        setting.type = SETTING_MENU;
                    } else {
                        setting.type = SETTING_INT;
                    }
                    
                    // Parse values (some may be optional)
                    QString minStr = match.captured(3);
                    QString maxStr = match.captured(4);
                    QString defaultStr = match.captured(6);
                    QString valueStr = match.captured(7);
                    
                    setting.min_value = minStr.isEmpty() ? 0 : minStr.toInt();
                    setting.max_value = maxStr.isEmpty() ? (typeStr == "bool" ? 1 : 100) : maxStr.toInt();
                    setting.default_value = defaultStr.isEmpty() ? 0 : defaultStr.toInt();
                    setting.current_value = valueStr.isEmpty() ? setting.default_value : valueStr.toInt();
                    
                    // For menu types, extract menu text if present
                    if (setting.type == SETTING_MENU) {
                        QRegularExpression menuTextRegex(R"(\(([^)]+)\)\s*$)");
                        QRegularExpressionMatch menuMatch = menuTextRegex.match(line);
                        if (menuMatch.hasMatch()) {
                            QString menuText = menuMatch.captured(1);
                            // Store menu text in display_name or add to menu_options
                            setting.display_name = (displayName + " (" + menuText + ")").toStdString();
                        }
                    }
                    
                    profile.settings.push_back(setting);
                    } else {
                    // Try a simpler regex for controls without min/max (like bool with just default and value)
                    // Also try pattern that matches controls with menu text before value
                    QRegularExpression simpleRegex(R"((\w+)\s+0x[0-9a-fA-F]+\s+\((int|bool|menu)\)\s*:\s*(?:min=(-?\d+)\s+)?(?:max=(-?\d+)\s+)?(?:default=(-?\d+)\s+)?value=(-?\d+))");
                    QRegularExpressionMatch simpleMatch = simpleRegex.match(line);
                    if (simpleMatch.hasMatch()) {
                        qDebug() << "Matched with simple regex:" << simpleMatch.captured(1);
                        CameraSetting setting;
                        setting.name = simpleMatch.captured(1).toStdString();
                        
                        QString displayName = QString::fromStdString(setting.name);
                        displayName = displayName.replace("_", " ");
                        if (!displayName.isEmpty()) {
                            displayName[0] = displayName[0].toUpper();
                        }
                        setting.display_name = displayName.toStdString();
                        setting.v4l2_id = simpleMatch.captured(1).toStdString();
                        
                        QString typeStr = simpleMatch.captured(2);
                        if (typeStr == "int") {
                            setting.type = SETTING_INT;
                        } else if (typeStr == "bool") {
                            setting.type = SETTING_BOOL;
                        } else if (typeStr == "menu") {
                            setting.type = SETTING_MENU;
                        } else {
                            setting.type = SETTING_INT;
                        }
                        
                        QString defaultStr = simpleMatch.captured(3);
                        QString valueStr = simpleMatch.captured(4);
                        
                        setting.min_value = (typeStr == "bool") ? 0 : 0;
                        setting.max_value = (typeStr == "bool") ? 1 : 100;
                        setting.default_value = defaultStr.isEmpty() ? 0 : defaultStr.toInt();
                        setting.current_value = valueStr.isEmpty() ? setting.default_value : valueStr.toInt();
                        
                        profile.settings.push_back(setting);
                    }
                }
            }
        }
        
        return profile;
    }
    
    // Define MindVision settings structure from actual camera capabilities
    CameraSettingsProfile defineMindVisionSettings(const string& cameraName, CameraHandle handle = 0) {
        CameraSettingsProfile profile;
        profile.camera_name = cameraName;
        profile.camera_type = "MindVision";
        
        // Query camera capabilities if handle is provided
        tSdkCameraCapbility cap;
        bool hasCapability = false;
        if (handle != 0) {
            CameraSdkStatus status = CameraGetCapability(handle, &cap);
            if (status == CAMERA_STATUS_SUCCESS) {
                hasCapability = true;
            }
        }
        
        if (hasCapability) {
            // Exposure - use actual camera range
            CameraSetting exposure;
            exposure.name = "exposure";
            exposure.display_name = "Exposure";
            exposure.type = SETTING_INT;
            // Map exposure time (microseconds) to slider (0-100)
            // Use camera's actual exposure range
            exposure.min_value = 0;
            exposure.max_value = 100;
            exposure.default_value = 50;
            exposure.current_value = 50;
            exposure.v4l2_id = "";
            profile.settings.push_back(exposure);
            
            // Analog Gain - use actual camera range
            CameraSetting analogGain;
            analogGain.name = "analog_gain";
            analogGain.display_name = "Analog Gain";
            analogGain.type = SETTING_INT;
            analogGain.min_value = cap.sExposeDesc.uiAnalogGainMin;
            analogGain.max_value = cap.sExposeDesc.uiAnalogGainMax;
            analogGain.default_value = (analogGain.min_value + analogGain.max_value) / 2;
            analogGain.current_value = analogGain.default_value;
            analogGain.v4l2_id = "";
            profile.settings.push_back(analogGain);
            
            // Digital Gain (RGB) - use actual camera range
            CameraSetting digitalGain;
            digitalGain.name = "digital_gain";
            digitalGain.display_name = "Digital Gain (RGB)";
            digitalGain.type = SETTING_INT;
            digitalGain.min_value = cap.sRgbGainRange.iRGainMin;
            digitalGain.max_value = cap.sRgbGainRange.iRGainMax;
            digitalGain.default_value = (digitalGain.min_value + digitalGain.max_value) / 2;
            digitalGain.current_value = digitalGain.default_value;
            digitalGain.v4l2_id = "";
            profile.settings.push_back(digitalGain);
            
            // Gain (legacy, maps to digital gain for compatibility)
            CameraSetting gain;
            gain.name = "gain";
            gain.display_name = "Gain";
            gain.type = SETTING_INT;
            gain.min_value = 0;
            gain.max_value = 100;
            gain.default_value = 50;
            gain.current_value = 50;
            gain.v4l2_id = "";
            profile.settings.push_back(gain);
            
            // Brightness (maps to analog gain for compatibility)
            CameraSetting brightness;
            brightness.name = "brightness";
            brightness.display_name = "Brightness";
            brightness.type = SETTING_INT;
            brightness.min_value = 0;
            brightness.max_value = 255;
            brightness.default_value = 128;
            brightness.current_value = 128;
            brightness.v4l2_id = "";
            profile.settings.push_back(brightness);
            
            // Contrast - use actual camera range
            CameraSetting contrast;
            contrast.name = "contrast";
            contrast.display_name = "Contrast";
            contrast.type = SETTING_INT;
            contrast.min_value = cap.sContrastRange.iMin;
            contrast.max_value = cap.sContrastRange.iMax;
            contrast.default_value = (contrast.min_value + contrast.max_value) / 2;
            contrast.current_value = contrast.default_value;
            contrast.v4l2_id = "";
            profile.settings.push_back(contrast);
            
            // Gamma - use actual camera range
            CameraSetting gamma;
            gamma.name = "gamma";
            gamma.display_name = "Gamma";
            gamma.type = SETTING_INT;
            gamma.min_value = cap.sGammaRange.iMin;
            gamma.max_value = cap.sGammaRange.iMax;
            gamma.default_value = (gamma.min_value + gamma.max_value) / 2;
            gamma.current_value = gamma.default_value;
            gamma.v4l2_id = "";
            profile.settings.push_back(gamma);
            
            // Saturation - use actual camera range (only for color cameras)
            if (!cap.sIspCapacity.bMonoSensor) {
                CameraSetting saturation;
                saturation.name = "saturation";
                saturation.display_name = "Saturation";
                saturation.type = SETTING_INT;
                saturation.min_value = cap.sSaturationRange.iMin;
                saturation.max_value = cap.sSaturationRange.iMax;
                saturation.default_value = (saturation.min_value + saturation.max_value) / 2;
                saturation.current_value = saturation.default_value;
                saturation.v4l2_id = "";
                profile.settings.push_back(saturation);
            }
            
            // Sharpness - use actual camera range
            CameraSetting sharpness;
            sharpness.name = "sharpness";
            sharpness.display_name = "Sharpness";
            sharpness.type = SETTING_INT;
            sharpness.min_value = cap.sSharpnessRange.iMin;
            sharpness.max_value = cap.sSharpnessRange.iMax;
            sharpness.default_value = (sharpness.min_value + sharpness.max_value) / 2;
            sharpness.current_value = sharpness.default_value;
            sharpness.v4l2_id = "";
            profile.settings.push_back(sharpness);
            
            // Auto Exposure Target - if supported
            if (cap.sIspCapacity.bAutoExposure) {
                CameraSetting aeTarget;
                aeTarget.name = "ae_target";
                aeTarget.display_name = "AE Target";
                aeTarget.type = SETTING_INT;
                aeTarget.min_value = cap.sExposeDesc.uiTargetMin;
                aeTarget.max_value = cap.sExposeDesc.uiTargetMax;
                aeTarget.default_value = (aeTarget.min_value + aeTarget.max_value) / 2;
                aeTarget.current_value = aeTarget.default_value;
                aeTarget.v4l2_id = "";
                profile.settings.push_back(aeTarget);
            }
            
            // Frame Speed - use available frame speeds
            if (cap.iFrameSpeedDesc > 0) {
                CameraSetting frameSpeed;
                frameSpeed.name = "frame_speed";
                frameSpeed.display_name = "Frame Speed";
                frameSpeed.type = SETTING_INT;
                frameSpeed.min_value = 0;
                frameSpeed.max_value = cap.iFrameSpeedDesc - 1;
                frameSpeed.default_value = 0;
                frameSpeed.current_value = 0;
                frameSpeed.v4l2_id = "";
                profile.settings.push_back(frameSpeed);
            }
            
            // Mode (resolution/FPS) - stored as index
            CameraSetting mode;
            mode.name = "mode_index";
            mode.display_name = "Mode";
            mode.type = SETTING_INT;
            mode.min_value = 0;
            mode.max_value = (cap.iImageSizeDesc > 0) ? (cap.iImageSizeDesc - 1) : 10;
            mode.default_value = 0;
            mode.current_value = 0;
            mode.v4l2_id = "";
            profile.settings.push_back(mode);
        } else {
            // Fallback to default values if capability query failed
            // Exposure (0-100 slider, maps to 1000-100000 microseconds)
            CameraSetting exposure;
            exposure.name = "exposure";
            exposure.display_name = "Exposure";
            exposure.type = SETTING_INT;
            exposure.min_value = 0;
            exposure.max_value = 100;
            exposure.default_value = 50;
            exposure.current_value = 50;
            exposure.v4l2_id = "";
            profile.settings.push_back(exposure);
            
            // Gain (0-100)
            CameraSetting gain;
            gain.name = "gain";
            gain.display_name = "Gain";
            gain.type = SETTING_INT;
            gain.min_value = 0;
            gain.max_value = 100;
            gain.default_value = 50;
            gain.current_value = 50;
            gain.v4l2_id = "";
            profile.settings.push_back(gain);
            
            // Brightness (0-255, maps to analog gain 0-100)
            CameraSetting brightness;
            brightness.name = "brightness";
            brightness.display_name = "Brightness";
            brightness.type = SETTING_INT;
            brightness.min_value = 0;
            brightness.max_value = 255;
            brightness.default_value = 128;
            brightness.current_value = 128;
            brightness.v4l2_id = "";
            profile.settings.push_back(brightness);
            
            // Contrast (0-100)
            CameraSetting contrast;
            contrast.name = "contrast";
            contrast.display_name = "Contrast";
            contrast.type = SETTING_INT;
            contrast.min_value = 0;
            contrast.max_value = 100;
            contrast.default_value = 50;
            contrast.current_value = 50;
            contrast.v4l2_id = "";
            profile.settings.push_back(contrast);
            
            // Saturation (0-100)
            CameraSetting saturation;
            saturation.name = "saturation";
            saturation.display_name = "Saturation";
            saturation.type = SETTING_INT;
            saturation.min_value = 0;
            saturation.max_value = 100;
            saturation.default_value = 50;
            saturation.current_value = 50;
            saturation.v4l2_id = "";
            profile.settings.push_back(saturation);
            
            // Sharpness (0-100)
            CameraSetting sharpness;
            sharpness.name = "sharpness";
            sharpness.display_name = "Sharpness";
            sharpness.type = SETTING_INT;
            sharpness.min_value = 0;
            sharpness.max_value = 100;
            sharpness.default_value = 50;
            sharpness.current_value = 50;
            sharpness.v4l2_id = "";
            profile.settings.push_back(sharpness);
            
            // Mode (resolution/FPS) - stored as index
            CameraSetting mode;
            mode.name = "mode_index";
            mode.display_name = "Mode";
            mode.type = SETTING_INT;
            mode.min_value = 0;
            mode.max_value = 10;
            mode.default_value = 0;
            mode.current_value = 0;
            mode.v4l2_id = "";
            profile.settings.push_back(mode);
        }
        
        return profile;
    }
    
    // Save camera settings profile to config file
    void saveCameraSettingsProfile(const CameraSettingsProfile& profile) {
        QString configPath = "camera_settings.txt";
        QFile file(configPath);
        
        // Read existing file to preserve other cameras
        map<string, CameraSettingsProfile> profiles = loadAllCameraSettingsProfiles();
        
        // Update or add this camera's profile
        profiles[profile.camera_name] = profile;
        
        // Write all profiles back
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QMessageBox::warning(this, "Error", "Failed to save camera settings to " + configPath);
            return;
        }
        
        QTextStream out(&file);
        out << "# Camera Settings Configuration\n";
        out << "# Format: Each camera has its own section with settings structure and values\n";
        out << "\n";
        
        for (const auto& pair : profiles) {
            const CameraSettingsProfile& prof = pair.second;
            out << "[Camera:" << QString::fromStdString(prof.camera_name) << "]\n";
            out << "type=" << QString::fromStdString(prof.camera_type) << "\n";
        out << "\n";
        
            out << "[Settings:" << QString::fromStdString(prof.camera_name) << "]\n";
            // Save settings structure
            out << "settings_count=" << prof.settings.size() << "\n";
            for (size_t i = 0; i < prof.settings.size(); i++) {
                const CameraSetting& s = prof.settings[i];
                out << "setting_" << i << "_name=" << QString::fromStdString(s.name) << "\n";
                out << "setting_" << i << "_display=" << QString::fromStdString(s.display_name) << "\n";
                out << "setting_" << i << "_type=" << static_cast<int>(s.type) << "\n";
                out << "setting_" << i << "_min=" << s.min_value << "\n";
                out << "setting_" << i << "_max=" << s.max_value << "\n";
                out << "setting_" << i << "_default=" << s.default_value << "\n";
                out << "setting_" << i << "_v4l2_id=" << QString::fromStdString(s.v4l2_id) << "\n";
            }
            out << "\n";
            
            // Save values
            out << "[Values:" << QString::fromStdString(prof.camera_name) << "]\n";
            for (const auto& valPair : prof.saved_values) {
                out << QString::fromStdString(valPair.first) << "=" << valPair.second << "\n";
            }
            out << "\n";
            
            // Save algorithm settings
            if (!prof.algorithm_settings.empty()) {
                out << "[Algorithm:" << QString::fromStdString(prof.camera_name) << "]\n";
                for (const auto& algoPair : prof.algorithm_settings) {
                    out << QString::fromStdString(algoPair.first) << "=" << algoPair.second << "\n";
                }
                out << "\n";
            }
        }
        
        file.close();
    }
    
    // Load all camera settings profiles from config file
    map<string, CameraSettingsProfile> loadAllCameraSettingsProfiles() {
        map<string, CameraSettingsProfile> profiles;
        QString configPath = "camera_settings.txt";
        QFile file(configPath);
        
        if (!file.exists() || !file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            return profiles;  // Return empty map if file doesn't exist
        }
        
        QTextStream in(&file);
        QString currentCamera;
        string currentSection;
        CameraSettingsProfile currentProfile;
        int settingsCount = 0;
        map<int, CameraSetting> tempSettings;
        
        while (!in.atEnd()) {
            QString line = in.readLine().trimmed();
            
            if (line.startsWith("#") || line.isEmpty()) continue;
            
            // Parse section headers: [Camera:name], [Settings:name], [Values:name], [Algorithm:name]
            QRegularExpression sectionRegex(R"(\[(\w+):([^\]]+)\])");
            QRegularExpressionMatch sectionMatch = sectionRegex.match(line);
            if (sectionMatch.hasMatch()) {
                string sectionType = sectionMatch.captured(1).toStdString();
                currentCamera = sectionMatch.captured(2);
                currentSection = sectionType;
                
                if (sectionType == "Camera") {
                    // Start new camera profile
                    if (!currentProfile.camera_name.empty()) {
                        profiles[currentProfile.camera_name] = currentProfile;
                    }
                    currentProfile = CameraSettingsProfile();
                    currentProfile.camera_name = currentCamera.toStdString();
                    tempSettings.clear();
                    settingsCount = 0;
                } else if (sectionType == "Settings") {
                    // Settings structure section
                } else if (sectionType == "Values") {
                    // Values section
                } else if (sectionType == "Algorithm") {
                    // Algorithm settings section
                }
                continue;
            }
            
            // Parse key=value pairs
            if (line.contains("=")) {
                QStringList parts = line.split("=");
                if (parts.size() == 2) {
                    QString key = parts[0].trimmed();
                    QString value = parts[1].trimmed();
                    
                    if (currentSection == "Camera") {
                        if (key == "type") {
                            currentProfile.camera_type = value.toStdString();
                        }
                    } else if (currentSection == "Settings") {
                        // Parse settings structure
                        if (key == "settings_count") {
                            settingsCount = value.toInt();
                        } else if (key.startsWith("setting_")) {
                            QStringList keyParts = key.split("_");
                            if (keyParts.size() >= 3) {
                                int index = keyParts[1].toInt();
                                QString field = keyParts[2];
                                
                                if (tempSettings.find(index) == tempSettings.end()) {
                                    tempSettings[index] = CameraSetting();
                                }
                                
                                if (field == "name") {
                                    tempSettings[index].name = value.toStdString();
                                } else if (field == "display") {
                                    tempSettings[index].display_name = value.toStdString();
                                } else if (field == "type") {
                                    tempSettings[index].type = static_cast<SettingType>(value.toInt());
                                } else if (field == "min") {
                                    tempSettings[index].min_value = value.toInt();
                                } else if (field == "max") {
                                    tempSettings[index].max_value = value.toInt();
                                } else if (field == "default") {
                                    tempSettings[index].default_value = value.toInt();
                                } else if (field == "v4l2_id") {
                                    tempSettings[index].v4l2_id = value.toStdString();
                                }
                            }
                        }
                    } else if (currentSection == "Values") {
                        // Parse saved values
                        bool ok;
                        int intValue = value.toInt(&ok);
                        if (ok) {
                            currentProfile.saved_values[key.toStdString()] = intValue;
                        }
                    } else if (currentSection == "Algorithm") {
                        // Parse algorithm settings (doubles)
                        bool ok;
                        double doubleValue = value.toDouble(&ok);
                        if (ok) {
                            currentProfile.algorithm_settings[key.toStdString()] = doubleValue;
                        }
                    }
                }
            }
        }
        
        // Add last profile and reconstruct settings vector
        if (!currentProfile.camera_name.empty()) {
            // Reconstruct settings vector from tempSettings map
            for (int i = 0; i < settingsCount; i++) {
                if (tempSettings.find(i) != tempSettings.end()) {
                    // Apply saved value if available
                    if (currentProfile.saved_values.find(tempSettings[i].name) != currentProfile.saved_values.end()) {
                        tempSettings[i].current_value = currentProfile.saved_values[tempSettings[i].name];
                    } else {
                        tempSettings[i].current_value = tempSettings[i].default_value;
                    }
                    currentProfile.settings.push_back(tempSettings[i]);
                }
            }
            profiles[currentProfile.camera_name] = currentProfile;
        }
        
        file.close();
        return profiles;
    }
    
    // Load camera settings profile for a specific camera
    void loadCameraSettingsProfileForCamera(const string& cameraName, const string& cameraType, int v4l2Index = -1) {
        // First, try to load from saved profiles
        map<string, CameraSettingsProfile> allProfiles = loadAllCameraSettingsProfiles();
        
        if (allProfiles.find(cameraName) != allProfiles.end()) {
            // Load from saved profile
            currentCameraSettings_ = allProfiles[cameraName];
        } else {
            // Create new profile based on camera type
            if (cameraType == "MindVision") {
                // Try to use camera handle if available
                CameraHandle handle = 0;
                if (useMindVision_ && mvHandle_ != 0 && cameraName == cameraList_[selectedCameraIndex_]) {
                    handle = mvHandle_;
                }
                currentCameraSettings_ = defineMindVisionSettings(cameraName, handle);
            } else if (cameraType == "Arducam" && v4l2Index >= 0) {
                currentCameraSettings_ = queryArducamSettings(v4l2Index, cameraName);
            } else {
                // Default V4L2 camera (not Arducam)
                currentCameraSettings_ = CameraSettingsProfile();
                currentCameraSettings_.camera_name = cameraName;
                currentCameraSettings_.camera_type = "V4L2";
            }
        }
        
        // Store in profiles map
        cameraSettingsProfiles_[cameraName] = currentCameraSettings_;
    }
    
    // Apply saved values to camera settings profile
    void applySavedValuesToProfile() {
        map<string, CameraSettingsProfile> allProfiles = loadAllCameraSettingsProfiles();
        if (allProfiles.find(currentCameraSettings_.camera_name) != allProfiles.end()) {
            const CameraSettingsProfile& savedProfile = allProfiles[currentCameraSettings_.camera_name];
            // Apply saved values to current profile
            for (size_t i = 0; i < currentCameraSettings_.settings.size(); i++) {
                const string& settingName = currentCameraSettings_.settings[i].name;
                if (savedProfile.saved_values.find(settingName) != savedProfile.saved_values.end()) {
                    currentCameraSettings_.settings[i].current_value = savedProfile.saved_values.at(settingName);
                    currentCameraSettings_.saved_values[settingName] = savedProfile.saved_values.at(settingName);
                }
            }
        }
    }
    
    // Clear dynamic camera controls
    void clearDynamicCameraControls() {
        // Remove all widgets from layout
        if (dynamicSettingsLayout_) {
            QLayoutItem* item;
            while ((item = dynamicSettingsLayout_->takeAt(0)) != nullptr) {
                if (item->widget()) {
                    item->widget()->deleteLater();
                }
                delete item;
            }
        }
        
        // Clear maps
        dynamicControlWidgets_.clear();
        dynamicSliders_.clear();
        dynamicSpinBoxes_.clear();
        dynamicCheckBoxes_.clear();
        dynamicComboBoxes_.clear();
    }
    
    // Populate dynamic camera controls based on current camera settings profile
    void populateDynamicCameraControls() {
        // Clear existing dynamic controls
        clearDynamicCameraControls();
        
        if (currentCameraSettings_.camera_name.empty() || !dynamicSettingsLayout_) {
            return;
        }
        
        // For MindVision cameras, show dynamic controls for additional settings
        // (beyond the standard exposure, gain, brightness, contrast, saturation, sharpness, mode)
        // Standard settings are handled by fixed controls in the top camera group
        // Additional settings from SDK (analog_gain, digital_gain, gamma, ae_target, frame_speed) go here
        
        // List of standard settings that have fixed controls (don't show in dynamic area)
        set<string> standardSettings = {"exposure", "gain", "brightness", "contrast", "saturation", "sharpness", "mode_index"};
        
        // Check if there are additional settings beyond standard ones
        bool hasAdditionalSettings = false;
        for (const auto& setting : currentCameraSettings_.settings) {
            if (standardSettings.find(setting.name) == standardSettings.end()) {
                hasAdditionalSettings = true;
                break;
            }
        }
        
        // For MindVision: only show dynamic controls if there are additional settings
        // For V4L2/Arducam: always show dynamic controls
        if (useMindVision_ && !hasAdditionalSettings) {
            if (dynamicSettingsScrollArea_) {
                dynamicSettingsScrollArea_->setVisible(false);
            }
            return;
        }
        
        // Show scroll area
        if (dynamicSettingsScrollArea_) {
            dynamicSettingsScrollArea_->setVisible(true);
        }
        
        // Create controls for each setting in the profile
        for (const auto& setting : currentCameraSettings_.settings) {
            // Skip standard settings that already have fixed controls (for both MindVision and V4L2)
            if (standardSettings.find(setting.name) != standardSettings.end()) {
                continue;
            }
            
            int value = setting.current_value;
            if (currentCameraSettings_.saved_values.find(setting.name) != currentCameraSettings_.saved_values.end()) {
                value = currentCameraSettings_.saved_values.at(setting.name);
            }
            
            // Create label - parent must be dynamicSettingsWidget_ not this
            QLabel* label = new QLabel(QString::fromStdString(setting.display_name) + ":", dynamicSettingsWidget_);
            dynamicSettingsLayout_->addWidget(label);
            
            if (setting.type == SETTING_INT) {
                // Create slider and spin box - parent must be dynamicSettingsWidget_
                QSlider* slider = new QSlider(Qt::Horizontal, dynamicSettingsWidget_);
                slider->setRange(setting.min_value, setting.max_value);
                slider->setValue(value);
                slider->setMinimumWidth(100);
                
                QSpinBox* spinBox = new QSpinBox(dynamicSettingsWidget_);
                spinBox->setRange(setting.min_value, setting.max_value);
                spinBox->setValue(value);
                spinBox->setMinimumWidth(60);
                
                // Connect signals
                connect(slider, &QSlider::valueChanged, spinBox, &QSpinBox::setValue);
                connect(spinBox, QOverload<int>::of(&QSpinBox::valueChanged), slider, &QSlider::setValue);
                connect(slider, &QSlider::valueChanged, this, &AprilTagDebugGUI::updateCameraSettings);
                
                // Store references
                dynamicSliders_[setting.name] = slider;
                dynamicSpinBoxes_[setting.name] = spinBox;
                dynamicControlWidgets_[setting.name] = slider;
                
                // Add to layout
                dynamicSettingsLayout_->addWidget(slider);
                dynamicSettingsLayout_->addWidget(spinBox);
                dynamicSettingsLayout_->addSpacing(15);
                
            } else if (setting.type == SETTING_BOOL) {
                // Create checkbox with text - parent must be dynamicSettingsWidget_
                QCheckBox* checkBox = new QCheckBox(QString::fromStdString(setting.display_name), dynamicSettingsWidget_);
                checkBox->setChecked(value != 0);
                connect(checkBox, &QCheckBox::toggled, this, &AprilTagDebugGUI::updateCameraSettings);
                
                // Store reference
                dynamicCheckBoxes_[setting.name] = checkBox;
                dynamicControlWidgets_[setting.name] = checkBox;
                
                // Add to layout
                dynamicSettingsLayout_->addWidget(checkBox);
                dynamicSettingsLayout_->addSpacing(15);
                
            } else if (setting.type == SETTING_MENU) {
                // Create combo box - parent must be dynamicSettingsWidget_
                QComboBox* comboBox = new QComboBox(dynamicSettingsWidget_);
                // For menu types, add options from min to max
                for (int i = setting.min_value; i <= setting.max_value; i++) {
                    QString optionText = QString("Option %1").arg(i);
                    // Try to get menu text if available (would need to query v4l2-ctl for menu items)
                    comboBox->addItem(optionText, i);
                }
                comboBox->setCurrentIndex(value - setting.min_value);
                comboBox->setMinimumWidth(120);
                connect(comboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), 
                        this, &AprilTagDebugGUI::updateCameraSettings);
                
                // Store reference
                dynamicComboBoxes_[setting.name] = comboBox;
                dynamicControlWidgets_[setting.name] = comboBox;
                
                // Add to layout
                dynamicSettingsLayout_->addWidget(comboBox);
                dynamicSettingsLayout_->addSpacing(15);
            }
        }
        
        dynamicSettingsLayout_->addStretch();
        
        // Update widget size to ensure scroll area shows content
        if (dynamicSettingsWidget_) {
            dynamicSettingsWidget_->adjustSize();
            dynamicSettingsWidget_->updateGeometry();
            // Force minimum width so content is visible
            dynamicSettingsWidget_->setMinimumWidth(dynamicSettingsLayout_->sizeHint().width());
        }
        
        // Ensure scroll area is visible and shows content
        if (dynamicSettingsScrollArea_) {
            dynamicSettingsScrollArea_->updateGeometry();
            dynamicSettingsScrollArea_->ensureWidgetVisible(dynamicSettingsWidget_);
        }
    }
    
    // Apply camera settings from profile to camera and UI
    void applyCameraSettingsFromProfile() {
        if (currentCameraSettings_.camera_name.empty()) return;
        
        if (useMindVision_ && mvHandle_ != 0) {
#ifdef HAVE_MINDVISION_SDK
            // Apply saved settings to MindVision camera
            for (const auto& setting : currentCameraSettings_.settings) {
                int value = setting.current_value;
                if (currentCameraSettings_.saved_values.find(setting.name) != currentCameraSettings_.saved_values.end()) {
                    value = currentCameraSettings_.saved_values.at(setting.name);
                }
                
                if (setting.name == "exposure") {
                    double min_exposure = 1000.0;
                    double max_exposure = 100000.0;
                    double exposure = max_exposure - (value / 100.0) * (max_exposure - min_exposure);
                    CameraSetExposureTime(mvHandle_, exposure);
                    
                    // Update UI slider
                    if (exposureSlider_ && exposureSpin_) {
                        exposureSlider_->blockSignals(true);
                        exposureSlider_->setValue(value);
                        exposureSpin_->setValue(value);
                        exposureSlider_->blockSignals(false);
                    }
                } else if (setting.name == "gain") {
                    CameraSetGain(mvHandle_, value, value, value);
                    
                    // Update UI slider
                    if (gainSlider_ && gainSpin_) {
                        gainSlider_->blockSignals(true);
                        gainSlider_->setValue(value);
                        gainSpin_->setValue(value);
                        gainSlider_->blockSignals(false);
                    }
                } else if (setting.name == "brightness") {
                    INT analogGain = (value * 100) / 255;
                    CameraSetAnalogGain(mvHandle_, analogGain);
                    
                    // Update UI slider
                    if (brightnessSlider_ && brightnessSpin_) {
                        brightnessSlider_->blockSignals(true);
                        brightnessSlider_->setValue(value);
                        brightnessSpin_->setValue(value);
                        brightnessSlider_->blockSignals(false);
                    }
                } else if (setting.name == "contrast") {
                    CameraSetContrast(mvHandle_, value);
                    
                    // Update UI slider
                    if (contrastSlider_ && contrastSpin_) {
                        contrastSlider_->blockSignals(true);
                        contrastSlider_->setValue(value);
                        contrastSpin_->setValue(value);
                        contrastSlider_->blockSignals(false);
                    }
                } else if (setting.name == "saturation") {
                    CameraSetSaturation(mvHandle_, value);
                    
                    // Update UI slider
                    if (saturationSlider_ && saturationSpin_) {
                        saturationSlider_->blockSignals(true);
                        saturationSlider_->setValue(value);
                        saturationSpin_->setValue(value);
                        saturationSlider_->blockSignals(false);
                    }
                } else if (setting.name == "sharpness") {
                    CameraSetSharpness(mvHandle_, value);
                    
                    // Update UI slider
                    if (sharpnessSlider_ && sharpnessSpin_) {
                        sharpnessSlider_->blockSignals(true);
                        sharpnessSlider_->setValue(value);
                        sharpnessSpin_->setValue(value);
                        sharpnessSlider_->blockSignals(false);
                    }
                }
            }
#endif
        } else if (cameraCap_.isOpened()) {
            // Apply saved settings to V4L2/Arducam camera using v4l2-ctl
            QString devicePath;
            int v4l2Index = -1;
            
            // Find V4L2 index
            for (int i = 0; i < selectedCameraIndex_; i++) {
                if (!isMindVision_[i]) v4l2Index++;
            }
            
            if (v4l2Index >= 0) {
                devicePath = QString("/dev/video%1").arg(v4l2Index);
                
                // Apply each setting using v4l2-ctl
                for (const auto& setting : currentCameraSettings_.settings) {
                    int value = setting.current_value;
                    if (currentCameraSettings_.saved_values.find(setting.name) != currentCameraSettings_.saved_values.end()) {
                        value = currentCameraSettings_.saved_values.at(setting.name);
                    }
                    
                    // Skip mode_index (handled separately)
                    if (setting.name == "mode_index") continue;
                    
                    // Apply using v4l2-ctl
                    if (!setting.v4l2_id.empty()) {
                        QProcess v4l2Process;
                        v4l2Process.start("v4l2-ctl", QStringList() << "--device" << devicePath 
                                         << "--set-ctrl" << QString("%1=%2").arg(QString::fromStdString(setting.v4l2_id)).arg(value));
                        v4l2Process.waitForFinished(1000);
                    }
                    
                    // Update UI sliders for standard settings
                    if (setting.name == "exposure" && exposureSlider_ && exposureSpin_) {
                        exposureSlider_->blockSignals(true);
                        exposureSlider_->setValue(value);
                        exposureSpin_->setValue(value);
                        exposureSlider_->blockSignals(false);
                    } else if (setting.name == "gain" && gainSlider_ && gainSpin_) {
                        gainSlider_->blockSignals(true);
                        gainSlider_->setValue(value);
                        gainSpin_->setValue(value);
                        gainSlider_->blockSignals(false);
                    } else if (setting.name == "brightness" && brightnessSlider_ && brightnessSpin_) {
                        brightnessSlider_->blockSignals(true);
                        brightnessSlider_->setValue(value);
                        brightnessSpin_->setValue(value);
                        brightnessSlider_->blockSignals(false);
                    } else if (setting.name == "contrast" && contrastSlider_ && contrastSpin_) {
                        contrastSlider_->blockSignals(true);
                        contrastSlider_->setValue(value);
                        contrastSpin_->setValue(value);
                        contrastSlider_->blockSignals(false);
                    } else if (setting.name == "saturation" && saturationSlider_ && saturationSpin_) {
                        saturationSlider_->blockSignals(true);
                        saturationSlider_->setValue(value);
                        saturationSpin_->setValue(value);
                        saturationSlider_->blockSignals(false);
                    } else if (setting.name == "sharpness" && sharpnessSlider_ && sharpnessSpin_) {
                        sharpnessSlider_->blockSignals(true);
                        sharpnessSlider_->setValue(value);
                        sharpnessSpin_->setValue(value);
                        sharpnessSlider_->blockSignals(false);
                    } else {
                        // Handle dynamic controls
                        if (dynamicSliders_.find(setting.name) != dynamicSliders_.end()) {
                            QSlider* slider = dynamicSliders_[setting.name];
                            QSpinBox* spinBox = dynamicSpinBoxes_[setting.name];
                            if (slider && spinBox) {
                                slider->blockSignals(true);
                                slider->setRange(setting.min_value, setting.max_value);
                                slider->setValue(value);
                                spinBox->setRange(setting.min_value, setting.max_value);
                                spinBox->setValue(value);
                                slider->blockSignals(false);
                            }
                        } else if (dynamicCheckBoxes_.find(setting.name) != dynamicCheckBoxes_.end()) {
                            QCheckBox* checkBox = dynamicCheckBoxes_[setting.name];
                            if (checkBox) {
                                checkBox->blockSignals(true);
                                checkBox->setChecked(value != 0);
                                checkBox->blockSignals(false);
                            }
                        } else if (dynamicComboBoxes_.find(setting.name) != dynamicComboBoxes_.end()) {
                            QComboBox* comboBox = dynamicComboBoxes_[setting.name];
                            if (comboBox) {
                                comboBox->blockSignals(true);
                                int index = value - setting.min_value;
                                if (index >= 0 && index < comboBox->count()) {
                                    comboBox->setCurrentIndex(index);
                                }
                                comboBox->blockSignals(false);
                            }
                        }
                    }
                }
            }
        }
    }
    
    void saveCameraSettings() {
        if (!cameraOpen_ || cameraList_.empty() || selectedCameraIndex_ < 0) return;
        
        string cameraName = cameraList_[selectedCameraIndex_];
        bool isMindVision = useMindVision_;
        
        // Get or create camera settings profile
        CameraSettingsProfile profile;
        if (!currentCameraSettings_.camera_name.empty() && currentCameraSettings_.camera_name == cameraName) {
            // Use current profile
            profile = currentCameraSettings_;
        } else {
            // Load from config file first
            map<string, CameraSettingsProfile> allProfiles = loadAllCameraSettingsProfiles();
            if (allProfiles.find(cameraName) != allProfiles.end()) {
                profile = allProfiles[cameraName];
            } else {
                // Create new profile based on camera type
                if (isMindVision) {
                    // Try to use camera handle if available
                    CameraHandle handle = 0;
                    if (useMindVision_ && mvHandle_ != 0 && cameraName == cameraList_[selectedCameraIndex_]) {
                        handle = mvHandle_;
                    }
                    profile = defineMindVisionSettings(cameraName, handle);
                } else {
                    profile.camera_name = cameraName;
                    profile.camera_type = "V4L2";
                }
            }
        }
        
        // Update saved values from UI controls
        if (exposureSlider_) profile.saved_values["exposure"] = exposureSlider_->value();
        if (gainSlider_) profile.saved_values["gain"] = gainSlider_->value();
        if (brightnessSlider_) profile.saved_values["brightness"] = brightnessSlider_->value();
        if (contrastSlider_) profile.saved_values["contrast"] = contrastSlider_->value();
        if (saturationSlider_) profile.saved_values["saturation"] = saturationSlider_->value();
        if (sharpnessSlider_) profile.saved_values["sharpness"] = sharpnessSlider_->value();
        if (modeCombo_ && modeCombo_->currentIndex() >= 0) {
            profile.saved_values["mode_index"] = modeCombo_->currentIndex();
        }
        
        // Update dynamic settings from currentCameraSettings_ if available
        if (!currentCameraSettings_.camera_name.empty() && currentCameraSettings_.camera_name == cameraName) {
            for (const auto& pair : currentCameraSettings_.saved_values) {
                // Only add if not already updated from UI
                if (profile.saved_values.find(pair.first) == profile.saved_values.end()) {
                    profile.saved_values[pair.first] = pair.second;
                }
            }
            // Preserve settings structure
            if (profile.settings.empty() && !currentCameraSettings_.settings.empty()) {
                profile.settings = currentCameraSettings_.settings;
            }
        }
        
        // Update current values in settings structure
        for (size_t i = 0; i < profile.settings.size(); i++) {
            const string& settingName = profile.settings[i].name;
            if (profile.saved_values.find(settingName) != profile.saved_values.end()) {
                profile.settings[i].current_value = profile.saved_values[settingName];
            }
        }
        
        // Ensure camera name and type are set correctly
        profile.camera_name = cameraName;
        profile.camera_type = isMindVision ? "MindVision" : "V4L2";
        
        // Save the profile (saves both structure and values per camera)
        saveCameraSettingsProfile(profile);
        
        // Update currentCameraSettings_
        currentCameraSettings_ = profile;
        cameraSettingsProfiles_[cameraName] = profile;
        
        QMessageBox::information(this, "Camera Settings", 
            QString("Camera settings saved for:\n%1\n\nSaved to: camera_settings.txt").arg(QString::fromStdString(cameraName)));
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
                                // Apply the mode based on camera type
                                if (useMindVision_) {
                                    applyMVMode(intValue);
                                } else {
                                    applyMode(intValue);
                                }
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
    
    void onSettingsCameraChanged(int index) {
        if (index < 0 || !settingsCameraCombo_) return;
        
        // Map settings camera index to actual camera list index (skip "None" at index 0)
        int actualCameraIndex = index + 1;  // +1 to skip "None"
        if (actualCameraIndex < 0 || actualCameraIndex >= (int)cameraList_.size()) {
            return;
        }
        
        // Always load from config file first (per camera settings)
        // This ensures we get the saved values for this specific camera
        loadCameraSettingsForSettingsTab(actualCameraIndex);
    }
    
    void loadCameraSettingsForSettingsTab(int cameraIndex) {
        if (cameraIndex < 0 || cameraIndex >= (int)cameraList_.size()) return;
        
        string cameraName = cameraList_[cameraIndex];
        bool isMindVision = isMindVision_[cameraIndex];
        
        // Load camera settings profile from config file (per camera)
        // This loads both the settings structure AND the saved values
        map<string, CameraSettingsProfile> allProfiles = loadAllCameraSettingsProfiles();
        
        CameraSettingsProfile profile;
        if (allProfiles.find(cameraName) != allProfiles.end()) {
            // Load existing profile from config file (includes both structure and values)
            profile = allProfiles[cameraName];
            qDebug() << "Loaded profile for camera:" << QString::fromStdString(cameraName) 
                     << "with" << profile.settings.size() << "settings and" 
                     << profile.saved_values.size() << "saved values";
        } else {
            // Create new profile based on camera type
            if (isMindVision) {
                // Try to use camera handle if available
                CameraHandle handle = 0;
                if (useMindVision_ && mvHandle_ != 0 && cameraName == cameraList_[selectedCameraIndex_]) {
                    handle = mvHandle_;
                }
                profile = defineMindVisionSettings(cameraName, handle);
            } else {
                profile.camera_name = cameraName;
                profile.camera_type = "V4L2";
                // For Arducam, try to use currentCameraSettings_ if available
                if (!currentCameraSettings_.camera_name.empty() && currentCameraSettings_.camera_name == cameraName) {
                    profile = currentCameraSettings_;
                }
            }
        }
        
        // Store in profiles map
        cameraSettingsProfiles_[cameraName] = profile;
        currentCameraSettings_ = profile;  // Update current settings
        
        // Populate UI with settings values from saved_values
        // First, apply saved values directly (these come from config file)
        // For standard settings, always try to load from saved_values
        if (profile.saved_values.find("exposure") != profile.saved_values.end() && settingsExposureSlider_ && settingsExposureSpin_) {
            int value = profile.saved_values.at("exposure");
            settingsExposureSlider_->blockSignals(true);
            settingsExposureSlider_->setValue(value);
            settingsExposureSpin_->setValue(value);
            settingsExposureSlider_->blockSignals(false);
        }
        if (profile.saved_values.find("gain") != profile.saved_values.end() && settingsGainSlider_ && settingsGainSpin_) {
            int value = profile.saved_values.at("gain");
            settingsGainSlider_->blockSignals(true);
            settingsGainSlider_->setValue(value);
            settingsGainSpin_->setValue(value);
            settingsGainSlider_->blockSignals(false);
        }
        if (profile.saved_values.find("brightness") != profile.saved_values.end() && settingsBrightnessSlider_ && settingsBrightnessSpin_) {
            int value = profile.saved_values.at("brightness");
            settingsBrightnessSlider_->blockSignals(true);
            settingsBrightnessSlider_->setValue(value);
            settingsBrightnessSpin_->setValue(value);
            settingsBrightnessSlider_->blockSignals(false);
        }
        if (profile.saved_values.find("contrast") != profile.saved_values.end() && settingsContrastSlider_ && settingsContrastSpin_) {
            int value = profile.saved_values.at("contrast");
            settingsContrastSlider_->blockSignals(true);
            settingsContrastSlider_->setValue(value);
            settingsContrastSpin_->setValue(value);
            settingsContrastSlider_->blockSignals(false);
        }
        if (profile.saved_values.find("saturation") != profile.saved_values.end() && settingsSaturationSlider_ && settingsSaturationSpin_) {
            int value = profile.saved_values.at("saturation");
            settingsSaturationSlider_->blockSignals(true);
            settingsSaturationSlider_->setValue(value);
            settingsSaturationSpin_->setValue(value);
            settingsSaturationSlider_->blockSignals(false);
        }
        if (profile.saved_values.find("sharpness") != profile.saved_values.end() && settingsSharpnessSlider_ && settingsSharpnessSpin_) {
            int value = profile.saved_values.at("sharpness");
            settingsSharpnessSlider_->blockSignals(true);
            settingsSharpnessSlider_->setValue(value);
            settingsSharpnessSpin_->setValue(value);
            settingsSharpnessSlider_->blockSignals(false);
        }
        if (profile.saved_values.find("mode_index") != profile.saved_values.end() && settingsModeCombo_) {
            int value = profile.saved_values.at("mode_index");
            if (value >= 0 && value < settingsModeCombo_->count()) {
                settingsModeCombo_->blockSignals(true);
                settingsModeCombo_->setCurrentIndex(value);
                settingsModeCombo_->blockSignals(false);
            }
        }
        
        // Also update ranges from settings structure if available
        // For MindVision, use standard UI elements with ranges from structure
        if (isMindVision) {
            for (const auto& setting : profile.settings) {
                int value = setting.current_value;
                if (profile.saved_values.find(setting.name) != profile.saved_values.end()) {
                    value = profile.saved_values.at(setting.name);
                }
                
                if (setting.name == "exposure" && settingsExposureSlider_ && settingsExposureSpin_) {
                    settingsExposureSlider_->blockSignals(true);
                    settingsExposureSlider_->setRange(setting.min_value, setting.max_value);
                    settingsExposureSlider_->setValue(value);
                    settingsExposureSpin_->setRange(setting.min_value, setting.max_value);
                    settingsExposureSpin_->setValue(value);
                    settingsExposureSlider_->blockSignals(false);
                } else if (setting.name == "gain" && settingsGainSlider_ && settingsGainSpin_) {
                    settingsGainSlider_->blockSignals(true);
                    settingsGainSlider_->setRange(setting.min_value, setting.max_value);
                    settingsGainSlider_->setValue(value);
                    settingsGainSpin_->setRange(setting.min_value, setting.max_value);
                    settingsGainSpin_->setValue(value);
                    settingsGainSlider_->blockSignals(false);
                } else if (setting.name == "brightness" && settingsBrightnessSlider_ && settingsBrightnessSpin_) {
                    settingsBrightnessSlider_->blockSignals(true);
                    settingsBrightnessSlider_->setRange(setting.min_value, setting.max_value);
                    settingsBrightnessSlider_->setValue(value);
                    settingsBrightnessSpin_->setRange(setting.min_value, setting.max_value);
                    settingsBrightnessSpin_->setValue(value);
                    settingsBrightnessSlider_->blockSignals(false);
                } else if (setting.name == "contrast" && settingsContrastSlider_ && settingsContrastSpin_) {
                    settingsContrastSlider_->blockSignals(true);
                    settingsContrastSlider_->setRange(setting.min_value, setting.max_value);
                    settingsContrastSlider_->setValue(value);
                    settingsContrastSpin_->setRange(setting.min_value, setting.max_value);
                    settingsContrastSpin_->setValue(value);
                    settingsContrastSlider_->blockSignals(false);
                } else if (setting.name == "saturation" && settingsSaturationSlider_ && settingsSaturationSpin_) {
                    settingsSaturationSlider_->blockSignals(true);
                    settingsSaturationSlider_->setRange(setting.min_value, setting.max_value);
                    settingsSaturationSlider_->setValue(value);
                    settingsSaturationSpin_->setRange(setting.min_value, setting.max_value);
                    settingsSaturationSpin_->setValue(value);
                    settingsSaturationSlider_->blockSignals(false);
                } else if (setting.name == "sharpness" && settingsSharpnessSlider_ && settingsSharpnessSpin_) {
                    settingsSharpnessSlider_->blockSignals(true);
                    settingsSharpnessSlider_->setRange(setting.min_value, setting.max_value);
                    settingsSharpnessSlider_->setValue(value);
                    settingsSharpnessSpin_->setRange(setting.min_value, setting.max_value);
                    settingsSharpnessSpin_->setValue(value);
                    settingsSharpnessSlider_->blockSignals(false);
                } else if (setting.name == "mode_index" && settingsModeCombo_) {
                    if (value >= 0 && value < settingsModeCombo_->count()) {
                        settingsModeCombo_->blockSignals(true);
                        settingsModeCombo_->setCurrentIndex(value);
                        settingsModeCombo_->blockSignals(false);
                    }
                }
            }
        } else {
            // For Arducam/V4L2, load settings from profile
            // Load standard settings that have UI controls
            for (const auto& setting : profile.settings) {
                int value = setting.current_value;
                if (profile.saved_values.find(setting.name) != profile.saved_values.end()) {
                    value = profile.saved_values.at(setting.name);
                }
                
                if (setting.name == "exposure" && settingsExposureSlider_ && settingsExposureSpin_) {
                    settingsExposureSlider_->blockSignals(true);
                    settingsExposureSlider_->setRange(setting.min_value, setting.max_value);
                    settingsExposureSlider_->setValue(value);
                    settingsExposureSpin_->setRange(setting.min_value, setting.max_value);
                    settingsExposureSpin_->setValue(value);
                    settingsExposureSlider_->blockSignals(false);
                } else if (setting.name == "gain" && settingsGainSlider_ && settingsGainSpin_) {
                    settingsGainSlider_->blockSignals(true);
                    settingsGainSlider_->setRange(setting.min_value, setting.max_value);
                    settingsGainSlider_->setValue(value);
                    settingsGainSpin_->setRange(setting.min_value, setting.max_value);
                    settingsGainSpin_->setValue(value);
                    settingsGainSlider_->blockSignals(false);
                } else if (setting.name == "brightness" && settingsBrightnessSlider_ && settingsBrightnessSpin_) {
                    settingsBrightnessSlider_->blockSignals(true);
                    settingsBrightnessSlider_->setRange(setting.min_value, setting.max_value);
                    settingsBrightnessSlider_->setValue(value);
                    settingsBrightnessSpin_->setRange(setting.min_value, setting.max_value);
                    settingsBrightnessSpin_->setValue(value);
                    settingsBrightnessSlider_->blockSignals(false);
                } else if (setting.name == "contrast" && settingsContrastSlider_ && settingsContrastSpin_) {
                    settingsContrastSlider_->blockSignals(true);
                    settingsContrastSlider_->setRange(setting.min_value, setting.max_value);
                    settingsContrastSlider_->setValue(value);
                    settingsContrastSpin_->setRange(setting.min_value, setting.max_value);
                    settingsContrastSpin_->setValue(value);
                    settingsContrastSlider_->blockSignals(false);
                } else if (setting.name == "saturation" && settingsSaturationSlider_ && settingsSaturationSpin_) {
                    settingsSaturationSlider_->blockSignals(true);
                    settingsSaturationSlider_->setRange(setting.min_value, setting.max_value);
                    settingsSaturationSlider_->setValue(value);
                    settingsSaturationSpin_->setRange(setting.min_value, setting.max_value);
                    settingsSaturationSpin_->setValue(value);
                    settingsSaturationSlider_->blockSignals(false);
                } else if (setting.name == "sharpness" && settingsSharpnessSlider_ && settingsSharpnessSpin_) {
                    settingsSharpnessSlider_->blockSignals(true);
                    settingsSharpnessSlider_->setRange(setting.min_value, setting.max_value);
                    settingsSharpnessSlider_->setValue(value);
                    settingsSharpnessSpin_->setRange(setting.min_value, setting.max_value);
                    settingsSharpnessSpin_->setValue(value);
                    settingsSharpnessSlider_->blockSignals(false);
                } else if (setting.name == "mode_index" && settingsModeCombo_) {
                    if (value >= 0 && value < settingsModeCombo_->count()) {
                        settingsModeCombo_->blockSignals(true);
                        settingsModeCombo_->setCurrentIndex(value);
                        settingsModeCombo_->blockSignals(false);
                    }
                }
            }
            
            // Store the full profile for saving (includes all dynamic settings)
            currentCameraSettings_ = profile;
        }
    }
    
    void saveCameraSettingsFromSettingsTab() {
        if (!settingsCameraCombo_ || settingsCameraCombo_->currentIndex() < 0) return;
        
        // Map settings camera index to actual camera list index (skip "None" at index 0)
        int actualCameraIndex = settingsCameraCombo_->currentIndex() + 1;  // +1 to skip "None"
        if (actualCameraIndex < 0 || actualCameraIndex >= (int)cameraList_.size()) {
            return;
        }
        
        string cameraName = cameraList_[actualCameraIndex];
        bool isMindVision = isMindVision_[actualCameraIndex];
        
        // Always load from config file first to get per-camera settings
        map<string, CameraSettingsProfile> allProfiles = loadAllCameraSettingsProfiles();
        
        // Get or create camera settings profile
        CameraSettingsProfile profile;
        if (allProfiles.find(cameraName) != allProfiles.end()) {
            // Load existing profile from config file (per camera)
            profile = allProfiles[cameraName];
        } else if (cameraSettingsProfiles_.find(cameraName) != cameraSettingsProfiles_.end()) {
            // Use cached profile if available
            profile = cameraSettingsProfiles_[cameraName];
        } else {
            // Create new profile based on camera type
            if (isMindVision) {
                // Try to use camera handle if available
                CameraHandle handle = 0;
                if (useMindVision_ && mvHandle_ != 0 && cameraName == cameraList_[selectedCameraIndex_]) {
                    handle = mvHandle_;
                }
                profile = defineMindVisionSettings(cameraName, handle);
            } else {
                profile.camera_name = cameraName;
                profile.camera_type = "V4L2";
                // For Arducam, try to use currentCameraSettings_ if available
                if (!currentCameraSettings_.camera_name.empty() && currentCameraSettings_.camera_name == cameraName) {
                    profile = currentCameraSettings_;
                }
            }
        }
        
        // Update saved values from UI
        if (isMindVision) {
            // MindVision settings
            if (settingsExposureSlider_) profile.saved_values["exposure"] = settingsExposureSlider_->value();
            if (settingsGainSlider_) profile.saved_values["gain"] = settingsGainSlider_->value();
            if (settingsBrightnessSlider_) profile.saved_values["brightness"] = settingsBrightnessSlider_->value();
            if (settingsContrastSlider_) profile.saved_values["contrast"] = settingsContrastSlider_->value();
            if (settingsSaturationSlider_) profile.saved_values["saturation"] = settingsSaturationSlider_->value();
            if (settingsSharpnessSlider_) profile.saved_values["sharpness"] = settingsSharpnessSlider_->value();
            if (settingsModeCombo_ && settingsModeCombo_->currentIndex() >= 0) {
                profile.saved_values["mode_index"] = settingsModeCombo_->currentIndex();
            }
        } else {
            // For Arducam/V4L2, update values from UI and currentCameraSettings_
            if (!currentCameraSettings_.camera_name.empty() && currentCameraSettings_.camera_name == cameraName) {
                profile = currentCameraSettings_;
            }
            
            // Update standard settings from UI
            if (settingsExposureSlider_) profile.saved_values["exposure"] = settingsExposureSlider_->value();
            if (settingsGainSlider_) profile.saved_values["gain"] = settingsGainSlider_->value();
            if (settingsBrightnessSlider_) profile.saved_values["brightness"] = settingsBrightnessSlider_->value();
            if (settingsContrastSlider_) profile.saved_values["contrast"] = settingsContrastSlider_->value();
            if (settingsSaturationSlider_) profile.saved_values["saturation"] = settingsSaturationSlider_->value();
            if (settingsSharpnessSlider_) profile.saved_values["sharpness"] = settingsSharpnessSlider_->value();
            if (settingsModeCombo_ && settingsModeCombo_->currentIndex() >= 0) {
                profile.saved_values["mode_index"] = settingsModeCombo_->currentIndex();
            }
            
            // Update all other settings from currentCameraSettings_ saved_values (preserves dynamic settings)
            for (const auto& pair : currentCameraSettings_.saved_values) {
                // Only add if not already updated from UI
                if (profile.saved_values.find(pair.first) == profile.saved_values.end()) {
                    profile.saved_values[pair.first] = pair.second;
                }
            }
        }
        
        // Update current values in settings structure
        for (size_t i = 0; i < profile.settings.size(); i++) {
            const string& settingName = profile.settings[i].name;
            if (profile.saved_values.find(settingName) != profile.saved_values.end()) {
                profile.settings[i].current_value = profile.saved_values[settingName];
            }
        }
        
        // Ensure camera name is set correctly (per camera)
        profile.camera_name = cameraName;
        profile.camera_type = isMindVision ? "MindVision" : "V4L2";
        
        // Save the profile (saves both structure and values per camera)
        // This function saves all cameras, preserving other cameras' settings
        saveCameraSettingsProfile(profile);
        
        // Update the profiles map
        cameraSettingsProfiles_[cameraName] = profile;
        
        // Update currentCameraSettings_ if this is the currently selected camera
        if (currentCameraSettings_.camera_name == cameraName) {
            currentCameraSettings_ = profile;
        }
        
        QMessageBox::information(this, "Camera Settings", 
            QString("Camera settings saved for:\n%1\n\nSaved to: camera_settings.txt").arg(QString::fromStdString(cameraName)));
    }
    
    // Get current algorithm parameter values from UI (shared by all algorithms)
    void getCurrentAlgorithmParameters(double& quad_decimate, double& quad_sigma, bool& refine_edges,
                                       double& decode_sharpening, int& nthreads,
                                       int& min_cluster_pixels, double& max_line_fit_mse,
                                       double& critical_angle_degrees, int& min_white_black_diff) {
        quad_decimate = quadDecimateSpin_ ? quadDecimateSpin_->value() : 2.0;
        quad_sigma = quadSigmaSpin_ ? quadSigmaSpin_->value() : 0.0;
        refine_edges = refineEdgesCheck_ ? refineEdgesCheck_->isChecked() : true;
        decode_sharpening = decodeSharpeningSpin_ ? decodeSharpeningSpin_->value() : 0.5;
        nthreads = nthreadsSpin_ ? nthreadsSpin_->value() : 4;
        min_cluster_pixels = minClusterPixelsSpin_ ? minClusterPixelsSpin_->value() : 4;
        max_line_fit_mse = maxLineFitMseSpin_ ? maxLineFitMseSpin_->value() : 12.0;
        critical_angle_degrees = criticalAngleSpin_ ? criticalAngleSpin_->value() : 10.0;
        min_white_black_diff = minWhiteBlackDiffSpin_ ? minWhiteBlackDiffSpin_->value() : 4;
    }
    
    // Apply algorithm settings to a specific algorithm instance
    void applyAlgorithmSettingsToAlgorithm(AprilTagAlgorithm* algorithm) {
        if (!algorithm) return;
        
        double quad_decimate, quad_sigma, decode_sharpening, max_line_fit_mse, critical_angle_degrees;
        bool refine_edges;
        int nthreads, min_cluster_pixels, min_white_black_diff;
        
        getCurrentAlgorithmParameters(quad_decimate, quad_sigma, refine_edges, decode_sharpening, nthreads,
                                     min_cluster_pixels, max_line_fit_mse, critical_angle_degrees, min_white_black_diff);
        
        algorithm->updateDetectorParameters(
            quad_decimate, quad_sigma, refine_edges, decode_sharpening, nthreads,
            min_cluster_pixels, max_line_fit_mse, critical_angle_degrees, min_white_black_diff
        );
    }
    
    // Apply algorithm settings to detector and processing functions
    void applyAlgorithmSettings() {
        // Get parameter values from UI (shared by all algorithms)
        double quad_decimate, quad_sigma, decode_sharpening, max_line_fit_mse, critical_angle_degrees;
        bool refine_edges;
        int nthreads, min_cluster_pixels, min_white_black_diff;
        
        getCurrentAlgorithmParameters(quad_decimate, quad_sigma, refine_edges, decode_sharpening, nthreads,
                                     min_cluster_pixels, max_line_fit_mse, critical_angle_degrees, min_white_black_diff);
        
        // Apply to CPU detector (for Capture tab)
        if (td_) {
            td_->quad_decimate = quad_decimate;
            td_->quad_sigma = quad_sigma;
            td_->refine_edges = refine_edges ? 1 : 0;
            td_->decode_sharpening = decode_sharpening;
            td_->nthreads = nthreads;
            // Recreate worker pool with new thread count
            if (td_->wp) {
                workerpool_destroy(td_->wp);
            }
            td_->wp = workerpool_create(nthreads);
            
            // Apply quad threshold parameters
            td_->qtp.min_cluster_pixels = min_cluster_pixels;
            td_->qtp.max_line_fit_mse = max_line_fit_mse;
            td_->qtp.cos_critical_rad = cos(critical_angle_degrees * M_PI / 180.0);
            td_->qtp.min_white_black_diff = min_white_black_diff;
        }
        
        // Apply to all algorithms (both CPU and Fast AprilTag) - use same parameters
        // Apply to current algorithm (if running in Processing tab)
        if (currentAlgorithm_) {
            applyAlgorithmSettingsToAlgorithm(currentAlgorithm_.get());
        }
        
        // Apply to capture algorithm (if running in Capture tab)
        if (captureAlgorithm_) {
            applyAlgorithmSettingsToAlgorithm(captureAlgorithm_.get());
        }
        
        // Save algorithm settings to current camera's profile
        saveAlgorithmSettingsToCamera();
        
        QMessageBox::information(this, "Algorithm Settings", 
            "Algorithm settings applied successfully!\n\nSettings saved for current camera.\nChanges will take effect for new detections.");
    }
    
    // Save algorithm settings to current camera's profile
    void saveAlgorithmSettingsToCamera() {
        if (currentCameraSettings_.camera_name.empty()) {
            qDebug() << "No camera selected, cannot save algorithm settings";
            return;
        }
        
        // Clear existing algorithm settings
        currentCameraSettings_.algorithm_settings.clear();
        
        // Save algorithm tuning settings to current camera profile
        // Preprocessing
        if (preprocessHistEqCheck_) {
            currentCameraSettings_.algorithm_settings["histogram_equalization"] = preprocessHistEqCheck_->isChecked() ? 1.0 : 0.0;
        }
        if (preprocessClaheClipSpin_) {
            currentCameraSettings_.algorithm_settings["clahe_clip"] = preprocessClaheClipSpin_->value();
        }
        if (preprocessGammaSpin_) {
            currentCameraSettings_.algorithm_settings["gamma"] = preprocessGammaSpin_->value();
        }
        if (preprocessContrastSpin_) {
            currentCameraSettings_.algorithm_settings["contrast_multiplier"] = preprocessContrastSpin_->value();
        }
        
        // Edge Detection
        if (cannyLowSpin_) {
            currentCameraSettings_.algorithm_settings["canny_low"] = cannyLowSpin_->value();
        }
        if (cannyHighSpin_) {
            currentCameraSettings_.algorithm_settings["canny_high"] = cannyHighSpin_->value();
        }
        if (adaptiveThreshBlockSpin_) {
            currentCameraSettings_.algorithm_settings["adaptive_thresh_block"] = adaptiveThreshBlockSpin_->value();
        }
        if (adaptiveThreshConstantSpin_) {
            currentCameraSettings_.algorithm_settings["adaptive_thresh_constant"] = adaptiveThreshConstantSpin_->value();
        }
        
        // Detection Parameters
        if (quadDecimateSpin_) {
            currentCameraSettings_.algorithm_settings["quad_decimate"] = quadDecimateSpin_->value();
        }
        if (quadSigmaSpin_) {
            currentCameraSettings_.algorithm_settings["quad_sigma"] = quadSigmaSpin_->value();
        }
        if (refineEdgesCheck_) {
            currentCameraSettings_.algorithm_settings["refine_edges"] = refineEdgesCheck_->isChecked() ? 1.0 : 0.0;
        }
        if (decodeSharpeningSpin_) {
            currentCameraSettings_.algorithm_settings["decode_sharpening"] = decodeSharpeningSpin_->value();
        }
        if (nthreadsSpin_) {
            currentCameraSettings_.algorithm_settings["nthreads"] = nthreadsSpin_->value();
        }
        
        // Quad Threshold Parameters
        if (minClusterPixelsSpin_) {
            currentCameraSettings_.algorithm_settings["min_cluster_pixels"] = minClusterPixelsSpin_->value();
        }
        if (maxLineFitMseSpin_) {
            currentCameraSettings_.algorithm_settings["max_line_fit_mse"] = maxLineFitMseSpin_->value();
        }
        if (criticalAngleSpin_) {
            currentCameraSettings_.algorithm_settings["critical_angle_degrees"] = criticalAngleSpin_->value();
        }
        if (minWhiteBlackDiffSpin_) {
            currentCameraSettings_.algorithm_settings["min_white_black_diff"] = minWhiteBlackDiffSpin_->value();
        }
        
        // Advanced Parameters
        if (cornerRefineWinSizeSpin_) {
            currentCameraSettings_.algorithm_settings["corner_refine_win_size"] = cornerRefineWinSizeSpin_->value();
        }
        if (cornerRefineMaxIterSpin_) {
            currentCameraSettings_.algorithm_settings["corner_refine_max_iter"] = cornerRefineMaxIterSpin_->value();
        }
        if (patternBorderSizeSpin_) {
            currentCameraSettings_.algorithm_settings["pattern_border_size"] = patternBorderSizeSpin_->value();
        }
        if (tagMinAreaSpin_) {
            currentCameraSettings_.algorithm_settings["tag_min_area"] = tagMinAreaSpin_->value();
        }
        if (tagMaxAreaSpin_) {
            currentCameraSettings_.algorithm_settings["tag_max_area"] = tagMaxAreaSpin_->value();
        }
        
        // Save to camera_settings.txt file
        saveCameraSettingsProfile(currentCameraSettings_);
        
        qDebug() << "Algorithm settings saved for camera:" << QString::fromStdString(currentCameraSettings_.camera_name);
    }
    
    // Load algorithm settings for current camera from camera profile
    void loadAlgorithmSettingsForCamera() {
        if (currentCameraSettings_.camera_name.empty()) {
            return;
        }
        
        // Load algorithm settings from current camera's profile
        const map<string, double>& algoSettings = currentCameraSettings_.algorithm_settings;
        
        if (algoSettings.empty()) {
            // No saved settings for this camera, reset to default values
            resetAlgorithmSettingsToDefaults();
            return;
        }
        
        // Preprocessing
        if (algoSettings.find("histogram_equalization") != algoSettings.end() && preprocessHistEqCheck_) {
            preprocessHistEqCheck_->setChecked(algoSettings.at("histogram_equalization") != 0.0);
        }
        if (algoSettings.find("clahe_clip") != algoSettings.end() && preprocessClaheClipSpin_) {
            preprocessClaheClipSpin_->setValue(algoSettings.at("clahe_clip"));
        }
        if (algoSettings.find("gamma") != algoSettings.end() && preprocessGammaSpin_) {
            preprocessGammaSpin_->setValue(algoSettings.at("gamma"));
        }
        if (algoSettings.find("contrast_multiplier") != algoSettings.end() && preprocessContrastSpin_) {
            preprocessContrastSpin_->setValue(algoSettings.at("contrast_multiplier"));
        }
        
        // Edge Detection
        if (algoSettings.find("canny_low") != algoSettings.end() && cannyLowSpin_) {
            cannyLowSpin_->setValue(static_cast<int>(algoSettings.at("canny_low")));
        }
        if (algoSettings.find("canny_high") != algoSettings.end() && cannyHighSpin_) {
            cannyHighSpin_->setValue(static_cast<int>(algoSettings.at("canny_high")));
        }
        if (algoSettings.find("adaptive_thresh_block") != algoSettings.end() && adaptiveThreshBlockSpin_) {
            adaptiveThreshBlockSpin_->setValue(static_cast<int>(algoSettings.at("adaptive_thresh_block")));
        }
        if (algoSettings.find("adaptive_thresh_constant") != algoSettings.end() && adaptiveThreshConstantSpin_) {
            adaptiveThreshConstantSpin_->setValue(static_cast<int>(algoSettings.at("adaptive_thresh_constant")));
        }
        
        // Detection Parameters
        if (algoSettings.find("quad_decimate") != algoSettings.end() && quadDecimateSpin_) {
            quadDecimateSpin_->setValue(algoSettings.at("quad_decimate"));
        }
        if (algoSettings.find("quad_sigma") != algoSettings.end() && quadSigmaSpin_) {
            quadSigmaSpin_->setValue(algoSettings.at("quad_sigma"));
        }
        if (algoSettings.find("refine_edges") != algoSettings.end() && refineEdgesCheck_) {
            refineEdgesCheck_->setChecked(algoSettings.at("refine_edges") != 0.0);
        }
        if (algoSettings.find("decode_sharpening") != algoSettings.end() && decodeSharpeningSpin_) {
            decodeSharpeningSpin_->setValue(algoSettings.at("decode_sharpening"));
        }
        if (algoSettings.find("nthreads") != algoSettings.end() && nthreadsSpin_) {
            nthreadsSpin_->setValue(static_cast<int>(algoSettings.at("nthreads")));
        }
        
        // Quad Threshold Parameters
        if (algoSettings.find("min_cluster_pixels") != algoSettings.end() && minClusterPixelsSpin_) {
            minClusterPixelsSpin_->setValue(static_cast<int>(algoSettings.at("min_cluster_pixels")));
        }
        if (algoSettings.find("max_line_fit_mse") != algoSettings.end() && maxLineFitMseSpin_) {
            maxLineFitMseSpin_->setValue(algoSettings.at("max_line_fit_mse"));
        }
        if (algoSettings.find("critical_angle_degrees") != algoSettings.end() && criticalAngleSpin_) {
            criticalAngleSpin_->setValue(algoSettings.at("critical_angle_degrees"));
        }
        if (algoSettings.find("min_white_black_diff") != algoSettings.end() && minWhiteBlackDiffSpin_) {
            minWhiteBlackDiffSpin_->setValue(static_cast<int>(algoSettings.at("min_white_black_diff")));
        }
        
        // Advanced Parameters
        if (algoSettings.find("corner_refine_win_size") != algoSettings.end() && cornerRefineWinSizeSpin_) {
            cornerRefineWinSizeSpin_->setValue(static_cast<int>(algoSettings.at("corner_refine_win_size")));
        }
        if (algoSettings.find("corner_refine_max_iter") != algoSettings.end() && cornerRefineMaxIterSpin_) {
            cornerRefineMaxIterSpin_->setValue(static_cast<int>(algoSettings.at("corner_refine_max_iter")));
        }
        if (algoSettings.find("pattern_border_size") != algoSettings.end() && patternBorderSizeSpin_) {
            patternBorderSizeSpin_->setValue(static_cast<int>(algoSettings.at("pattern_border_size")));
        }
        if (algoSettings.find("tag_min_area") != algoSettings.end() && tagMinAreaSpin_) {
            tagMinAreaSpin_->setValue(static_cast<int>(algoSettings.at("tag_min_area")));
        }
        if (algoSettings.find("tag_max_area") != algoSettings.end() && tagMaxAreaSpin_) {
            tagMaxAreaSpin_->setValue(static_cast<int>(algoSettings.at("tag_max_area")));
        }
        
        // Apply loaded settings to detector
        applyAlgorithmSettings();
        
        qDebug() << "Algorithm settings loaded for camera:" << QString::fromStdString(currentCameraSettings_.camera_name);
    }
    
    // Load default algorithm settings (public slot for button)
    void loadDefaultAlgorithmSettings() {
        resetAlgorithmSettingsToDefaults();
        QMessageBox::information(this, "Algorithm Settings", 
            "Default algorithm settings loaded and applied!\n\nAll algorithms (CPU and Fast AprilTag) will use these default values.");
    }
    
    // Reset algorithm settings to default values
    void resetAlgorithmSettingsToDefaults() {
        // Preprocessing defaults
        if (preprocessHistEqCheck_) {
            preprocessHistEqCheck_->setChecked(false);
        }
        if (preprocessClaheClipSpin_) {
            preprocessClaheClipSpin_->setValue(3.0);
        }
        if (preprocessGammaSpin_) {
            preprocessGammaSpin_->setValue(1.5);
        }
        if (preprocessContrastSpin_) {
            preprocessContrastSpin_->setValue(1.5);
        }
        
        // Edge Detection defaults
        if (cannyLowSpin_) {
            cannyLowSpin_->setValue(50);
        }
        if (cannyHighSpin_) {
            cannyHighSpin_->setValue(150);
        }
        if (adaptiveThreshBlockSpin_) {
            adaptiveThreshBlockSpin_->setValue(11);
        }
        if (adaptiveThreshConstantSpin_) {
            adaptiveThreshConstantSpin_->setValue(2);
        }
        
        // Detection Parameters defaults
        if (quadDecimateSpin_) {
            quadDecimateSpin_->setValue(2.0);
        }
        if (quadSigmaSpin_) {
            quadSigmaSpin_->setValue(0.0);
        }
        if (refineEdgesCheck_) {
            refineEdgesCheck_->setChecked(true);
        }
        if (decodeSharpeningSpin_) {
            decodeSharpeningSpin_->setValue(0.5);  // Increased from 0.25 to 0.5 for better decoding in various lighting
        }
        if (nthreadsSpin_) {
            nthreadsSpin_->setValue(4);
        }
        
        // Quad Threshold Parameters defaults (more sensitive for better detection)
        // Lower values = more sensitive (detects more tags, including lower quality ones)
        if (minClusterPixelsSpin_) {
            minClusterPixelsSpin_->setValue(4);  // Lowered from 6 to 4 for better sensitivity
        }
        if (maxLineFitMseSpin_) {
            maxLineFitMseSpin_->setValue(12.0);  // Increased from 8.0 to 12.0 (more lenient line fitting)
        }
        if (criticalAngleSpin_) {
            criticalAngleSpin_->setValue(10.0);  // Increased from 7.0 to 10.0 (more angle tolerance)
        }
        if (minWhiteBlackDiffSpin_) {
            minWhiteBlackDiffSpin_->setValue(4);  // Lowered from 6 to 4 (detect tags with lower contrast)
        }
        
        // Advanced Parameters defaults
        if (cornerRefineWinSizeSpin_) {
            cornerRefineWinSizeSpin_->setValue(5);
        }
        if (cornerRefineMaxIterSpin_) {
            cornerRefineMaxIterSpin_->setValue(30);
        }
        if (patternBorderSizeSpin_) {
            patternBorderSizeSpin_->setValue(4);
        }
        if (tagMinAreaSpin_) {
            tagMinAreaSpin_->setValue(500);
        }
        if (tagMaxAreaSpin_) {
            tagMaxAreaSpin_->setValue(50000);
        }
        
        // Apply default settings to detector
        applyAlgorithmSettings();
        
        qDebug() << "Algorithm settings reset to defaults for camera:" << QString::fromStdString(currentCameraSettings_.camera_name);
    }
    
    void loadCameraSettingsForAlgorithm() {
        QString configPath = "camera_settings.txt";
        QFile file(configPath);
        if (!file.exists() || !file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            // Config file doesn't exist, use default values
            return;
        }
        
        QTextStream in(&file);
        // Use the shared camera from top-level controls
        if (!cameraOpen_ || selectedCameraIndex_ < 0) {
            file.close();
            return;
        }
        
        QString cameraName = QString::fromStdString(cameraList_[selectedCameraIndex_]);
        QString currentType = useMindVision_ ? "MindVision" : "V4L2";
        
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
            if (useMindVision_) {
#ifdef HAVE_MINDVISION_SDK
                if (mvHandle_ != 0) {
                    // Apply exposure
                    if (savedExposure >= 0) {
                        double min_exposure = 1000.0;
                        double max_exposure = 100000.0;
                        double exposure = max_exposure - (savedExposure / 100.0) * (max_exposure - min_exposure);
                        CameraSetExposureTime(mvHandle_, exposure);
                    }
                    
                    // Apply gain
                    if (savedGain >= 0) {
                        CameraSetGain(mvHandle_, savedGain, savedGain, savedGain);
                    }
                    
                    // Apply brightness (analog gain)
                    if (savedBrightness >= 0) {
                        INT analogGain = (savedBrightness * 100) / 255;
                        CameraSetAnalogGain(mvHandle_, analogGain);
                    }
                    
                    // Apply contrast
                    if (savedContrast >= 0) {
                        CameraSetContrast(mvHandle_, savedContrast);
                    }
                    
                    // Apply saturation
                    if (savedSaturation >= 0) {
                        CameraSetSaturation(mvHandle_, savedSaturation);
                    }
                    
                    // Apply sharpness
                    if (savedSharpness >= 0) {
                        CameraSetSharpness(mvHandle_, savedSharpness);
                    }
                }
#endif
            } else {
                // Apply V4L2 settings (use shared camera)
                if (cameraCap_.isOpened()) {
                    if (savedExposure >= 0) {
                        cameraCap_.set(CAP_PROP_EXPOSURE, savedExposure);
                    }
                    if (savedGain >= 0) {
                        cameraCap_.set(CAP_PROP_GAIN, savedGain);
                    }
                    if (savedBrightness >= 0) {
                        cameraCap_.set(CAP_PROP_BRIGHTNESS, savedBrightness);
                    }
                    if (savedContrast >= 0) {
                        cameraCap_.set(CAP_PROP_CONTRAST, savedContrast);
                    }
                    if (savedSaturation >= 0) {
                        cameraCap_.set(CAP_PROP_SATURATION, savedSaturation);
                    }
                    if (savedSharpness >= 0) {
                        cameraCap_.set(CAP_PROP_SHARPNESS, savedSharpness);
                    }
                }
            }
        }
    }
};

#include "apriltag_debug_gui.moc"

int main(int argc, char *argv[]) {
    // CRITICAL: Disable OpenGL/GPU rendering to prevent interference with CUDA
    // Force Qt to use software rendering instead of OpenGL
    // These attributes MUST be set BEFORE creating QApplication
    QCoreApplication::setAttribute(Qt::AA_UseSoftwareOpenGL, true);
    QCoreApplication::setAttribute(Qt::AA_DisableShaderDiskCache, true);
    
    // Also set environment variables to force software rendering
    setenv("QT_XCB_FORCE_SOFTWARE_OPENGL", "1", 1);
    setenv("LIBGL_ALWAYS_SOFTWARE", "1", 1);
    setenv("QT_QUICK_BACKEND", "software", 1);
    
    QApplication app(argc, argv);
    
    AprilTagDebugGUI window;
    window.show();
    
    return app.exec();
}

