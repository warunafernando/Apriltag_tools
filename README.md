# AprilTag Debug Tools

A comprehensive Qt-based GUI toolkit for debugging and analyzing AprilTag detection on images and camera feeds.

## Features

### Processing Tab
- **Side-by-side Image Comparison**: Load and compare two images simultaneously
- **Stage-by-Stage Debugging**: Visualize each stage of the AprilTag detection pipeline
  - Preprocessing: Histogram Equalization, CLAHE, Gamma Correction, Contrast Enhancement
  - Edge Detection: Canny, Sobel, Laplacian, Adaptive Threshold
  - Detection: Full detection pipeline with tag IDs and quality metrics
  - Advanced Visualizations:
    - Corner Refinement: Visualize corner refinement process
    - Warped Tags: View individual warped tag candidates
    - Pattern Extraction: Extract and visualize 6x6 binary patterns
    - Hamming Decode: Decode and verify tag patterns
- **Independent Controls**: Separate preprocessing, edge detection, and mirror options for each image
- **Quality Metrics**: Display detailed quality metrics for each stage
- **Quad Selection**: Select specific quadrilaterals/tags for detailed analysis

### Capture Tab
- **Multi-Camera Support**: 
  - V4L2 cameras (USB webcams, etc.)
  - MindVision cameras (with SDK)
- **Camera Controls**:
  - Resolution and FPS selection
  - Exposure control
  - Gain control
  - Brightness, Contrast, Saturation, Sharpness adjustments
- **Live Preview**: Real-time video feed from selected camera
- **Image Capture**: Capture frames with automatic filename generation (camera name + timestamp)
- **Fisheye Correction**: Apply fisheye distortion correction to live feed and captured images

### Fisheye Correction Tab
- **Calibration Management**: Load existing fisheye calibration files (YAML format)
- **Visualization**: Side-by-side preview of original vs. corrected images with reference lines
- **Checkerboard Calibration**: 
  - Interactive 6x6 grid-based calibration process
  - Automatic capture when checkerboard is stable
  - Visual feedback (yellow borders for captured tiles, small dots for captured positions)
  - Progress tracking (stops after 34 unique grid positions)
- **Enable/Disable**: Toggle fisheye correction for captures and live feed

### Status Indicator
- **Top-Level Status**: Always-visible fisheye correction status at the top of the window
  - Green: Correction Applied (calibration loaded and enabled)
  - Yellow: Calibration loaded but disabled
  - Red: No calibration loaded

## Dependencies

- **OpenCV** (>= 4.0): For image processing and camera access
- **Qt5** (Core, Widgets): For GUI framework
- **AprilTag Library**: For tag detection
- **MindVision SDK** (optional): For MindVision camera support

## Building

### Prerequisites

Install required packages:

```bash
# Ubuntu/Debian
sudo apt-get install libopencv-dev qtbase5-dev libapriltag-dev cmake build-essential

# Or use pkg-config for AprilTag
sudo apt-get install libapriltag-dev
```

### Build Instructions

1. Clone or download this repository
2. Create build directory:
```bash
cd Apriltag_tools
mkdir build && cd build
```

3. Configure and build:
```bash
cmake ..
make
```

4. The executable will be in `build/bin/apriltag_debug_gui`

### Optional: MindVision Camera Support

If you have MindVision cameras and SDK:

1. Update the SDK path in `CMakeLists.txt` (line 49 and 68) to match your installation
2. The SDK will be automatically detected during cmake configuration
3. If found, MindVision camera support will be enabled

## Usage

### Basic Usage

```bash
./build/bin/apriltag_debug_gui
```

### Processing Tab

1. Click "Load Image 1" and/or "Load Image 2" to load images
2. Use the dropdown menus to select:
   - **Preprocessing method**: Original, Histogram Equalization, CLAHE, Gamma Correction, Contrast Enhancement
   - **Edge Detection method**: None, Canny, Sobel, Laplacian, Adaptive Threshold
   - **Detection stage**: Original, Detection, Contours, Quadrilaterals, etc.
   - **Advanced visualization**: Corner Refinement, Warped Tags, Pattern Extraction, Hamming Decode
3. Use independent mirror checkboxes for each image
4. Select specific quads/tags from dropdown menus for detailed analysis
5. View quality metrics in the boxes below each image

### Capture Tab

1. Select camera from dropdown (V4L2 or MindVision)
2. Select resolution/FPS mode
3. Adjust camera settings (exposure, gain, brightness, etc.)
4. Click "Capture" to save a frame (default location: `input/` directory)
5. Enable fisheye correction if calibration is loaded

### Fisheye Correction Tab

1. **Load Calibration**:
   - Enter path to calibration YAML file
   - Click "Load Calibration"
   - Or use default path: `/home/nav/9202/Hiru/Apriltag/calibration_data/camera_params.yaml`

2. **Preview Correction**:
   - Click "Load Test Image" to see original vs. corrected side-by-side
   - Reference lines help visualize the correction effect

3. **Create Calibration**:
   - Print a 6x6 checkerboard pattern
   - Open a camera in the Capture tab
   - Click "Start Calibration" in Fisheye Correction tab
   - Show the checkerboard in different grid positions
   - System automatically captures when stable (up to 34 positions)
   - Click "Save Calibration" when done

4. **Enable Correction**:
   - Select "Use Corrected" radio button
   - Correction will be applied to:
     - Live camera feed in Capture tab
     - Captured images
     - Test images in Fisheye Correction tab

## Calibration File Format

The fisheye calibration file should be a YAML file with the following format:

```yaml
%YAML:1.0
---
camera_matrix: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ fx, 0, cx, 0, fy, cy, 0, 0, 1 ]
distortion_coefficients: !!opencv-matrix
   rows: 4
   cols: 1
   dt: d
   data: [ k1, k2, k3, k4 ]
image_width: 1280
image_height: 1024
```

## File Structure

```
Apriltag_tools/
├── CMakeLists.txt           # Build configuration
├── README.md               # This file
├── .gitignore              # Git ignore rules
├── apriltag_debug_gui.cpp  # Main application source
└── build/                  # Build directory (created during build)
    └── bin/
        └── apriltag_debug_gui  # Executable
```

## Troubleshooting

### Camera not detected
- Check camera permissions: `sudo usermod -a -G video $USER` (logout/login required)
- Verify camera is connected and not in use by another application
- For MindVision cameras, ensure SDK is properly installed and path is correct in CMakeLists.txt

### AprilTag library not found
- Install `libapriltag-dev` package
- Or update `CMakeLists.txt` to point to your AprilTag installation path

### Fisheye correction not working
- Verify calibration file path is correct
- Check that calibration file contains valid camera matrix and distortion coefficients
- Ensure image dimensions match the calibration data

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- Built with OpenCV for image processing
- Uses Qt5 for GUI framework
- AprilTag library for tag detection
