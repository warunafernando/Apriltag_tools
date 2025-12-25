#!/bin/bash
# Setup script for GitHub repository

echo "Setting up git repository for Apriltag_tools..."

# Check if git user is configured
if ! git config user.email > /dev/null 2>&1; then
    echo "Git user email not configured."
    read -p "Enter your GitHub email: " email
    git config user.email "$email"
fi

if ! git config user.name > /dev/null 2>&1; then
    echo "Git user name not configured."
    read -p "Enter your GitHub username: " name
    git config user.name "$name"
fi

echo "Git configuration:"
echo "  Name: $(git config user.name)"
echo "  Email: $(git config user.email)"
echo ""
echo "Making initial commit..."

cd "$(dirname "$0")"
git commit -m "Initial commit: AprilTag Debug GUI Tool

- Multi-tab interface (Processing, Capture, Fisheye Correction)
- Side-by-side image comparison with stage-by-stage debugging
- Camera support (V4L2 and MindVision)
- Fisheye distortion correction with calibration tools
- Quality metrics and advanced visualizations
- Interactive checkerboard calibration process"

echo ""
echo "✓ Repository initialized and committed!"
echo ""
echo "Next steps:"
echo "1. Create a repository on GitHub named 'Apriltag_tools'"
echo "2. Run: git remote add origin https://github.com/YOUR_USERNAME/Apriltag_tools.git"
echo "3. Run: git branch -M main"
echo "4. Run: git push -u origin main"
echo ""
echo "Or see PUSH_TO_GITHUB.md for detailed instructions."













