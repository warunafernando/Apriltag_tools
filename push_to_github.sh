#!/bin/bash
# Script to push Apriltag_tools repository to GitHub

set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

echo "========================================="
echo "AprilTag Tools - GitHub Push Script"
echo "========================================="
echo ""

# Check if remote already exists
if git remote get-url origin > /dev/null 2>&1; then
    echo "Remote 'origin' already exists: $(git remote get-url origin)"
    read -p "Do you want to update it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter GitHub repository URL (https://github.com/USERNAME/Apriltag_tools.git): " repo_url
        git remote set-url origin "$repo_url"
    else
        echo "Using existing remote."
    fi
else
    read -p "Enter your GitHub username: " username
    repo_url="https://github.com/${username}/Apriltag_tools.git"
    
    echo ""
    echo "Adding remote: $repo_url"
    git remote add origin "$repo_url"
fi

echo ""
echo "Renaming branch to 'main'..."
git branch -M main

echo ""
echo "Pushing to GitHub..."
git push -u origin main

echo ""
echo "✓ Successfully pushed to GitHub!"
echo "Repository URL: $(git remote get-url origin)"



