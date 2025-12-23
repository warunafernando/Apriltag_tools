# Pushing to GitHub

Follow these steps to push this repository to GitHub:

## 1. Create Repository on GitHub

1. Go to https://github.com/new
2. Repository name: `Apriltag_tools`
3. Description: "Qt-based GUI toolkit for debugging and analyzing AprilTag detection"
4. Choose Public or Private
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

## 2. Add Remote and Push

Run these commands in the Tools directory:

```bash
cd /home/nav/Apriltag/StandAlone/Tools

# Add GitHub remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/Apriltag_tools.git

# Or if using SSH:
# git remote add origin git@github.com:YOUR_USERNAME/Apriltag_tools.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## 3. Verify

Check https://github.com/YOUR_USERNAME/Apriltag_tools to verify all files are uploaded.

## Alternative: Using GitHub CLI

If you have GitHub CLI installed:

```bash
cd /home/nav/Apriltag/StandAlone/Tools
gh repo create Apriltag_tools --public --source=. --remote=origin --push
```



