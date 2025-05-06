#!/bin/bash
set -e

echo "===== MLOps Project Environment Setup (Linux/macOS) ====="

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
echo "Checking Python version..."
if command_exists python3; then
    python_cmd="python3"
elif command_exists python; then
    python_cmd="python"
else
    echo "Error: Python not found"
    exit 1
fi

python_version=$($python_cmd --version 2>&1 | cut -d ' ' -f 2)
python_major=$(echo $python_version | cut -d '.' -f 1)
python_minor=$(echo $python_version | cut -d '.' -f 2)

if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 9 ]); then
    echo "Error: Python 3.9+ is required but found: $python_version"
    exit 1
else
    echo "Python 3.9+ is installed: $python_version"
fi

# Check Git
echo "Checking Git..."
if command_exists git; then
    git_version=$(git --version)
    echo "Git is installed: $git_version"
else
    echo "Error: Git is required but not found"
    exit 1
fi

# Check Git LFS
echo "Checking Git LFS..."
if command_exists git-lfs || (command_exists git && git lfs --version >/dev/null 2>&1); then
    git_lfs_version=$(git lfs --version)
    echo "Git LFS is installed: $git_lfs_version"
else
    echo "Error: Git LFS is required but not found"
    exit 1
fi

# Check for UV
echo "Checking UV..."
if command_exists uv; then
    uv_version=$(uv --version)
    echo "UV is installed: $uv_version"
else
    echo "UV not found, will install..."
    $python_cmd -m pip install uv
    uv_version=$(uv --version)
    echo "UV installed successfully: $uv_version"
fi

# Initialize Git LFS
echo "Initializing Git LFS..."
git lfs install

# Create virtual environment
echo "Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "Virtual environment already exists"
else
    uv venv .venv
    echo "Virtual environment created successfully"
fi

# Install dependencies
echo "Installing dependencies..."
if [ "$(uname)" == "Darwin" ]; then
    # macOS
    source .venv/bin/activate
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Linux
    source .venv/bin/activate
else
    echo "Unsupported OS for activation script"
    exit 1
fi

uv pip install -r requirements-dev.txt
echo "Dependencies installed successfully"

echo "===== Environment Setup Complete ====="
echo "To activate the environment, run: source .venv/bin/activate"
echo "To deactivate, run: deactivate"

