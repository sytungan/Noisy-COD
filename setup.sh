#!/bin/bash

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install base dependencies first
pip3 install --upgrade pip
pip3 install numpy==1.21.5 setuptools wheel

# Install PyTorch and torchvision (as they might need specific CUDA versions)
pip3 install torch==1.8.0 torchvision==0.9.0

# Install all other dependencies from requirements.txt
pip3 install -r requirements.txt

# Install apex separately as it needs to be built from source
pip3 install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git

echo "Setup completed. Virtual environment activated."
echo "To activate the virtual environment in the future, run: source venv/bin/activate" 