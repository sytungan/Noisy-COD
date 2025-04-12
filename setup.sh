#!/bin/bash

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip first
pip install --upgrade pip

# Install PyTorch and torchvision for macOS
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install basic dependencies first
pip install numpy matplotlib pillow tqdm pyyaml

# Install remaining dependencies from requirements.txt
pip install -r requirements.txt

echo "Setup completed. Virtual environment activated."
echo "To activate the virtual environment in the future, run: source venv/bin/activate" 