#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Check if data directory exists
if [ ! -d "code/data" ]; then
    echo "Error: Please download the dataset and place it in code/data directory"
    echo "Download links:"
    echo "Google Drive: https://drive.google.com/drive/folders/1nHD-d3FanT6-ORsZTEeGgGzQ2CUKyWSe?usp=drive_link"
    echo "BaiDu Drive: https://pan.baidu.com/s/1xAe4s6vqONcmwQIAzKOMCQ (Passwd: ECCV)"
    exit 1
fi

# Step 1: Train ANet
echo "Step 1: Training ANet..."
python code/TrainANet/TrainDDP.py --gpu_id 0 --ration 1

# Step 2: Generate Pseudo Labels
echo "Step 2: Generating Pseudo Labels..."
python code/TrainANet/Test.py --ration 1

# Step 3: Train PNet
echo "Step 3: Training PNet..."
python code/TrainPNet/TrainDDP.py --gpu_id 0 --ration 1 --q_epoch 20 --batchsize_fully 6 --batchsize_weakly 24

echo "Training completed!" 