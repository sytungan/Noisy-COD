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

# Run testing
echo "Running testing..."
python3 code/TrainPNet/Test.py --ration 1

echo "Testing completed!" 