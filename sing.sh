#!/bin/bash


echo "Running Python script"
python3 main.py \
    --attack_type "$1" \
    --image_folder_path "$2" \
    --model "$3" \
    --epochs "$4"