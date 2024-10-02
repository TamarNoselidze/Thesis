#!/bin/bash

apt-get update
apt-get install -y python3-pip

pip3 install torch torchvision matplotlib numpy Pillow

echo "Running Python script"
python3 main.py
