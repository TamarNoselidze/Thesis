#!/bin/bash

pip install python-dotenv

if [[ -f "$SCRATCHDIR/.env" ]]; then
    export $(grep -v '^#' "$SCRATCHDIR/.env" | xargs)
fi

echo "Running Python script"
python3 main.py \
    --attack "$1" \
    --image_folder_path "$2" \
    --models "$3" \
    --epochs "$4"
