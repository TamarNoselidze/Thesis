#!/bin/bash

pip install python-dotenv

if [[ -f "$SCRATCHDIR/.env" ]]; then
    export $(grep -v '^#' "$SCRATCHDIR/.env" | xargs)
fi

echo "Running Python script"
python3 main.py \
    --attack_type "$1" \
    --image_folder_path "$2" \
    --model "$3" \
    --patch_size "$4" \
    --epochs "$5" \
    --brightness "$6" \
    --color_transfer "$7"
