#!/bin/bash

pip install python-dotenv

if [[ -f "$SCRATCHDIR/.env" ]]; then
    export $(grep -v '^#' "$SCRATCHDIR/.env" | xargs)
fi

TRAINING_MODELS="$3"
TARGET_MODELS="$4"

training_models_list=$(echo "$TRAINING_MODELS" | tr ',' ' ')
target_models_list=$(echo "$TARGET_MODELS" | tr ',' ' ')

echo "Running Python script"
python3 main.py \
    --image_folder_path "$1" \
    --transfer_mode "$2" \
    --training_models "$training_models_list" \
    --target_models "$target_models_list" \
    --patch_size "$5" \
    --epochs "$6" \
    --num_of_classes "$7" \
    --brightness "$8" \
    --color_transfer "$9"
