#!/bin/bash

pip install python-dotenv

if [[ -f "$SCRATCHDIR/.env" ]]; then
    export $(grep -v '^#' "$SCRATCHDIR/.env" | xargs)
fi

echo "Running Python script"
python3 main.py \
    --image_folder_path "$1" \
    --transfer_mode "$2" \
    --training_models "$3" \
    --target_models "$4" \
    --patch_size "$5" \
    --epochs "$6" \
    --brightness "$7" \
    --color_transfer "$8"
