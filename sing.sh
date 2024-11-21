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
    --num_of_train_classes "$7" \
    --num_of_target_classes "$8" \
    --brightness "$9" \
    --color_transfer "$10"
