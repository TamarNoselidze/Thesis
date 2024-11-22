#!/bin/bash

pip install python-dotenv

if [[ -f "$SCRATCHDIR/.env" ]]; then
    export $(grep -v '^#' "$SCRATCHDIR/.env" | xargs)
fi


echo "Running Python script"
python3 main.py \
    --image_folder_path "$1" \
    --checkpoint_folder_path "$2" \
    --transfer_mode "$3" \
    --training_models "$4" \
    --target_models "$5" \
    --patch_size "$6" \
    --epochs "$7" \
    --num_of_train_classes "$8" \
    --num_of_target_classes "$9" \
    --brightness "$10" \
    --color_transfer "$11"
