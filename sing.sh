#!/bin/bash

pip install python-dotenv

if [[ -f "$SCRATCHDIR/.env" ]]; then
    export $(grep -v '^#' "$SCRATCHDIR/.env" | xargs)
fi

echo "Arguments passed to main.py: $@"
echo "Running Python script"
python3 main.py \
    --image_folder_path "$1" \
    --checkpoint_folder_path "$2" \
    --attack_mode "$3" \
    --num_of_patches "$4" \
    --transfer_mode "$5" \
    --training_models "$6" \
    --target_models "$7" \
    --patch_size "$8" \
    --epochs "$9" \
    --num_of_train_classes "${10}" \
    --target_class "${11}" \
    --mini_type "${12}"