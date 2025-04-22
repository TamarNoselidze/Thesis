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
    --run_mode "$3" \
    --attack_mode "$4" \
    --training_models "$5" \
    --target_models "$6" \
    --target_class "$7" \
    --patch_size "${8}" \
    --num_of_patches "${9}" \
    --epochs "${10}" \
    --num_of_train_classes "${11}"