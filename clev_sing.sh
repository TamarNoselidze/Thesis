#!/bin/bash

pip install python-dotenv

if [[ -f "$SCRATCHDIR/.env" ]]; then
    export $(grep -v '^#' "$SCRATCHDIR/.env" | xargs)
fi

echo "Running Python script"
python3 main.py \
    --image_folder_path "$1" \
    --attack "$2" \
    --model "$3" \
    --target "$4" \
    --epsilon "$5"