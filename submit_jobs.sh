#!/bin/bash

# arrays for different values
# TRAINING_MODELS_LIST=("vit_b_16" "vit_l_16" "swin_b" "resnet50" "vgg16_bn" "resnet152" "" )
# TARGET_CLASS_LIST=(153 309 481 746 932)
# MINI_TYPE_LIST=(0 1 2)

# Loop over all combinations of TRAINING_MODELS, TARGET_CLASS, and MINI_TYPE
# for TRAINING_MODELS in "${TRAINING_MODELS_LIST[@]}"; do
#     for TARGET_CLASS in "${TARGET_CLASS_LIST[@]}"; do
#         for MINI_TYPE in "${MINI_TYPE_LIST[@]}"; do
#             echo "Submitting job for TRAINING_MODELS=$TRAINING_MODELS, TARGET_CLASS=$TARGET_CLASS, MINI_TYPE=$MINI_TYPE"
#             qsub -v "ATTACK_MODE=mini,NUMBER_OF_PATCHES=8,TRANSFER_MODE=src-tar,TRAINING_MODELS='$TRAINING_MODELS',PATCH_SIZE=16,EPOCHS=40,CLASSES=1000,TARGET_CLASS=$TARGET_CLASS,MINI_TYPE=$MINI_TYPE" job.sh
#         done
#     done
# done


# for TARGET_CLASS in "${TARGET_CLASS_LIST[@]}"; do
#     for MINI_TYPE in "${MINI_TYPE_LIST[@]}"; do
#         echo "Submitting job for TRAINING_MODELS=vit_b_32, TARGET_CLASS=$TARGET_CLASS, MINI_TYPE=$MINI_TYPE"
#         qsub -v "ATTACK_MODE=mini,NUMBER_OF_PATCHES=4,TRANSFER_MODE=src-tar,TRAINING_MODELS=vit_b_32,PATCH_SIZE=32,EPOCHS=40,CLASSES=1000,TARGET_CLASS=$TARGET_CLASS,MINI_TYPE=$MINI_TYPE" job.sh
#     done
# done





# training_models=("vit_b_16 vit_b_32" "vit_b_16 vit_l_16" "vit_b_16 swin_b" "vit_b_32 vit_l_16" "vit_b_32 swin_b" "vit_l_16 swin_b")


# training_models=("vit_b_16 vit_b_32 vit_l_16" "vit_b_16 vit_b_32 swin_b" "vit_l_16 vit_b_32 swin_b" "vit_b_16 vit_b_32 vit_l_16 swin_b")



training_models=("vit_l_16")
# ("vit_b_16" "vit_b_32" "resnet50" "resnet152" "swin_b" "vgg16_bn")

target_classes=(153 309 481 746 932)
#  (153 309 481 746 932)

# Other constant variables

run_mode="train"
attack_mode="mini_0"
patch_size=16
patches=8
epochs=40
classes=1000

# Loop over each model and each target class
for model in "${training_models[@]}"; do
  for target_class in "${target_classes[@]}"; do
    echo "Submitting job for model=${model}, target_class=${target_class}"
    qsub -v "RUN_MODE=${run_mode},ATTACK_MODE=${attack_mode},TRAINING_MODELS='${model}',TARGET_CLASS=${target_class},PATCH_SIZE=${patch_size},PATCHES=${patches},EPOCHS=${epochs},CLASSES=${classes}" job.sh
    sleep 1  # optional: short delay to avoid overwhelming the scheduler
  done
done