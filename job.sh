#!/bin/bash
#PBS -q gpu
#PBS -l walltime=15:00:00
#PBS -l select=1:ncpus=1:ngpus=1:mem=8gb:gpu_mem=10gb:scratch_local=100gb

trap 'clean_scratch' TERM EXIT

HOME_DIR=${1:-'/storage/brno2/home/takonoselidze'}
PROJECT_DIR="$HOME_DIR/Thesis/Random_Position_Patch"
DATASET_DIR="$HOME_DIR/imagenetv2-top-images"
CONTAINER_PATH="$HOME_DIR/containers/my_container.sif"

CHECKPOINT_DIR="$SCRATCHDIR/checkpoints"
set -o allexport; source "$HOME_DIR/Thesis/.env"; set +o allexport


# Split TRAINING_MODELS and TARGET_MODELS into arrays
IFS=',' read -r -a training_models_array <<< "$TRAINING_MODELS"
IFS=',' read -r -a target_models_array <<< "$TARGET_MODELS"

# Convert arrays to space-separated strings for loggings
TRAINING_MODELS_STR="${training_models_array[@]}"
TARGET_MODELS_STR="${target_models_array[@]}"

TRAINING_MODELS_STR=$(echo "${training_models_array[@]}" | xargs)
TARGET_MODELS_STR=$(echo "${target_models_array[@]}" | xargs)

RUN_MODE=${RUN_MODE}
ATTACK_MODE=${ATTACK_MODE}

TARGET_CLASS=${TARGET_CLASS}
PATCH_SIZE=${PATCH_SIZE:-64}
PATCHES=${PATCHES:-1}

EPOCHS=${EPOCHS:-40}           # Number of epochs
CLASSES=${CLASSES:-1000}        # Number of classes to load for training

SCRATCH_RESULTS="$SCRATCHDIR/results"
SCRATCH_LOGS="$SCRATCHDIR/logs"
RESULTS_TAR="$SCRATCHDIR/results.tar.gz"

# scratch directory
cd $SCRATCHDIR || exit 1

# cpying project and dataset to scratch
echo "Copying project and dataset to scratch..."
cp -r "$PROJECT_DIR"/* "$SCRATCHDIR" || { echo "Project copy failed"; exit 1; }
cp -r "$DATASET_DIR" "$SCRATCHDIR" || { echo "Dataset copy failed"; exit 1; }
cp $HOME_DIR/Thesis/sing.sh $SCRATCHDIR
cp $HOME_DIR/Thesis/.env $SCRATCHDIR

mkdir -p "$CHECKPOINT_DIR"  # Create checkpoint directory in scratch

# echo "Files in scratch directory:"
# ls -R $SCRATCHDIR

# creating directories for results and logs
mkdir -p "$SCRATCH_RESULTS" "$SCRATCH_LOGS" || { echo "Failed to create result/log directories"; exit 1; }

# run the singularity container
echo "Running singularity container..."
singularity exec --nv "$CONTAINER_PATH" bash sing.sh \
	"$SCRATCHDIR/imagenetv2-top-images/imagenetv2-top-images-format-val" \
	"$CHECKPOINT_DIR" \
	"$RUN_MODE" \
	"$ATTACK_MODE" \
	"$TRAINING_MODELS_STR" \
	"$TARGET_MODELS_STR" \
	"$TARGET_CLASS" \
	"$PATCH_SIZE"\
	"$PATCHES" \
	"$EPOCHS" \
	"$CLASSES" || { echo "Singularity execution failed"; exit 1; }


# archive the results
# echo "Archiving results..."
# tar -czf "$RESULTS_TAR" "$SCRATCH_RESULTS" "$SCRATCH_LOGS" || { echo "Failed to create results archive"; exit 1; }

# # copy results back to home directory
# echo "Copying results to home directory..."
# cp "$RESULTS_TAR" "$HOME_DIR/results" || { echo "Failed to copy results, keeping scratch"; export CLEAN_SCRATCH=false; exit 1; }
# # cp -r $SCRATCHDIR/pics $HOME_DIR/result_images/$PBS_JOBID || export CLEAN_SCRATCH=false
# ------------------------------------------------------------



# Copy saved generators from scratch to home directory
# echo "Copying the best generator(s) to home directory..."
# mkdir -p "$HOME_DIR/generators/$PBS_JOBID"
# cp -r "$CHECKPOINT_DIR"/best_generators/* "$HOME_DIR/generators/$PBS_JOBID" || { echo "Failed to copy generators"; exit 1; }

# echo "Generator saved to $HOME_DIR/generators/$PBS_JOBID"


echo "Job completed successfully!"
