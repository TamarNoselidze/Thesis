#!/bin/bash
#PBS -q gpu
#PBS -l walltime=4:00:00
#PBS -l select=1:ncpus=1:ngpus=1:mem=5gb:scratch_local=100gb

trap 'clean_scratch' TERM EXIT

# Set environment variables

HOME_DIR=${1:-'/storage/brno2/home/takonoselidze'}
PROJECT_DIR="$HOME_DIR/Thesis/CleverHans"

RESULT_DIR="$HOME_DIR/Thesis/results"
CONTAINER_PATH="$HOME_DIR/containers/pytorch_container_cleverhans.sif"
PROJECT_DIR="$HOME_DIR/Thesis/CleverHans"
# DATASET_DIR="$HOME_DIR/Thesis/imagenetv2-top-images"
DATASET_DIR="$HOME_DIR/imagenetv2-top-images"


set -o allexport; source "$HOME_DIR/Thesis/.env"; set +o allexport

ATTACK=${ATTACK:-'FGSM'}  # Default attack type
MODEL=${MODEL:-'vit_b_16'}     # Default model
TARGET=${TARGET}
EPSILON=${EPSILON}


SCRATCH_RESULTS="$SCRATCHDIR/results"
SCRATCH_LOGS="$SCRATCHDIR/logs"
RESULTS_TAR="$SCRATCHDIR/results.tar.gz"


cd $SCRATCHDIR || exit 1

echo "Copying project and dataset to scratch..."
# Copy project and dataset to scratch
cp -r "$PROJECT_DIR"/* "$SCRATCHDIR" || { echo "Project copy failed"; exit 1; }
cp -r "$DATASET_DIR" "$SCRATCHDIR" || { echo "Dataset copy failed"; exit 1; }
cp $HOME_DIR/Thesis/clev_sing.sh $SCRATCHDIR
cp $HOME_DIR/Thesis/.env $SCRATCHDIR


# Go to the scratch directory
mkdir -p "$SCRATCH_RESULTS" "$SCRATCH_LOGS" || { echo "Failed to create result/log directories"; exit 1; }

echo "Running singularity container..."

# Run your script inside the Singularity container
singularity exec --nv "$CONTAINER_PATH" bash clev_sing.sh \
	"$SCRATCHDIR/imagenetv2-top-images/imagenetv2-top-images-format-val" \
	"$ATTACK" \
	"$MODEL" \
	"$TARGET" \
	"$EPSILON" || { echo "Singularity execution failed"; exit 1; }


# echo "Copying results to home directory..."
# mkdir -p "$PROJECT_DIR/results" || { echo "Failed to create results directory"; exit 1; }
# tar -czf "$RESULTS_TAR" "$SCRATCH_RESULTS" "$SCRATCH_LOGS" || { echo "Failed to create results archive"; exit 1; }

# cp "$RESULTS_TAR" "$PROJECT_DIR/results/$PBS_JOBID_results" || { echo "Failed to copy results, keeping scratch"; export CLEAN_SCRATCH=false; exit 1; }

echo "Job completed successfully!"
