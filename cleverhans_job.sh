#!/bin/bash
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -l walltime=4:00:00
#PBS -l select=1:ncpus=1:ngpus=1:mem=30gb:scratch_local=100gb

trap 'clean_scratch' TERM EXIT

# Set environment variables

HOME_DIR=${1:-'/storage/brno2/home/takonoselidze'}
PROJECT_DIR="$HOME_DIR/Thesis/CleverHans"

RESULT_DIR="/storage/brno2/home/takonoselidze/Thesis/results"
CONTAINER_PATH="$HOME_DIR/containers/pytorch_container_cleverhans.sif"
PROJECT_DIR="/storage/brno2/home/takonoselidze/Thesis/CleverHans"
DATASET_DIR="/storage/brno2/home/takonoselidze/Thesis/imagenetv2-top-images"

set -o allexport; source "$HOME_DIR/Thesis/.env"; set +o allexport

ATTACK=${ATTACK:-'FGSM'}  # Default attack type
MODELS=${MODELS:-'vit_b_16'}     # Default model
EPOCHS=${EPOCHS:-0}           # Number of epochs

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
echo "Files in scratch directory:"
ls -R $SCRATCHDIR
mkdir -p "$SCRATCH_RESULTS" "$SCRATCH_LOGS" || { echo "Failed to create result/log directories"; exit 1; }

echo "Running singularity container..."

# Run your script inside the Singularity container
singularity exec --nv "$CONTAINER_PATH" bash clev_sing.sh \
	"$ATTACK" \
	"$SCRATCHDIR/imagenetv2-top-images/imagenetv2-top-images-format-val" \
	"$MODELS" \
	"$EPOCHS" || { echo "Singularity execution failed"; exit 1; }



# singularity exec --nv $CONTAINER_PATH \
#     python3 main.py --attack FGSM --image_folder_path "$SCRATCH_DIR/imagenetv2-top-images/imagenetv2-top-images-format-val" \
#     --models resnet50 vit_b_16 --epochs 5 --brightness 1.5

echo "Copying results to home directory..."
mkdir -p "$PROJECT_DIR/results" || { echo "Failed to create results directory"; exit 1; }
tar -czf "$RESULTS_TAR" "$SCRATCH_RESULTS" "$SCRATCH_LOGS" || { echo "Failed to create results archive"; exit 1; }

cp "$RESULTS_TAR" "$PROJECT_DIR/results/$PBS_JOBID_results" || { echo "Failed to copy results, keeping scratch"; export CLEAN_SCRATCH=false; exit 1; }
# cp -r $SCRATCHDIR/pics $HOME_DIR/result_images/$PBS_JOBID || export CLEAN_SCRATCH=false

echo "Job completed successfully!"
