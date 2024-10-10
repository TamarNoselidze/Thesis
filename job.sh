#!/bin/bash
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -l walltime=1:00:00
#PBS -l select=1:ncpus=1:ngpus=1:mem=30gb:scratch_local=100gb

trap 'clean_scratch' TERM EXIT

# accepting command-line arguments
## experiment name
# EXPERIMENT_NAME=${1:-''}

HOME_DIR=${1:-'/storage/brno2/home/takonoselidze'}
PROJECT_DIR=${2:-"$HOME_DIR/Thesis/Random_Position_Patch"}
DATASET_DIR=${3:-"$HOME_DIR/Thesis/imagenetv2-top-images"}
CONTAINER_PATH=${4:-"$HOME_DIR/containers/pytorch_container.sif"}

EPOCHS=${5:-40}  # Number of epochs
ATTACK_TYPE=${6:-0}  # Default attack type
MODEL=${7:-'vit_b_16'}  # Default model

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


echo "Files in scratch directory:"
ls -R $SCRATCHDIR

# creating directories for results and logs
mkdir -p "$SCRATCH_RESULTS" "$SCRATCH_LOGS" || { echo "Failed to create result/log directories"; exit 1; }

# run the singularity container
echo "Running singularity container..."
singularity exec --nv "$CONTAINER_PATH" bash sing.sh \
  --attack_type "$ATTACK_TYPE" \
  --image_folder_path "$SCRATCHDIR/imagenetv2-top-images/imagenetv2-top-images-format-val" \
  --model "$MODEL" \
  --epochs "$EPOCHS" || { echo "Singularity execution failed"; exit 1; }

# archive the results
echo "Archiving results..."
tar -czf "$RESULTS_TAR" "$SCRATCH_RESULTS" "$SCRATCH_LOGS" || { echo "Failed to create results archive"; exit 1; }

# copy results back to home directory
echo "Copying results to home directory..."
cp "$RESULTS_TAR" "$HOME_DIR/results" || { echo "Failed to copy results, keeping scratch"; export CLEAN_SCRATCH=false; exit 1; }
cp -r $SCRATCHDIR/pics $HOME_DIR/result_images/$PBS_JOBID || export CLEAN_SCRATCH=false

echo "Job completed successfully!"
