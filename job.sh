#!/bin/bash
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -l walltime=6:00:00
#PBS -l select=1:ncpus=1:ngpus=1:mem=30gb:scratch_local=100gb

trap 'clean_scratch' TERM EXIT

# accepting command-line arguments
HOME_DIR=${1:-'/storage/brno2/home/takonoselidze'}
PROJECT_DIR=${2:-"$HOME_DIR/Thesis/Random_Position_Patch"}
DATASET_DIR=${3:-"$HOME_DIR/Thesis/imagenetv2-top-images"}
CONTAINER_PATH=${4:-"$HOME_DIR/containers/pytorch_container.sif"}

SCRATCH_RESULTS="$SCRATCHDIR/results"
SCRATCH_LOGS="$SCRATCHDIR/logs"
RESULTS_TAR="$SCRATCHDIR/results.tar.gz"

# scratch directory
cd $SCRATCHDIR || exit 1

# cpying project and dataset to scratch
echo "Copying project and dataset to scratch..."
cp -r "$PROJECT_DIR"/* "$SCRATCHDIR" || { echo "Project copy failed"; exit 1; }
cp -r "$DATASET_DIR" "$SCRATCHDIR" || { echo "Dataset copy failed"; exit 1; }


echo "Files in scratch directory:"
ls -R $SCRATCHDIR

# creating directories for results and logs
mkdir -p "$SCRATCH_RESULTS" "$SCRATCH_LOGS" || { echo "Failed to create result/log directories"; exit 1; }

# run the singularity container
echo "Running singularity container..."
singularity exec --nv "$CONTAINER_PATH" bash sing.sh || { echo "Singularity execution failed"; exit 1; }

# archive the results
echo "Archiving results..."
tar -czf "$RESULTS_TAR" "$SCRATCH_RESULTS" "$SCRATCH_LOGS" || { echo "Failed to create results archive"; exit 1; }

# copy results back to home directory
echo "Copying results to home directory..."
cp "$RESULTS_TAR" "$HOME_DIR/results" || { echo "Failed to copy results, keeping scratch"; export CLEAN_SCRATCH=false; exit 1; }
cp -r $SCRATCHDIR/pics $HOME_DIR/results || export CLEAN_SCRATCH=false

echo "Job completed successfully!"



# trap 'clean_scratch' TERM EXIT

# cd $SCRATCHDIR

# HOME_DIR='/storage/brno2/home/takonoselidze'
# PROJECT_DIR='/storage/brno2/home/takonoselidze/Thesis/Random_Position_Patch'
# DATASET_DIR='/storage/brno2/home/takonoselidze/Thesis/imagenetv2-top-images'


# # Copy project and data to the scratch directory
# cp -r $HOME_DIR/Thesis/sing.sh $SCRATCHDIR
# cp -r $PROJECT_DIR/* $SCRATCHDIR
# #cp -r $DATASET_DIR/* $SCRATCHDIR
# cp -r $DATASET_DIR $SCRATCHDIR

# #for debugging
# ls -R $SCRATCHDIR

# # Make directories for outputs
# mkdir -p $SCRATCHDIR/results
# mkdir -p $SCRATCHDIR/logs


# # Run the singularity container and execute singularity.sh inside it
# singularity exec --nv /storage/brno2/home/takonoselidze/containers/pytorch_container.sif bash sing.sh


# # Archive the results
# tar -czf results.tar.gz $SCRATCHDIR/results $SCRATCHDIR/logs


# # Copy results back to the storage directory
# cp results.tar.gz $HOME_DIR/results || export CLEAN_SCRATCH=false

