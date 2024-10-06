#!/bin/bash
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -l walltime=4:00:00
#PBS -l select=1:ncpus=1:ngpus=1:mem=30gb:scratch_local=100gb

trap 'clean_scratch' TERM EXIT

cd $SCRATCHDIR

HOME_DIR='/storage/brno2/home/takonoselidze'
PROJECT_DIR='/storage/brno2/home/takonoselidze/Thesis/Random_Position_Patch'
DATASET_DIR='/storage/brno2/home/takonoselidze/Thesis/imagenetv2-top-images'


# Copy project and data to the scratch directory
cp -r $HOME_DIR/Thesis/sing.sh $SCRATCHDIR
cp -r $PROJECT_DIR/* $SCRATCHDIR
#cp -r $DATASET_DIR/* $SCRATCHDIR
cp -r $DATASET_DIR $SCRATCHDIR

#for debugging
ls -R $SCRATCHDIR

# Make directories for outputs
mkdir -p $SCRATCHDIR/results
mkdir -p $SCRATCHDIR/logs


# Run the singularity container and execute singularity.sh inside it
singularity exec --nv /storage/brno2/home/takonoselidze/containers/pytorch_container.sif bash sing.sh


# Archive the results
tar -czf results.tar.gz $SCRATCHDIR/results $SCRATCHDIR/logs


# Copy results back to the storage directory
cp results.tar.gz $HOME_DIR/results || export CLEAN_SCRATCH=false
