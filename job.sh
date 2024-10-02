#!/bin/bash
#PBS -q gpu@meta-pbs.metacentrum.cz
#PBS -l walltime=4:00:00
#PBS -l select=1:ncpus=1:ngpus=1:mem=10gb:scratch_local=100gb

trap 'clean_scratch' TERM EXIT

cd $SCRATCHDIR

HOME_DIR='/storage/brno2/home/takonoselidze'
PROJECT_DIR='/storage/brno2/home/takonoselidze/Random_Position_Patch'
DATASET_DIR='/storage/brno2/home/takonoselidze/imagenetv2-top-images'


# Copy project and data to the scratch directory
cp -r $HOME_DIR/sing.sh $SCRATCHDIR
cp -r $PROJECT_DIR/* $SCRATCHDIR
cp -r $DATASET_DIR/* $SCRATCHDIR


# Make directories for outputs
mkdir $SCRATCHDIR/results
mkdir $SCRATCHDIR/logs


# Run the singularity container and execute singularity.sh inside it
singularity exec --nv /storage/brno2/home/takonoselidze/containers/pytorch_with_libraries.sif bash sing.sh


# Archive the results
tar -czf results.tar.gz $SCRATCHDIR/results $SCRATCHDIR/logs


# Copy results back to the storage directory
cp results.tar.gz $DATADIR/results || export CLEAN_SCRATCH=false