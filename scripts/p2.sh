#!/usr/bin/env bash

#SBATCH --partition=short
#SBATCH -c 32
#SBATCH -o logs/latest_p2.log

container="/data/courses/2026_dat471_dit066/containers/assignment7.sif"
glove_dataset="/data/courses/2026_dat471_dit066/datasets/glove"
pubs_dataset="/data/courses/2026_dat471_dit066/datasets/pubs"

BASE_DIR="$HOME/a7-gpu"

# DATASET="data/pubs/pubs.csv"
# QUERIES="data/pubs/pub_queries_small.txt"
# LABELS="data/pubs/pub_queries_small_names.txt"

DATASET="data/glove/glove.6B.50d.txt"
QUERIES="data/glove/glove.6B.50d_queries_medium.txt"
LABELS="data/glove/glove.6B.50d_queries_medium_names.txt"

# DATASET="data/glove/glove.840B.300d.txt"
# QUERIES="data/glove/glove.840B.300d_queries_small.txt"
# LABELS="data/glove/glove.840B.300d_queries_small_names.txt"

mkdir -p $BASE_DIR/data/glove $BASE_DIR/data/pubs

apptainer exec \
    --bind "$HOME" \
    --bind "$glove_dataset:$BASE_DIR/data/glove" \
    --bind "$pubs_dataset:$BASE_DIR/data/pubs" \
    --nv \
    $container \
    bash -c "src/assignment7_problem2.py --dataset $DATASET --queries $QUERIES --labels $LABELS --batch-size 128;"
