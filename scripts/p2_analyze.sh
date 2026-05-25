#!/usr/bin/env bash

#SBATCH --partition=long
#SBATCH -c 64
#SBATCH --gres=gpu:1
#SBATCH -o logs/latest_p2_analyze.log

BASE_DIR="$HOME/a7-gpu"
container="/data/courses/2026_dat471_dit066/containers/assignment7.sif"
glove_dataset="/data/courses/2026_dat471_dit066/datasets/glove"
pubs_dataset="/data/courses/2026_dat471_dit066/datasets/pubs"

mkdir -p $BASE_DIR/data/glove $BASE_DIR/data/pubs

DATASET_TYPES=("pubs" "glove.6B.50d")
SIZES=("small" "medium" "big")

BATCH_SIZE=256

RESULT="results/p2_analyze.csv"

PUBS_ROOT="data/pubs"
TMP_LOG="logs/tmp.log"

echo "dataset,num_queries,batch_size,total_time,host_to_device,device_to_host,num_errors,throughput" > $RESULT

for dataset_type in "${DATASET_TYPES[@]}"; do
    for size in "${SIZES[@]}"; do
        echo "starting run: $dataset_type, size=$size"

        if [ "$dataset_type" == "pubs" ]; then
            DATASET="$PUBS_ROOT/pubs.csv"
            QUERIES="$PUBS_ROOT/pub_queries_${size}.txt"
            LABELS="$PUBS_ROOT/pub_queries_${size}_names.txt"
        else
            DATASET="data/glove/${dataset_type}.txt"
            QUERIES="data/glove/${dataset_type}_queries_${size}.txt"
            LABELS="data/glove/${dataset_type}_queries_${size}_names.txt"
        fi

        if [ "$size" == "big" ]; then
            num_queries=10000
        elif [ "$size" == "medium" ]; then
            num_queries=1000
        else
            num_queries=100
        fi

        apptainer exec \
            --bind "$HOME" \
            --bind "$glove_dataset:$BASE_DIR/data/glove" \
            --bind "$pubs_dataset:$BASE_DIR/data/pubs" \
            --nv \
            $container \
            bash -c "src/assignment7_problem2.py --dataset $DATASET --queries $QUERIES --labels $LABELS --batch-size $BATCH_SIZE;" > $TMP_LOG

        echo "content of the log:"
        cat $TMP_LOG

        total_time=$(awk '/queries took/ {print $6}' $TMP_LOG)
        throughput=$(awk '/Throughput/ {print $2}' $TMP_LOG)
        errors=$(awk '/erroneous queries:/ {print $5}' $TMP_LOG)
        h_to_d=$(awk '/host to device:/ {print $7}' $TMP_LOG)
        d_to_h=$(awk '/device to host:/ {print $7}' $TMP_LOG)

        echo "$dataset_type,$num_queries,$BATCH_SIZE,$total_time,$h_to_d,$d_to_h,$errors,$throughput" >> $RESULT
    done
done

echo "completed, results can be found in $RESULT"
rm -f logs/tmp.log
