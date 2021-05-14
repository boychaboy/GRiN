#!/bin/bash
TASK=$1
VER=$2
GPU_ID=$3

MAX_SEED=1
TEST_LEN=2000
MODEL=boychaboy/MNLI_distilbert-base-cased_2
NAME=distilbert-base-cased_2

if [ ! -d result/${TASK}/${VER} ]; then
    mkdir result/${TASK}/${VER}
fi

# echo "Keep Going Keep Shival!"

SET=$(seq 1 $MAX_SEED)
for i in $SET
do
    SEED=$i
    echo "SEED $SEED"
    mkdir result/${TASK}/${VER}/${NAME}
    echo "${MODEL} running..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python src/grin.py \
        --seed $SEED \
        --subtype_len $TEST_LEN \
        --model_name $MODEL \
        --male_terms terms/male_terms2.json \
        --female_terms terms/female_terms2.json \
        --racial_terms terms/racial_terms4.csv \
        --crowspairs_gender sents/crowspairs-gender2.json \
        --crowspairs_race sents/crowspairs-race2.json \
        --save_dir result/${TASK}/${VER}/${NAME}/
done
