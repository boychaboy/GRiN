#!/bin/bash
TASK=$1
VER=$2
GPU_ID=$3

MAX_SEED=1
TEST_LEN=2000
MODEL=howey/electra-large-mnli
NAME=electra-large-mnli
if [ ! -d result/${TASK}/${VER} ]; then
    mkdir result/${TASK}/${VER}
    mkdir templates/${TASK}/${VER}
fi

echo "Keep Going Keep Shival!"
echo ""

SET=$(seq 1 $MAX_SEED)
for i in $SET
do
    SEED=$i
    echo "SEED $SEED"
    echo "(${MODEL}) running..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python src/grin.py \
        --seed $SEED \
        --subtype_len $TEST_LEN \
        --model_name $MODEL \
        --racial_terms terms/racial_terms4.csv \
        --crowspairs_gender sents/crowspairs-gender2.json \
        --crowspairs_race sents/crowspairs-race2.json \
        --template_A templates/${TASK}/${VER}/${NAME}_A.csv \
        --template_B templates/${TASK}/${VER}/${NAME}_B.csv \
        --template_C templates/${TASK}/${VER}/${NAME}_C.csv \
        --save_dir result/${TASK}/${VER}/${NAME}.csv
done
