#!/bin/bash
TASK=$1
VER=$2
GPU_ID=$3

MAX_SEED=1
TEST_LEN=4000

if [ ! -d result/${TASK}/${VER} ]; then
    mkdir result/${TASK}/${VER}
    mkdir templates/${TASK}/${VER}
fi

echo "Keep Going Keep Shival!"
echo ""

SET=$(seq 0 $MAX_SEED)
for i in $SET
do
    SEED=$i
    echo "SEED $SEED"
    MODELS=( "bert-base-cased" "bert-large-cased" "roberta-base" "roberta-large" "distilbert-base-cased" "distilroberta-base" )
    for MODEL in ${MODELS[@]}; do
        echo "(${MODEL}) running..."
        CUDA_VISIBLE_DEVICES=$GPU_ID python src/grin.py \
            --seed $SEED \
            --subtype_len $TEST_LEN \
            --model_name boychaboy/SNLI_${MODEL} \
            --racial_terms terms/racial_terms3.csv \
            --crowspairs_gender sents/crowspairs-gender2.json \
            --crowspairs_race sents/crowspairs-race2.json \
            --template_A templates/${TASK}/${VER}/${MODEL}_${TEST_LEN}.csv \
            --template_B templates/${TASK}/${VER}/${MODEL}_${TEST_LEN}.csv \
            --template_C templates/${TASK}/${VER}/${MODEL}_${TEST_LEN}.csv \
            --save_dir result/${TASK}/${VER}/${MODEL}_${TEST_LEN}_${SEED}
    done
done
