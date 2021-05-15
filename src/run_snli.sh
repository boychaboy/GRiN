#!/bin/bash
GPU_ID=$1
TASK=SNLI

MAX_SEED=3
TEST_LEN=99999

SET=$(seq 1 $MAX_SEED)
for i in $SET
do
    SEED=$i
    echo "SEED $SEED"
    VER=${TEST_LEN}_${SEED}

    MODELS=( "bert-base-uncased" "bert-large-uncased" "distilbert-base-cased" "roberta-base" "roberta-large" "distilroberta-base" )

    for MODEL in ${MODELS[@]}; do
        mkdir result/${TASK}/${VER}
        mkdir result/${TASK}/${VER}/${MODEL}
        echo "${MODEL} running..."
        CUDA_VISIBLE_DEVICES=$GPU_ID python src/grin.py \
            --seed $SEED \
            --subtype_len $TEST_LEN \
            --model_name boychaboy/${TASK}_${MODEL} \
            --male_terms terms/male_terms2.json \
            --female_terms terms/female_terms2.json \
            --racial_terms terms/racial_terms4.csv \
            --crowspairs_gender sents/crowspairs-gender2.json \
            --crowspairs_race sents/crowspairs-race2.json \
            --save_dir result/${TASK}/${VER}/${MODEL}/
    done
done
