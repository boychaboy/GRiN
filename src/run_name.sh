#!/bin/bash
TASK=$1
GPU_ID=$2

MAX_SEED=1
TEST_LEN=2000

SET=$(seq 1 $MAX_SEED)
for i in $SET
do
    SEED=$i
    echo "SEED $SEED"
    VER=name_${TEST_LEN}_${SEED}

    MODELS=( "bert-base-uncased" "bert-base-uncased_2" "bert-large-uncased" "roberta-base" "roberta-large" "distilbert-base-cased" "distilbert-base-cased_2" "distilroberta-base" )
    # MODELS=( "distilbert-base-cased" )

    for MODEL in ${MODELS[@]}; do
        mkdir result/${TASK}/${VER}
        mkdir result/${TASK}/${VER}/${MODEL}
        echo "${MODEL} running..."
        CUDA_VISIBLE_DEVICES=$GPU_ID python src/grin_name.py \
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
