#!/bin/bash
GPU_ID=$1
TASK=MNLI

TEST_LEN=10000

SET=$(seq 1 $MAX_SEED)
for i in $SET
do
    SEED=$i
    echo "SEED $SEED"
    VER=${TEST_LEN}_4

    MODELS=( "bert-base-uncased_2" "bert-large-uncased" "distilbert-base-cased_2" "distilbert-base-uncased" "roberta-base" "roberta-large" "distilroberta-base" )
    # MODELS=( "distilbert-base-cased_2" )

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
            --racial_terms terms/racial_terms.json \
            --crowspairs_gender sents/crowspairs-gender2.json \
            --crowspairs_race sents/crowspairs-race2.json \
            --save_dir result/${TASK}/${VER}/${MODEL}/
    done
done
