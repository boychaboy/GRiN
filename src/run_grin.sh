#!/bin/bash
TASK=$1
VER=$2
GPU_ID=$3

MAX_SEED=1
TEST_LEN=2000

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
    MODELS=( "bert-large-uncased" "bert-large-cased" "roberta-base" "roberta-large" "distilbert-base-cased" "distilroberta-base" )
    for MODEL in ${MODELS[@]}; do
        echo "(${MODEL}) running..."
        CUDA_VISIBLE_DEVICES=$GPU_ID python src/grin.py \
            --seed $SEED \
            --subtype_len $TEST_LEN \
            --model_name boychaboy/${TASK}_${MODEL} \
            --male_terms terms/male_terms2.json \
            --female_terms terms/female_terms2.json \
            --racial_terms terms/racial_terms4.csv \
            --crowspairs_gender sents/crowspairs-gender2.json \
            --crowspairs_race sents/crowspairs-race2.json \
            --template_A templates/${TASK}/${VER}/${MODEL}_${TEST_LEN}.csv \
            --template_B templates/${TASK}/${VER}/${MODEL}_${TEST_LEN}.csv \
            --template_C templates/${TASK}/${VER}/${MODEL}_${TEST_LEN}.csv \
            --save_dir result/${TASK}/${VER}/${MODEL}_${TEST_LEN}
    done
done
