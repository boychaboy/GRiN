GPU_ID=$1
MAX_SEED=6
TEST_LEN=4000
VER=

SET=$(seq 1 $MAX_SEED)
for i in $SET
do
    SEED=$i
    echo "SEED $SEED"
    MODELS=( "bert-base-cased" "bert-large-cased" "roberta-base" "roberta-large" "distilbert-base-cased" "distilroberta-base" )
    for MODEL in ${MODELS[@]}; do
        echo "This is ${MODEL}"
        CUDA_VISIBLE_DEVICES=$GPU_ID python src/grin.py \
            --seed $SEED \
            --subtype_len $TEST_LEN \
            --model_name boychaboy/MNLI_${MODEL} \
            --racial_terms terms/racial_terms3.csv \
            --crowspairs_gender sents/crowspairs-gender2.json \
            --crowspairs_race sents/crowspairs-race2.json \
            --template_A templates/mnli/template_A_${TEST_LEN}_${MODEL}${VER}.csv \
            --template_B templates/mnli/template_B_${TEST_LEN}_${MODEL}${VER}.csv \
            --template_C templates/mnli/template_C_${TEST_LEN}_${MODEL}${VER}.csv \
            --save_dir result/mnli/${MODEL}_${TEST_LEN}_${SEED}${VER}
    done
done
