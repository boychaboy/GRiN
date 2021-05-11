GPU_ID=$1
MAX_SEED=1
TEST_LEN=4000
VER=r_nocap/

if [ ! -d result/mnli/$VER ]; then
    mkdir result/mnli/$VER
    mkdir templates/mnli/$VER
    echo "$VER directory made"
fi

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
            --racial_terms terms/racial_terms4.csv \
            --crowspairs_gender sents/crowspairs-gender2.json \
            --crowspairs_race sents/crowspairs-race2.json \
            --template_A templates/mnli/${VER}template_A_${TEST_LEN}_${MODEL}.csv \
            --template_B templates/mnli/${VER}template_B_${TEST_LEN}_${MODEL}.csv \
            --save_dir result/mnli/${VER}${MODEL}_${TEST_LEN}_${SEED}
    done
done

# --template_C templates/mnli/${VER}template_C_${TEST_LEN}_${MODEL}.csv \
