GPU_ID=$1
SEED=1
TEST_LEN=2000
VER=_CP3
MODEL=roberta-base

CUDA_VISIBLE_DEVICES=$GPU_ID python src/grin.py \
    --seed $SEED \
    --subtype_len $TEST_LEN \
    --model_name boychaboy/mnli_${MODEL} \
    --racial_terms terms/racial_terms2.csv \
    --crowspairs_gender sents/crowspairs-gender2.json \
    --crowspairs_race sents/crowspairs-race2.json \
    --template_A templates/template_A_${TEST_LEN}_${MODEL}${VER}.csv \
    --template_B templates/template_B_${TEST_LEN}_${MODEL}${VER}.csv \
    --template_C templates/template_C_${TEST_LEN}_${MODEL}${VER}.csv \
    --save_dir result/${MODEL}_2.0_${TEST_LEN}_${SEED}${VER}
