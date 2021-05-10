GPU_ID=$1
SEED=1
TEST_LEN=2000
MODEL=roberta-base

CUDA_VISIBLE_DEVICES=$GPU_ID python src/grin.py \
    --seed $SEED \
    --subtype_len $TEST_LEN \
    --model_name boychaboy/mnli_${MODEL} \
    --template_A templates/template_A_${TEST_LEN}_${MODEL}.csv \
    --template_B templates/template_B_${TEST_LEN}_${MODEL}.csv \
    --template_C templates/template_C_${TEST_LEN}_${MODEL}.csv \
    --save_dir result/${MODEL}_2.0_${TEST_LEN}_${SEED}
