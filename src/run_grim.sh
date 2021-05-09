GPU_ID=$1
SEED=1
SPLIT=0.005
TEST_LEN=2000

MODEL=bert-base-cased

CUDA_VISIBLE_DEVICES=$GPU_ID python grim.py \
    --seed $SEED \
    --split \
    --split_ratio $SPLIT \
    --subtype_len $TEST_LEN \
    --model_name boychaboy/mnli_${MODEL} \
    --template_A templates/template_A_${TEST_LEN}_${MODEL}.csv \
    --template_B templates/template_B_${TEST_LEN}_${MODEL}.csv \
    --template_C templates/template_C_${TEST_LEN}_${MODEL}.csv \
    --save_dir ../result/${MODEL}_${SPLIT}_${SEED}
