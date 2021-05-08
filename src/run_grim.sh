GPU_ID=$1
SEED=2
SPLIT=0.0025
KEY_SPLIT=1

MODEL=roberta-base

CUDA_VISIBLE_DEVICES=$GPU_ID python grim.py \
    --seed $SEED \
    --split \
    --split_ratio $SPLIT \
    --key_split \
    --key_split_ratio $KEY_SPLIT \
    --model_name boychaboy/mnli_${MODEL} \
    --save_dir ../result/${MODEL}_${SPLIT}_${SEED}
