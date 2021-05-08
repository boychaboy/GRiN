GPU_ID=$1
SEED=1
SPLIT=0.0025
MODEL=roberta-base

CUDA_VISIBLE_DEVICES=$GPU_ID python grim.py \
    --seed $SEED \
    --split \
    --split_ratio $SPLIT \
    --model_name boychaboy/mnli_${MODEL} \
    --save_dir ../result/${MODEL}_${SPLIT}_${SEED}
