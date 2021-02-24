CUDA_VISIBLE_DEVICES=$1 \
python flat_main.py \
    --use_bert 0 \
    --after_bert mlp \
    --lexicon_name yj \
    --batch 10 \
    --optim sgd \
    --dataset $2 \
    --status $3
