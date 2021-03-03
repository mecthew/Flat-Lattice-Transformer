CUDA_VISIBLE_DEVICES=$1 \
python flat_main.py \
    --use_bert 0 \
    --after_bert mlp \
    --lexicon_name yj \
    --batch 10 \
    --epoch 100 \
    --optim sgd \
    --dataset $2 \
    --status $3 \
    --new_tag_scheme \
    $4
