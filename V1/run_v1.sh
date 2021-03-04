CUDA_VISIBLE_DEVICES=$1 \
python flat_main.py \
    --use_bert 1 \
    --after_bert mlp \
    --lexicon_name yj \
    --lr 1e-2 \
    --batch 20 \
    --epoch 200 \
    --early_stop 100 \
    --optim sgd \
    --dataset $2 \
    --status $3 \
    --ple_channel_num 1 \
    --new_tag_scheme \
    $4
