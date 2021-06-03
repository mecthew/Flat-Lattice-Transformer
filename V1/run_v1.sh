datasets=($1)
schemes=(1)
ple_channel_nums=(1)

for dataset in ${datasets[*]}
do
  for scheme in ${schemes[*]}
  do
    for pnum in ${ple_channel_nums[*]}
    do
      echo "$dataset, scheme=$scheme, ple_channel_num=$pnum"
      CUDA_VISIBLE_DEVICES=$2 \
      python3 flat_main.py \
          --use_bert 0 \
          --after_bert mlp \
          --lexicon_name yj \
          --lr 1e-2 \
          --batch 12 \
          --epoch 150 \
          --early_stop 25 \
          --optim sgd \
          --dataset $dataset \
          --status train \
          --ple_channel_num $pnum \
          --new_tag_scheme $scheme
    done
  done
done