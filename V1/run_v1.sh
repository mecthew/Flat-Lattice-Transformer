dataset=$1
schemes=(1)
ple_channel_nums=(1)
plstm=(0)
use_bert=(1)
random_seeds=(1080956)    #
optim=adam
status=train

if [ $1 == resume -o $1 == weibo ]; then
  eps=100
  early_stop=10
  batch=10
  update_every=3
elif [ $1 == ontonotes4 -o $1 == msra ];then
  eps=30
  early_stop=10
  batch=10
  update_every=3
fi


for scheme in ${schemes[*]}
do
  for pnum in ${ple_channel_nums[*]}
  do
    for pl in ${plstm[*]}
    do
      for seed in ${random_seeds[*]}
      do
        echo "$dataset, bz=$batch, update_every=$update_every, bert=$use_bert, scheme=$scheme, ple_channel_num=$pnum, use_ple_lstm=$pl, seed=$seed"
        CUDA_VISIBLE_DEVICES=$2 \
        python3 flat_main.py \
            --use_bert $use_bert \
            --after_bert mlp \
            --lexicon_name yj \
            --lr 1e-2 \
            --batch $batch \
            --update_every $update_every \
            --epoch $eps \
            --early_stop $early_stop \
            --optim $optim \
            --dataset $dataset \
            --status $status \
            --ple_channel_num $pnum \
            --new_tag_scheme $scheme \
            --use_ple_lstm $pl \
            --seed $seed
      done
    done
  done
done

#   --loadmodel_path /home/liujian/github/Flat-Lattice-Transformer/output/ckpt/ontonotes4/bert1_scheme1_ple1_plstm0/best_Lattice_Transformer_SeqLabel_f_2021-06-09-21-16-06
