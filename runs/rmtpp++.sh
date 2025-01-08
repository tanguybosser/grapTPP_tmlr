#!/bin/bash
  

datasets=(
    'lastfm'
    'mooc'
    'github'
    'reddit'
    'stack_overflow'
)

for dataset in ${datasets[@]}
do
    for split in {0..4}
    do
    python3 -u scripts/train.py --dataset $dataset --load-from-dir 'data' \
    --save-results-dir "results/${dataset}" --save-check-dir "checkpoints/${dataset}" \
    --eval-metrics True --include-poisson False --patience 50 --batch-size 8 --split $split \
    --encoder-histtime 'gru' --encoder-histmark 'gru' \
    --encoder-histtime-encoding 'temporal_with_labels' --encoder-histmark-encoding 'temporal_with_labels' --encoder-emb-dim 8 \
    --encoder-units-rnn 27 --encoder-layers-rnn 1 \
    --encoder-units-mlp 16 --encoder-activation-mlp 'relu' \
    --decoder 'rmtpp-dd' \
    --decoder-encoding 'log_times_only' \
    --decoder-units-mlp 32 \
    --separate-training True \
    --decoder-hist-time-grouping 'concatenation'
    done 
done
