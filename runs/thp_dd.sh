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
    --eval-metrics True --include-poisson False --patience 50 --batch-size 4 --split $split \
    --encoder-histtime 'gru' --encoder-histmark 'gru' \
    --encoder-histtime-encoding 'temporal_with_labels' --encoder-histmark-encoding 'temporal_with_labels' --encoder-emb-dim 8 \
    --encoder-units-rnn 28 --encoder-layers-rnn 1 \
    --encoder-units-mlp 18 --encoder-activation-mlp 'relu' \
    --decoder 'thp-double-dd' \
    --decoder-encoding 'log_times_only' --decoder-emb-dim 8 \
    --decoder-units-mlp 48 \
    --decoder-hist-time-grouping 'concatenation' \
    --separate-training True \
    --decoder-mc-prop-est 50
    done 
done
