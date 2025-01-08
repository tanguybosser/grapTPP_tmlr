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
    --encoder 'gru' --encoder-encoding 'temporal_with_labels' --encoder-emb-dim 8 \
    --encoder-units-rnn 36 --encoder-layers-rnn 1 \
    --encoder-units-mlp 32 --encoder-activation-mlp 'relu' \
    --decoder 'thp' \
    --decoder-encoding 'log_times_only' --decoder-emb-dim 8 \
    --decoder-units-mlp 96 \
    --decoder-hist-time-grouping 'concatenation' \
    --separate-training False \
    --decoder-mc-prop-est 50
    done
done 



