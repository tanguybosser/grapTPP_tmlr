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
    --encoder-histtime-encoding 'temporal_with_labels' --encoder-histmark-encoding 'temporal_with_labels' \
    --encoder-emb-dim 8 --encoder-embedding-constraint 'nonneg' \
    --encoder-units-rnn 30 --encoder-layers-rnn 1 --encoder-constraint-rnn 'nonneg' \
    --encoder-units-mlp 18 --encoder-activation-mlp 'relu' --encoder-constraint-mlp 'nonneg' \
    --decoder 'mlp-cm-double-dd' --decoder-encoding 'log_times_only' --decoder-emb-dim 8 --decoder-embedding-constraint 'nonneg' \
    --decoder-units-mlp 36 --decoder-units-mlp 30 --decoder-activation-mlp 'gumbel_softplus' \
    --decoder-activation-final-mlp 'parametric_softplus' --decoder-constraint-mlp 'nonneg' \
    --decoder-hist-time-grouping 'concatenation' \
    --separate-training True
    done 
done 
