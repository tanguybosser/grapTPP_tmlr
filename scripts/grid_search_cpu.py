import pickle 
import subprocess
import numpy as np
import os 
import random

def random_search(param_dic, num_config=10):
    configs= [dict.fromkeys(param_dic.keys()) for i in range(num_config)]
    for i in range(num_config):
        for param, values in param_dic.items():
            j = random.randint(0, len(values)-1)
            configs[i][param] = values[j]
    return configs



datasets = ['reddit_filtered', 'mooc_filtered', 'wikipedia_filtered', 'mimic2_filtered', 'github', 'lastfm']
datasets = ['reddit_filtered_seqshort']
batch_sizes = [8, 64, 64, 64, 64, 64]
splits = [0]

encoder_emb_dims = decoder_emb_dims = [4, 8, 16, 32]

encoder_units_rnns = [16, 32, 64]
encoder_layers_rnns = [1, 2]

encoder_units_mlps = decoder_units_mlps = [8, 16, 32]

param_dic = {'encoder_emb_dim':encoder_emb_dims,
            'encoder_units_rnn':encoder_units_rnns,
            'encoder_layers_rnn':encoder_layers_rnns,
            'encoder_units_mlp':encoder_units_mlps,
            'decoder_units_mlp':decoder_units_mlps
            }

for i, dataset in enumerate(datasets):
    save_results_dir = 'results/model_selection/marked_filtered_seqshort/' + dataset + '/candidates/'
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)
    for split in splits:
        configs = random_search(param_dic)
        load_dir = 'baseline3/' + dataset + '/split_' + str(split)
        cmd = ''
        for j, config in enumerate(configs):
            cmd += "srun --ntasks=1 --exclusive python3 -u scripts/train.py --no-mlflow --dataset={} --load-from-dir={} --save-results-dir={} " \
                    "--eval-metrics={} --include-poisson={} --patience={} --batch-size={} --split={} --config={} --train-epochs={} " \
                    "--model-selection={} " \
                    "--encoder={} --encoder-encoding={} --encoder-emb-dim={} " \
                    "--encoder-units-rnn={} --encoder-layers-rnn={} " \
                    "--encoder-units-mlp={} --encoder-activation-mlp={} " \
                    "--decoder={} --decoder-encoding={} --decoder-emb-dim={} " \
                    "--decoder-units-mlp={} --decoder-units-mlp={} --decoder-activation-mlp={} --decoder-activation-final-mlp={} & " \
                    .format(dataset, load_dir, save_results_dir,
                            False, False, 20, batch_sizes[i], split, j, 5,
                            True,
                            'gru', 'temporal_with_labels', config["encoder_emb_dim"],
                            config["encoder_units_rnn"], config["encoder_layers_rnn"],
                            config["encoder_units_mlp"], "relu",
                            'conditional-poisson', 'temporal_with_labels', config["encoder_emb_dim"],
                            config["decoder_units_mlp"], config["decoder_units_mlp"], 'relu', 'parametric_softplus')
        cmd += "wait"
        while True:
            try:  
                print(cmd)
                out = subprocess.run([cmd], shell=True, check=True)
                break
            except subprocess.CalledProcessError as e:
                    print('error occured')
                    break 