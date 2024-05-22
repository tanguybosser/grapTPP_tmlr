import os, sys 
import json 
import torch as th
import matplotlib.pyplot as plt 
import numpy as np 
import pickle as pkl

sys.path.remove(os.path.abspath(os.path.join('..', 'neuralTPPs')))
sys.path.append(os.path.abspath(os.path.join('..', 'neurips')))

from argparse import ArgumentParser, Namespace

from data_processing.simu_hawkes import simulate_seqs, plot_process_multi_axes

from tpps.utils.cli import parse_args
from tpps.utils.data import get_loader
from tpps.processes.multi_class_dataset import MultiClassDataset as Dataset
from tpps.models import get_model
from tpps.utils.events import get_events, get_window

def load_models(args):
    if args.model_name == 'mlp-cm':
        model_name_base = f'{args.dataset}_poisson_gru_{args.model_name}_temporal_with_labels_adjust_param_split{args.split}'
        model_name_dd = f'{args.dataset}_poisson_gru_temporal_with_labels_gru_temporal_with_labels_{args.model_name}-dd_separate_split{args.split}'
    else:
        model_name_base = f'{args.dataset}_gru_{args.model_name}_temporal_with_labels_adjust_param_split{args.split}'
        model_name_dd = f'{args.dataset}_gru_temporal_with_labels_gru_temporal_with_labels_{args.model_name}-dd_separate_split{args.split}'
    model_path_base = os.path.join(args.save_check_dir, model_name_base + '.pth')
    model_path_dd = os.path.join(args.save_check_dir, model_name_dd + '.pth')
    
    model_args_base = load_model_args(args, model_name_base)
    model_args_dd = load_model_args(args, model_name_dd)
    
    args.include_poisson = model_args_base.include_poisson
    
    model_base = get_model(model_args_base)
    model_dd = get_model(model_args_dd)

    model_base.load_state_dict(th.load(model_path_base, map_location=args.device))
    model_dd.load_state_dict(th.load(model_path_dd, map_location=args.device))
    return model_base, model_dd

def load_model(args):
    model_name = f'{args.dataset}_{args.model_name}_split0'
    model_path = os.path.join(args.save_check_dir, model_name + '.pth')    
    model_args = load_model_args(args, model_name)
    
    args.include_poisson = model_args.include_poisson
    
    model = get_model(model_args)

    model.load_state_dict(th.load(model_path, map_location=args.device))
    return model

def load_model_args(args, model_name):
    args_path = os.path.join(args.save_check_dir, 'args')
    args_path = os.path.join(args_path, model_name + '.json')
    with open(args_path, 'r') as f:
        model_args = json.load(f)
    model_args = Namespace(**model_args)
    model_args.decoder_mc_prop_est = 250
    model_args.device = args.device
    return model_args

def get_test_sequence(args, n_seq=1):
    if 'n_processes' not in vars(args):
        args_dic = vars(args)
        args_dic['n_processes'] = 1
        args = Namespace(**args_dic)
    sequence = []
    artifacts = {'marked_intensity':[], 'ground_intensity':[], 'true_mark_pmf':[], 'intensity_times':[]}
    seq_process, artifacts_process = simulate_seqs(args)
    sequence.extend(seq_process)
    artifacts['marked_intensity'].extend(artifacts_process['marked_intensity'])
    artifacts['ground_intensity'].extend(artifacts_process['ground_intensity'])
    artifacts['true_mark_pmf'].extend(artifacts_process['true_mark_pmf'])
    artifacts['intensity_times'].extend(artifacts_process['intensity_times'])
    data_sequence = Dataset(
    args=args, size=n_seq, seed=args.seed, data=sequence)
    sequence = get_loader(data_sequence, args=args, shuffle=False)
    return sequence, artifacts



def plot_hawkes_intensity(args):
    loader, artifacts_seq = get_test_sequence(args) #[np, L]
    models = load_models(args)
    fig, axes = plt.subplots(args.marks,1, figsize=(20,20))
    fig_g, axes_g = plt.subplots(1,1, figsize=(20,10))
    for i, batch in enumerate(loader):
        times, labels = batch["times"], batch["labels"]
        labels_idx = th.argmax(labels, dim=-1).squeeze(-1).cpu().numpy()
        axes = plot_process_multi_axes(times.cpu().numpy(), axes, labels_idx=labels_idx, marked=True)
        axes_g = plot_process_multi_axes(times.cpu().numpy(), axes_g, labels_idx=labels_idx, marked=False)
        labels = (labels != 0).type(labels.dtype) #?
        mask = (times != args.padding_id).type(times.dtype)
        times = times * args.time_scale #time_scale=1 by default
        window_start, window_end = get_window(times=times, window=args.window)
        events = get_events(
        times=times, mask=mask, labels=labels,
        window_start=window_start, window_end=window_end)
        
        intensity_times = artifacts_seq['intensity_times'][0][1:]
        true_marked_intensity = artifacts_seq['marked_intensity'][0][100:]
        true_ground_intensity = artifacts_seq['ground_intensity'][0][100:]
        intensity_times = th.tensor(intensity_times, dtype=th.float32).unsqueeze(0).to(args.device)
        marked_intensities, ground_intensities, diffs_marked, diffs_ground = [], [], [], []
        for model in models:
            marked_log_intensity, _, intensity_mask, _ = model.artifacts(
                query=intensity_times, events=events)
            marked_intensity = th.exp(marked_log_intensity)
            ground_intensity = th.sum(marked_intensity, dim=-1)
            marked_intensity = marked_intensity.squeeze(0).detach().cpu().numpy()
            ground_intensity = ground_intensity.squeeze(0).detach().cpu().numpy()
            diffs_marked.append(mean_absolute_deviation(marked_intensity, true_marked_intensity))
            diffs_ground.append(mean_absolute_deviation(ground_intensity, true_ground_intensity.squeeze()))
            marked_intensities.append(marked_intensity)
            ground_intensities.append(ground_intensity)
        intensity_times = intensity_times.squeeze(0).detach().cpu().numpy()
        for i, ax in enumerate(axes):
            ax.plot(intensity_times, marked_intensities[0][:,i], color='green', label=f'Base ({np.round(diffs_marked[0][i],3)})')
            ax.plot(intensity_times, marked_intensities[1][:,i], color='blue', label=f'DD ({np.round(diffs_marked[1][i],3)})')
            ax.plot(intensity_times, true_marked_intensity[:,i], color='red', label='True')
            ax.legend()
        axes_g.plot(intensity_times, true_ground_intensity, color='red', label='True')
        axes_g.plot(intensity_times, ground_intensities[0], color='blue', label=f'Base ({np.round(diffs_ground[0],3)})')
        axes_g.plot(intensity_times, ground_intensities[1], color='green', label=f'DD ({np.round(diffs_ground[1],3)})')
        axes_g.legend()
        fig.savefig(f'figures/simulations/marked_{args.dataset}_{args.model_name}_{args.split}.png', bbox_inches='tight')
        fig_g.savefig(f'figures/simulations/ground_{args.dataset}_{args.model_name}_{args.split}.png', bbox_inches='tight')

#carefull seed 
def compute_intensity_differences(args):
    loader, artifacts_seq = get_test_sequence(args) #[np, L]
    print(f'MODEL: {args.model_name}')
    model = load_model(args)
    n_seq = 0
    diffs_marked = np.zeros(args.marks)
    diffs_ground = 0
    for i, batch in enumerate(loader):
        times, labels = batch["times"].to(args.device), batch["labels"].to(args.device)
        labels = (labels != 0).type(labels.dtype) #?
        mask = (times != args.padding_id).type(times.dtype)
        times = times * args.time_scale #time_scale=1 by default
        window_start, window_end = get_window(times=times, window=args.window)
        events = get_events(
        times=times, mask=mask, labels=labels,
        window_start=window_start, window_end=window_end)
        
        first_event = float(min(times[0]))
        intensity_times = artifacts_seq['intensity_times'][i][1:] 
        mask = intensity_times > first_event + 0.001
        intensity_times = intensity_times[mask]

        true_marked_intensity = artifacts_seq['marked_intensity'][i][1:]
        true_ground_intensity = artifacts_seq['ground_intensity'][i][1:]
        true_marked_intensity = true_marked_intensity[mask]
        true_ground_intensity = true_ground_intensity[mask]
        intensity_times = th.tensor(intensity_times, dtype=th.float32).unsqueeze(0).to(args.device)

        log_ground_intensity, log_mark_pmf, _ , intensity_mask, _ = model.artifacts(
            query=intensity_times, events=events)
        
        ground_intensity = th.exp(log_ground_intensity)
        mark_pmf = th.exp(log_mark_pmf)
        marked_intensity = ground_intensity.unsqueeze(-1) * mark_pmf 

        marked_intensity = marked_intensity.squeeze(0).detach().cpu().numpy()
        ground_intensity = ground_intensity.squeeze(0).detach().cpu().numpy()
        diff_marked = mean_absolute_deviation(marked_intensity, true_marked_intensity)
        diff_ground = mean_absolute_deviation(ground_intensity, true_ground_intensity.squeeze())
        diffs_marked += diff_marked
        diffs_ground += float(diff_ground)
        n_seq +=1
    diffs_marked = list(diffs_marked / n_seq)
    diffs_ground = diffs_ground / n_seq
    results = {'marked':diffs_marked, 'ground':diffs_ground}
    save_path = f'results/simulations3/{args.dataset}_{args.model_name}_split{args.split}.txt'
    with open(save_path, "wb") as fp: 
        pkl.dump(results, fp)
    print('end')

def plot_hawkes_intensity_unique(args):
    loader, artifacts_seq = get_test_sequence(args, n_seq=10) #[np, L]
    model = load_model(args)
    print(args.model_name)
    for j, batch in enumerate(loader):
        fig, axes = plt.subplots(args.marks,1, figsize=(20,20))
        fig_g, axes_g = plt.subplots(1,1, figsize=(20,10))
        times, labels = batch["times"].to(args.device), batch["labels"].to(args.device)
        labels_idx = th.argmax(labels, dim=-1).squeeze(-1).cpu().numpy()
        axes = plot_process_multi_axes(times.cpu().numpy(), axes, labels_idx=labels_idx, marked=True)
        axes_g = plot_process_multi_axes(times.cpu().numpy(), axes_g, labels_idx=labels_idx, marked=False)
        labels = (labels != 0).type(labels.dtype) #?
        mask = (times != args.padding_id).type(times.dtype)
        times = times * args.time_scale #time_scale=1 by default
        window_start, window_end = get_window(times=times, window=args.window)
        events = get_events(
        times=times, mask=mask, labels=labels,
        window_start=window_start, window_end=window_end)
        
        first_event = float(min(times[0]))
        intensity_times = artifacts_seq['intensity_times'][j][1:] 
        mask = intensity_times > first_event + 0.001
        intensity_times = intensity_times[mask]

        true_marked_intensity = artifacts_seq['marked_intensity'][j][1:]
        true_ground_intensity = artifacts_seq['ground_intensity'][j][1:]
        true_marked_intensity = true_marked_intensity[mask]
        true_ground_intensity = true_ground_intensity[mask]
        
        intensity_times = th.tensor(intensity_times, dtype=th.float32).unsqueeze(0).to(args.device)
        log_ground_intensity, log_mark_pmf, _ , intensity_mask, _ = model.artifacts(
            query=intensity_times, events=events)
        
        ground_intensity = th.exp(log_ground_intensity)
        mark_pmf = th.exp(log_mark_pmf)
        marked_intensity = ground_intensity.unsqueeze(-1) * mark_pmf 

        marked_intensity = marked_intensity.squeeze(0).detach().cpu().numpy()
        ground_intensity = ground_intensity.squeeze(0).detach().cpu().numpy()
        diff_marked = mean_absolute_deviation(marked_intensity, true_marked_intensity)
        diff_ground = mean_absolute_deviation(ground_intensity, true_ground_intensity.squeeze())

        intensity_times = intensity_times.squeeze(0).detach().cpu().numpy()
        print(f'seq {j}, {diff_marked}')
        for i, ax in enumerate(axes):
            ax.plot(intensity_times, marked_intensity[:,i], color='green', label=f'Mod. ({np.round(diff_marked[i],3)})')
            ax.plot(intensity_times, true_marked_intensity[:,i], color='red', label='True')
            ax.legend(fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.tick_params(axis='both', which='minor', labelsize=16)
        fig.suptitle(map_model_name(args.model_name), fontsize=16, y=0.9)
        axes_g.plot(intensity_times, true_ground_intensity, color='red', label='True')
        axes_g.plot(intensity_times, ground_intensity, color='green', label=f'Mod. ({np.round(diff_ground,3)})')
        axes_g.legend()
        save_dir = f'figures/simulations4/{args.model_name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = f'{save_dir}/{j}.pdf'
        fig.savefig(save_path, bbox_inches='tight')
        #fig_g.savefig(f'figures/simulations2/ground_{args.dataset}_{args.model_name}_{args.split}.png', bbox_inches='tight')

def mean_absolute_deviation(intensity, true_intensity):
    diff = np.abs(true_intensity-intensity)
    diff = np.mean(diff, axis=0)
    return diff

def map_model_name(model_name):
    mapping={
    'gru_thp_temporal_with_labels':'THP',
    'gru_temporal_with_labels_gru_temporal_with_labels_thp-dd':'THP-DD',

    'gru_sahp_temporal_with_labels':'SAHP',
    'gru_temporal_with_labels_gru_temporal_with_labels_sahp-dd':'SAHP-DD',

    'gru_mlp-cm_temporal_with_labels':'FNN',
    'gru_temporal_with_labels_gru_temporal_with_labels_mlp-cm-dd':'FNN-DD',

    'gru_log-normal-mixture_temporal_with_labels':'LNM',
    'gru_temporal_with_labels_gru_temporal_with_labels_log-normal-mixture-dd':'LNM-DD',

    'gru_rmtpp_temporal_with_labels':'RMTPP',
    'gru_temporal_with_labels_gru_temporal_with_labels_rmtpp-dd':'RMTPP-DD'
    }
    return mapping[model_name]

if __name__ == "__main__":
    parsed_args = parse_args()
    seed = parsed_args.seed
    json_dir = os.path.join(os.getcwd(), parsed_args.load_from_dir)
    json_dir = os.path.join(json_dir, parsed_args.dataset)
    json_dir = os.path.join(json_dir, f'split_{parsed_args.split}')    
    json_path = os.path.join(json_dir, 'args.json')
    with open(json_path, 'r') as fp:
        args_dict_json = json.load(fp)
    args_dict = vars(parsed_args)
    args_dict.update(args_dict_json)
    args = Namespace(**args_dict) 
    args.device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    args.seed = seed
    #plot_hawkes_intensity(args)
    #plot_hawkes_intensity_unique(args)
    compute_intensity_differences(args)
    print('stop')