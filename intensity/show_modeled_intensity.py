
from cProfile import label
import numpy as np 
import os
import importlib, sys
sys.path.remove(os.path.abspath(os.path.join('..', 'neuralTPPs')))
sys.path.append(os.path.abspath(os.path.join('..', 'icml')))

import torch as th 
from pathlib import Path
import json
import pickle
import matplotlib.pyplot as plt 
from plot.acronyms import get_acronym
from plot.plots import map_datasets_name

from data_processing.simu_hawkes import simulate_process, plot_process, simulate_seqs
from data_processing.tick_plot_process import plot_point_process, _extract_process_interval, _plot_tick_bars, _plot_tick_intensity

from tpps.processes.multi_class_dataset import MultiClassDataset as Dataset
from tpps.utils.data import get_loader
from tpps.utils.events import get_events, get_window 
from tpps.models import get_model

from intensity_plots import plot_modelled_intensity, plot_modelled_density, plot_modelled_pmf, plot_entropy, plot_modelled_ground_density

from argparse import ArgumentParser, Namespace
from distutils.util import strtobool
from tpps.models.base.process import Process

import pickle as pkl

def parse_args_intensity():
    parser = ArgumentParser(allow_abbrev=False)
    # Model dir
    parser.add_argument("--model", type=str, required=True, 
                        help="Path of the saved model")
    # Run configuration
    parser.add_argument("--seed", type=int, default=0, help="The random seed.")
    parser.add_argument("--padding-id", type=float, default=-1.,
                        help="The value used in the temporal sequences to "
                             "indicate a non-event.")
    # Common model hyperparameters
    parser.add_argument("--batch-size", type=int, default=1,
                        help="The batch size to use for parametric model"
                             " training and evaluation.")
    parser.add_argument("--time-scale", type=float, default=1.,
                        help='Time scale used to prevent overflow')
    parser.add_argument("--multi-labels",
                        type=lambda x: bool(strtobool(x)), default=False,
                        help="Whether the likelihood is computed on "
                             "multi-labels events or not")
    parser.add_argument("--window", type=int, default=10,
                        help="The window of the simulated process.py. Also "
                             "taken as the window of any parametric Hawkes "
                             "model if chosen.")
    # Dirs
    parser.add_argument("--load-from-dir", type=str, required=True,
                        help="If not None, load data from a directory")
    parser.add_argument("--dataset", type=str, required=True,
                        help="If not None, load data from a directory")
    parser.add_argument("--split", type=str, default='split_0',
                        help="If not None, load data from a directory")
    parser.add_argument("--plots-dir", type=str,
                        default="~/neural-tpps/plots",
                        help="Directory to save the plots")
    parser.add_argument("--save-fig-dir", type=str, default=None,
                        help="Directory to save the preprocessed data")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="The top-level directory where the model is saved.")
    parser.add_argument("--save-results-dir", type=str, default=None,
                        help="The top-level directory where the model is saved.")
                
    #plot
    parser.add_argument("--seq-num", type=int, default=None,
                        help="Test sequence to plot. If specified, overides arguments below.")
    parser.add_argument("--min-event-per-seq", type=int, default=1,
                        help="Minimum number required for the sequences")
    parser.add_argument("--max-event-per-seq", type=int, default=1,
                        help="Maximum number required for the sequences")
    parser.add_argument("--mark-per-seq", type=int, default=4,
                        help="Number of different mark per sequence")
    parser.add_argument("--ground-intensity", default=True,
                        type=lambda x: bool(strtobool(x)),
                        help="If True, shows the ground intensity.")
    parser.add_argument("--x-lims", default=[0,10], nargs='+', 
                        type=float,
                        help="Limits on x-axis")
    parser.add_argument("--y-lims", default=1, 
                        type=float,
                        help="Upper limit on y-axis")                  
    #Simulation
    parser.add_argument("--window-end", type=int, default=10,
                        help="End window of the simulated process")
    parser.add_argument("--size", type=str, default=1000,
                        help="Number of independent sequences to simulate")
    parser.add_argument("--marked", default=True,
                        type=lambda x: bool(strtobool(x)))
    parser.add_argument("--type", choices=['homogeneous', 'heterogeneous'],
                        type=str, default='homogeneous')
    parser.add_argument("--n-seq", type=int, default=1)
    parser.add_argument("--plot", type=lambda x: bool(strtobool(x)), default=True)
    #Monte Carlo
    parser.add_argument("--decoder-mc-prop-est", type=float, default=1.,
                        help="Proportion of MC samples, "
                             "compared to dataset size")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Name of the experiment")
    args, _ = parser.parse_known_args()
    
    cwd = Path.cwd()
    if 'hawkes_commontest' in args.dataset:
        data_args_path = os.path.join(args.load_from_dir, 'args.json')
    else:
        data_args_path = os.path.join(args.load_from_dir, args.dataset)
        data_args_path = os.path.join(data_args_path, args.split)
        data_args_path = os.path.join(data_args_path, 'args.json')
        data_args_path = os.path.join(cwd, data_args_path)
    model_path = os.path.join(args.model_dir, args.dataset)
    model_dir = os.path.join(cwd, model_path)
    with open(data_args_path, 'r') as fp:
        args_dict_json = json.load(fp)
    args_dict_json.update(vars(args))
    #args_dict = vars(args)
    #args_dict.update(args_dict_json)
    args_dict_json["model_dir"] = model_dir
    model_split = args.split.split('_')[0] + args.split.split('_')[1]
    args_dict_json["model_split"] = model_split
    args = Namespace(**args_dict_json)
    return args

def update_args(args, model_file):
    mc = args.decoder_mc_prop_est
    cwd = Path.cwd()
    model_name = model_file.split('/')[-1][:-3]
    model_args_path = os.path.join(Path(model_file).parent, 'args')
    model_args_path = os.path.join(model_args_path, model_name + 'json')
    model_args_path = os.path.join(cwd, model_args_path)
    args_dict = vars(args)
    with open(model_args_path, 'r') as f:
        model_args_dic = json.load(f)
    model_args_dic.update(args_dict)
    args = Namespace(**model_args_dic)
    args.device = th.device('cpu')
    args.verbose = False
    args.batch_size = 1
    args.decoder_mc_prop_est = mc
    return args, model_name



def reformat(process):
    labelled_process = [[[time, label] for label, dimension in enumerate(seq) for time in dimension] for seq in process]
    sorted_process = [sorted(seq, key= lambda seq: seq[0]) for seq in labelled_process]
    dic_process = [[{'time':event[0], 'labels':[event[1]]} for event in seq] for seq in sorted_process]
    return dic_process 

def reformat_timestamps(sequence):
    marks = np.unique([event['labels'][0] for event in sequence])
    timestamps = [[event['time'] for event in sequence if event['labels'][0] == mark] for mark in marks]
    return np.array(timestamps)


def get_test_sequence(args):
    if 'hawkes' in args.dataset:
        args_file = os.path.join(args.load_from_dir, 'args.json')
        print(args_file)
        with open(args_file, 'r') as f:
            args_dataset = json.load(f)
        args_dataset.update(vars(args))     
        args = Namespace(**args_dataset)
        if 'n_processes' not in vars(args):
            args_dic = vars(args)
            args_dic['n_processes'] = 1
            args = Namespace(**args_dic)
        sequence = []
        artifacts = {'marked_intensity':[], 'ground_intensity':[], 'true_mark_pmf':[], 'intensity_times':[]}
        for i in range(args.n_processes):
            print(f'simulating process {i+1}')
            args_process = vars(args).copy()
            args_process['decays'] = args.decays[i]
            args_process['adjacency'] = args.adjacency[i]
            args_process['baselines'] = args.baselines[i]
            args_process = Namespace(**args_process)
            
            seq_process, artifacts_process = simulate_seqs(args_process, n_seq=args.n_seq)
            sequence.extend(seq_process)
            artifacts['marked_intensity'].extend(artifacts_process['marked_intensity'])
            artifacts['ground_intensity'].extend(artifacts_process['ground_intensity'])
            artifacts['true_mark_pmf'].extend(artifacts_process['true_mark_pmf'])
            artifacts['intensity_times'].extend(artifacts_process['intensity_times'])
    else:
        dataset_path = os.path.join(args.load_from_dir, args.dataset)
        dataset_path = os.path.join(dataset_path, args.split) 
        sequence = None
        dataset_path = os.path.join(dataset_path, 'test.json') 
        with open(dataset_path, 'r') as f:
            print(dataset_path)
            dataset = json.load(f)  
        if args.seq_num is not None:
            sequence = dataset[args.seq_num]
        else:
            for i, seq in enumerate(dataset):
                unique_marks = np.unique([event['labels'][0] for event in seq])
                if (len(unique_marks) == args.mark_per_seq) and (args.min_event_per_seq <= len(seq) <= args.max_event_per_seq):
                    sequence = seq
                    print(f'Sequence {i}')
                    print(len(seq))
                    break
        if sequence is None:
            raise ValueError('Not sequence of minimum length found.')
        artifacts = {}
    return sequence, artifacts

def plot_sequence_intensity(args):
    sequence, artifacts_seq = get_test_sequence(args)
    print(f'NUM EVENTS: {len(sequence)}')
    unique_marks = np.unique([event['labels'][0] for event in sequence])
    print(f'NUM MARKS: {len(unique_marks)}')
    timestamps =  [[event['time'] for event in sequence]]
    print(timestamps)
    sequence = [sequence]
    t_min, t_max = args.x_lims[0], args.x_lims[1]
    step = 0.000001
    all_models_files = os.listdir(args.model_dir)
    file_to_find = args.model + '_' + args.model_split
    print(file_to_find)
    model_file = None
    for file in all_models_files:
        if file.startswith(file_to_find):
            model_file = os.path.join(args.model_dir, file)
            break
    if model_file is None:
        raise ValueError('Checkpoint not found')
    args, model_name = update_args(args, model_file)
    model_name = model_file.split('/')[-1][:-3]
    data_sequence = Dataset(
    args=args, size=1, seed=args.seed, data=sequence)
    loader = get_loader(data_sequence, args=args, shuffle=False)
    model = get_model(args)
    model.load_state_dict(th.load(model_file, map_location=args.device))
    for batch in loader:
        times, labels = batch["times"], batch["labels"]
        labels_idx = th.argmax(labels, dim=-1).squeeze(-1).cpu().numpy()
        if 'hawkes' in args.dataset:
            fig, ax = plt.subplots(1,2, figsize=(15,10))
            fig_en, ax_en = plt.subplots(1,2, figsize=(20,10))
            fig_pmf, ax_pmf = plt.subplots(1,2, figsize=(25,15))
            for i in range(2): 
                ax_pmf[i] = plot_process(times.cpu().numpy(), ax_pmf[i], labels_idx=labels_idx)
                ax_en[i] = plot_process(times.cpu().numpy(), ax_en[i], labels_idx=labels_idx) 
                ax[i] = plot_process(times.cpu().numpy(), ax[i], labels_idx=labels_idx)
        else:
            fig, ax = plt.subplots(1,1)
            fig_en, ax_en = plt.subplots(1,1)
            fig_pmf, ax_pmf = plt.subplots(1,1, figsize=(20,8))
            ax_pmf = plot_process(times.cpu().numpy(), ax_pmf, labels_idx=labels_idx)
            ax_en = plot_process(times.cpu().numpy(), ax_en, labels_idx=labels_idx) 
            ax = plot_process(times.cpu().numpy(), ax, labels_idx=labels_idx)       
        labels = (labels != 0).type(labels.dtype) #?
        mask = (times != args.padding_id).type(times.dtype)
        times = times * args.time_scale #time_scale=1 by default
        window_start, window_end = get_window(times=times, window=args.window_end)
        events = get_events(
        times=times, mask=mask, labels=labels,
        window_start=window_start, window_end=window_end)
        query_times = th.arange(t_min, t_max+step, step=step, dtype=th.float32).unsqueeze(0)        
        #print(times)

        ground_density, log_ground_density, mark_pmf, _ = get_density(model, query_times, events, args) 
        ground_density_events, log_ground_density_events, mark_pmf_events, log_mark_pmf_events = get_density(model, times, events, args, is_event=True)
        ground_intensity, log_ground_intensity, ground_intensity_integral, intensity_mask, artifacts, intensity = get_intensity(model, query_times, events, args)

        #print(np.argmax(mark_pmf_events, axis=-1))
        true_events_pmf = th.tensor(mark_pmf_events) * labels
        true_events_pmf = th.sum(true_events_pmf[0,1:,:], dim=-1)

        print(true_events_pmf)
        results = {'events_pmf': -th.log(true_events_pmf)}
        save_file = f'results/plot/{args.model}.txt'
        cwd = Path.cwd()
        save_file = os.path.join(cwd, save_file)
        with open(save_file, 'wb') as f:    
            pkl.dump(results, f)
        total_events_pmf = np.round(-th.sum(th.log(true_events_pmf)).cpu().numpy().astype(np.float64),2)
        query_times = query_times.squeeze().cpu().numpy()
        print('Total Sequence NLL-M : {}'.format(total_events_pmf))
        accuracy = get_accuracy(mark_pmf_events, labels_idx[0])
        print(f'Predicted sequence accuracy: {accuracy}')
        if 'hawkes' in args.dataset:
            accuracy_true = get_accuracy(artifacts_seq['true_mark_pmf'], labels_idx[0], true_pmf=True, 
                                         event_times=times.cpu().numpy()[0], all_times=artifacts_seq['intensity_times'])
            print(f'True sequence accuracy: {accuracy_true}')
            nll_true = get_true_nll(artifacts_seq['true_mark_pmf'], labels_idx[0], 
                                args.marks,event_times=times.cpu().numpy(), 
                                all_times=artifacts_seq['intensity_times'])
            brier = brier_score(artifacts_seq['true_mark_pmf'], mark_pmf, 
                            all_times=artifacts_seq['intensity_times'], 
                            event_times=times.cpu().numpy()[0])
        if 'hawkes' in args.dataset:
            plot_modelled_intensity(query_times, intensity, ax[0], args, model_name, label_idxs=labels_idx)
            plot_modelled_intensity(artifacts_seq['intensity_times'], artifacts_seq['marked_intensity'], ax[1], args, model_name, label_idxs=labels_idx)
            
            plot_modelled_pmf(query_times, mark_pmf, ax_pmf[0], labels_idx)
            plot_modelled_pmf(artifacts_seq['intensity_times'], artifacts_seq['true_mark_pmf'], ax_pmf[1], labels_idx)
            plot_entropy(query_times, mark_pmf, ax_en[0])
            plot_entropy(artifacts_seq['intensity_times'], artifacts_seq['true_mark_pmf'], ax_en[1])
        else:
            plot_modelled_intensity(query_times, intensity, ax, args, model_name, label_idxs=labels_idx)
            plot_modelled_pmf(query_times, mark_pmf, ax_pmf, labels_idx)
            plot_entropy(query_times, mark_pmf, ax_en)
        title = args.model
        title_pmf = 'pmf_' + title 
        print(title)
        if 'LastFM' in title:
            title = title.split('LastFM-')[1]
            fig.suptitle(title, fontsize=24)
            title = 'LastFM-' + title
        else:
            fig.suptitle(title, fontsize=24)
        if 'hawkes' in args.dataset:
            for i in range(2):
                ax[i].legend()
                ax[i].set_xlim(t_min, t_max)
                ax_pmf[i].set_xlim(t_min, t_max)
                ax_en[i].set_xlim(t_min, t_max)
                ax[i].grid(True)
                ax_pmf[i].grid(True)
                ax[i].set_ylabel(r'$\lambda^\ast(t)~\backslash~f(t|\mathcal{H}_t)$', fontsize=24)
                ax_pmf[i].set_ylabel(r'$p^\ast(k|t)$', fontsize=24)
                ax[i].tick_params(axis='both', which='major', labelsize=24 )
                ax_pmf[i].tick_params(axis='both', which='major', labelsize=24 )
                ax[i].set_xlabel(r'$t$', fontsize=24) 
                ax_pmf[i].set_xlabel(r'$t$', fontsize=24) 
                ax_en[i].set_ylabel('Entropy', fontsize=24)
                ax_en[i].tick_params(axis='both', which='major', labelsize=20)
            ax_pmf[0].set_title(f'NLL: {total_events_pmf}, Acc: {np.round(accuracy,2)}, Brier: {np.round(brier,4)}', fontsize=24)
            ax_pmf[1].set_title(f'NLL: {np.round(nll_true,2)}, Acc: {np.round(accuracy_true,2)}', fontsize=24) 
        else:
            #ax.legend()
            ax.set_xlim(t_min, t_max)
            ax_pmf.set_xlim(t_min, t_max)
            ax_en.set_xlim(t_min, t_max)
            ax_pmf.set_ylim(ax_pmf.get_ylim()[0],args.y_lims)
            ax.grid(True)
            ax_pmf.grid(True)
            ax.set_ylabel(r'$\lambda^\ast(t)~\backslash~f(t|\mathcal{H}_t)$', fontsize=24)
            ax_pmf.set_ylabel(r'$p^\ast(k|t)$', fontsize=24)
            ax.tick_params(axis='both', which='major', labelsize=24 )
            ax_pmf.tick_params(axis='both', which='major', labelsize=24 )
            ax.set_xlabel(r'$t$', fontsize=24) 
            ax_pmf.set_xlabel(r'$t$', fontsize=24) 
            ax_en.set_ylabel('Entropy', fontsize=24)
            ax_en.tick_params(axis='both', which='major', labelsize=20)
            ticks = [tick for tick in plt.gca().get_yticks() if tick >=0]
            ax_pmf.set_yticks(ticks)
        if args.model.startswith('poisson'):
            model_name = args.model.replace('poisson_', '').replace(args.dataset + '_', '') + '_base'
        else:
            model_name = args.model.replace(args.dataset + '_', '')
        pmf_fig_title = map_datasets_name(args.dataset) + ' ' + get_acronym([model_name])[0]
        fig_pmf.suptitle(pmf_fig_title, fontsize=24)
        title_en = 'entropy_' + title
        fig.savefig('figures/intensities/{}.png'.format(title.replace('/', '_')), bbox_inches='tight')
        fig_pmf.savefig('figures/intensities/{}.png'.format(title_pmf.replace('/', '_')), bbox_inches='tight')
        fig_en.savefig('figures/intensities/{}.png'.format(title_en.replace('/', '_')), bbox_inches='tight')
        print('DONE')

###########################################################################@
############################################################################

def plot_sequence_intensity_iclr(args):
    sequence, artifacts_seq = get_test_sequence(args)
    print(f'NUM EVENTS: {len(sequence)}')
    unique_marks = np.unique([event['labels'][0] for event in sequence])
    print(f'NUM MARKS: {len(unique_marks)}')
    timestamps =  [[event['time'] for event in sequence]]
    print(timestamps)
    sequence = [sequence]
    t_min, t_max = args.x_lims[0], args.x_lims[1]
    step = 0.0000001
    all_models_files = os.listdir(args.model_dir)
    file_to_find = args.model + '_' + args.model_split
    print(file_to_find)
    model_file = None
    for file in all_models_files:
        if file.startswith(file_to_find):
            model_file = os.path.join(args.model_dir, file)
            break
    if model_file is None:
        raise ValueError('Checkpoint not found')
    args, model_name = update_args(args, model_file)
    model_name = model_file.split('/')[-1][:-3]
    data_sequence = Dataset(
    args=args, size=1, seed=args.seed, data=sequence)
    loader = get_loader(data_sequence, args=args, shuffle=False)
    model = get_model(args)
    model.load_state_dict(th.load(model_file, map_location=args.device))
    for batch in loader:
        times, labels = batch["times"], batch["labels"]
        labels_to_show_mask1 =  t_min < times 
        labels_to_show_mask2 = times < t_max
        labels_to_show_mask = labels_to_show_mask1 * labels_to_show_mask2
        labels_to_show = labels[labels_to_show_mask]
        times_to_show = times[labels_to_show_mask]
        #times_mask = times < t_max
        #times_process = times[times_mask]
        #times = times_process[:-1].unsqueeze(0)
        #labels_process = labels[times_mask]
        #labels = labels_process[:-1].unsqueeze(0)
        labels_idx = th.argmax(labels_to_show, dim=-1).squeeze(-1).cpu().numpy()        
        fig, axes = plt.subplots(3,1, figsize=(12,8), gridspec_kw={'height_ratios': [2, 2, 1]})
        #fig_en, ax_en = plt.subplots(1,1)
        #fig_pmf, ax_pmf = plt.subplots(1,1, figsize=(20,8))
        #ax_pmf = plot_process(times.cpu().numpy(), ax_pmf, labels_idx=labels_idx)
        #ax_en = plot_process(times.cpu().numpy(), ax_en, labels_idx=labels_idx) 
        axes[2] = plot_process(times_to_show.cpu().numpy(), axes[2], labels_idx=labels_idx)       
        labels = (labels != 0).type(labels.dtype) #?
        mask = (times != args.padding_id).type(times.dtype)
        times = times * args.time_scale #time_scale=1 by default
        window_start, window_end = get_window(times=times, window=args.window_end)
        events = get_events(
        times=times, mask=mask, labels=labels,
        window_start=window_start, window_end=window_end)
        query_times = th.arange(t_min, t_max+step, step=step, dtype=th.float32).unsqueeze(0)        
        ground_density, log_ground_density, mark_pmf, _ = get_density(model, query_times, events, args) 
        ground_density_events, log_ground_density_events, mark_pmf_events, log_mark_pmf_events = get_density(model, times, events, args, is_event=True)
        #ground_intensity, log_ground_intensity, ground_intensity_integral, intensity_mask, artifacts, intensity = get_intensity(model, query_times, events, args)
        
        predicted_event = mark_pmf_events.argmax(-1)
        true_event = labels.argmax(-1).numpy().squeeze()
        print(ground_density_events)
        print(predicted_event)
        print(true_event)
        print('ACC:',np.mean(predicted_event == true_event))
        #print(np.argmax(mark_pmf_events, axis=-1))
        #true_events_pmf = th.tensor(mark_pmf_events) * labels
        #true_events_pmf = th.sum(true_events_pmf[0,1:,:], dim=-1)

        #print(true_events_pmf)
        #results = {'events_pmf': -th.log(true_events_pmf)}
        #save_file = f'results/plot/{args.model}.txt'
        #cwd = Path.cwd()
        #save_file = os.path.join(cwd, save_file)
        #with open(save_file, 'wb') as f:    
        #    pkl.dump(results, f)
        #total_events_pmf = np.round(-th.sum(th.log(true_events_pmf)).cpu().numpy().astype(np.float64),2)
        query_times = query_times.squeeze().cpu().numpy()
        #print('Total Sequence NLL-M : {}'.format(total_events_pmf))
        #accuracy = get_accuracy(mark_pmf_events, labels_idx[0])
        #print(f'Predicted sequence accuracy: {accuracy}')
        #plot_modelled_intensity(query_times, intensity, ax, args, model_name, label_idxs=labels_idx)
        
        
        plot_modelled_pmf(query_times, mark_pmf, axes[0], labels_idx)
        plot_modelled_ground_density(query_times, ground_density, axes[1])
        #plot_entropy(query_times, mark_pmf, ax_en)
        title = args.model
        title_pmf = 'pmf_' + title 
        print(title)
        '''
        if 'LastFM' in title:
            title = title.split('LastFM-')[1]
            fig.suptitle(title, fontsize=24)
            title = 'LastFM-' + title
        else:
            fig.suptitle(title, fontsize=24)
        '''
        for ax in axes:
            ax.set_xlim(t_min, t_max)
            ax.grid(True)
            ax.tick_params(axis='both', which='major', labelsize=24)
        #ax_pmf.set_xlim(t_min, t_max)
        #ax_en.set_xlim(t_min, t_max)
        axes[0].set_ylim(axes[0].get_ylim()[0],args.y_lims)
        #axes[1].set_ylim(0,1000)
        axes[0].set_xticks([])
        axes[1].set_xticks([])
        axes[2].set_yticks([])
        #ax.grid(True)
        #ax_pmf.grid(True)
        axes[1].set_ylabel(r'$f^\ast(t)$', fontsize=24)
        axes[0].set_ylabel(r'$p^\ast(k|t)$', fontsize=24)
        '''
        ax_pmf.tick_params(axis='both', which='major', labelsize=24 )
        ax.set_xlabel(r'$t$', fontsize=24) 
        ax_pmf.set_xlabel(r'$t$', fontsize=24) 
        ax_en.set_ylabel('Entropy', fontsize=24)
        ax_en.tick_params(axis='both', which='major', labelsize=20)
        ticks = [tick for tick in plt.gca().get_yticks() if tick >=0]
        ax_pmf.set_yticks(ticks)
        '''
        if args.model.startswith('poisson'):
            model_name = args.model.replace('poisson_', '').replace(args.dataset + '_', '') + '_base'
        else:
            model_name = args.model.replace(args.dataset + '_', '')
        pmf_fig_title = map_datasets_name(args.dataset) + ' ' + get_acronym([model_name])[0]
        #fig_pmf.suptitle(pmf_fig_title, fontsize=24)
        title_en = 'entropy_' + title
        fig.savefig('figures/iclr/{}_bis.png'.format(title.replace('/', '_')), bbox_inches='tight')
        #fig_pmf.savefig('figures/intensities/{}.png'.format(title_pmf.replace('/', '_')), bbox_inches='tight')
        #fig_en.savefig('figures/intensities/{}.png'.format(title_en.replace('/', '_')), bbox_inches='tight')
        print('DONE')


#########################################################################################
#########################################################################################


def show_joint_distribution(args):
    sequence, artifacts_seq = get_test_sequence(args)
    print(f'NUM EVENTS: {len(sequence)}')
    unique_marks = np.unique([event['labels'][0] for event in sequence])
    print(f'NUM MARKS: {len(unique_marks)}')
    timestamps =  [[event['time'] for event in sequence]]
    print(timestamps)
    sequence = [sequence]
    t_min, t_max = args.x_lims[0], args.x_lims[1]
    step = 0.0000001
    all_models_files = os.listdir(args.model_dir)
    file_to_find = args.model + '_' + args.model_split
    print(file_to_find)
    model_file = None
    for file in all_models_files:
        if file.startswith(file_to_find):
            model_file = os.path.join(args.model_dir, file)
            break
    if model_file is None:
        raise ValueError('Checkpoint not found')
    args, model_name = update_args(args, model_file)
    model_name = model_file.split('/')[-1][:-3]
    data_sequence = Dataset(
    args=args, size=1, seed=args.seed, data=sequence)
    loader = get_loader(data_sequence, args=args, shuffle=False)
    model = get_model(args)
    model.load_state_dict(th.load(model_file, map_location=args.device))
    for batch in loader:
        times, labels = batch["times"], batch["labels"]
        num_marks = labels.shape[-1]
        labels_to_show_mask1 =  t_min < times 
        labels_to_show_mask2 = times < t_max
        labels_to_show_mask = labels_to_show_mask1 * labels_to_show_mask2
        labels_to_show = labels[labels_to_show_mask]
        times_to_show = times[labels_to_show_mask]
        #times_mask = times < t_max
        #times_process = times[times_mask]
        #times = times_process[:-1].unsqueeze(0)
        #labels_process = labels[times_mask]
        #labels = labels_process[:-1].unsqueeze(0)
        labels_idx = th.argmax(labels_to_show, dim=-1).squeeze(-1).cpu().numpy()        
        
        #fig, axes = plt.subplots(num_marks+1,1, figsize=(12,8), gridspec_kw={'height_ratios': [2, 2, 1]})
        fig, axes = plt.subplots(num_marks+1,1, figsize=(12,80))
        #fig_en, ax_en = plt.subplots(1,1)
        #fig_pmf, ax_pmf = plt.subplots(1,1, figsize=(20,8))
        #ax_pmf = plot_process(times.cpu().numpy(), ax_pmf, labels_idx=labels_idx)
        #ax_en = plot_process(times.cpu().numpy(), ax_en, labels_idx=labels_idx) 
        axes[-1] = plot_process(times_to_show.cpu().numpy(), axes[-1], labels_idx=labels_idx)       
        labels = (labels != 0).type(labels.dtype) #?
        mask = (times != args.padding_id).type(times.dtype)
        times = times * args.time_scale #time_scale=1 by default
        window_start, window_end = get_window(times=times, window=args.window_end)
        events = get_events(
        times=times, mask=mask, labels=labels,
        window_start=window_start, window_end=window_end)
        query_times = th.arange(t_min, t_max+step, step=step, dtype=th.float32).unsqueeze(0)        
        density, ground_density, log_ground_density, mark_pmf, _ = get_density(model, query_times, events, args) 
        _, ground_density_events, log_ground_density_events, mark_pmf_events, log_mark_pmf_events = get_density(model, times, events, args, is_event=True)
        #ground_intensity, log_ground_intensity, ground_intensity_integral, intensity_mask, artifacts, intensity = get_intensity(model, query_times, events, args)
        
        #predicted_event = mark_pmf_events.argmax(-1)
        #true_event = labels.argmax(-1).numpy().squeeze()
        #print(ground_density_events)
        #print(predicted_event)
        #print(true_event)
        #print('ACC:',np.mean(predicted_event == true_event))
        #print(np.argmax(mark_pmf_events, axis=-1))
        #true_events_pmf = th.tensor(mark_pmf_events) * labels
        #true_events_pmf = th.sum(true_events_pmf[0,1:,:], dim=-1)

        #print(true_events_pmf)
        #results = {'events_pmf': -th.log(true_events_pmf)}
        #save_file = f'results/plot/{args.model}.txt'
        #cwd = Path.cwd()
        #save_file = os.path.join(cwd, save_file)
        #with open(save_file, 'wb') as f:    
        #    pkl.dump(results, f)
        #total_events_pmf = np.round(-th.sum(th.log(true_events_pmf)).cpu().numpy().astype(np.float64),2)
        query_times = query_times.squeeze().cpu().numpy()
        #print('Total Sequence NLL-M : {}'.format(total_events_pmf))
        #accuracy = get_accuracy(mark_pmf_events, labels_idx[0])
        #print(f'Predicted sequence accuracy: {accuracy}')
        #plot_modelled_intensity(query_times, intensity, ax, args, model_name, label_idxs=labels_idx)
        
        
        #plot_modelled_pmf(query_times, mark_pmf, axes[0], labels_idx)
        #plot_modelled_ground_density(query_times, ground_density, axes[1])
        plot_joint_distribution(query_times, density, axes)
        #plot_entropy(query_times, mark_pmf, ax_en)
        title = args.model
        title_pmf = 'pmf_' + title 
        print(title)
        '''
        if 'LastFM' in title:
            title = title.split('LastFM-')[1]
            fig.suptitle(title, fontsize=24)
            title = 'LastFM-' + title
        else:
            fig.suptitle(title, fontsize=24)
        '''
        for ax in axes:
            ax.set_xlim(t_min, t_max)
            ax.grid(True)
            ax.tick_params(axis='both', which='major', labelsize=24)
            ax.set_xticks([])
        axes[-1].set_yticks([])
        #ax_pmf.set_xlim(t_min, t_max)
        #ax_en.set_xlim(t_min, t_max)
        #axes[0].set_ylim(axes[0].get_ylim()[0],args.y_lims)
        #axes[1].set_ylim(0,1000)
        #ax.grid(True)
        #ax_pmf.grid(True)
        #axes[1].set_ylabel(r'$f^\ast(t)$', fontsize=24)
        #axes[0].set_ylabel(r'$p^\ast(k|t)$', fontsize=24)
        '''
        ax_pmf.tick_params(axis='both', which='major', labelsize=24 )
        ax.set_xlabel(r'$t$', fontsize=24) 
        ax_pmf.set_xlabel(r'$t$', fontsize=24) 
        ax_en.set_ylabel('Entropy', fontsize=24)
        ax_en.tick_params(axis='both', which='major', labelsize=20)
        ticks = [tick for tick in plt.gca().get_yticks() if tick >=0]
        ax_pmf.set_yticks(ticks)
        '''
        if args.model.startswith('poisson'):
            model_name = args.model.replace('poisson_', '').replace(args.dataset + '_', '') + '_base'
        else:
            model_name = args.model.replace(args.dataset + '_', '')
        pmf_fig_title = map_datasets_name(args.dataset) + ' ' + get_acronym([model_name])[0]
        #fig_pmf.suptitle(pmf_fig_title, fontsize=24)
        title_en = 'entropy_' + title
        if args.exp_name is None:
            fig.savefig('figures/iclr/{}_joint.png'.format(title.replace('/', '_')), bbox_inches='tight')
        else:
            fig.savefig('figures/iclr/{}_{}_joint.png'.format(title.replace('/', '_'), args.exp_name), bbox_inches='tight')
        #fig_pmf.savefig('figures/intensities/{}.png'.format(title_pmf.replace('/', '_')), bbox_inches='tight')
        #fig_en.savefig('figures/intensities/{}.png'.format(title_en.replace('/', '_')), bbox_inches='tight')
        print('DONE')


def plot_joint_distribution(times, density, axes):
    for i, ax in enumerate(axes[:-1]):
        ax.plot(times, density[:,i], color='forestgreen')



def plot_hawkes_intensity(args):
    sequence, artifacts_seq = get_test_sequence(args) #[np, L]
    t_min, t_max = args.x_lims[0], args.x_lims[1]
    step = 0.001
    all_models_files = os.listdir(args.model_dir)
    file_to_find = args.model + '_' + args.model_split
    print(file_to_find)
    model_file = None
    for file in all_models_files:
        if file.startswith(file_to_find):
            model_file = os.path.join(args.model_dir, file)
            break
    if model_file is None:
        raise ValueError('Checkpoint not found')
    args, model_name = update_args(args, model_file)
    model_name = model_file.split('/')[-1][:-3]
    data_sequence = Dataset(
    args=args, size=args.n_processes, seed=args.seed, data=sequence)
    loader = get_loader(data_sequence, args=args, shuffle=False)
    model = get_model(args)
    model.load_state_dict(th.load(model_file, map_location=args.device))
    #fig, ax = plt.subplots(2,3, figsize=(15,10))
    #fig_en, ax_en = plt.subplots(2,3, figsize=(20,10))
    fig_pmf, ax_pmf = plt.subplots(2,3, figsize=(35,20))
    for i, batch in enumerate(loader):
        times, labels = batch["times"], batch["labels"]
        labels_idx = th.argmax(labels, dim=-1).squeeze(-1).cpu().numpy()
        print(labels_idx)
        for j in range(2): 
            ax_pmf[j,i] = plot_process(times.cpu().numpy(), ax_pmf[j,i], labels_idx=labels_idx)
            #ax_en[i] = plot_process(times.cpu().numpy(), ax_en[i], labels_idx=labels_idx) 
            #ax[i] = plot_process(times.cpu().numpy(), ax[i], labels_idx=labels_idx)
        labels = (labels != 0).type(labels.dtype) #?
        mask = (times != args.padding_id).type(times.dtype)
        times = times * args.time_scale #time_scale=1 by default
        window_start, window_end = get_window(times=times, window=args.window_end)
        events = get_events(
        times=times, mask=mask, labels=labels,
        window_start=window_start, window_end=window_end)
        query_times = th.arange(t_min, t_max+step, step=step, dtype=th.float32).unsqueeze(0)        

        ground_density, log_ground_density, mark_pmf, _ = get_density(model, query_times, events, args) 
        ground_density_events, log_ground_density_events, mark_pmf_events, log_mark_pmf_events = get_density(model, times, events, args, is_event=True)
        ground_intensity, log_ground_intensity, ground_intensity_integral, intensity_mask, artifacts, intensity = get_intensity(model, query_times, events, args)

        true_events_pmf = th.tensor(mark_pmf_events) * labels
        true_events_pmf = th.sum(true_events_pmf[0,1:,:], dim=-1)

        total_events_pmf = -th.sum(th.log(true_events_pmf)).cpu().numpy().astype(np.float64)
        total_events_pmf = np.round(total_events_pmf, 2)
        query_times = query_times.squeeze().cpu().numpy()
        print('Total Sequence PMF : {}'.format(total_events_pmf))
        accuracy = get_accuracy(mark_pmf_events, labels_idx[0])
        print(f'Predicted sequence accuracy: {accuracy}')
        accuracy_true = get_accuracy(artifacts_seq['true_mark_pmf'][i], labels_idx[0], true_pmf=True, 
                                         event_times=times.cpu().numpy()[0], all_times=artifacts_seq['intensity_times'][i])
        print(f'True sequence accuracy: {accuracy_true}')
        nll_true = get_true_nll(artifacts_seq['true_mark_pmf'][i], labels_idx[0], 
                                args.marks,event_times=times.cpu().numpy(), 
                                all_times=artifacts_seq['intensity_times'][i])        
        brier = brier_score(artifacts_seq['true_mark_pmf'][i], mark_pmf, 
                            all_times=artifacts_seq['intensity_times'][i], 
                            event_times=times.cpu().numpy()[0])
        

        print(f'Brier score: {brier}')
        #plot_modelled_intensity(query_times, intensity, ax[0], args, model_name, label_idxs=labels_idx)
        #plot_modelled_intensity(artifacts_seq['intensity_times'], artifacts_seq['marked_intensity'], ax[1], args, model_name, label_idxs=labels_idx)
        if args.plot:
            plot_modelled_pmf(query_times, mark_pmf, ax_pmf[0,i], labels_idx)
            plot_modelled_pmf(artifacts_seq['intensity_times'][i], artifacts_seq['true_mark_pmf'][i], ax_pmf[1,i], labels_idx)
            #plot_entropy(query_times, mark_pmf, ax_en[0])
            #plot_entropy(artifacts_seq['intensity_times'], artifacts_seq['true_mark_pmf'], ax_en[1])
            
            title = args.model
            title_pmf = 'pmf_' + title 
            print(title)
            '''
            if 'LastFM' in title:
                title = title.split('LastFM-')[1]
                fig.suptitle(title, fontsize=24)
                title = 'LastFM-' + title
            else:
                fig.suptitle(title, fontsize=24)
            '''
            for j in range(2):
                ax_pmf[j,i].set_xlim(t_min, t_max)
                ax_pmf[j,i].grid(True)
                ax_pmf[j,i].set_ylabel(r'$p^\ast(k|t)$', fontsize=24)
                ax_pmf[j,i].tick_params(axis='both', which='major', labelsize=24 )
                ax_pmf[j,i].set_xlabel(r'$t$', fontsize=24)  
            ax_pmf[0,i].set_title(f'NLL: {np.round(total_events_pmf,2)}, Acc: {np.round(accuracy,2)}, Brier: {np.round(brier,4)}', fontsize=24)
            ax_pmf[1,i].set_title(f'NLL: {np.round(nll_true,2)}, Acc: {np.round(accuracy_true,2)}', fontsize=24)

            #for ax in ax_pmf:
            #   for ax2 in ax:
                    #ax[i].legend()
                    #ax[i].set_xlim(t_min, t_max)
                    #ax_en[i].set_xlim(t_min, t_max)
                    #ax[i].grid(True)
                    #ax[i].set_ylabel(r'$\lambda^\ast(t)~\backslash~f(t|\mathcal{H}_t)$', fontsize=24)
                    #ax[i].tick_params(axis='both', which='major', labelsize=24 )
                    #ax[i].set_xlabel(r'$t$', fontsize=24) 
                    #ax_en[i].set_ylabel('Entropy', fontsize=24)
                    #ax_en[i].tick_params(axis='both', which='major', labelsize=20)
            if args.model.startswith('poisson'):
                model_name = args.model.replace('poisson_', '').replace(args.dataset + '_', '') + '_base'
            else:
                model_name = args.model.replace(args.dataset + '_', '')
            pmf_fig_title = map_datasets_name(args.dataset) + ' ' + get_acronym([model_name])[0]
            fig_pmf.suptitle(pmf_fig_title, fontsize=24, y=0.95)
            #title_en = 'entropy_' + title
            #fig.savefig('figures/intensities/{}.png'.format(title.replace('/', '_')), bbox_inches='tight')
            fig_pmf.savefig('figures/intensities/{}.png'.format(title_pmf.replace('/', '_')), bbox_inches='tight')
            #fig_en.savefig('figures/intensities/{}.png'.format(title_en.replace('/', '_')), bbox_inches='tight')
        print('DONE')

def evaluate(args):
    sequence, artifacts_seq = get_test_sequence(args) #[L*np, n]
    t_min, t_max = args.x_lims[0], args.x_lims[1]
    step = 0.001
    all_models_files = os.listdir(args.model_dir)
    file_to_find = args.model + '_' + args.model_split
    print(file_to_find)
    model_file = None
    for file in all_models_files:
        if file.startswith(file_to_find):
            model_file = os.path.join(args.model_dir, file)
            break
    if model_file is None:
        raise ValueError('Checkpoint not found')
    args, model_name = update_args(args, model_file)
    data_sequence = Dataset(
    args=args, size=len(sequence), seed=args.seed, data=sequence)
    loader = get_loader(data_sequence, args=args, shuffle=False)
    model = get_model(args)
    model.load_state_dict(th.load(model_file, map_location=args.device))
    metrics = {'Model NLL':[], 'True NLL': [], 'True Accuracy':[], 'Model Accuracy':[], 'Brier':[]}
    for i, batch in enumerate(loader):
        times, labels = batch["times"], batch["labels"]
        labels_idx = th.argmax(labels, dim=-1).squeeze(-1).cpu().numpy()
        
        labels = (labels != 0).type(labels.dtype) #?
        mask = (times != args.padding_id).type(times.dtype)
        times = times * args.time_scale 
        window_start, window_end = get_window(times=times, window=args.window_end)
        events = get_events(
        times=times, mask=mask, labels=labels,
        window_start=window_start, window_end=window_end)
        #print(events)
        query_times = th.arange(t_min, t_max+step, step=step, dtype=th.float32).unsqueeze(0)        

        ground_density, log_ground_density, mark_pmf, _ = get_density(model, query_times, events, args) 
        ground_density_events, log_ground_density_events, mark_pmf_events, log_mark_pmf_events = get_density(model, times, events, args, is_event=True)
        ground_intensity, log_ground_intensity, ground_intensity_integral, intensity_mask, artifacts, intensity = get_intensity(model, query_times, events, args)

        true_events_pmf = th.tensor(mark_pmf_events) * labels
        true_events_pmf = th.sum(true_events_pmf[0,1:,:], dim=-1)

        total_events_pmf = -th.sum(th.log(true_events_pmf)).cpu().numpy().astype(np.float64)
        total_events_pmf = np.round(total_events_pmf, 2)
        query_times = query_times.squeeze().cpu().numpy()
        if labels_idx.ndim == 1:
            labels_idx = labels_idx[np.newaxis,:]
        accuracy = get_accuracy(mark_pmf_events, labels_idx[0])
        accuracy_true = get_accuracy(artifacts_seq['true_mark_pmf'][i], labels_idx[0], true_pmf=True, 
                                         event_times=times.cpu().numpy()[0], all_times=artifacts_seq['intensity_times'][i])
        nll_true = get_true_nll(artifacts_seq['true_mark_pmf'][i], labels_idx[0], 
                                args.marks,event_times=times.cpu().numpy()[0], 
                                all_times=artifacts_seq['intensity_times'][i])        
        #print(times.cpu().numpy()[0])
        #print(artifacts_seq['intensity_times'][i][340:360])
        brier = brier_score(artifacts_seq['true_mark_pmf'][i], mark_pmf, 
                            all_times=artifacts_seq['intensity_times'][i], 
                            event_times=times.cpu().numpy()[0])
        metrics['True NLL'].append(nll_true)
        metrics['Model NLL'].append(total_events_pmf)
        metrics['True Accuracy'].append(accuracy_true)
        metrics['Model Accuracy'].append(accuracy)
        metrics['Brier'].append(brier)
    print(f'NLL TRUE: {np.mean(metrics["True NLL"])}')
    print(f'NLL MODEL: {np.mean(metrics["Model NLL"])}')
    print(f'ACC TRUE: {np.mean(metrics["True Accuracy"])}')
    print(f'ACC MODEL: {np.mean(metrics["Model Accuracy"])}')
    print(f'BRIER: {np.mean(metrics["Brier"])}')
    if args.save_results_dir is not None:
        save_results(args, metrics)    


def save_results(args, metrics):
    save_dir = args.save_results_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = args.model + '.txt'
    save_path = os.path.join(save_dir, file_name)
    with open(save_path, 'wb') as f:
        pickle.dump(metrics, f)
    print('Results saved to {}'.format(save_path, flush=True))

def get_true_nll(true_pmf, true_labels, num_marks, event_times, all_times):
    #floored_event_times = (np.floor(1000*event_times[0])/1000)
    #all_times = all_times.astype(np.float32)
    #event_idxs = np.array([int(np.where(all_times == event)[0]) for event in floored_event_times]) 
    event_idxs = np.array([np.isclose(all_times.astype(np.float64), event.astype(np.float64), rtol=0, atol=1e-6).nonzero()[0][0]
                          for event in event_times])
    if len(np.unique(event_idxs)) != len(event_times):
        unique, counts = np.unique(event_idxs, return_counts=True)
        idx_to_add = unique[np.where(counts > 1)] + counts[np.where(counts > 1)] - 1
        event_idxs = np.append(unique, idx_to_add)
    assert(len(np.unique(event_idxs)) == len(event_times))
    events_true_pmf = true_pmf[event_idxs, :]
    events_true_log_pmf = np.log(events_true_pmf)
    label_mask = np.eye(num_marks)[true_labels].astype(np.bool)
    events_true_log_pmf = -np.sum(events_true_log_pmf[label_mask])
    return events_true_log_pmf

def get_accuracy(intensity, true_labels, true_pmf=False, event_times=None, all_times=None):
    if not true_pmf: 
        predicted_marks = np.argmax(intensity, axis=-1)
        accuracy = np.sum(predicted_marks == true_labels)/len(true_labels)
    else:        
        #floored_event_times = (np.floor(1000*event_times)/1000)
        #all_times = all_times.astype(np.float32)
        #event_idxs = np.array([int(np.where(all_times == event)[0]) for event in floored_event_times]) 
        event_idxs = np.array([np.isclose(all_times.astype(np.float64), event.astype(np.float64), rtol=0, atol=1e-6).nonzero()[0][0]
                          for event in event_times])
        if len(np.unique(event_idxs)) != len(event_times):
            unique, counts = np.unique(event_idxs, return_counts=True)
            idx_to_add = unique[np.where(counts > 1)] + counts[np.where(counts > 1)] - 1
            event_idxs = np.append(unique, idx_to_add)
        assert(len(np.unique(event_idxs)) == len(event_times))
        events_true_intensity = intensity[event_idxs, :]
        predicted_marks = np.argmax(events_true_intensity, axis=-1)
        #print(true_labels, 'True')
        #print(predicted_marks, 'Precited')
        accuracy = np.sum(predicted_marks == true_labels)/len(true_labels)
    return accuracy

def brier_score(true_pmf, estimated_pmf, all_times, event_times):
    #all_times_round = np.round(all_times, 4)
    #event_times_round = np.round(event_times, 4) - 1e-5
    #mask_idx = np.where()
    #mask_idx = np.searchsorted(all_times.astype(np.float64), event_times.astype(np.float64), side="left").astype(np.int)
    mask_idx = np.array([np.isclose(all_times.astype(np.float64), event.astype(np.float64), rtol=0, atol=1e-6).nonzero()[0][0]
                          for event in event_times])
    if len(np.unique(mask_idx)) != len(event_times):
        unique, counts = np.unique(mask_idx, return_counts=True)
        idx_to_add = unique[np.where(counts > 1)] + counts[np.where(counts > 1)] - 1
        mask_idx = np.append(unique, idx_to_add)
    assert(len(np.unique(mask_idx)) == len(event_times))
    mask = np.ones_like(all_times, dtype=np.bool)
    mask[mask_idx] = False    
    #Remove event pmf so that both arrays have the same length (pmf at events are added by default by tick.)
    true_pmf = true_pmf[mask] 
    diff = true_pmf - estimated_pmf
    num_points = true_pmf.shape[0]
    brier_score = np.sum(np.power(diff, 2))/num_points
    return brier_score

def sequence_loss_distribution(args):
    dataset_path = os.path.join(args.load_from_dir, args.dataset)
    dataset_path = os.path.join(dataset_path, args.split)
    dataset_path = os.path.join(dataset_path, 'test.json')
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)  
    all_models_files = os.listdir(args.model_dir)
    file_to_find = args.model + '_' + args.model_split
    print(file_to_find)
    model_file = None
    for file in all_models_files:
        if file.startswith(file_to_find):
            model_file = os.path.join(args.model_dir, file)
            break
    if model_file is None:
        raise ValueError('File no found !')
    args, model_name = update_args(args, model_file)
    model_name = model_file.split('/')[-1][:-3]
    data_sequence = Dataset(
    args=args, size=1, seed=args.seed, data=dataset)
    args
    loader = get_loader(data_sequence, args=args, shuffle=False)
    model = get_model(args)
    model.load_state_dict(th.load(model_file, map_location=args.device))
    intensities, densities, pmfs = [], [], []
    for i, batch in enumerate(loader):
        times, labels = batch["times"], batch["labels"]
        labels = (labels != 0).type(labels.dtype) #?
        mask = (times != args.padding_id).type(times.dtype)
        times = times * args.time_scale #time_scale=1 by default
        #times_mask = (times > t_min) * (times < t_max)
        #times_masked = times[times_mask]
        
        window_start, window_end = get_window(times=times, window=args.window)
        events = get_events(
        times=times, mask=mask, labels=labels,
        window_start=window_start, window_end=window_end)
        #query_times = th.tensor([0.91940, 1.097]).unsqueeze(0)
        #log_intensity, intensity_integral, intensity_mask, artifacts = model.artifacts(query=times, events=events)
        #log_intensity = log_intensity * intensity_mask.unsqueeze(-1)        
        
        #ground_intensity, log_ground_intensity, _, _ = get_intensity(model, times, events, args)
        #log_ground_intensity_sum = np.sum(log_ground_intensity)
        
        ground_density, log_ground_density, mark_pmf, log_mark_pmf = get_density(model, times, events, args, is_event=True)
        log_ground_density_sum = np.sum(log_ground_density)
        true_log_mark_pmf = log_mark_pmf * labels
        true_log_mark_pmf_sum = float(th.sum(true_log_mark_pmf).detach().cpu().numpy())
        pmfs.append(true_log_mark_pmf_sum)


        #intensities.append(float(log_ground_intensity_sum))
        #densities.append(float(log_ground_density_sum))
    print(pmfs)
    print('Mean log intensity:', np.mean(intensities))
    print('Mean log density:', np.mean(densities))
    print('Mean log pmf:', np.mean(pmfs))
    '''
    fig, ax = plt.subplots(1,2)
    ax[0].hist(intensities, bins=10)
    ax[1].hist(densities, bins=10)
    if 'poisson_' + args.dataset in model_file:
        fig.savefig('figures/intensities/test_hist_base.png', bbox_inches='tight')
    else:
        fig.savefig('figures/intensities/test_hist.png', bbox_inches='tight')
    '''
    print('DONE')


def comp_base_no_base(args):
    dataset_path = os.path.join(args.load_from_dir, args.dataset)
    dataset_path = os.path.join(dataset_path, 'split_0/test.json') 
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)  
    sequence = dataset[0]
    for seq in dataset:
        if len(seq) >= args.min_event_per_seq and len(seq) <= args.max_event_per_seq:
            sequence = seq
            break
    print(len(sequence))
    if sequence is None:
        raise ValueError('Not sequence of minimum length found.')
    if args.ground_intensity is False:
        timestamps = reformat_timestamps(sequence)
        n_marks = len(np.unique([event['labels'][0] for event in sequence]))
        fig, ax = plt.subplots(n_marks,1)
    else:
        timestamps =  [[event['time'] for event in sequence]]
        fig, ax = plt.subplots(1,1)
    sequence = [sequence]
    #print(timestamps)
    ax = plot_process(timestamps, ax)
    t_min, t_max = args.x_lims[0], args.x_lims[1]
    step = (t_max-t_min)/100000 
    all_models_files = os.listdir(args.model_dir)
    file_to_find_poisson = 'poisson_' + args.model.split('_base')[0] + '_config'
    file_to_find = args.model + '_config'
    for file in all_models_files:
        if file.startswith(file_to_find):
            model_file = os.path.join(args.model_dir, file)
            break
    for file in all_models_files:
        if file.startswith(file_to_find_poisson):
            model_file_poisson = os.path.join(args.model_dir, file)
            break
    args, model_name = update_args(args, model_file)
    args_poisson, model_name_poisson = update_args(args, model_file_poisson)
    model_name = model_file.split('/')[-1][:-3]
    data_sequence = Dataset(
    args=args, size=1, seed=args.seed, data=sequence)
    loader = get_loader(data_sequence, args=args, shuffle=False)
    model = get_model(args)
    model_poisson = get_model(args_poisson)
    model.load_state_dict(th.load(model_file, map_location=args.device))
    model_poisson.load_state_dict(th.load(model_file_poisson, map_location=args.device))
    for batch in loader:
        times, labels = batch["times"], batch["labels"]
        labels = (labels != 0).type(labels.dtype) #?
        mask = (times != args.padding_id).type(times.dtype)
        times = times * args.time_scale #time_scale=1 by default
        #times_mask = (times > t_min) * (times < t_max)
        #times_masked = times[times_mask]
        window_start, window_end = get_window(times=times, window=args.window)
        events = get_events(
        times=times, mask=mask, labels=labels,
        window_start=window_start, window_end=window_end)
        query_times = th.arange(t_min, t_max, step=step, dtype=th.float32).unsqueeze(0)        
        #query_times = th.tensor([0.91940, 1.097]).unsqueeze(0)
        print(times)
        intensity = get_intensity(model, query_times, events, args)
        density = get_density(model, query_times, events, args)
        intensity_p = get_intensity(model_poisson, query_times, events, args_poisson)
        density_p = get_density(model_poisson, query_times, events, args_poisson)
        query_times = query_times.squeeze().cpu().numpy()
        plot_modelled_intensity(query_times, density, ax, args, model_name)
        plot_modelled_intensity(query_times, intensity, ax, args, model_name)
        title = fig_title(model_name, args.dataset)
        print(title)
        if 'LastFM' in title:
            title = title.split('LastFM-')[1]
            ax.set_title(title, fontsize=24)
            title = 'LastFM-' + title
        else:
            ax.set_title(title, fontsize=24)
        ax.set_xlim(t_min, t_max)
        ax.grid(True)
        ax.set_ylabel(r'$\lambda^\ast(t)~\backslash~f(t|\mathcal{H}_t)$', fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=24 )
        ax.set_xticklabels('')
        ax.set_xlabel(r'$t$', fontsize=24) 
        fig.savefig('figures/intensities/base_{}.png'.format(title.replace('/', '_')), bbox_inches='tight')
        print('DONE')

def get_intensity(model, query_times, events, args, poisson=False, is_event=False):
    log_intensity, marked_intensity_integral , intensity_mask, artifacts = model.artifacts(query=query_times, events=events)
    if is_event:
        log_intensity = log_intensity * intensity_mask.unsqueeze(-1)
    intensity = th.exp(log_intensity).squeeze(0)
    ground_intensity = th.sum(th.exp(log_intensity), dim=-1).squeeze(-1)
    log_ground_intensity = th.logsumexp(log_intensity, dim=-1).squeeze(-1)
    ground_intensity_integral = th.sum(marked_intensity_integral, dim=-1).squeeze(-1)
    
    if is_event:
        ground_intensity = ground_intensity * intensity_mask
        log_ground_intensity = log_ground_intensity * intensity_mask
        ground_intensity_integral = ground_intensity_integral * intensity_mask
    ground_intensity = ground_intensity.squeeze(0)
    log_ground_intensity = log_ground_intensity.squeeze()
    if poisson:
        log_intensity_0 = artifacts["log_intensity_0"]
        if is_event:
            log_intensity_0 = log_intensity_0 * intensity_mask.unsqueeze(-1)
        intensity_0 = th.sum(th.exp(log_intensity_0),dim=-1).squeeze(-1)
        if is_event:
            intensity_0 = intensity_0 * intensity_mask
        intensity_0 = intensity_0.squeeze()
        log_intensity_1 = artifacts["log_intensity_1"]
        if is_event:
            log_intensity_1 = log_intensity_1 * intensity_mask.unsqueeze(-1)
        intensity_1 = th.sum(th.exp(log_intensity_1),dim=-1).squeeze(-1)
        if is_event:
            intensity_1 = intensity_1 * intensity_mask
        intensity_1 = intensity_1.squeeze()
        alpha = th.tensor(artifacts["alpha"])     
        intensity_0 = alpha[1] * intensity_0 
        intensity_1 = alpha[0] * intensity_1 
        intensity_0 = intensity_0.detach().cpu().numpy()
        intensity_1 = intensity_1.detach().cpu().numpy()
    else:
        intensity_0, intensity_1 = None, None
    ground_intensity = ground_intensity.detach().cpu().numpy()
    log_ground_intensity = log_ground_intensity.detach().cpu().numpy()
    ground_intensity_integral = ground_intensity_integral.detach().cpu().numpy()
    intensity = intensity.detach().cpu().numpy()
    artifacts = {}
    artifacts['intensity_0'] = intensity_0
    artifacts['intensity_1'] = intensity_1
    return ground_intensity, log_ground_intensity, ground_intensity_integral, intensity_mask, artifacts, intensity

def get_density(model, query_times, events, args, poisson=False, is_event=False):
    log_density, log_mark_pmf, density_mask = model.log_density(query=query_times, events=events)
    #print('log_density', log_density)
    if is_event:
        log_density = log_density * density_mask.unsqueeze(-1)
        log_mark_pmf = log_mark_pmf * density_mask.unsqueeze(-1)
    ground_density = th.sum(th.exp(log_density), dim=-1).squeeze(-1)
    log_ground_density = th.logsumexp(log_density, dim=-1).squeeze(-1)
    mark_pmf = th.exp(log_mark_pmf)
    if is_event:
        ground_density = ground_density * density_mask
        log_ground_density = log_ground_density * density_mask
        mark_pmf = mark_pmf * density_mask.unsqueeze(-1)
    ground_density = ground_density.squeeze(0)
    log_ground_density = log_ground_density.squeeze(0)
    mark_pmf = mark_pmf.squeeze(0)
    ground_density = ground_density.detach().cpu().numpy()
    log_ground_density = log_ground_density.detach().cpu().numpy()
    mark_pmf = mark_pmf.detach().cpu().numpy()
    density = th.exp(log_density).squeeze(0)
    density = density.detach().cpu().numpy()
    print('density', density.shape)
    return density, ground_density, log_ground_density, mark_pmf, log_mark_pmf

def fig_title(model_name, dataset):
    model_short = model_name.split(dataset + '_')[1].split('_split')[0]
    if 'lnmk1' in model_name:
        model_short += '_lnmk1'
    model_acr = get_acronym([model_short])[0]
    if model_acr != 'Poisson' and model_acr != 'Hawkes':
        model_acr_short = model_acr.split('-')[0] + '-' + model_acr.split('-')[1]
    else:
        model_acr_short = model_acr
    dataset_acr = map_datasets_name(dataset)
    title = dataset_acr + '-' + model_acr_short
    return title

if __name__ == "__main__":
    parsed_args = parse_args_intensity()
    #plot_intensity(args=parsed_args)
    if parsed_args.type == 'homogeneous':   
        #plot_sequence_intensity(parsed_args)
        #plot_sequence_intensity_iclr(parsed_args)
        show_joint_distribution(parsed_args)
    else:
        plot_hawkes_intensity(parsed_args)
        evaluate(parsed_args)
    #sequence_loss_distribution(parsed_args)