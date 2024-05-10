from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser, Namespace
import os
import torch as th
import seaborn as sns

from tick.plot import plot_hawkes_kernels
from tick.hawkes import SimuHawkesSumExpKernels, SimuHawkesMulti, SimuHawkesExpKernels, \
    HawkesSumExpKern
#from tick.plot import plot_point_process
from data_processing.tick_plot_process import plot_point_process, _extract_process_interval

    
def initialize_kernel(kernel_name, marks, baselines, window, 
                    decays=None, self_decays=None, mutual_decays=None, adjacency=None, self_adjacency=None, mutual_adjacency=None, noise=None, seed=None):
    artifacts = {}
    if kernel_name in ['hawkes_exponential_mutual', 'hawkes_exponential_independent']:
        if decays is None:
            decays = [[self_decays[i] if i == j else mutual_decays[0] for i in range(marks)] for j in range(marks)]
        if adjacency is None:
            adjacency = [[self_adjacency[i] if i == j else mutual_adjacency[0] for i in range(marks)] for j in range(marks)]
        kernel = SimuHawkesExpKernels(adjacency=adjacency, decays=decays, baseline=baselines, end_time=window, verbose=False, seed=seed)
    elif kernel_name in ['hawkes_sum_exponential_mutual', 'hawkes_sum_exponential_independent']:
        if decays is None:
            decays = np.array(self_decays)
        if adjacency is None: #Problem if decay matrix is specified, but not adjacency one.
            adjacency = np.array([[self_adjacency[i] if i == j else mutual_adjacency for i in range(marks)] for j in range(marks)])
            adjacency_mask = adjacency != 0
            adjacency = np.repeat(adjacency[:, :, np.newaxis], decays.shape[0] , axis=2)
            
            adjacency_mask = np.repeat(adjacency_mask[:, :, np.newaxis], decays.shape[0] , axis=2)
            if noise is not None:
                adj_noise = np.random.uniform(0, noise, adjacency.shape) * adjacency_mask
                adjacency = adjacency + adj_noise 
        kernel = SimuHawkesSumExpKernels(adjacency=adjacency, decays=decays, baseline=baselines, end_time=window, verbose=False, seed=seed)
    if kernel.spectral_radius() >= 1:
        print(kernel.spectral_radius())
        kernel.adjust_spectral_radius(0.99)
        print('Spectral radius adjusted !')
    artifacts['decays'] = kernel.decays.tolist()
    artifacts['adjacency'] = kernel.adjacency.tolist()
    artifacts['baselines'] = kernel.baseline.tolist()
    return kernel, artifacts

def simulate_process(kernel_name, window, n_seq,
                      marks, baselines, self_decays=None, mutual_decays=None, self_adjacency=None, 
                     mutual_adjacency=None, track_intensity=False, decays=None, adjacency=None, noise=None, seed=None):
    if decays is None:
        assert(self_decays is not None and mutual_decays is not None), 'Either specify self decays and mutual decays, or overall decays, but not both.'
    if adjacency is None:
        assert(self_adjacency is not None and mutual_adjacency is not None), 'Either specify self adjacency and mutual adjacency, or overall adjacency, but not both.'
    kernel, artifacts = initialize_kernel(kernel_name=kernel_name, marks=marks, self_decays=self_decays, mutual_decays=mutual_decays, 
                                          self_adjacency=self_adjacency, mutual_adjacency=mutual_adjacency, baselines=baselines, window=window,
                                          decays=decays, adjacency=adjacency, noise=noise, seed=seed)
    if track_intensity:
        dt = 0.001
        kernel.track_intensity(dt)
    if n_seq is None:
        n_seq = n_seq_train + n_seq_val + n_seq_cal + n_seq_test
    process = SimuHawkesMulti(kernel, n_simulations=n_seq)
    process.end_time = [window] * n_seq  
    process.simulate()
    return process, artifacts


def get_kernel(name):
    if name == 'hawkes_exponential':
        kernel = SimuHawkesExpKernels
    elif name == 'hawkes_sum_exponential':
        kernel = SimuHawkesSumExpKernels
    return kernel

if __name__ == "__main__":
    #self_decays = [4.1, 2.5, 6.2, 4.9, 4.1]
    self_decays = [4.1, 2.5, 6.2, 4.9, 4.1, 5.3, 3.7, 4.3, 3.8, 6.4]
    mutual_decays = 1
    baselines = [.2, .3, 0.25, .35, .45]
    self_adjacency = [.15, .25, .2, .15, .1]
    mutual_adjacency = 0.001
    window = 10
    n_seq = 1000
    n_processes = 5
    noise = 0.1
    name = 'hawkes_sum_exponential_mutual'
    
    #get_process_name(name, process_adjacency)
    process,_ = simulate_process(name, window=window, n_seq=1, marks=5, baselines=baselines, self_decays=self_decays, mutual_decays=mutual_decays, 
                                self_adjacency=self_adjacency, mutual_adjacency=mutual_adjacency, track_intensity=True, noise=noise)
    fig, ax = plt.subplots(5, 1, figsize=(16, 8), sharex=True, sharey=True)
    process = process._simulations[0]
    plot_point_process(process, n_points=50000, t_min=0, ax=ax, plot_intensity=True, show_points=True)
    plt.show()
    #print(type(process.hawkes_simu))


def plot_process(timestamps, ax, labels_idx=None):
    if labels_idx is not None:
        label_set = np.unique(labels_idx)
        cm = plt.get_cmap('gist_rainbow')
        #ax.set_prop_cycle(color=[cm(1.*i/len(label_set)) for i in range(len(label_set))])
        ax.set_prop_cycle('color', sns.color_palette("Set2", len(label_set)))
        for i, label in enumerate(label_set):
            label_mask = labels_idx == label
            timestamps_to_plot = timestamps[label_mask]
            #y_point_pos = np.array([-100-100*i] * len(timestamps_to_plot))
            y_point_pos = np.array([-100] * len(timestamps_to_plot))
            print(y_point_pos)
            ax.scatter(timestamps_to_plot, y_point_pos, s=100, label=f'mark {label}')
            ax.set_xticklabels([])
    return ax

def plot_process_multi_axes(timestamps, axes, labels_idx=None, marked=True):
    if marked:
        label_set = np.sort(np.unique(labels_idx))
        cm = plt.get_cmap('gist_rainbow')
        #ax.set_prop_cycle(color=[cm(1.*i/len(label_set)) for i in range(len(label_set))])
        #axes.set_prop_cycle('color', sns.color_palette("Set2", len(label_set)))
        for i, label in enumerate(label_set):
            label_mask = labels_idx == label
            timestamps_to_plot = timestamps[label_mask]
            #y_point_pos = np.array([-100-100*i] * len(timestamps_to_plot))
            y_point_pos = np.array([-1] * len(timestamps_to_plot))
            axes[i].scatter(timestamps_to_plot, y_point_pos, s=100)
            #axes[i].set_xticklabels([])
    else:
        y_point_pos = np.array([-1] * len(timestamps[0]))
        axes.scatter(timestamps[0], y_point_pos, s=100)
    return axes

def simulate_seqs(args:Namespace, n_seq=1):
    if 'self_decays' not in vars(args):
        process, artifacts = simulate_process(kernel_name=args.kernel_name, window=args.window,
                                            n_seq=n_seq, marks=args.marks, baselines=args.baselines, 
                                            decays=args.decays, 
                                            adjacency=args.adjacency, 
                                            track_intensity=True,
                                            seed=args.seed)
    else:
        process, artifacts = simulate_process(kernel_name=args.kernel_name, window=args.window, 
                                            n_seq=n_seq, marks=args.marks, self_decays=args.self_decays, 
                                            mutual_decays=args.mutual_decays, 
                                            baselines=args.baselines, self_adjacency=args.self_adjacency, 
                                            mutual_adjacency=args.mutual_adjacency, track_intensity=True,
                                            seed=args.seed)
    all_process = process._simulations
    marked_intensities, ground_intensities, mark_pmfs, intensity_timess, sequences = [], [], [], [], []
    for process in all_process:
        timestamps, intensity_times, intensities = _extract_process_interval(
                                                range(process.n_nodes), process.end_time, process.timestamps,
                                                intensity_times=process.intensity_tracked_times, 
                                                intensities=process.tracked_intensity)
        marked_intensity = np.transpose(np.array(intensities)) #[L,K]
        ground_intensity = np.sum(marked_intensity, axis=-1)
        ground_intensity = ground_intensity[:,np.newaxis]
        mark_pmf = marked_intensity/ground_intensity    
        sequence = reformat([timestamps])
        if len(sequence) == 0:
            print('Empty sequence simulated')
            continue
        elif len(sequence[0]) < 2:
            print('Sequence of length smaller than 2 simulated')
            continue 
        sequence = sequence[0]
        marked_intensities.append(marked_intensity)
        ground_intensities.append(ground_intensity)
        mark_pmfs.append(mark_pmf)
        intensity_timess.append(intensity_times)
        sequences.append(sequence)
    artifacts['marked_intensity'] = marked_intensities
    artifacts['ground_intensity'] = ground_intensities
    artifacts['true_mark_pmf'] = mark_pmfs
    artifacts['intensity_times'] = intensity_timess
    return sequences, artifacts

def reformat(process):
    labelled_process = [[[time, label] for label, dimension in enumerate(seq) for time in dimension] for seq in process]
    sorted_process = [sorted(seq, key= lambda seq: seq[0]) for seq in labelled_process]
    dic_process = [[{'time':event[0], 'labels':[event[1]]} for event in seq] for seq in sorted_process]
    dic_mask = np.array([len(seq) for seq in dic_process]) != 0
    dic_process = np.array(dic_process, dtype='object')[dic_mask].tolist()
    return dic_process 



""" 
    end_time = 1000
    n_realizations = 10

    n_processes = 5
    self_decays = [1.0, 1.5, 1.2, .9, 1.1]
    decays = [[self_decays[i] if i == j else 10.0 for i in range(n_processes)] for j in range(n_processes)]

    baseline = [1, .8, 1.1, .7, .9]

    self_adjacency = [0.7, 0.5, 0.6, 0.8, 0.75]

    adjacency = [[self_adjacency[i] if i == j else 0.0 for i in range(n_processes)] for j in range(n_processes)]

    hawkes_exp_kernels = SimuHawkesExpKernels(
        adjacency=adjacency, decays=decays, baseline=baseline,
        end_time=end_time, verbose=False, seed=1039)

    dt = 0.01
    hawkes_exp_kernels.track_intensity(dt)


    multi = SimuHawkesMulti(hawkes_exp_kernels, n_simulations=n_realizations)
    multi.end_time = [10] * n_realizations
    multi.simulate()
    hawkes = multi._simulations[0]
    print(hawkes.timestamps) 
    print()
    #print(len(hawkes.timestamps))
    #learner = HawkesSumExpKern(decays, penalty='elasticnet',
    #                           elastic_net_ratio=0.8)
    #learner.fit(multi.timestamps)




    #fig, ax = plt.subplots(5, 1, figsize=(16, 8), sharex=True, sharey=True)
    #plot_point_process(hawkes, n_points=50000, t_min=0, ax=ax, plot_intensity=True, show_points=False)
    #fig.tight_layout()
    #plt.show() """