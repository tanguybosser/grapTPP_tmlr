from argparse import ArgumentParser, Namespace
import json
import sys, os 
from pathlib import Path
import numpy as np
import random
from collections import defaultdict
sys.dont_write_bytecode = True

sys.path.remove(os.path.abspath(os.path.join('..', 'neuralTPPs')))
sys.path.append(os.path.abspath(os.path.join('..', 'neurips')))

from data_processing.simu_hawkes import simulate_process, reformat, plot_process
#from data_processing.prepare_datasets import make_splits

from intensity.intensity_plots import plot_modelled_intensity, plot_modelled_pmf, plot_entropy

from distutils.util import strtobool
import matplotlib.pyplot as plt 

def parse_args():
    parser = ArgumentParser(allow_abbrev=False)
    parser.add_argument("--kernel-name", type=str, required=True,
                        help="Type of Process to simulate.")
    parser.add_argument("--dataset-name", type=str, default=None, 
                        help="Name to give to the simulated dataset. If default, kernel-name is passed as name.")
    parser.add_argument("--window", type=float, default=10,
                        help="End window of the simulated process.")
    parser.add_argument("--n-seq", type=int, default=None,
                        help="Number of independent sequences to simulate")
    parser.add_argument("--n-seq-train", type=int, default=1000,
                        help="Number of independent training sequences to simulate")
    parser.add_argument("--n-seq-val", type=int, default=1000,
                        help="Number of independent validation sequences to simulate")
    parser.add_argument("--n-seq-cal", type=int, default=1000,
                        help="Number of independent calibration sequences to simulate")
    parser.add_argument("--n-seq-test", type=int, default=1000,
                        help="Number of independent test sequences to simulate")
    parser.add_argument("--marks", type=int, default=5,
                        help="Number of dimensions to simulate.")
    parser.add_argument("--self-decays", type=float, nargs="+", default=[],
                        help="If sum of exponential, the list length defines the number of exponentila kernels.")
    parser.add_argument("--mutual-decays", type=float, nargs="+", default=[], 
                        help="")
    parser.add_argument("--baselines", type=float, nargs="+", default=[],
                        help="")
    parser.add_argument("--self-adjacency", type=float, nargs="+", default=[],
                        help="")
    parser.add_argument("--mutual-adjacency", type=float, nargs="+", default=[],
                        help="")
    parser.add_argument("--noise", type=float, default=None,
                        help="Upper bound of noise to add to the adjacency matrix. Only for Hawkes sum of exponentials. Defaults to None")
    parser.add_argument("--save-dir", type=str, required=True,
                        help="Directory in which to save the simulated dataset")
    parser.add_argument("--num-splits", type=int, default=None,
                        help="Number of splits to make out of the simulated dataset. If None, no splits get created.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--simu-type", type=str, choices=['dataset', 'visualization'], default='visualization',
                        help="If 'dataset', simulates a complete dataset with split. If 'visualization', simulate and show a unique sequence.")
    parser.add_argument("--heterogeneous-dataset", type=lambda x: bool(strtobool(x)), default=False,
                        help="If True, generates a heterogeneous Hawkes dataset containing multiple different processes.")
    parser.add_argument("--n-processes", type=int, default=1,
                        help="Number of different heterogeneous processes to simulate.")
    parser.add_argument("--n-exp", type=int, default=None,
                        help="Number of different heterogeneous processes to simulate.")
    parser.add_argument("--from-file", type=str, default=None,
                        help="If provided, processes will be generated from the parameters indicated in the file.")
    args, _ = parser.parse_known_args()
    if args.heterogeneous_dataset and len(args.baselines) != 0:
        args = update_heterogeneous_args(args)
    return args


def update_heterogeneous_args(args):
    assert(len(args.self_decays)/(args.marks * args.n_processes) == 1), 'Wrong dim. passed'
    assert(len(args.baselines)/(args.marks * args.n_processes) == 1), 'Wrong dim. passed'
    assert(len(args.self_adjacency)/(args.marks * args.n_processes) == 1), 'Wrong dim. passed'
    assert(len(args.mutual_decays)/args.n_processes == 1), 'Wrong dim. passed'
    assert(len(args.mutual_adjacency)/args.n_processes == 1), 'Wrong dim. passed'
    self_decays = [args.self_decays[i*args.marks:(i+1)*args.marks] for i in range(args.n_processes)]
    mutual_decays = [args.mutual_decays[i] for i in range(args.n_processes)]
    self_adjacency = [args.self_adjacency[i*args.marks:(i+1)*args.marks] for i in range(args.n_processes)]
    mutual_adjacency = [args.mutual_adjacency[i] for i in range(args.n_processes)]
    baselines = [args.baselines[i*args.marks:(i+1)*args.marks] for i in range(args.n_processes)]
    n_seq = [args.n_seq[i] for i in range(args.n_processes)]
    args_dic = vars(args)
    args_dic['self_decays'] = self_decays
    args_dic['mutual_decays'] = mutual_decays
    args_dic['self_adjacency'] = self_adjacency
    args_dic['mutual_adjacency'] = mutual_adjacency
    args_dic['baselines'] = baselines
    args_dic['n_seq'] = n_seq
    args = Namespace(**args_dic)
    return args 


def simulate_dataset(args):
    if len(args.baselines) == 0:
        args = get_simulation_parameters(args)
    #if 'independent' in args.kernel_name:
    #    assert(args.mutual_adjacency == 0.), 'Independent Hawkes must have mutual adjacency set to 0'
    #    assert(args.mutual_decays == 0.), 'Independent Hawkes must have mutual decays set to 0'
    #if 'mutual' in args.kernel_name:
    #    assert(args.mutual_adjacency[0] > 0.), 'Dependent Hawkes must have mutual adjacency different of 0'
    #    assert(args.mutual_decays[0] > 0.), 'Dependent Hawkes must have mutual decays different of 0'        
    process, artifacts = simulate_process(args, track_intensity=False)
    process_reformat = reformat(process.timestamps)
    return process_reformat, artifacts    

def simulate_heterogeneous_dataset(args, shuffle=True):
    all_processes = []
    all_artifacts = {'decays':[], 'adjacency':[], 'baselines':[]}
    for i in range(args.n_processes):
        if (args.heterogeneous_dataset and len(args.baselines) == 0):
            params= get_simulation_parameters(args.marks)
            n_seq = int(args.n_seq[0]/args.n_processes) 
            process, artifacts = simulate_process(kernel_name=args.kernel_name, window=args.window, 
                               n_seq=n_seq, marks=args.marks, self_decays=params['self_decays'], mutual_decays=params['mutual_decays'], 
                               baselines=params['baselines'], self_adjacency=params['self_adjacency'], mutual_adjacency=params['mutual_adjacency'], track_intensity=False)
        
        else:
            process, artifacts = simulate_process(kernel_name=args.kernel_name, window=args.window, 
                               n_seq=args.n_seq[i], marks=args.marks, self_decays=args.self_decays[i], mutual_decays=args.mutual_decays[i], 
                               baselines=args.baselines[i], self_adjacency=args.self_adjacency[i], mutual_adjacency=args.mutual_adjacency[i], track_intensity=False)
        process_reformat = reformat(process.timestamps)
        all_processes.extend(process_reformat) 
        all_artifacts['decays'].append(artifacts['decays'])
        all_artifacts['adjacency'].append(artifacts['adjacency'])
        all_artifacts['baselines'].append(artifacts['baselines'])
    if shuffle:
        random.shuffle(all_processes)
    return all_processes, all_artifacts

def simulate_heterogeneous_dataset_from_file(args, shuffle=True):
    all_processes = []
    file = os.path.join('data/baseline3/hawkes_hete/hawkes_basetest_1000', 'args.json')
    with open(file, 'r') as f:
        all_params = json.load(f)
    all_artifacts = {'decays':[], 'adjacency':[], 'baselines':[]}
    for i in range(args.n_processes):
        baselines = all_params['baselines'][i]
        decays = all_params['decays'][i]
        adjacency = all_params['adjacency'][i]
        n_seq = int(args.n_seq[0]/args.n_processes) 
        process, artifacts = simulate_process(kernel_name=args.kernel_name, window=args.window, 
                            n_seq=n_seq, marks=args.marks, decays=decays,
                            baselines=baselines, 
                            adjacency=adjacency, track_intensity=False)
        process_reformat = reformat(process.timestamps)
        all_processes.extend(process_reformat) 
        all_artifacts['decays'].append(artifacts['decays'])
        all_artifacts['adjacency'].append(artifacts['adjacency'])
        all_artifacts['baselines'].append(artifacts['baselines'])
    if shuffle:
        random.shuffle(all_processes)
    return all_processes, all_artifacts


def get_simulation_parameters(args):
    num_marks = args.marks
    np.random.seed(args.seed)
    #args.self_decays = np.random.uniform(0, 10, size=num_marks)
    #args.mutual_decays = np.random.uniform(0,1, size=1)
    if args.kernel_name == 'hawkes_exponential_mutual':
        args.baselines = np.random.uniform(0,1, size=num_marks)
        args.adjacency = np.random.uniform(0,1, size=(num_marks,num_marks))
        args.decays = np.random.uniform(0,10, size=(num_marks,num_marks))
    else:
        args.baselines = np.random.uniform(0,1, size=num_marks)
        args.adjacency = np.random.uniform(0,1, size=(num_marks,num_marks, args.n_exp))
        args.decays = np.random.uniform(0,10, size=args.n_exp)
    print(args.baselines)
    #args.self_adjacency = np.random.uniform(0,.8, size=num_marks)
    #args.mutual_adjacency = np.random.uniform(0,0.3, size=1)
    args.seed += 1
    #print(f'SIMULATED PARAMETERS : {params}')
    return args

def make_splits(process, artifacts, args, split_idx, train_prop=0.6, val_prop=0.2):
    #end_train_idx = int(len(process)*train_prop)
    #end_val_idx = end_train_idx + int(len(process)*val_prop)
    train_idx = args.n_seq_train
    val_idx = args.n_seq_train + args.n_seq_val
    cal_idx = args.n_seq_train + args.n_seq_val + args.n_seq_cal
    train = process[0:train_idx]
    val = process[train_idx:val_idx]
    cal = process[val_idx:cal_idx]
    test = process[cal_idx:]
    data_args = {'seed':args.seed, 'window':args.window, 'marks':args.marks, 'train_size':len(train), 'val_size':len(val), 
                'cal_size':len(cal), 'test_size':len(test), 'decays':artifacts['decays'], 'baselines':artifacts['baselines'],
                'adjacency':artifacts['adjacency'], 'kernel_name':args.kernel_name, 'n_processes':args.n_processes}
    save_dir = os.path.join(Path.cwd(), args.save_dir)
    if args.dataset_name is None:
        save_dir = os.path.join(save_dir, args.kernel_name)
    else:
        save_dir = os.path.join(save_dir, args.dataset_name)
    save_dir = os.path.join(save_dir, 'split_{}'.format(split_idx))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)   
    save_train_path = os.path.join(save_dir, 'train.json')
    save_val_path = os.path.join(save_dir, 'val.json')
    save_cal_path = os.path.join(save_dir, 'cal.json')
    save_test_path = os.path.join(save_dir, 'test.json')
    save_args_path = os.path.join(save_dir, 'args.json')
    paths = [save_train_path, save_val_path, save_cal_path, save_test_path, save_args_path]
    data = [train, val, cal, test, data_args]
    for i, path in enumerate(paths):
        with open(path, 'w') as f:
            json.dump(data[i], f)
    print('Successfully created split {} for simulated dataset'.format(split_idx))

def make_test_set(process, artifacts, args):
    data_args = {'seed':args.seed, 'window':args.window, 'marks':args.marks, 'train_size':0, 'val_size':0, 'test_size':len(process),
            'decays':artifacts['decays'], 'baselines':artifacts['baselines'], 'adjacency':artifacts['adjacency'], 'kernel_name':args.kernel_name, 
            'n_processes':args.n_processes}
    save_dir = os.path.join(Path.cwd(), args.save_dir)
    save_dir = os.path.join(save_dir, args.dataset_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)   
    save_test_path = os.path.join(save_dir, 'test.json')
    save_args_path = os.path.join(save_dir, 'args.json')
    paths = [save_test_path, save_args_path]
    data = [process, data_args]
    for i, path in enumerate(paths):
        with open(path, 'w') as f:
            json.dump(data[i], f)
    print('Successfully created split for simulated dataset')

def make_hetero_splits(process, artifacts, args, train_prop=0.8, split=0):
    end_train_idx = int(len(process)*train_prop)
    train = process[0:end_train_idx]
    val = process[end_train_idx:]
    file = os.path.join(args.from_file, 'test.json')
    with open(file, 'r') as f:
        test = json.load(f)
    data_args = {'seed':args.seed, 'window':args.window, 'marks':args.marks, 'train_size':len(train), 'val_size':len(val), 'test_size':len(test),
            'decays':artifacts['decays'], 'baselines':artifacts['baselines'], 'adjacency':artifacts['adjacency'], 'kernel_name':args.kernel_name, 
            'n_processes':args.n_processes}
    save_dir = os.path.join(Path.cwd(), args.save_dir)
    if args.dataset_name is None:
        save_dir = os.path.join(save_dir, args.kernel_name)
    else:
        save_dir = os.path.join(save_dir, args.dataset_name)
    save_dir = os.path.join(save_dir, f'split_{split}')
    #save_dir = os.path.join(save_dir, 'split_{}'.format(split_idx))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)   
    save_train_path = os.path.join(save_dir, 'train.json')
    save_val_path = os.path.join(save_dir, 'val.json')
    save_test_path = os.path.join(save_dir, 'test.json')
    save_args_path = os.path.join(save_dir, 'args.json')
    paths = [save_train_path, save_val_path, save_test_path, save_args_path]
    data = [train, val, test, data_args]
    for i, path in enumerate(paths):
        with open(path, 'w') as f:
            json.dump(data[i], f)
    print('Successfully created split for simulated dataset')

def plot_intensity(args):
    sequence, artifacts = simulate_unique_seq(args)
    fig_pmf, ax_pmf = plt.subplots(1,2, figsize=(16, 8))
    times = np.array([event['time'] for event in sequence])
    labels_idx = [event['labels'][0] for event in sequence]
    ax_pmf[0] = plot_process(times, ax_pmf[0], labels_idx=labels_idx)
    plot_modelled_pmf(artifacts['intensity_times'], artifacts['true_mark_pmf'], ax_pmf[0], 
    labels_idx)
    plot_entropy(artifacts['intensity_times'], artifacts['true_mark_pmf'], ax_pmf[1])
    fig_pmf.savefig('figures/intensities/test.png', bbox_inches='tight')

def plot_heterogeneous_intensity(args):
    fig_pmf, ax_pmf = plt.subplots(1,args.n_processes, figsize=(16, 8))
    for i in range(args.n_processes):
        args_process = vars(args).copy()
        args_process['self_decays'] = args.self_decays[i]
        args_process['mutual_decays'] = args.mutual_decays[i]
        args_process['self_adjacency'] = args.self_adjacency[i]
        args_process['mutual_adjacency'] = args.mutual_adjacency[i]
        args_process['baselines'] = args.baselines[i]
        args_process = Namespace(**args_process)
        sequence, artifacts = simulate_unique_seq(args_process)
        times = np.array([event['time'] for event in sequence])
        labels_idx = [event['labels'][0] for event in sequence]
        ax_pmf[i] = plot_process(times, ax_pmf[i], labels_idx=labels_idx)
        plot_modelled_pmf(artifacts['intensity_times'], artifacts['true_mark_pmf'], ax_pmf[i], 
        labels_idx)
        #plot_entropy(artifacts['intensity_times'], artifacts['true_mark_pmf'], ax_pmf[i])
        
        fig_pmf.savefig('figures/intensities/test.png', bbox_inches='tight')



if __name__ == "__main__":
    args = parse_args()
    print('Simulating {} dataset'.format(args.kernel_name))
    if args.simu_type == 'dataset':
        if args.from_file is not None and args.num_splits != 0:
            for split in range(args.num_splits):
                process, artifacts = simulate_heterogeneous_dataset_from_file(args)
                make_hetero_splits(process, artifacts, args)
        else:
            if args.num_splits != 0:
                for split in range(args.num_splits):
                    if args.heterogeneous_dataset:
                        print(f'SIMULATING DATASET FOR {args.n_processes} PROCESSES')
                        process, artifacts = simulate_heterogeneous_dataset(args)
                    else:
                        process, artifacts = simulate_dataset(args)
                    make_splits(process, artifacts, args, split)
                print(f'Successfully saved simulated dataset for split {split}')
            else:
                print('here!')
                if args.n_processes == 1:
                    #Take the first process of the test set made of a 1000 processes to create an homogeneous test set 
                    process, artifacts = simulate_heterogeneous_dataset_from_file(args)
                else:
                    process, artifacts = simulate_heterogeneous_dataset(args)
                make_test_set(process, artifacts, args)
                print('DONE')
    else:
        if args.heterogeneous_dataset:
            plot_heterogeneous_intensity(args)
        else:
            plot_intensity(args)
        print('DONE')