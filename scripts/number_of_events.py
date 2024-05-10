from cProfile import label
import numpy as np 
import os
import torch as th 
from pathlib import Path
import json
import matplotlib.pyplot as plt 
from plots.acronyms import get_acronym
from plots.plots import map_datasets_name
import importlib, sys
import pickle as pkl

from tpps.processes.multi_class_dataset import MultiClassDataset as Dataset
from tpps.utils.data import get_loader
from tpps.utils.events import get_events, get_window 
from tpps.models import get_model


from argparse import ArgumentParser, Namespace
from distutils.util import strtobool
from tpps.models.base.process import Process

from show_modeled_intensity import get_intensity



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
    parser.add_argument("--window", type=int, default=100,
                        help="The window of the simulated process.py. Also "
                             "taken as the window of any parametric Hawkes "
                             "model if chosen.")
    # Dirs
    parser.add_argument("--load-from-dir", type=str, required=True,
                        help="If not None, load data from a directory")
    parser.add_argument("--dataset", type=str, required=True,
                        help="If not None, load data from a directory")
    parser.add_argument("--splits", type=int, default=5,
                        help="Number of dataset splits on which to evaluate the model.")
    parser.add_argument("--save-events-results-dir", type=str, required=True,
                        help="Directory to save the results.")

    #Simulation
    parser.add_argument("--window-end", type=int, default=10,
                        help="End window of the simulated process")
    args, _ = parser.parse_known_args()
    cwd = Path.cwd()
    
    model_path = os.path.join('checkpoints/model_selection/marked_filtered', args.dataset)
    model_dir = os.path.join(model_path, 'best')
    model_dir = os.path.join(cwd, model_dir)
    args_dict = vars(args)
    args_dict["model_dir"] = model_dir
    args = Namespace(**args_dict)
    return args

def update_dataset_args(args, dataset_path):
    cwd = Path.cwd()    
    data_args_path = os.path.join(dataset_path, 'args.json')
    data_args_path = os.path.join(cwd, data_args_path)
    save_results_path = os.path.join(cwd, args.save_events_results_dir)
    save_results_path = os.path.join(save_results_path, args.dataset)
    save_results_path = os.path.join(save_results_path, 'best')
    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)
    with open(data_args_path, 'r') as fp:
        args_dict_json = json.load(fp)
    args_dict = vars(args)
    args_dict.update(args_dict_json)
    args = Namespace(**args_dict)
    args.save_results_path = save_results_path
    return args

def update_model_args(args, model_file):
    cwd = Path.cwd()
    model_name = model_file.split('/')[-1][:-4]
    model_args_path = os.path.join(Path(model_file).parent, 'args')
    model_args_path_1 = os.path.join(model_args_path, model_name + '.json')
    model_args_path_1 = os.path.join(cwd, model_args_path_1)
    model_args_path_2 = os.path.join(model_args_path, model_name + '_args.json')
    model_args_path_2 = os.path.join(cwd, model_args_path_2)
    save_results_path = os.path.join(args.save_results_path, model_name + '.txt')
    args.save_results_path = save_results_path
    args_dict = vars(args)
    try:
        with open(model_args_path_1, 'r') as f:
            model_args_dic = json.load(f)
    except:
        with open(model_args_path_2, 'r') as f:
            model_args_dic = json.load(f)
    args_dict.update(model_args_dic)
    args = Namespace(**args_dict)
    cuda = th.cuda.is_available() 
    if cuda:
        args.device = th.device('cuda')
    else:
        args.device = th.device('cpu')
    args.verbose = False
    return args, model_name


def number_of_events(args):
    dataset_path = os.path.join(args.load_from_dir, args.dataset)
    for split in range(args.splits):
        split_path = os.path.join(dataset_path, 'split_{}'.format(str(split)))
        args = update_dataset_args(args, split_path)
        split_path = os.path.join(split_path, 'test.json')
        with open(split_path, 'r') as f:
            dataset = json.load(f)  
        all_models_files = os.listdir(args.model_dir)
        if 'base' in args.model:
            if 'lnmk1' in args.model:
                file_to_find = 'poisson_' + args.dataset + '_' + args.model.split('_lnmk1')[0] + '_split{}_lnmk1_config'.format(str(split))    
            else:    
                file_to_find = 'poisson_' + args.dataset + '_' + args.model.split('_base')[0] + '_split{}_config'.format(str(split))
        else:
            if 'lnmk1' in args.model:
                file_to_find = args.dataset + '_' + args.model.split('_lnmk1')[0] + '_split{}_lnmk1_config'.format(str(split))
            else:
                file_to_find = args.dataset + '_' + args.model + '_split{}_config'.format(str(split))
        print(file_to_find)
        model_file = None
        for file in all_models_files:
            if file.startswith(file_to_find):
                model_file = os.path.join(args.model_dir, file)
                break
        if model_file is None:
            raise ValueError('Model checkpoint not found')
        args, model_name = update_model_args(args, model_file)
        model_name = model_file.split('/')[-1][:-3]
        data_sequence = Dataset(
        args=args, size=args.batch_size, seed=args.seed, data=dataset)
        loader = get_loader(data_sequence, args=args, shuffle=False)
        model = get_model(args)
        model.load_state_dict(th.load(model_file, map_location=args.device))
        n_seq = 0
        diff_events_total, diff_events_total_squared = 0, 0
        #results = {}
        #print('EVALUATION STARTING', flush=True)
        for batch in loader:
            times, labels = batch["times"], batch["labels"]
            labels = (labels != 0).type(labels.dtype) #?
            mask = (times != args.padding_id).type(times.dtype)
            times = times * args.time_scale #time_scale=1 by default
            window_start, window_end = get_window(times=times, window=args.window)
            events = get_events(
                times=times, mask=mask, labels=labels,
                window_start=window_start, window_end=window_end)
            event_times = events.get_times(postpend_window=True)
            _, _, ground_intensity_integral_events, intensity_mask , _ = get_intensity(model, event_times, events, args, is_event=True) #[B,L+1]
        
            sum_ground_integral = np.sum(ground_intensity_integral_events, axis=-1) #[B]
            times = times * intensity_mask[:,:-1]
            times_mask = times != 0
            num_events = th.sum(times_mask, dim=-1) #[B]
            num_events = num_events.detach().cpu().numpy()
            n_seq_batch = len(num_events)
            diff_events_batch_squared = np.sum(np.power(num_events-sum_ground_integral, 2)) #[1]
            #diff_events_batch = np.sum(num_events-sum_ground_integral)
            diff_events_batch = np.sum(num_events-sum_ground_integral)
            n_seq += n_seq_batch
            diff_events_total_squared += diff_events_batch_squared
            diff_events_total += diff_events_batch
        diff_events_total = diff_events_total/n_seq
        diff_events_total_squared = diff_events_total_squared/n_seq
        #results['diff_num_events'] = float(diff_events_total)
        with open(args.save_results_path, 'rb') as f:
            old_results = pkl.load(f)
        old_results['test']['num_events_squared'] = float(diff_events_total_squared)
        old_results['test']['num_events_total'] = float(diff_events_total)
        with open(args.save_results_path, 'wb') as f:
            pkl.dump(old_results, f)


if __name__ == "__main__":
    parsed_args = parse_args_intensity()
    number_of_events(parsed_args)