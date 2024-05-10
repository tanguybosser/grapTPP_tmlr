#import mlflow
#import mlflow.pytorch
from ast import arg
from calendar import c
from cmath import exp, log
from logging.config import valid_ident
from unittest import result
#import imageio
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'esann2023')))

import pdb
import json
import numpy as np
import os
import stat
import time
import torchvision  
import pickle as pkl
import datetime 

import torch as th
from torch.optim import Adam
from torch.utils.data import DataLoader

from argparse import Namespace
from copy import deepcopy
from typing import Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from tpps.utils.events import get_events, get_window
from tpps.utils.mlflow import params_log_dict, get_epoch_str, log_metrics

from tpps.models import get_model
from tpps.models.base.process import Process
from tpps.utils.cli import parse_args
from tpps.utils.metrics import eval_metrics
from tpps.utils.plot import log_figures
from tpps.utils.data import get_loader, load_data
from tpps.utils.logging import get_status
from tpps.utils.lr_scheduler import create_lr_scheduler
from tpps.utils.run import make_deterministic
from tpps.utils.stability import check_tensor

from scripts.train2 import get_loss

print(th.cuda.is_available(), flush=True)


torchvision.__version__ = '0.4.0'


def detach(x: th.Tensor):
    return x.cpu().detach().numpy()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(model: Process, args: Namespace, loader: DataLoader, test: Optional[bool] = False
             ) -> Dict[str, float]:
    """Evaluate a model on a specific dataset.

    Args:
        model: The model to evaluate.
        args: Arguments for evaluation
        loader: The loader corresponding to the dataset to evaluate on.
        test: If False, put what is not returned by log-likelihhod. 

    Returns:
        Dictionary containing all metrics evaluated and averaged over total
            sequences.

    """
    model.eval()

    t0, epoch_loss, epoch_loss_per_time, n_seqs = time.time(), 0., 0., 0.
    epoch_loss_t, epoch_loss_m, epoch_loss_w = 0, 0, 0 
    pred_labels, gold_labels, mask_labels, probas, ranks = [], [], [], [], []
    results = {}
    epoch_ground_density_loss, epoch_marks_loss, epoch_window_loss = 0, 0, 0
    epoch_intensity_0, epoch_intensity_1, alpha = 0, 0, 0
    epoch_mu, epoch_sigma, epoch_weight, epoch_pmc, epoch_pm = [], [], [], [], []
    epoch_h_t, epoch_h_m, epoch_h = [], [], []
    cumulative_density = []
    all_log_density_per_seq, all_log_mark_density_per_seq, n_valid_events = [], [], []
    num_batch = 0
    for batch in tqdm(loader) if args.verbose else loader:
        batch['times'], batch['labels'], batch['seq_lens'] = batch['times'].to(args.device), batch['labels'].to(args.device), batch['seq_lens'].to(args.device)
        loss_t, loss_m, loss_w, loss_mask, artifacts = get_loss(  # [B]
            model, batch=batch, eval_metrics=args.eval_metrics, args=args,
            dynamic_batch_length=False, test=test) 
        loss_t = loss_t * loss_mask #[B]
        loss_m = loss_m * loss_mask
        loss_w = loss_w * loss_mask
        epoch_loss_t += detach(th.sum(loss_t))
        epoch_loss_m += detach(th.sum(loss_m))
        epoch_loss_w += detach(th.sum(loss_w))
        epoch_loss += epoch_loss_t + epoch_loss_m + epoch_loss_w
        if test:
            cdf = artifacts['cumulative density'].cpu().numpy()
            valid_cdf = [cdf[i][cdf[i] >=0].tolist() for i in range(cdf.shape[0])]
            cumulative_density.extend(valid_cdf)
            log_mark_density_per_seq =  artifacts["log mark density per seq"].tolist()
            log_density_per_seq =  artifacts["log density per seq"].tolist()
            all_log_mark_density_per_seq.extend(log_mark_density_per_seq)
            all_log_density_per_seq.extend(log_density_per_seq)
            n_valid_events.append(int(artifacts['n valid events']))
        #winwow_loss = artifacts["window integral"] * loss_mask
        #epoch_ground_density_loss += detach(th.sum(ground_density_loss))
        #epoch_marks_loss += detach(th.sum(marks_loss))
        #epoch_window_loss += detach(th.sum(winwow_loss))
        #loss_per_time = loss / artifacts["interval"] 
        #epoch_loss_per_time += detach(th.sum(loss_per_time))
        if 'alpha' in artifacts:
            epoch_intensity_0 += detach(th.sum(artifacts['intensity_0'] * loss_mask))
            epoch_intensity_1 += detach(th.sum(artifacts['intensity_1'] * loss_mask))
            alpha += artifacts['alpha']
        if test and 'mu' in artifacts:
            epoch_mu.append(artifacts["mu"])
            epoch_sigma.append(artifacts["sigma"])
            epoch_weight.append(artifacts["w"])
        if test and 'pm' in artifacts:
            epoch_pm.append(artifacts['pm'])
            if 'pmc' in artifacts:
                epoch_pmc.append(artifacts['pmc'])
        if test and 'last_h_t' in artifacts:
            epoch_h_t.append(artifacts['last_h_t'])
            epoch_h_m.append(artifacts['last_h_m'])
        if test and 'last_h' in artifacts:
            epoch_h.append(artifacts['last_h'])
        n_seqs_batch = detach(th.sum(loss_mask))
        n_seqs += n_seqs_batch
        num_batch += 1

        if args.eval_metrics:
            pred_labels.append(detach(artifacts['y_pred']))
            gold_labels.append(detach(artifacts['y_true']))
            mask_labels.append(detach(artifacts['y_pred_mask']))
            probas.append(detach(artifacts['max proba']))
            ranks.append(artifacts['ranks proba'])
    if args.eval_metrics: 
        results = eval_metrics(
            pred=pred_labels,
            gold=gold_labels,
            probas=probas,
            ranks=ranks,
            mask=mask_labels,
            results=results,
            n_class=args.marks,
            multi_labels=args.multi_labels,
            test=test)

    dur = time.time() - t0
    results["dur"] = dur
    results["loss"] = float(epoch_loss / n_seqs)
    #results["loss_per_time"] = float(epoch_loss_per_time / n_seqs)
    results["log ground density"] = float(-epoch_loss_t/n_seqs)
    results["log mark density"] = float(-epoch_loss_m/n_seqs)
    results["window integral"] = float(-epoch_loss_w/n_seqs)
    if test:
        results["cdf"] = cumulative_density
        results['log density per seq'] = all_log_density_per_seq
        results['log mark density per seq'] = all_log_mark_density_per_seq
        results['n valid events'] = n_valid_events
        results['median log density'] = np.median(all_log_density_per_seq)
        results['median log mark density'] = np.median(all_log_mark_density_per_seq)
        if 'last_h_t' in artifacts:
            last_h_t = np.concatenate(epoch_h_t)
            last_h_m = np.concatenate(epoch_h_m)
            results['last_h_t'] = last_h_t
            results['last_h_m'] = last_h_m
        if 'last_h' in artifacts:
            last_h = np.concatenate(epoch_h)
            results['last_h'] = last_h
    results['intensity_0'] = float(epoch_intensity_0/n_seqs)
    results['intensity_1'] = float(epoch_intensity_1/n_seqs)
    if args.use_coefficients:
        results['alpha'] = alpha/num_batch
    results["mu"] = epoch_mu
    results["sigma"] = epoch_sigma
    results["mixture weights"] = epoch_weight
    results['pmc'] = epoch_pmc
    results['pm'] = epoch_pm
    
    return results



def set_model_state(args, model, best_state, best_state_time, best_state_mark):
    if args.separate_training:
        model_dict = model.state_dict()
        mark_dict = {k: v for k, v in best_state_mark.items() if 'mark' in k}
        model_dict.update(mark_dict)
        time_dict = {k: v for k, v in best_state_time.items() if not 'mark' in k}
        model_dict.update(time_dict)
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(best_state)
    return model


def early_stopping(val_metrics, best_loss, best_loss_time, best_loss_mark, args):
    if not args.separate_training: 
        new_best_time, new_best_mark = True, True
        val_loss = -val_metrics['log ground density'] - val_metrics['log mark density'] - val_metrics['window integral']
        new_best = val_loss < best_loss
        if args.loss_relative_tolerance is not None:
            abs_rel_loss_diff = (val_loss - best_loss) / best_loss
            abs_rel_loss_diff = abs(abs_rel_loss_diff)
            above_numerical_tolerance = (abs_rel_loss_diff >
                                            args.loss_relative_tolerance)
            new_best = new_best and above_numerical_tolerance
    else:
        val_loss_time = -val_metrics['log ground density']- val_metrics['window integral']
        val_loss_mark = -val_metrics['log mark density']
        new_best_time = val_loss_time < best_loss_time
        new_best_mark = val_loss_mark < best_loss_mark
        new_best = True
        if args.loss_relative_tolerance is not None:
            abs_rel_loss_diff_time = abs((val_loss_time - best_loss_time) / best_loss_time)
            abs_rel_loss_diff_mark = abs((val_loss_mark - best_loss_mark) / best_loss_mark)
            above_numerical_tolerance_time = (abs_rel_loss_diff_time >
                                            args.loss_relative_tolerance)
            above_numerical_tolerance_mark = (abs_rel_loss_diff_mark >
                                            args.loss_relative_tolerance)
            new_best_time = new_best_time and above_numerical_tolerance_time
            new_best_mark = new_best_mark and above_numerical_tolerance_mark    
    return new_best, new_best_time, new_best_mark
        
    
def main(args: Namespace):
    model_path = os.path.join(args.save_check_dir, args.model_name + '.pth')    
    print(model_path)
    model_name  = args.model_name
    args = load_args(args)
    datasets = load_data(args=args) 
    loaders = dict()
    loaders["train"] = get_loader(datasets["train"], args=args, shuffle=True)
    loaders["val"] = get_loader(datasets["val"], args=args, shuffle=False)
    loaders["test"] = get_loader(datasets["test"], args=args, shuffle=False)
    model = get_model(args)
    model.load_state_dict(th.load(model_path, map_location=args.device))
    num_params = count_parameters(model)
    print('NUM OF TRAINABLE PARAMETERS : {}'.format(num_params))
    print("INSTATIATED MODEL : {}/{}/{} on dataset {}".format(args.encoder_encoding, args.encoder, args.decoder, args.dataset))
    '''
    if args.mu_cheat and "poisson" in model.processes: 
        poisson = model.processes["poisson"].decoder
        mu = th.from_numpy(args.mu).type(
            poisson.mu.dtype).to(poisson.mu.device)
        poisson.mu.data = mu
    model, images_urls, train_metrics, val_metrics = train(
        model, args=args, loader=loaders["train"],
        val_loader=loaders["val"], test_loader=loaders["test"]) 
    
    if args.save_check_dir is not None:
        file_path = os.path.join(args.save_check_dir, exp_name + '.pth')
        th.save(model.state_dict(), file_path)
        print('Model saved to disk')
    '''
    print("EVALUATING MODEL")
    metrics = {
        k: evaluate(model=model, args=args, loader=l, test=True)
        for k, l in loaders.items()}
    train_metrics = metrics['train'] 
    val_metrics = metrics['val']
    test_metrics = metrics['test']
    if args.verbose:
        print(metrics, flush=True)
    if args.save_results_dir is not None:
        save_evaluation_results(train_metrics, val_metrics, test_metrics,save_path=args.save_results_dir, exp_name=model_name ,args=args)

def load_args(args:Namespace):
    batch_size = args.batch_size
    load_dir = args.load_from_dir
    save_dir = args.save_results_dir
    args_path = os.path.join(args.save_check_dir, 'args')
    args_path = os.path.join(args_path, args.model_name + '.json')
    args = vars(args)
    with open(args_path, 'r') as f:
        model_args = json.load(f)
    args.update(model_args)
    args = Namespace(**args)
    cuda = th.cuda.is_available() and not parsed_args.disable_cuda
    if cuda:
        args.device = th.device('cuda')
    else:
        args.device = th.device('cpu')
    args.batch_size = batch_size
    args.load_from_dir = load_dir
    args.save_results_dir = save_dir    
    return args 


def save_evaluation_results(train_metrics: Dict[str, list], val_metrics: Dict[str, list], test_metrics: Dict[str, list], 
                save_path: str, exp_name: str, args: Namespace):
    results = {'train':train_metrics, 'val':val_metrics, 'test':test_metrics, 'args':args}
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    exp_name = exp_name + '_evaluation.txt'
    save_path = os.path.join(save_path, exp_name)
    with open(save_path, "wb") as fp: 
        pkl.dump(results, fp)
    print('Results saved to {}'.format(save_path, flush=True))

def get_state_dic(path:str):
    state_dic = th.load(path, map_location=th.device('cpu'))
    print(state_dic)


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.load_from_dir is not None:
        json_dir = os.path.join(os.getcwd(), parsed_args.load_from_dir)
        json_dir = os.path.join(json_dir, parsed_args.dataset)
        if parsed_args.split is not None:
            split = 'split_{}'.format(parsed_args.split)
            json_dir = os.path.join(json_dir, split)
        json_path = os.path.join(json_dir, 'args.json')
        #parsed_args.data_dir = os.path.join(os.getcwd(), parsed_args.data_dir)#os.path.expanduser(parsed_args.data_dir)
        #parsed_args.save_dir = os.path.join(parsed_args.data_dir,
        #                                parsed_args.load_from_dir)
        #json_path = parsed_args.save_dir + '/args.json'
        with open(json_path, 'r') as fp:
            args_dict_json = json.load(fp)
        args_dict = vars(parsed_args)
        #print("Warning: overriding some args from json:", flush=True)
        shared_keys = set(args_dict_json).intersection(set(args_dict))
        for k in shared_keys:
            v1, v2 = args_dict[k], args_dict_json[k]
            is_equal = np.allclose(v1, v2) if isinstance(
                v1, np.ndarray) else v1 == v2
            if not is_equal:
                print(f"    {k}: {v1} -> {v2}", flush=True)
        args_dict.update(args_dict_json)
        parsed_args = Namespace(**args_dict)
        parsed_args.mu = np.array(parsed_args.mu, dtype=np.float32)
        parsed_args.alpha = np.array(
            parsed_args.alpha, dtype=np.float32).reshape(
            parsed_args.mu.shape * 2)
        parsed_args.beta = np.array(
            parsed_args.beta, dtype=np.float32).reshape(
            parsed_args.mu.shape * 2)

    else:
        parsed_args.data_dir = os.path.expanduser(parsed_args.data_dir)
        parsed_args.save_dir = os.path.join(parsed_args.data_dir, "None")
        Path(parsed_args.save_dir).mkdir(parents=True, exist_ok=True)

    # check_repo(allow_uncommitted=not parsed_args.use_mlflow)
    make_deterministic(seed=parsed_args.seed)
    main(args=parsed_args)
