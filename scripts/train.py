import sys, os
sys.dont_write_bytecode = True

import json
import numpy as np
import time
import pickle as pkl
from collections import defaultdict

import torch as th
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from torch.linalg import vector_norm

from argparse import Namespace
from copy import deepcopy
from typing import Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from tpps.utils.events import get_events, get_window

from tpps.models import get_model
from tpps.models.base.process import Process
from tpps.utils.cli import parse_args
from tpps.utils.metrics import eval_metrics
from tpps.utils.data import get_loader, load_data
from tpps.utils.run import make_deterministic
from tpps.utils.stability import check_tensor

from tpps.utils.utils import detach, count_parameters, count_parameters_enc


print('cuda', th.cuda.is_available(), flush=True)

def get_loss(
        model: Process,
        batch: Dict[str, th.Tensor],
        args: Namespace,
        eval_metrics: Optional[bool] = False,
        dynamic_batch_length: Optional[bool] = True,
        test: Optional[bool] = False
) -> Tuple[th.Tensor, th.Tensor, Dict]:
    times, labels = batch["times"], batch["labels"]
    labels = (labels != 0).type(labels.dtype) 
    mask = (times != args.padding_id).type(times.dtype)
    times = times * args.time_scale 
    window_start, window_end = get_window(times=times, window=args.window) 
    events = get_events(
        times=times, mask=mask, labels=labels,
        window_start=window_start, window_end=window_end)
    loss_t, loss_m, loss_w, loss_mask, artifacts = model.neg_log_likelihood(events=events, test=test, time_scaling=args.nll_scaling)  # [B]
    if eval_metrics:
        events_times = events.get_times()
        log_p, log_mark_density, y_pred_mask = model.log_density(
        query=events_times, events=events)
        proba = th.exp(log_mark_density)
        if args.multi_labels:
            y_pred = log_p  # [B,L,M]
            labels = events.labels
        else:
            y_pred = log_mark_density.argmax(-1).type(events.labels.dtype)  # [B,L]
            labels = events.labels.argmax(-1).type(events.labels.dtype)
            max_proba  = proba.max(-1).values.type(log_mark_density.dtype)
            proba = proba.detach().cpu().numpy()
            ranks = (-proba).argsort(-1).argsort(-1)
            ranks_idx = labels.unsqueeze(-1) #[B,L,1]
            ranks_idx = ranks_idx.detach().cpu().numpy().astype(np.int32)            
            ranks_proba = np.take_along_axis(ranks, ranks_idx, axis=-1).squeeze(-1) #[B,L]
            artifacts['ranks proba'] = ranks_proba
        artifacts['y_pred'] = y_pred
        artifacts['y_true'] = labels
        artifacts['y_pred_mask'] = y_pred_mask
        artifacts['max proba'] = max_proba
    return loss_t, loss_m, loss_w, loss_mask, artifacts


def evaluate(model: Process, args: Namespace, loader: DataLoader, test: Optional[bool] = False, eval_met: Optional[bool] =False
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
    epoch_h_t, epoch_h_m, epoch_h = [], [], []
    cumulative_density = []
    num_batch = 0
    for batch in tqdm(loader) if args.verbose else loader:
        batch['times'], batch['labels'], batch['seq_lens'] = batch['times'].to(args.device), batch['labels'].to(args.device), batch['seq_lens'].to(args.device)
        loss_t, loss_m, loss_w, loss_mask, artifacts = get_loss(  # [B]
            model, batch=batch, eval_metrics=eval_met, args=args,
            dynamic_batch_length=False, test=test) 
        loss_t = loss_t * loss_mask #[B]
        loss_m = loss_m * loss_mask
        loss_w = loss_w * loss_mask
        
        true_loss_t = artifacts['loss_t'] * loss_mask
        true_loss_m = artifacts['loss_m'] * loss_mask      
        true_loss_w = artifacts['loss_w'] * loss_mask

        epoch_loss_t += detach(th.sum(true_loss_t))
        epoch_loss_m += detach(th.sum(true_loss_m))
        epoch_loss_w += detach(th.sum(true_loss_w))
        epoch_loss += epoch_loss_t + epoch_loss_m + epoch_loss_w
        
        if test:
            cdf = artifacts['cumulative density'].cpu().numpy()
            valid_cdf = [cdf[i][cdf[i] >=0].tolist() for i in range(cdf.shape[0])]
            cumulative_density.extend(valid_cdf)
        if test and 'last_h_t' in artifacts:
            epoch_h_t.append(artifacts['last_h_t'])
            epoch_h_m.append(artifacts['last_h_m'])
        if test and 'last_h' in artifacts:
            epoch_h.append(artifacts['last_h'])
        n_seqs_batch = detach(th.sum(loss_mask))
        n_seqs += n_seqs_batch
        num_batch += 1

        if eval_met:
            pred_labels.append(detach(artifacts['y_pred']))
            gold_labels.append(detach(artifacts['y_true']))
            mask_labels.append(detach(artifacts['y_pred_mask']))
            probas.append(detach(artifacts['max proba']))
            ranks.append(artifacts['ranks proba'])
    if eval_met: 
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
    results['loss_t'] = float(epoch_loss_t / n_seqs)
    results['loss_m'] = float(epoch_loss_m / n_seqs)
    results['loss_w'] = float(epoch_loss_w / n_seqs)
    if test:
        results["cdf"] = cumulative_density
        if 'last_h_t' in artifacts:
            last_h_t = np.concatenate(epoch_h_t)
            last_h_m = np.concatenate(epoch_h_m)
            results['last_h_t'] = last_h_t
            results['last_h_m'] = last_h_m
        if 'last_h' in artifacts:
            last_h = np.concatenate(epoch_h)
            results['last_h'] = last_h
    return results

def compute_grad_metrics(time_grads, mark_grads, dot_dic, sim_dic, tpi_dic):
    for k in time_grads.keys():
        g_t = time_grads[k]
        g_m = mark_grads[k]
        norm_g_t = vector_norm(g_t, ord=2)
        norm_g_m = vector_norm(g_m, ord=2)
        tpi = int(norm_g_m < norm_g_t)
        #Gradient magnitude similarity. 
        gms = (2 * vector_norm(g_t, ord=2) * vector_norm(g_m, ord=2))
        gms = gms/(th.square(vector_norm(g_t, ord=2)) + th.square(vector_norm(g_m, ord=2)))                
        #Avoids numerical instabilities 
        g_t[th.abs(g_t) < 1e-6] = 0
        g_m[th.abs(g_m) < 1e-6] = 0
        g_t = F.normalize(g_t, dim=0)
        g_m = F.normalize(g_m, dim=0)
        dot = float((g_t * g_m).sum(dim=0))
        dot_dic[k].append(dot)
        sim_dic[k].append(float(gms))
        tpi_dic[k].append(tpi)        
    return dot_dic, sim_dic, tpi_dic
        

def train(
        model: Process,
        args: Namespace,
        loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader) -> Tuple[Process, dict, list, list]:
    """Train a model.

    Args:
        model: Model to be trained.
        args: Arguments for training.
        loader: The dataset for training.
        val_loader: The dataset for evaluation.
        test_loader: The dataset for testing

    Returns:
        Best trained model from early stopping.

    """
    print("NEW TRAINING HAS BEGUN")
    train_metrics_list, val_metrics_list = [], []
    optimizer = Adam(model.parameters(), lr=args.lr_rate_init)
    lr_wait, cnt_wait, cnt_wait_time, cnt_wait_mark, best_epoch = 0, 0, 0, 0, 0 
    best_loss, best_loss_time, best_loss_mark = 1e9, 1e9, 1e9
    if args.separate_training:
        best_state = deepcopy(model.state_dict())
        best_state_time, best_state_mark = None, None  
    else:
        best_state = None 
        best_state_time = deepcopy(model.state_dict())
        best_state_mark = deepcopy(model.state_dict())
    train_dur, val_dur, images_urls = list(), list(), dict()
    epochs = range(args.train_epochs)
    if args.verbose:
        epochs = tqdm(epochs)
    t_start = time.time()
    epoch_dot_products = {k:[] for k,_ in model.named_parameters()}
    epoch_grad_sim = {k:[] for k,_ in model.named_parameters()}
    epoch_grad_tpi = {k:[] for k,_ in model.named_parameters()}
    for j, epoch in enumerate(epochs):
        print(f'Epoch: {j}')
        t0, _ = time.time(), model.train()
        train_metrics = {}
        epoch_loss_time, epoch_loss_mark, epoch_loss_window, epoch_loss = 0, 0, 0, 0 
        n_seqs = 0
        time_grads, mark_grads = defaultdict(), defaultdict()
        dot_products = {k:[] for k,_ in model.named_parameters()}
        grad_sim = {k:[] for k,_ in model.named_parameters()}
        grad_tpi = {k:[] for k,_ in model.named_parameters()}
        for i, batch in enumerate((tqdm(loader)) if args.verbose else loader):
            batch['times'], batch['labels'], batch['seq_lens'] = batch['times'].to(args.device), batch['labels'].to(args.device), batch['seq_lens'].to(args.device)
            optimizer.zero_grad()
            loss_t, loss_m, loss_w, loss_mask, artifacts = get_loss(model, batch=batch, args=args, test=False)  # [B]
            loss_t = th.sum(loss_t * loss_mask)
            loss_mark = th.sum(loss_m * loss_mask)
            loss_w = th.sum(loss_w * loss_mask)
            loss_time = loss_t + loss_w 
            check_tensor(loss_time)
            check_tensor(loss_mark)
            loss_time.backward(retain_graph=True)
            for name, p in model.named_parameters():
                if p.requires_grad:            
                    if p.grad is None:
                        grad = th.zeros_like(p.data, device='cpu').detach().view(-1)
                    else:
                        grad = p.grad.data.detach().clone().cpu().view(-1)
                    time_grads[name] = grad
            loss_mark.backward()
            for name, p in model.named_parameters():
                if p.requires_grad: 
                    #Grads are accumulated, so we must subtract the time grads to get the mark grads. 
                    mark_grads[name] = p.grad.data.detach().clone().cpu().view(-1) - time_grads[name]            
            dot_products, grad_sim, grad_tpi = compute_grad_metrics(time_grads,
                                                                    mark_grads,
                                                                    dot_products,
                                                                    grad_sim,
                                                                    grad_tpi) 
            optimizer.step()
            epoch_loss_time += detach(loss_time)
            epoch_loss_mark += detach(loss_mark)
            epoch_loss_window += detach(loss_w)
            epoch_loss += epoch_loss_time + epoch_loss_mark + epoch_loss_window
            n_seqs += detach(th.sum(loss_mask))
        train_metrics['dur'] = time.time() - t0
        train_metrics['loss'] = float(epoch_loss/n_seqs)
        train_metrics['loss_t'] = float(epoch_loss_time/n_seqs)
        train_metrics['loss_m'] = float(epoch_loss_mark/n_seqs)
        train_metrics['loss_w'] = float(epoch_loss_window/n_seqs)
        train_metrics['dot'] = dot_products
        train_metrics['sim'] = grad_sim
        train_metrics['ind'] = grad_tpi
        train_metrics_list.append(train_metrics)
        val_metrics = evaluate(model, args=args, loader=val_loader, test=False, eval_met=False)
        val_dur.append(val_metrics["dur"])
        val_metrics_list.append(val_metrics)
        new_best, new_best_time, new_best_mark = early_stopping(val_metrics, best_loss, best_loss_time, best_loss_mark, parsed_args)
        
        for k, v in dot_products.items():
            epoch_dot_products[k].extend(v) 
            epoch_grad_sim[k].extend(grad_sim[k])
            epoch_grad_tpi[k].extend(grad_tpi[k])
        #Early stopping
        if not args.separate_training:
            if new_best:
                val_loss = val_metrics['loss_t'] + val_metrics['loss_m'] + val_metrics['loss_w']
                best_loss, best_t = val_loss, epoch
                cnt_wait, lr_wait = 0, 0
                best_state = deepcopy(model.state_dict())
            else:
                cnt_wait, lr_wait = cnt_wait + 1, lr_wait + 1
            if cnt_wait == args.patience:
                print("Early stopping! Stopping at epoch {}".format(str(epoch)), flush=True)
                break
        else:
            if new_best_time and cnt_wait_time < args.patience:
                best_loss_time = val_metrics['loss_t'] + val_metrics['loss_w']
                cnt_wait_time, lr_wait_time = 0, 0
                best_state_time = deepcopy(model.state_dict())
            else:
                cnt_wait_time, lr_wait_time = cnt_wait_time + 1, lr_wait_time + 1
            if new_best_mark and cnt_wait_mark < args.patience:
                best_loss_mark = val_metrics['loss_m']
                cnt_wait_mark, lr_wait_mark = 0, 0 
                best_state_mark = deepcopy(model.state_dict())
            else:
                cnt_wait_mark, lr_wait_mark = cnt_wait_mark + 1, lr_wait_mark + 1
            if new_best_time or new_best_mark:
                best_state = deepcopy(model.state_dict())
            if cnt_wait_time > args.patience and cnt_wait_mark > args.patience:
                print("Early stopping! Stopping at epoch {}".format(str(epoch)), flush=True)
                print(f'Time training stopped at epoch {epoch-cnt_wait_time}')
                print(f'Mark training stopped at epoch {epoch-cnt_wait_mark}')
                train_metrics_list.append({'stop epoch time':epoch-cnt_wait_time,
                                           'stop epoch mark':epoch-cnt_wait_mark})
                break
            if cnt_wait_time > args.patience:
                for name, param in model.named_parameters():
                    if 'mark' not in name:
                        param.requires_grad = False
            if cnt_wait_mark > args.patience:
                for name, param in model.named_parameters():
                    if 'mark' in name:
                        param.requires_grad = False
            optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_rate_init)
        lr = optimizer.param_groups[0]['lr']
        train_metrics["lr"] = lr
    train_metrics_list.append({'dot_products':epoch_dot_products, 'grad_sim':epoch_grad_sim, 'grad_ind': epoch_grad_tpi})
    model = set_model_state(args, model, best_state, best_state_time, best_state_mark)
    delta_t = time.time() - t_start
    hours, mins, sec = int(delta_t/3600), int((delta_t%3600)/60), int((delta_t%3600)%60)
    print("Total training time : {}:{}:{}".format(hours,mins,sec))
    return model, images_urls, train_metrics_list, val_metrics_list



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
        val_loss = val_metrics['loss_t'] + val_metrics['loss_m'] + val_metrics['loss_w']
        new_best = val_loss < best_loss
        if args.loss_relative_tolerance is not None:
            abs_rel_loss_diff = (val_loss - best_loss) / best_loss
            abs_rel_loss_diff = abs(abs_rel_loss_diff)
            above_numerical_tolerance = (abs_rel_loss_diff >
                                            args.loss_relative_tolerance)
            new_best = new_best and above_numerical_tolerance
    else:
        val_loss_time = val_metrics['loss_t'] + val_metrics['loss_w']
        val_loss_mark = val_metrics['loss_m']
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
    datasets = load_data(args=args) 
    loaders = dict()
    if args.limit_train_size is not None:
        train_size = int(args.train_size*args.limit_train_size)
        subset_indices = list(range(train_size))
        datasets['train'] = Subset(datasets['train'], subset_indices)
    loaders["train"] = get_loader(datasets["train"], args=args, shuffle=True)
    loaders["val"] = get_loader(datasets["val"], args=args, shuffle=False)
    loaders["test"] = get_loader(datasets["test"], args=args, shuffle=False)
    exp_name = get_exp_name(args)
    save_args(args, exp_name)
    print('ARGS SAVED')
    if args.decoder == 'hawkes_fixed':
        args = update_args_hawkes(args)
        save_args(args, exp_name)
        model = get_model(args)
        save_model(model, args, exp_name)
        return 
    model = get_model(args)
    print("INSTATIATED MODEL : {}/{}/{} on dataset {}".format(args.encoder_encoding, args.encoder, args.decoder, args.dataset))
    model, images_urls, train_metrics, val_metrics = train(
        model, args=args, loader=loaders["train"],
        val_loader=loaders["val"], test_loader=loaders["test"]) 
        
    save_model(model, args, exp_name)
    print("EVALUATING MODEL")
    t_val = time.time()
    metrics = {
    'train':evaluate(model=model, args=args, loader=loaders["train"], test=True, eval_met=False),
    'val':evaluate(model=model, args=args, loader=loaders["val"], test=True, eval_met=False),
    'test':evaluate(model=model, args=args, loader=loaders["test"], test=True, eval_met=args.eval_metrics)
    }
    delta_t = time.time() - t_val
    hours, mins, sec = int(delta_t/3600), int((delta_t%3600)/60), int((delta_t%3600)%60)
    print("Total validation time : {}:{}:{}".format(hours,mins,sec))
    train_metrics.append(metrics['train']) 
    val_metrics.append(metrics['val'])
    test_metrics = metrics['test']
    if args.verbose:
        print(metrics, flush=True)
    if args.save_results_dir is not None:
        save_results(train_metrics, val_metrics, test_metrics,save_path=args.save_results_dir, exp_name=exp_name ,args=args)

def save_model(model, args, exp_name):
    if args.save_check_dir is not None:
        file_path = os.path.join(args.save_check_dir, exp_name + '.pth')
        th.save(model.state_dict(), file_path)
        print('Model saved to disk')

def update_args_hawkes(args:Namespace):
    assert args.dataset == 'hawkes_exponential_mutual'
    args_file = os.path.join(args.load_from_dir, args.dataset)
    args_file = os.path.join(args_file, 'split_0/args.json')
    with open(args_file, 'r') as f:
        data_args = json.load(f)
    args_to_update = {
        'decoder_baselines': data_args['baselines'],
        'decoder_decays': data_args['decays'],
        'decoder_adjacency':data_args['adjacency'],   
    }
    args = vars(args)
    args.update(args_to_update)
    args = Namespace(**args)
    return args

def save_args(args:Namespace, exp_name:str):
    args_dic = vars(args).copy()
    args_dic.pop('device')
    for key, values in args_dic.items():
        if type(values) == np.ndarray:
            args_dic[key] = values.tolist()
    save_args_dir = os.path.join(args.save_check_dir, 'args')
    if not os.path.exists(save_args_dir):
        os.makedirs(save_args_dir)
    args_path = os.path.join(save_args_dir, exp_name + '.json')
    with open(args_path , 'w') as f:
        json.dump(args_dic, f)

def get_exp_name(args: Namespace) -> str:
    if args.encoder is not None:
        exp_name = args.encoder + '_' + args.decoder 
    elif args.encoder_histtime is not None:
        exp_name = args.encoder_histtime + '_' + args.encoder_histtime_encoding + '_' + args.encoder_histmark + '_' +  args.encoder_histmark_encoding + '_' +  args.decoder    
    if args.encoder_encoding is not None:
        exp_name += '_' + args.encoder_encoding
    if args.dataset is None:
        exp_name = 'Hawkes_' + exp_name
        if args.include_poisson:
            exp_name = 'poisson_' + exp_name
    else:
        if args.include_poisson:
            exp_name = args.dataset + '_poisson_' + exp_name
        else:
            exp_name = args.dataset + '_' +  exp_name
    if args.limit_train_size is not None:
        exp_name += '_trainsize' + str(args.limit_train_size)
    if args.exp_name is not None:
        exp_name += '_' + args.exp_name
    if args.split is not None:
        exp_name = exp_name + '_split' + str(args.split)
    if args.config is not None:
        exp_name += '_config' + str(args.config)
    print('EXP_NAME ', exp_name)
    return exp_name

def save_results(train_metrics: Dict[str, list], val_metrics: Dict[str, list], test_metrics: Dict[str, list], 
                save_path: str, exp_name: str, args: Namespace):
    results = {'train':train_metrics, 'val':val_metrics, 'test':test_metrics, 'args':args}
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    exp_name = exp_name + '.txt'
    save_path = os.path.join(save_path, exp_name)
    with open(save_path, "wb") as fp: 
        pkl.dump(results, fp)
    print('Results saved to {}'.format(save_path, flush=True))

def get_state_dic(path:str):
    state_dic = th.load(path, map_location=th.device('cpu'))
    print(state_dic)


if __name__ == "__main__":
    parsed_args = parse_args()
    json_dir = os.path.join(os.getcwd(), parsed_args.load_from_dir)
    json_dir = os.path.join(json_dir, parsed_args.dataset)
    if parsed_args.split is not None:
        split = 'split_{}'.format(parsed_args.split)
        json_dir = os.path.join(json_dir, split)
    json_path = os.path.join(json_dir, 'args.json')
    with open(json_path, 'r') as fp:
        args_dict_json = json.load(fp)
    args_dict = vars(parsed_args)
    shared_keys = set(args_dict_json).intersection(set(args_dict))
    for k in shared_keys:
        v1, v2 = args_dict[k], args_dict_json[k]
        is_equal = np.allclose(v1, v2) if isinstance(
            v1, np.ndarray) else v1 == v2
        if not is_equal:
            print(f"    {k}: {v1} -> {v2}", flush=True)
    args_dict.update(args_dict_json)
    parsed_args = Namespace(**args_dict)
    cuda = th.cuda.is_available() and not parsed_args.disable_cuda
    if cuda:
        parsed_args.device = th.device('cuda')
    else:
        parsed_args.device = th.device('cpu')        
    if not parsed_args.include_window:
        parsed_args.window = None  
    make_deterministic(seed=parsed_args.seed)
    main(args=parsed_args)
