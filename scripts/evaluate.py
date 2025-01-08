import sys, os

import json
import numpy as np
import time
import pickle as pkl

import torch as th
from torch.utils.data import DataLoader

from argparse import Namespace
from typing import Dict, Optional
from pathlib import Path
from tqdm import tqdm

from tpps.models import get_model
from tpps.models.base.process import Process
from tpps.utils.cli import parse_args
from tpps.utils.metrics import eval_metrics
from tpps.utils.data import get_loader, load_data
from tpps.utils.run import make_deterministic
from tpps.utils.history_bst import get_repeated_prev_times, get_history_and_target_all

from tpps.utils.utils import detach, count_parameters, count_parameters_enc

from scripts.train import get_loss

print('cuda', th.cuda.is_available(), flush=True)


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
    pred_labels, gold_labels, mask_labels, probas, ranks, target_times, predicted_times = [], [], [], [], [], [], []
    results = {}
    epoch_h_t, epoch_h_m, epoch_h = [], [], []
    cumulative_density = []
    num_batch = 0
    for batch in tqdm(loader) if args.verbose else loader:
        batch['times'], batch['labels'], batch['seq_lens'] = batch['times'].to(args.device), batch['labels'].to(args.device), batch['seq_lens'].to(args.device)
        predictions, target = get_time_prediction(model, batch, args)
        predicted_times.append(detach(predictions))
        target_times.append(detach(target))
        loss_t, loss_m, loss_w, loss_mask, artifacts = get_loss(  # [B]
            model, batch=batch, eval_metrics=eval_met, args=args,
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
        mae = mae_time_prediction(predicted_times, target_times)
        results["mae"] = mae

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

def get_time_prediction(model, batch, args, alpha_val=0.5):
    past_events, target_time, target_label, target_mask = get_history_and_target_all(batch, args)
    target_mask = target_mask.bool()
    prev_times = get_repeated_prev_times(past_events, n_repeats=1)
    b, l = prev_times[0].shape
    alpha = th.ones(b, l) * alpha_val
    alpha = alpha.to(args.device)
    time_prediction = model.icdf(alpha, past_events=past_events, prev_times=prev_times)
    cdf, cdf_mask = model.one_minus_cdf(query=time_prediction, events=past_events, prev_times=prev_times)
    cdf_mask = cdf_mask.bool()
    valid_cdf = cdf[target_mask.bool()]
    low = valid_cdf > 0.45
    high = valid_cdf < 0.55
    valid = low * high
    if valid_cdf[~valid].shape[0] != 0:
        print('Invalid CDF value encountered')
        print('non valid cdf shape', valid_cdf[~valid].shape)
        print('non valid times', time_prediction[target_mask][~valid])
    return time_prediction[target_mask.bool()][valid], target_time[target_mask.bool()][valid]

def mae_time_prediction(predictions, target):
    predictions = np.concatenate([p.reshape(-1) for p in predictions])
    target = np.concatenate([t.reshape(-1) for t in target])
    mae = np.mean(np.abs(predictions-target))
    return mae 

def main(args: Namespace):
    model_path = f'{args.save_check_dir}/{args.dataset}_{args.model_name}_split{args.split}.pth'
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
    count_parameters_enc(model)
    print('NUM OF TRAINABLE PARAMETERS : {}'.format(num_params))
    print("INSTATIATED MODEL : {}/{}/{} on dataset {}".format(args.encoder_encoding, args.encoder, args.decoder, args.dataset))
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
    train_metrics = metrics['train']
    val_metrics = metrics['val']
    test_metrics = metrics['test']
    if args.save_results_dir is not None:
        save_evaluation_results(train_metrics, val_metrics, test_metrics,save_path=args.save_results_dir, exp_name=model_name ,args=args)

def load_args(args:Namespace):
    batch_size = args.batch_size
    load_dir = args.load_from_dir
    save_dir = args.save_results_dir
    mc_prop = args.decoder_mc_prop_est
    args_path = f'{args.save_check_dir}/args/{args.dataset}_{args.model_name}_split{args.split}.json'
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
    args.decoder_mc_prop_est = mc_prop
    return args 


def save_evaluation_results(train_metrics: Dict[str, list], val_metrics: Dict[str, list], test_metrics: Dict[str, list], 
                save_path: str, exp_name: str, args: Namespace):
    results = {'train':train_metrics, 'val':val_metrics, 'test':test_metrics, 'args':args}
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    exp_name = f'{exp_name}_evaluation2_split{args.split}.txt'
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
    make_deterministic(seed=parsed_args.seed)
    main(args=parsed_args)
