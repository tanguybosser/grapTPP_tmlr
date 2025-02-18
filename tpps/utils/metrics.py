from re import S
import numpy as np

from sklearn.metrics import roc_auc_score

from collections import Counter

def eval_metrics(pred, gold, probas, ranks, mask, results, n_class, multi_labels=False, test=False):
    if multi_labels:
        pred_labels = np.concatenate([p.reshape(-1, n_class) for p in pred])
        gold_labels = np.concatenate([g.reshape(-1, n_class) for g in gold])
    else:
        pred_labels = np.concatenate([p.reshape(-1) for p in pred])
        gold_labels = np.concatenate([g.reshape(-1) for g in gold])
        probas = np.concatenate([p.reshape(-1) for p in probas])
        ranks = np.concatenate([r.reshape(-1) for r in ranks])
    mask_labels = np.concatenate([m.reshape(-1) for m in mask])
    pred_labels = pred_labels[mask_labels == 1]
    gold_labels = gold_labels[mask_labels == 1]
    probas = probas[mask_labels == 1]
    ranks = ranks[mask_labels == 1]
    if multi_labels:
        pred_labels = pred_labels[:, gold_labels.sum(0) > 0]
        gold_labels = gold_labels[:, gold_labels.sum(0) > 0]
        if n_class < 20:
            roc_auc = roc_auc_score(
                y_true=gold_labels, y_score=pred_labels,
                multi_class="ovo", average=None)
            results["roc_auc"] = roc_auc
        roc_auc_macro = roc_auc_score(
            y_true=gold_labels, y_score=pred_labels,
            multi_class="ovo", average="macro")
        roc_auc_micro = roc_auc_score(
            y_true=gold_labels, y_score=pred_labels,
            multi_class="ovo", average="micro")
        roc_auc_weighted = roc_auc_score(
            y_true=gold_labels, y_score=pred_labels,
            multi_class="ovo", average="weighted")
        results["roc_auc_macro"] = float(roc_auc_macro)
        results["roc_auc_micro"] = float(roc_auc_micro)
        results["roc_auc_weighted"] = float(roc_auc_weighted)
    else:
        precision, recall, f1, acc = metrics_per_class(
            pred=pred_labels, gold=gold_labels, n_class=n_class) #metrics for each class.
        pre_macro, rec_macro, f1_macro, acc_macro = macro_metrics(
            pred=pred_labels, gold=gold_labels, n_class=n_class) #average of metrics for all class.
        pre_micro, rec_micro, f1_micro, acc_micro = micro_metrics(
            pred=pred_labels, gold=gold_labels, n_class=n_class) #metrics computed using all TP, TN, etc amongst all classes.
        pre_weighted, rec_weighted, f1_weighted, acc_weighted = \
            weighted_metrics(
                pred=pred_labels, gold=gold_labels, n_class=n_class) #metrics per class, weighted by the occurence of that specific class.
        acc_1 = accuracy_at_n(ranks=ranks, n=1)
        acc_3 = accuracy_at_n(ranks=ranks, n=3)
        if n_class > 5:
            acc_5 = accuracy_at_n(ranks=ranks, n=5)
            results["acc at 5"] = acc_5
        if n_class > 10:
            acc_10 = accuracy_at_n(ranks=ranks, n=10)
            results["acc at 10"] = acc_10
        mrr = mean_reciprocal_rank(ranks=ranks)
        #mae = mae_time_prediction(times, )
        results["precision"] = precision
        results["pre_macro"] = float(pre_macro)
        results["pre_micro"] = float(pre_micro)
        results["pre_weighted"] = float(pre_weighted)
        results["recall"] = recall
        results["rec_macro"] = float(rec_macro)
        results["rec_micro"] = float(rec_micro)
        results["rec_weighted"] = float(rec_weighted)
        results["f1"] = f1
        results["f1_macro"] = float(f1_macro)
        results["f1_micro"] = float(f1_micro)
        results["f1_weighted"] = float(f1_weighted)
        results["acc at 1"] = float(acc_1)
        results["acc at 3"] = float(acc_3)
        results["mean reciprocal rank"] = float(mrr) 
        results["acc_macro"] = float(acc_macro)
        results["acc_micro"] = float(acc_micro)
        results["acc_weighted"] = float(acc_weighted)
        if test:
            calibration, samples_per_bin, conf_per_bin  = discrete_calibration(pred_labels, gold_labels, probas)
            results["calibration"] = calibration
            results["samples per bin"] = samples_per_bin
            results["confidence per bin"] = conf_per_bin
    return results


def accuracy_at_n(ranks, n=5):
    acc = np.mean(ranks < n)
    return acc

def mean_reciprocal_rank(ranks):
    mrr = np.mean(1/(ranks+1))
    return mrr

def discrete_calibration(pred, gold, proba, n_bins = 10):
    acc_per_bin, conf_per_bin, samples_per_bin = [], [], []
    for i in range(1,n_bins+1):
        idx = np.logical_and(proba <= i/n_bins, proba > (i-1)/n_bins)
        samples_per_bin.append(idx.sum())
        pred_bin = np.array(pred)[idx]
        gold_bin = np.array(gold)[idx]
        proba_bin = np.array(proba)[idx]
        if len(pred_bin) != 0:
            acc = np.mean(pred_bin == gold_bin)
            conf = np.mean(proba_bin)
        else:
            acc, conf = 0, 0
        acc_per_bin.append(acc)
        conf_per_bin.append(conf)
    return acc_per_bin, samples_per_bin, conf_per_bin

def rates_per_class(pred, gold, n_class):
    true_positive, true_negative, = np.zeros(n_class), np.zeros(n_class)
    false_positive, false_negative = np.zeros(n_class), np.zeros(n_class)
    for i in range(n_class):
        true_positive[i] = np.logical_and(gold == i, pred == i).sum()
        true_negative[i] = np.logical_and(gold != i, pred != i).sum()
        false_positive[i] = np.logical_and(gold != i, pred == i).sum()
        false_negative[i] = np.logical_and(gold == i, pred != i).sum()
    return true_positive, true_negative, false_positive, false_negative


def metrics_per_class(pred, gold, n_class):
    true_positive, true_negative, false_positive, false_negative = \
        rates_per_class(pred=pred, gold=gold, n_class=n_class)
    precision = np.divide(
        true_positive, true_positive + false_positive,
        out=np.zeros_like(true_positive),
        where=true_positive + false_positive != 0)
    recall = np.divide(
        true_positive, true_positive + false_negative,
        out=np.zeros_like(true_positive),
        where=true_positive + false_negative != 0)
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros_like(2 * precision * recall),
        where=precision + recall != 0)
    acc = np.divide(
        true_positive + true_negative,
        true_positive + true_negative + false_positive + false_negative,
        out=np.zeros_like(true_positive),
        where=true_positive + true_negative +
              false_positive + false_negative != 0)
    test_sum = true_positive + true_negative + false_positive + false_negative
    if 0 in test_sum:
        print("Warning: the sum of TP, TN, FP, FN should not be zero")
    return precision, recall, f1, acc


def macro_metrics(pred, gold, n_class):
    precision, recall, f1, acc = metrics_per_class(
        pred=pred, gold=gold, n_class=n_class)
    precision = precision.sum() / n_class
    recall = recall.sum() / n_class
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros_like(2 * precision * recall),
        where=precision + recall != 0)
    acc = acc.sum() / n_class
    return precision, recall, f1, acc


def micro_metrics(pred, gold, n_class):
    true_positive, true_negative, false_positive, false_negative = \
        rates_per_class(pred=pred, gold=gold, n_class=n_class)
    precision = np.divide(
        true_positive.sum(), true_positive.sum() + false_positive.sum(),
        out=np.zeros_like(true_positive.sum()),
        where=true_positive.sum() + false_positive.sum() != 0)
    recall = np.divide(
        true_positive.sum(), true_positive.sum() + false_negative.sum(),
        out=np.zeros_like(true_positive.sum()),
        where=true_positive.sum() + false_negative.sum() != 0)
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros_like(2 * precision * recall),
        where=precision + recall != 0)
    acc = np.divide(
        true_positive.sum() + true_negative.sum(),
        true_positive.sum() + true_negative.sum() +
        false_positive.sum() + false_negative.sum(),
        out=np.zeros_like(true_positive.sum()),
        where=true_positive.sum() + true_negative.sum() +
              false_positive.sum() + false_negative.sum() != 0)
    return precision, recall, f1, acc

def weighted_metrics(pred, gold, n_class):
    precision, recall, f1, acc = metrics_per_class(
        pred=pred, gold=gold, n_class=n_class)
    gold_counter = Counter(gold)
    gold_count = np.zeros(n_class)
    for i in range(n_class):
        gold_count[i] = gold_counter[i]
    precision = np.sum(precision * gold_count) / gold_count.sum()
    recall = np.sum(recall * gold_count) / gold_count.sum()
    f1 = np.sum(f1 * gold_count) / gold_count.sum()
    acc = np.sum(acc * gold_count) / gold_count.sum()
    return precision, recall, f1, acc

