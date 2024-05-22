import os 
import pickle 
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import re
import json
import seaborn as sns 
import pandas as pd
from plot.acronyms import get_acronym, map_dataset_name, map_model_name_cal
from matplotlib.ticker import PercentFormatter
from matplotlib.collections import PolyCollection
from data_processing.simu_hawkes import plot_process

#import matplotlib as mpl
#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import paired_distances

def plot_loss(models, dataset, split, results_dir):
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    results_dir = os.path.join(results_dir, dataset)
    files = os.listdir(results_dir)
    markers = ['o', 'v', '*']
    colors = ['green', 'red', 'orange']
    for i, model in enumerate(models):
        file_to_find = dataset + '_' + model + f'_split{split}'
        results_file = None
        for file in files:
            if file.startswith(file_to_find):
                results_file = file
                break
        if results_file is None:
            raise ValueError('File not found!')
        results_file = os.path.join(results_dir, results_file)
        with open(results_file, 'rb') as f:
            results = pickle.load(f)    
        train_pmf = [-results['train'][i]['log mark density'] for i in range(len(results['train']))]
        val_pmf = [-results['val'][i]['log mark density'] for i in range(len(results['val']))]
        train_pmf = train_pmf[:-1]
        val_pmf = np.array(val_pmf[:-1])[np.newaxis,:]
        
        train_pdf = [-results['train'][i]['log ground density'] for i in range(len(results['train']))]
        val_pdf = [-results['val'][i]['log ground density'] for i in range(len(results['val']))]
        train_pdf = train_pdf[:-1]
        val_pdf = np.array(val_pdf[:-1])[np.newaxis,:]
        
        train_loss = [results['train'][i]['loss'] for i in range(len(results['train']))]
        val_loss = [results['val'][i]['loss'] for i in range(len(results['val']))]
        train_loss = train_loss[:-1]
        val_loss = np.array(val_loss[:-1])[np.newaxis,:]

        all_val_losses = np.concatenate((val_pmf, val_pdf, val_loss), axis=0)
        cum_val_losses = np.cumsum(all_val_losses, axis=0)    
#        print(cum_val_losses[:,:5])
        epochs = np.arange(1, len(train_pmf)+1)
        model_short = get_acronym([model])[0]
        title = dataset + f'-S{split}'
        zeros = np.zeros_like(val_pdf.squeeze())
        #print(all_val_losses[:,:5])
        ax[i].plot(epochs, all_val_losses[0,:], label='Val NLL-M')
        ax[i].plot(epochs, all_val_losses[1,:], label='Val NLL-T')
        ax[i].plot(epochs, all_val_losses[2,:], label='Val NLL')
        
        #ax[i].fill_between(epochs, all_val_losses[0,:], zeros, label='Val NLL-M')
        #ax[i].fill_between(epochs, all_val_losses[1,:], all_val_losses[0,:], label='Val NLL-T')
        #ax[i].fill_between(epochs, all_val_losses[2,:], all_val_losses[1,:], label='Val NLL')
        #ax[i].set_title(model_short, fontsize=20)
        #title = model
        #ax[0].plot(epochs, train_pmf, label=f'Train {model_short}', color=colors[i])
        #ax[0].plot(epochs, val_pmf, label=f'Val {model_short}', color=colors[i])
        #ax[1].plot(epochs, train_pdf, label=f'Train {model_short}', color=colors[i])
        #ax[1].plot(epochs, val_pdf, label=f'Val {model_short}', color=colors[i])
        #ax[2].plot(epochs, train_loss, label=f'Train {model_short}', color=colors[i])
        #ax[2].plot(epochs, val_loss, label=f'Val {model_short}', color=colors[i])
    fig.suptitle(title)

    #ax[0].set_ylabel(r'$-\mathrm{log}~p^\ast(k|t)$')
    #ax[1].set_ylabel(r'$-\mathrm{log}~f^\ast(t)$')
    #ax[2].set_ylabel(r'$-\mathcal{L}$')

    ax[0].legend()
    save_dir = 'figures/training_curves'
    save_dir = os.path.join(save_dir, title + '.png')
    fig.tight_layout()

    plt.savefig(save_dir, bbox_inches='tight')
    #plt.show()

def load_file(file_to_find, results_dir):
    results_file = None
    files = os.listdir(results_dir)
    for file in files:
        if file.startswith(file_to_find):
            results_file = file
            break
    if results_file is None:
        print(file_to_find)
        raise ValueError('File not found!')
    results_file = os.path.join(results_dir, results_file)
    with open(results_file, 'rb') as f:
        results = pickle.load(f) 
    return results

def mark_calibration(models, dataset, results_dir, splits, save_dir=None, title=None, appendix=False):
    results_dir = os.path.join(results_dir, dataset)
    df_all = pd.DataFrame()
    for model in models:
        keys = ['accuracy', 'samples', 'bins']
        data = dict.fromkeys(keys)
        accuracy_all_splits, samples_all_split = [], []
        for split in range(splits):
            file_to_find = f'{dataset}_{model}_split{split}'
            results = load_file(file_to_find, results_dir)    
            accuracy = results['test']['calibration']
            samples = results['test']['samples per bin']
            tot_samples = np.sum(samples)
            prop_samples = [s/tot_samples for s in samples]
            accuracy_all_splits.append(accuracy)
            samples_all_split.append(prop_samples)
        accuracy = np.array(accuracy_all_splits)
        ste = np.std(accuracy, axis=0)/(np.sqrt(accuracy.shape[0]))
        accuracy = np.mean(accuracy, axis=0)
        samples = np.array(samples_all_split)
        samples = np.mean(samples, axis=0)
        bins = [round(1/(2*len(accuracy)) + i/len(accuracy),2) for i in range(len(accuracy))]
        data['error'] = ste
        data['accuracy'] = accuracy
        data['perfect calibration'] = bins
        data['samples'] = samples
        data['bins'] = bins
        model_name = get_acronym([model])[0]
        data['model'] = [model_name] * len(accuracy) 
        df = pd.DataFrame(data=data)
        df_all = pd.concat([df_all, df], axis=0)
    sns.set_theme(style="white", rc={"font.size":16, "axes.facecolor": (0, 0, 0, 0), 'axes.linewidth':2})
    sns.set_style("whitegrid")
    palette = sns.color_palette("Set2", 12)
    g2 = sns.FacetGrid(df_all, palette=palette, col="model", hue="model", aspect=0.7)
    g1 = sns.FacetGrid(df_all, palette=palette, col="model", hue="model", aspect=.7)

    g2.map_dataframe(sns.barplot, x='bins', y='accuracy', fill=True, alpha=0.7)
    g2.map_dataframe(sns.pointplot, x=bins, y=bins, color='black', markers='')

    g1.map_dataframe(sns.barplot, x='bins', y='samples', fill=True, alpha=0.7)
    g2.fig.subplots_adjust(hspace=-.4)
    g2.fig.subplots_adjust(wspace=.1)
    #g2.fig.legend.remove()
    #g2.fig.legend(handles=g2.fig.legend.legendHandles[0], loc=7)
    g1.fig.subplots_adjust(hspace=-.4)
    g1.fig.subplots_adjust(wspace=.1)
    #g2.set_titles(list(titles))
    dataset_name = map_dataset_name(dataset)
    g2.set_titles("")
    #g2
    ax = g2.fig.axes[0]
    new_ticks = np.append(ax.get_xticks(),10) - 0.5
    xxx = np.round(np.arange(0.1,1.1,.1),1)
    #g2.set(xticks=new_ticks)
    #g2.set(xticks=[])
    g2.set_xticklabels(xxx, fontsize=12)
    g2.set_yticklabels(np.round(ax.get_yticks(),1),fontsize=14)
    g2.set_ylabels('Accuracy', fontsize=20)
    g2.set(xlabel=" ")
    ax = g2.fig.axes[0]

    axes = g2.fig.axes
    for i, ax in enumerate(axes):
        ax.set_xlabel(" ")
    if not appendix:
        mid_ax = 1
    else:
        mid_ax = 2
    mid_ax = axes[mid_ax] 
    mid_ax.annotate('Confidence', xy=(0.55,-0.2), ha='left', va='top',
        xycoords='axes fraction', textcoords='offset points', fontsize=20)
    #mid_ax = axes[mid_ax]   
    #mid_ax.set_xlabel('Confidence', fontsize=20)
    new_ticks = np.append(ax.get_xticks(),10) - 0.5
    #g1.set(xticks=new_ticks)
    #g1.set(xticks=[])
    for ax in g2.axes.ravel():
        hand, labl = ax.get_legend_handles_labels()
        ax.legend(np.unique(labl))
        ax.set_ylim(0,1)
    for ax in axes:
        for label in ax.get_xticklabels()[::2]:
            label.set_visible(False)
    #g2.set_xticklabels(np.round(ax.get_xticks(),1),fontsize=14)
    #ax.set_ylim(0,0.6)
    g1.set_titles("")
    #g1.set_yticklabels(np.round(ax.get_yticks(),1),fontsize=14)
    g1.set_ylabels('p% Samples', fontsize=20)
    #g1.set_xlabels('')
    #g2.fig.suptitle(map_dataset_name(dataset), fontsize=16)
    #g1.savefig(f'figures/calibration_neurips/{map_dataset_name(dataset)}_samples.pdf' , bbox_inches='tight')
    for i, ax in enumerate(axes):
        x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
        y_coords = [p.get_height() for p in ax.patches]
        ax.errorbar(x=x_coords, y=y_coords, yerr=df_all["error"][i*10:i*10+10], fmt="none", c="k")
    if title == '1' and appendix:
        g2.fig.suptitle(dataset_name, y=0.95)
    g2.savefig(f'figures/calibration_neurips/mark/{map_dataset_name(dataset)}_cal{title}.pdf' , bbox_inches='tight')  
    
    
    
    plt.show()


def time_calibration(models, dataset, results_dir, splits, save_dir=None, title=None, appendix=False):
    results_dir = os.path.join(results_dir, dataset)
    df_all = pd.DataFrame()
    #plt.rcParams["figure.figsize"] = (12,12)    
    for model in models:
        keys = ['True quantiles', 'Predicted quantiles', 'model']
        data = dict.fromkeys(keys)
        all_calibration = []
        for split in range(splits):
            file_to_find = f'{dataset}_{model}_split{split}'
            results = load_file(file_to_find, results_dir)
            cdf = results['test']['cdf']            
            bins, x = calibration(cdf, num_bins=50)
            all_calibration.append(x)
        obs_freq = np.mean(np.array(all_calibration), axis=0)
        data['Observed frequency'] = obs_freq
        data['Predicted probability'] = bins
        model = get_acronym([model])[0]
        data['model'] = [model]*len(obs_freq)
        df = pd.DataFrame(data=data)
        df_all = pd.concat([df_all, df], axis=0, ignore_index=True)
    sns.set_theme(style="white", rc={"font.size":16, "axes.facecolor": (0, 0, 0, 0), 'axes.linewidth':2})
    sns.set_style("whitegrid")
    palette = sns.color_palette("Set2", 12)
    
    g1 = sns.FacetGrid(df_all, palette=palette, col="model", hue="model", aspect=0.7)
    g1.map_dataframe(sns.lineplot, x='Predicted probability', y='Observed frequency', linewidth=3)
    g1.map_dataframe(sns.lineplot, x='Predicted probability', y='Predicted probability', color='black')
    g1.fig.subplots_adjust(wspace=.2)
    g1.set(xlim=[0,1], ylim=[0,1])
    axes = g1.fig.axes
    for i, ax in enumerate(axes):
        ax.set_xlabel(" ")
    if appendix: 
        mid_ax = 2
    else:
        mid_ax = 1
    mid_ax = axes[mid_ax]
    #mid_ax.set_xlabel('Predicted probability', fontsize=20)
    mid_ax.annotate('Predicted Probability', xy=(0.35,-0.2), ha='left', va='top',
        xycoords='axes fraction', textcoords='offset points', fontsize=20)
    #g1.set_xlabels('Predicted probability', fontsize=16)
    g1.set_ylabels('Frequency', fontsize=20)
    g1.set_titles("")
    #g1.set(xticks=[])
    #g1.set_xlabels('')
    g1.set_yticklabels(np.round(ax.get_yticks(),1),fontsize=14)
    #xxx = np.round(np.arange(0.1,1.1,.1),1)
    #g2.set(xticks=new_ticks)
    #g2.set(xticks=[])
    #g1.set_xticklabels(xxx, fontsize=12)
    #for ax in axes:
    #    for label in ax.get_xticklabels()[::2]:
    #        label.set_visible(False)
    #g1.fig.suptitle(map_dataset_name(dataset), fontsize=16)
    for ax in g1.axes.ravel():
        hand, labl = ax.get_legend_handles_labels()
        ax.legend(np.unique(labl))
    dataset_name = map_dataset_name(dataset)
    #if title == '1':
    #    g1.fig.suptitle(dataset_name, y=0.95)
    g1.savefig(f'figures/calibration_neurips/time/{map_dataset_name(dataset)}_{title}.pdf' , bbox_inches='tight')

def calibration(cdf, num_bins):
    cdf = np.array([item for seq in cdf for item in seq])  
    bins = [i / num_bins for i in range(1, num_bins + 1)]
    counts_cdf = []
    for i, bin in enumerate(bins):
        cond = cdf <= bin
        counts_cdf.append(cond.sum())
    x = np.array([count / len(cdf) for count in counts_cdf])
    return bins, x

    
def show_weighted_loss(result_dir, model, dataset, weights, n_split=5, save=False):
    results_dir = os.path.join(result_dir, dataset)
    files = os.listdir(results_dir)
    all_weighted_loss, all_weighted_density, all_weighted_pmf = [], [], []
    for weight in weights:  
        weight_loss, weight_density, weight_pmf = [], [], []
        for split in range(n_split):
            result_file = None
            file_to_find = dataset + '_' + model + '_weight' + str(weight) + '_split' + str(split) 
            if 'base' in model:
                file_to_find = 'poisson_' +  dataset + '_' + model.split('_base')[0] + '_weight' + str(weight) + '_split' + str(split)
            for file in files:
                if file.startswith(file_to_find):
                    result_file = file 
                    break
            if result_file is None:
                print(file_to_find)
                raise ValueError('File not found!') 
            result_file = os.path.join(results_dir, result_file)
            with open(result_file, 'rb') as f:
                results = pickle.load(f)
            density = -results['test']['log ground density']
            pmf =  -results['test']['log mark density']
            loss = pmf + density
            if weight == 0.7:
                print(pmf)
            weight_loss.append(loss)
            weight_density.append(density)
            weight_pmf.append(pmf)
        if weight == 0.9:
            print(np.mean(weight_loss), 'loss')
            print(np.mean(weight_density), 'density')
            print(np.mean(weight_pmf), 'pmf')
        all_weighted_loss.append(np.mean(weight_loss))
        all_weighted_density.append(np.mean(weight_density))
        all_weighted_pmf.append(np.mean(weight_pmf))
    fig, axes = plt.subplots(1,3, figsize=(20,5))
    axes[0].plot(weights, all_weighted_loss, label='Loss')
    axes[0].set_ylabel('Loss', fontsize=24)
    axes[1].plot(weights, all_weighted_density, label='Density')
    axes[1].set_ylabel('Density', fontsize=24)
    axes[1].set_xlabel('Weight', fontsize=24)
    axes[2].plot(weights, all_weighted_pmf, label='Pmf')
    axes[2].set_ylabel('Pmf', fontsize=24)
    model_name = get_acronym([model])[0]
    dataset_name = map_dataset_name(dataset)
    fig.suptitle('{}_{}'.format(dataset_name, model_name), fontsize=24)
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=16)
    fig.tight_layout()
    if save:
        fig.savefig('figures/weights/weights_{}_{}.png'.format(dataset, model) , bbox_inches='tight')
    plt.show()

def joint_distribution(dataset_dir, dataset, max_marks=None):
    dataset_dir = os.path.join(dataset_dir, dataset) 
    dataset_file = os.path.join(dataset_dir, dataset + '.json')
    with open(dataset_file, 'r') as f:
        data = json.load(f)
    num_marks = len(np.unique([event['labels'][0] for seq in data for event in seq]))   
    arrrival_time_per_mark = [[] for i in range(num_marks)] 
    for j, seq in enumerate(data):
        for i, event in enumerate(seq):
            inter_time = event['time'] if i == 0 else event['time'] - seq[i-1]['time']
            if inter_time == 0:
                continue
            arrrival_time_per_mark[event['labels'][0]].append(np.log(inter_time))
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(projection='3d')
    yticks = np.arange(num_marks)
    for k in yticks:
        kde = sns.kdeplot(data=arrrival_time_per_mark[k], zs=k, zdir='y', alpha=0.7, fill='True')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_yticks(yticks)

    plt.show()


def joint_distribution_bis(dataset_dir, dataset, marks_to_plot=None, title=True):
    dataset_dir = os.path.join(dataset_dir, dataset) 
    dataset_file = os.path.join(dataset_dir, dataset + '.json')
    df_dic = {}
    with open(dataset_file, 'r') as f:
        data = json.load(f)
    num_marks = len(np.unique([event['labels'][0] for seq in data for event in seq]))   
    arrrival_time_per_mark = [[] for i in range(num_marks)] 
    for j, seq in enumerate(data):
        for i, event in enumerate(seq):
            inter_time = event['time'] if i == 0 else event['time'] - seq[i-1]['time']
            if inter_time == 0:
                continue
            arrrival_time_per_mark[event['labels'][0]].append(np.log(inter_time))
    mark_list = []
    if marks_to_plot is None:
        marks_to_plot = np.arange(0,len(arrrival_time_per_mark))
        time_list = [time for mark_seq in arrrival_time_per_mark for time in mark_seq]
    else:
        marks_to_plot = np.arange(marks_to_plot[0],marks_to_plot[1])
        time_list = [time for mark_seq in arrrival_time_per_mark[marks_to_plot[0]:marks_to_plot[-1]+1] for time in mark_seq]
    for i in marks_to_plot:
        mark_list += [i]*len(arrrival_time_per_mark[i])
    df_dic['times'] = time_list
    df_dic['mark'] = mark_list
    df = pd.DataFrame(data=df_dic)
    plt.figure(figsize=(10,10))
    def label(x, color, label):
        ax = plt.gca()
        ax.text(.1, .07, label, color='black', fontsize=40,
            ha="left", va="center", transform=ax.transAxes)
    sns.set_theme(style="white", rc={"font.size":20, "axes.facecolor": (0, 0, 0, 0), 'axes.linewidth':2})
    palette = sns.color_palette("Set2", 12)
    g1 = sns.FacetGrid(df, palette=palette, row='mark', hue='mark', aspect=8)
    g1.map_dataframe(sns.kdeplot, x='times', fill=True, alpha=0.5)
    g1.map(label, "mark")
    g1.fig.subplots_adjust(hspace=-0.85)
    g1.set_titles("")
    g1.set_xticklabels(fontsize = 40)
    g1.set(yticks=[], xlabel=" " , ylabel=" ")
    g1.despine( left=True)
    dataset = map_dataset_name(dataset)
    if title:
        g1.fig.suptitle(dataset, fontsize=50)
    if marks_to_plot is None:
        save_title = 'figures/dataset/jointdis_' + dataset + '.png'
    else:
        save_title = 'figures/dataset/jointdis_' + dataset + f'{str(marks_to_plot[0])}_{str(marks_to_plot[-1])}' + '.png'
    g1.savefig(save_title , bbox_inches='tight') 

def error_heterogeneous(result_dir, models):
    datasets = [f'hawkes_commontest_{i}' for i in [1, 5, 10, 50, 100, 200, 500, 1000]]
    brier_scores = {model:[] for model in models}
    nll_model_scores = {model:[] for model in models}
    nll_true_scores = {model:[] for model in models}
    acc_model_scores = {model:[] for model in models}
    acc_true_scores = {model:[] for model in models}
    fig, ax = plt.subplots(1,1, figsize=(10, 10))
    #x_axis = np.arange(0, len(datasets))
    for model in models:
        for dataset in datasets:
            result_dir = os.path.join(result_dir, dataset)
            files = os.listdir(result_dir)
            model_file = None
            if 'base' in model:
                file_to_find = 'poisson_' + dataset + '_' + model.replace('_base', '')
            else:
                file_to_find =  dataset + '_' + model
            #file_to_find = dataset + '_' + file_to_find
            for file in files:
                if file.startswith(file_to_find):
                    model_file = file
                    break
            if model_file is None:
                print(file_to_find)
                raise ValueError('File not found')
            model_file = os.path.join(result_dir, model_file)
            with open(model_file, 'rb') as f:
                results = pickle.load(f)
            brier_scores[model].append(np.mean(results['Brier']))
            nll_model_scores[model].append(np.mean(results['Model NLL']))
            nll_true_scores[model].append(np.mean(results['True NLL']))
            acc_model_scores[model].append(np.mean(results['Model Accuracy']))
            acc_true_scores[model].append(np.mean(results['True Accuracy']))
        ax.plot(datasets, brier_scores[model], label=model)
    ax.legend()    
    plt.show()

def plot_mixture_errors(result_dir, n_mixs, dataset):
    result_dir = os.path.join(result_dir, dataset)
    files = os.listdir(result_dir)
    fig, ax = plt.subplots(1, 2)
    all_nll, all_ece, all_f1 = [], [], []
    for n_mix in n_mixs:
        file_to_find = dataset + '_gru_cond-mm-log-normal-mixture_temporal_with_labels_times_only_relu_summation_mm' + str(n_mix) + '_split0'
        model_file = None 
        for file in files:
            if file.startswith(file_to_find):
                model_file = file 
                break
        if model_file is None:
            print(file_to_find)
            raise ValueError('File not found')
        model_file = os.path.join(result_dir, model_file)
        with open(model_file, 'rb') as f:
            results = pickle.load(f)
        all_nll.append(-results['test']['log mark density'])
        #all_ece.append(results['test']['mark calibration'])
        all_f1.append(results['test']['f1_weighted'])
    n_mixs = [str(mix) for mix in n_mixs]
    print(all_nll)
    ax[0].plot(n_mixs, all_nll)
    ax[1].plot(n_mixs, all_f1)
    #ax[2].plot(n_mixs, all_ece)
    plt.show()

def plot_pc(result_dir, n_mix, dataset):
    result_dir = os.path.join(result_dir, dataset)
    files = os.listdir(result_dir)
    fig, ax = plt.subplots(1,3, figsize=(20,5))
    fig2, ax2 = plt.subplots(figsize=(5,5))
    fig3, ax3 = plt.subplots(figsize=(5,5))
    file_to_find = dataset + '_gru_cond-mm-log-normal-mixture_temporal_with_labels_times_only_relu_summation_mm' + str(n_mix) + '_split0'
    file_to_find2 = dataset + '_gru_cond-log-normal-mixture_temporal_with_labels_times_only_relu_summation_test_split0.txt'
    model_file = None 
    for file in files:
        if file.startswith(file_to_find):
            model_file = file 
            break
    if model_file is None:
        print(file_to_find)
        raise ValueError('File not found')
    model_file = os.path.join(result_dir, model_file)
    model_file2 = os.path.join(result_dir, file_to_find2)
    with open(model_file, 'rb') as f:
        results = pickle.load(f)
    with open(model_file2, 'rb') as f:
        results2 = pickle.load(f)
    for j in range(5):
        for i in range(3):
            x = [str(j) for j in range(8)]
            print(results['test']['pc'][0][0,j,i])
        #ax[i].bar(x=x,height=results['test']['pmc'][0][0,i,:])
    #ax2.bar(x=x, height=results['test']['pm'][0][0,:])
    #ax3.bar(x=x, height=results2['test']['pm'][0][0,:])


def plot_error_distribution(result_dir, dataset, model, split=0):
    result_dir = os.path.join(result_dir, dataset)
    files = os.listdir(result_dir)
    fig, ax = plt.subplots(1, 2)
    file_to_find = dataset + '_' + model + f'_split{str(split)}'
    model_file = None 
    for file in files:
        if file.startswith(file_to_find):
            model_file = file 
            break
    if model_file is None:
        print(file_to_find)
        raise ValueError('File not found')
    model_file = os.path.join(result_dir, model_file)
    with open(model_file, 'rb') as f:
        results = pickle.load(f)
    time_nll = results['test']['log density per seq']
    #med_time = results['test']['median log density']
    #med_mark = results['test']['median log mark density']
    #print(np.mean(time_nll), 'Mean time')
    #print(np.median(time_nll), 'Median time')
    mark_nll = results['test']['log mark density per seq']
    print(mark_nll[0:10])
    print(np.mean(mark_nll), 'Mean mark')
    #print(np.median(mark_nll), 'Median mark')
    ax[0].hist(time_nll)
    ax[0].set_title('Time NLL')
    bins = ax[1].hist(mark_nll)
    #print(bins)
    ax[1].set_title('Mark NLL')


def plot_error_per_seq(result_dir, files, dataset, seq_num, split):
    dataset_file = os.path.join('../neuralTPPs/data/baseline3', dataset)
    dataset_file = os.path.join(dataset_file, f'split_{split}')
    dataset_file = os.path.join(dataset_file, 'test.json')
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
    sequence = dataset[seq_num]
    times = np.array([event['time'] for event in sequence])
    marks = np.array([event['labels'][0] for event in sequence])
    print(marks)
    mark_mask = marks == 6
    fig, ax = plt.subplots(figsize=(20,10))
    ax = plot_process(times, ax, labels_idx=marks)
    ax.set_prop_cycle('color', sns.color_palette("Set1"))
    for file in files:
        model_file = os.path.join(result_dir, file)
        with open(model_file, 'rb') as f:
            results = pickle.load(f)
        pmf = np.array(results['events_pmf'])
        
        
        print(pmf[mark_mask[:-1]])
        ax.scatter(times[1:], pmf, label=file)
    ax.legend()
    fig.savefig('figures/intensities/test.png', bbox_inches='tight')

def tsne_history(result_dir, dataset, models_group ,split=0, seed=0, save_dir=None):
    result_dir = os.path.join(result_dir, dataset)
    files = os.listdir(result_dir)
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    for i, models in enumerate(models_group):
        for model in models:
            if 'evaluation' in model:
                file_to_find = dataset + '_' + model.replace('_evaluation', '') + f'_split{split}_evaluation'
            else:
                file_to_find = dataset + '_' + model + f'_split{split}'
            model_file = None
            for file in files:
                if file.startswith(file_to_find):
                    model_file = file
                    break
            if model_file is None:
                print(file_to_find)
                raise ValueError('Model not found')
            model_file = os.path.join(result_dir, model_file)
            with open(model_file, 'rb') as f:
                results = pickle.load(f)
            #print(results['test'].keys())
            if '-dd' in model:
                h_t = results['test']['last_h_t'].astype(np.float32)
                h_m = results['test']['last_h_m'].astype(np.float32)
            else:
                h_c = results['test']['last_h'].astype(np.float32)
        h_dd = np.concatenate([h_t, h_m], axis=0)    
        h_dd_transformed = TSNE(n_components=2,
                        init='pca', perplexity=5, random_state=seed).fit_transform(h_dd)
        h_c_transformed = TSNE(n_components=2,
                        init='pca', perplexity=5, random_state=seed).fit_transform(h_c)
        
        #shape = h_t.shape[0]
        h_t_transformed = h_dd_transformed[:h_t.shape[0], :]
        h_m_transformed = h_dd_transformed[h_t.shape[0]:, :]
        #h_c_transformed = h_transformed[2*h_t.shape[0]:, :]
        if i == 0:
            ax[i].scatter(h_t_transformed[:,0],h_t_transformed[:,1], color='blue', label=r'$h^t$', alpha=0.2)
            ax[i].scatter(h_m_transformed[:,0],h_m_transformed[:,1], color='red', label=r'$h^m$', alpha=0.2)
            ax[i].scatter(h_c_transformed[:,0],h_c_transformed[:,1], color='green', label=r'$h$', alpha=0.2)
        else:
            ax[i].scatter(h_t_transformed[:,0],h_t_transformed[:,1], color='blue', alpha=0.2)
            ax[i].scatter(h_m_transformed[:,0],h_m_transformed[:,1], color='red', alpha=0.2)
            ax[i].scatter(h_c_transformed[:,0],h_c_transformed[:,1], color='green', alpha=0.2)
        #ax[i].legend(loc='upper right')
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        #ax[i].set_xlabel('t-SNE component 1', fontsize=20)
        #ax[i].set_ylabel('t-SNE component 2', fontsize=20)
    fig.text(0.5, 0.04, 't-SNE component 1', va='center', ha='center', fontsize=20)
    fig.text(0.1, 0.5, 't-SNE component 2', va='center', ha='center', rotation='vertical', fontsize=20)
    dataset_name = map_dataset_name(dataset)
    #fig.suptitle(dataset_name, y=1.07, fontsize=20)
    bb = (fig.subplotpars.left, fig.subplotpars.top+0.02, 
      fig.subplotpars.right-fig.subplotpars.left,.1)

    fig.legend(bbox_to_anchor=bb, mode="expand", loc="lower left",
               ncol=3, borderaxespad=0., bbox_transform=fig.transFigure, fontsize=20)
    if save_dir is not None:
        save_path = f'{save_dir}/{dataset}.pdf'
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def euclidean_history(result_dir, dataset, models_group ,split=0):
    result_dir = os.path.join(result_dir, dataset)
    files = os.listdir(result_dir)
    fig, ax = plt.subplots(1,3, figsize=(10,5))
    for i, models in enumerate(models_group):
        for model in models:
            if 'evaluation' in model:
                file_to_find = dataset + '_' + model.replace('_evaluation', '') + f'_split{split}_evaluation'
            else:
                file_to_find = dataset + '_' + model + f'_split{split}'
            model_file = None
            for file in files:
                if file.startswith(file_to_find):
                    model_file = file
                    break
            if model_file is None:
                print(file_to_find)
                raise ValueError('Model not found')
            model_file = os.path.join(result_dir, model_file)
            with open(model_file, 'rb') as f:
                results = pickle.load(f)
            if 'sep' in model:
                h_t = results['test']['last_h_t'].astype(np.float32)
                h_m = results['test']['last_h_m'].astype(np.float32)
            else:
                h_c = results['test']['last_h'].astype(np.float32)
        dist_h_h_t = paired_distances(h_t, h_c)
        dist_h_h_m = paired_distances(h_m, h_c)
        dist_h_t_h_m = paired_distances(h_t, h_m)
        ax[0].hist(dist_h_h_t, label='h vs. h_t')
        ax[1].hist(dist_h_h_m, label='h vs. h_m')
        ax[2].hist(dist_h_t_h_m, label= 'h_t vs. h_m')
        for axes in ax:
            axes.legend()
    fig.savefig(f'figures/tsne/distances_{dataset}.png', bbox_inches='tight')
    plt.show()


def find_model_file(result_dir, model, dataset ,split):
    result_dir = os.path.join(result_dir, dataset)
    files = os.listdir(result_dir)
    model_file = None
    if 'base' in model:
        file_to_find = 'poisson_' + dataset + '_' + model.replace('_base', '') + '_split' + str(split)
    else:
        file_to_find = dataset + '_' + model + '_split' + str(split)
    for file in files:
        if file.startswith(file_to_find):
            model_file = file
            break
    if model_file is None:
        print(file_to_find)
        raise ValueError('model not found')    
    return os.path.join(result_dir, model_file)

def plot_training_losses(result_dir, models, dataset ,split, save_dir=None):
    fig, axes = plt.subplots(1,2,figsize=(10,5), constrained_layout=True)
    for model in models:
        model_file = find_model_file(result_dir, model, dataset, split)
        with open(model_file, 'rb') as f:
            results = pickle.load(f)
        time_loss_train = [-results['train'][i]['log ground density'] for i in range(len(results['train'])-1)]
        mark_loss_train = [-results['train'][i]['log mark density'] for i in range(len(results['train'])-1)]
        time_loss_val = [-results['val'][i]['log ground density'] for i in range(len(results['val'])-1)]
        mark_loss_val = [-results['val'][i]['log mark density'] for i in range(len(results['val'])-1)]
        model_name = map_model_name_cal(model)
        axes[0].plot(time_loss_train, label=model_name)
        axes[1].plot(mark_loss_train, label=model_name)
        #axes[0].plot(time_loss_val, label='Time val')
        #axes[1].plot(mark_loss_val, label='Val')
        axes[0].legend(fontsize=20) 
        for ax in axes:
            ax.set_xlabel('Epoch', fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.tick_params(axis='both', which='minor', labelsize=20)
            ax.set_yticklabels([])
        axes[0].set_ylabel('NLL-T', fontsize=20)
        axes[1].set_ylabel('NLL-M', fontsize=20)
    if save_dir is not None:
        save_file = f'{model}_{dataset}'
        save_file = os.path.join(save_dir, save_file)
        fig.savefig(save_file, bbox_inches='tight')


def computational_time(result_dir, model, dataset):
    #model_file = find_model_file(result_dir, model, dataset, split)
    if model == 'smurf-thp':
        model_types = ['-jd','-dd']
    else:
        model_types = ['','-jd','-dd']
    for types in model_types:
        if types == '-dd':
            file = f'{result_dir}/{dataset}/{dataset}_gru_temporal_with_labels_gru_temporal_with_labels_{model}{types}_split0.txt'
        else:
            file = f'{result_dir}/{dataset}/{dataset}_gru_{model}{types}_temporal_with_labels_split0.txt'
        with open(file, 'rb') as f:
                results = pickle.load(f)
        dur = [results['train'][i]['dur'] for i in range(len(results['train'])-2)]
        mean_dur = np.mean(dur)
        ste_dur = np.std(dur)/(np.sqrt(len(dur)))
        print(f'{model}{types}')
        print(f'Mean:{np.round(mean_dur,2)}, Std.:{np.round(ste_dur,2)}')
    
def plot_marked_sequences(data_dir, dataset, num_seq, min_events=5, max_events=100):
    file = os.path.join(data_dir, dataset)
    file = os.path.join(file, dataset + '.json')
    with open(file, 'r') as f:
        data = json.load(f)
    all_times = [[event['time'] for event in seq] for seq in data]
    all_marks = [[event['labels'][0] for event in seq] for seq in data]
    times_to_show, marks_to_show = [], []
    for i, times in enumerate(all_times):
        if len(times_to_show) == num_seq:
            break
        if len(times) > min_events and len(times) < max_events:
            times_to_show.append(times)  
            marks_to_show.append(all_marks[i])        
    fig, ax = plt.subplots(num_seq, 1)
    cm = plt.get_cmap('gist_rainbow')
    unique_marks = np.unique([mark for seq in marks_to_show for mark in seq])
    max_time = np.max(np.max(times_to_show))
    end_time = max_time + 0.01
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors=[cm(1.*i/len(unique_marks)) for i in range(len(unique_marks))]
    colors_mapping = {mark:colors[i] for i, mark in enumerate(unique_marks)}
    for i, times in enumerate(times_to_show):
        #ax[i].set_prop_cycle('color', sns.color_palette("Set2", len(unique_marks)))
        marks = marks_to_show[i]
        colors_to_show = [colors_mapping[mark] for mark in marks]
        ax[i].vlines(x=times, ymin=0, ymax=1, colors=colors_to_show)
        ax[i].set_xlim(left=0, right=end_time+0.1)
        ax[i].set_yticks([])
    for i in range(len(ax)-1):
        ax[i].set_xticks([])
    fig.suptitle(map_dataset_name(dataset), y=0.95)
    ax[-1].set_xlabel('Time', fontsize=13)
    fig.savefig(f'figures/sequences/{dataset}', bbox_inches='tight')
    plt.show()