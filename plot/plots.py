
import os 
import pickle 
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import re
import json
import seaborn as sns 
import pandas as pd
from plot.acronyms import get_acronym
from matplotlib.ticker import PercentFormatter

def barplot_datasets(models, datasets, result_dir, num_splits=5, mark=True):
    models_time, models_mark = models
    all_datasets_time = initialize_results_dic(datasets, models_time, marks=False, barplot=True)
    if mark:
        all_datasets_marks = initialize_results_dic(datasets, models_mark, marks=True, barplot=True)    
    for dataset in datasets:
        model_result_dir = os.path.join(result_dir, dataset)
        model_result_dir = os.path.join(model_result_dir, 'best')
        file_names = os.listdir(model_result_dir)
        if mark:
            for model in models_mark:
                for split in range(num_splits):
                    file_to_find = dataset + '_' + model + '_split' + str(split) + '_config'
                    if 'base' in model:
                        if 'base_fc' in model:
                            file_to_find = 'poisson_coefficients_' +  dataset + '_' + model.replace('_base_fc', '') + '_split' + str(split)
                        elif 'lnmk1' in model:
                            model_to_find = model.split('_lnmk1')[0]
                            file_to_find = 'poisson_' + dataset + '_' + model_to_find + '_split' + str(split) + '_lnmk1'
                        else:
                            file_to_find = 'poisson_' +  dataset + '_' + model.replace('_base', '') + '_split' + str(split)
                    elif 'lnmk1' in model:
                        model_to_find = model.split('_lnmk1')[0]
                        file_to_find = dataset + '_' + model_to_find + '_split' + str(split) + '_lnmk1'
                    for file_name in file_names:
                        if file_name.startswith(file_to_find):
                            path = os.path.join(model_result_dir, file_name)
                            all_datasets_marks = fill_result_dict(path, all_datasets_marks, dataset, model, barplot=True)
        for model in models_time:
            for split in range(num_splits):
                file_to_find = dataset + '_' + model + '_split' + str(split) + '_config'
                if 'base' in model:
                    if 'base_fc' in model:
                        file_to_find = 'poisson_coefficients_' +  dataset + '_' + model.replace('_base_fc', '') + '_split' + str(split)
                    elif 'lnmk1' in model:
                        model_to_find = model.split('_lnmk1')[0]
                        file_to_find = 'poisson_' + dataset + '_' + model_to_find + '_split' + str(split) + '_lnmk1'
                    else:
                        file_to_find = 'poisson_' +  dataset + '_' + model.replace('_base', '') + '_split' + str(split)
                elif 'lnmk1' in model:
                    model_to_find = model.split('_lnmk1')[0]
                    file_to_find = dataset + '_' + model_to_find + '_split' + str(split) + '_lnmk1'
                for file_name in file_names:
                    if file_name.startswith(file_to_find):
                        path = os.path.join(model_result_dir, file_name)
                        all_datasets_time = fill_result_dict(path, all_datasets_time, dataset, model, barplot=True)
    df_all = pd.DataFrame()
    for dataset in datasets:
        data = {}
        #data = dict.fromkeys(['Time NLL', 'Mark NLL', 'TCE', 'F1-score', 'ECE', 'model'])
        data = {key:None for key in ['Time NLL', 'Mark NLL', 'model', 'dataset']}
        models = [map_model(model, barplot=True) for model in models_time]
        for model in models:
            model_time_nll = np.mean(all_datasets_time[dataset][model]["NLL Time"]) 
            #model_ste_time_nll = np.std(model_dic["NLL Time"])/np.sqrt(len(model_dic["NLL Time"]))
            #model_time_cal = np.mean(all_datasets_time[dataset][model]["Time Cal."])
            if mark:
                model_mark_nll = np.mean(all_datasets_marks[dataset][model]["NLL Mark"])
                #model_mark_cal = np.mean(all_datasets_marks[dataset][model]["Mark Cal."])
                #model_f1 = np.mean(all_datasets_marks[dataset][model]["F1-score"])
                data["Mark NLL"] = model_mark_nll
                #data["ECE"] = model_mark_cal
                #data["F1-score"] = model_f1   
            #model_ste_time_cal = np.std(model_dic["Time Cal."])/np.sqrt(len(model_dic["Time Cal."]))
            data["Time NLL"] = model_time_nll
            #data["TCE"] = model_time_cal
            data["model"] =  model
            data["dataset"] = dataset
            df = pd.DataFrame(data=[data])
            df_all = pd.concat([df_all, df], axis=0, ignore_index=True)        
    palette = sns.color_palette("Set2", 12)  
    #data1 = df_all[["model", "Time NLL"]]
    #data2 = df_all[['model', 'TCE']]
    g1 = sns.FacetGrid(df_all, palette=palette, col="dataset", hue="dataset", sharey=False)
    g1.map_dataframe(sns.barplot, x="model", y='Time NLL', fill=True, alpha=0.7, palette=palette)
    #g1.map(label, "dataset")
    #g1.fig.subplots_adjust(hspace=-.4)
    titles = [map_dataset(dataset) for dataset in datasets]
    for i, ax in enumerate(g1.fig.axes):
        ax.set_title(titles[i], fontsize=24, y=1.12)
    ax = g1.fig.axes[0]
    ax.set_ylabel('NLL-T', fontsize=24)
    g1.set_xticklabels(fontsize = 18)
    g1.set_yticklabels(fontsize = 14)
    g1.set_xlabels('')

    if mark:
        g2 = sns.FacetGrid(df_all, palette=palette, col="dataset", hue="dataset", sharey=False)
        g2.map_dataframe(sns.barplot, x="model", y='Mark NLL', fill=True, alpha=0.7, palette=palette)
        #g1.map(label, "dataset")
        #g1.fig.subplots_adjust(hspace=-.4)
        titles = [map_dataset(dataset) for dataset in datasets]
        for i, ax in enumerate(g2.fig.axes):
            ax.set_title(" ")
        ax = g2.fig.axes[0]
        ax.set_ylabel('NLL-M', fontsize=24)
        g2.set_xticklabels(fontsize = 15)
        g2.set_yticklabels(fontsize = 14)
        g2.set_xlabels('')    
    if mark:
        g1.savefig('figures/dataset_comp/dataset_comp_marked_time', bbox_inches='tight')
        g2.savefig('figures/dataset_comp/dataset_comp_marked_mark', bbox_inches='tight')
    else:
        g1.savefig('figures/dataset_comp/dataset_comp_unmarked', bbox_inches='tight')
    #g1.set(yticks=[], xlabel=" " , ylabel=" ")
    #g1.despine( left=True)
        #g1.map(label, "dataset")
        #g1.fig.subplots_adjust(hspace=-.4)
        #g1.set_titles("")
        #g1.set_xticklabels(fontsize = 40)
        #g1.set(yticks=[], xlabel=" " , ylabel=" ")
        #g1.despine( left=True)  
    '''
        plt.figure(figsize=(8, 8))
        ax = sns.barplot(x='model', y='Time NLL', data=data1, palette=palette)
        width_scale = 0.45
        for bar in ax.containers[0]:
            bar.set_width(bar.get_width() * width_scale)    
        ax2 = ax.twinx()
        sns.barplot(x='model', y='TCE', data=data2, alpha=0.7, hatch='//', ax=ax2, palette=palette)
        for bar in ax2.containers[0]:
            x = bar.get_x()
            w = bar.get_width()
            bar.set_x(x + w * (1- width_scale))
            bar.set_width(w * width_scale)
        ax.set_ylabel('Time NLL', fontsize=32)  
        ax2.set_ylabel('TCE', fontsize=32)
        ax.tick_params(axis='both', which='major', labelsize=24 )
        ax2.tick_params(axis='both', which='major', labelsize=24)  
        a=ax.get_yticks().tolist()
        labels = [str(-int(label)) for label in a]
        ax.set_yticklabels(labels)
        ax.set_xlabel('')
        plt.suptitle(map_dataset(dataset), y=.95, fontsize=32)
        plt.savefig('figures/dataset_comp/dataset_comp_{}'.format(dataset) , bbox_inches='tight')
        if mark:
            data1 = df_all[["model", "Mark NLL"]]
            data2 = df_all[['model', 'ECE']]
            data3 = df_all[['model', 'F1-score']]

            plt.figure(figsize=(8, 8))
            ax = sns.barplot(x='model', y='Mark NLL', data=data1, palette=palette)
            width_scale = 0.3  
            ax2 = ax.twinx()
            sns.barplot(x='model', y='ECE', data=data2, alpha=0.7, hatch='//', ax=ax2, palette=palette)
            for bar in ax2.containers[0]:
                x = bar.get_x()
                w = bar.get_width()
                bar.set_x(x + 1.2*w*width_scale)
                bar.set_width(w * width_scale) 
            for bar in ax.containers[0]:
                x = bar.get_x()
                w = bar.get_width()
                bar.set_x(x  + 0.5*w* width_scale - 0.1)
                bar.set_width(w * width_scale) 
            ax3 = ax.twinx()
            sns.barplot(x='model', y='F1-score', data=data3, alpha=0.7, hatch='xx', ax=ax3, palette=palette)
            for bar in ax3.containers[0]:
                x = bar.get_x()
                w = bar.get_width()
                bar.set_x(x + 1.9*w* width_scale + 0.1)
                bar.set_width(w * width_scale) 
            ax3.spines['right'].set_position(('axes', 1.25))
            
            ax.set_ylabel('Mark NLL', fontsize=32)  
            ax2.set_ylabel('ECE', fontsize=32)
            ax3.set_ylabel('F1-score', fontsize=32)
            ax.tick_params(axis='both', which='major', labelsize=24 )
            ax2.tick_params(axis='both', which='major', labelsize=24) 
            ax3.tick_params(axis='both', which='major', labelsize=24)  
            #a=ax.get_yticks().tolist()
            #labels = [str(-int(label)) for label in a]
            #ax.set_yticklabels(labels)
            ax.set_xlabel('')
            plt.suptitle(map_dataset(dataset), y=.95, fontsize=32)
            plt.savefig('figures/dataset_comp/dataset_comp_mark_{}'.format(dataset) , bbox_inches='tight')
        '''
    plt.show()
        

def fill_result_dict(path, all_dic_dataset, dataset, m, mrc=False, split=None, barplot=False):
    """
    if mrc is True:
        assert(split is not None)
        for key in all_dic_dataset[dataset].keys():
            if key == 'F1-score':
                pre, rec, f1 = f1_most_present_class(dataset, split=split)
                all_dic_dataset[dataset][key][m].append(f1)
            else:
                all_dic_dataset[dataset][key][m].append(np.nan)
    """
    m = map_model(m, barplot)
    with open(path, 'rb') as fp:
        while True:
            try:
                e = pickle.load(fp)
                r = e['test']
                for key in all_dic_dataset[dataset][m].keys():
                    if key == 'NLL Time':
                        all_dic_dataset[dataset][m][key].append(-r['log ground density'])
                    elif key == 'NLL Mark':
                        all_dic_dataset[dataset][m][key].append(-r['log mark density'])
                    elif key == 'Time Cal.':
                        bins, cal = calibration(r, num_bins=50, tabular=True)
                        _ , c_cal_mae = calibration_errors(bins, cal)
                        all_dic_dataset[dataset][m][key].append(c_cal_mae)
                    elif key == 'Mark Cal.':
                        d_cal = r['calibration']
                        _ , d_cal_mae = discrete_calibration_error(d_cal)
                        all_dic_dataset[dataset][m][key].append(d_cal_mae)
                    elif key == 'Precision':
                        all_dic_dataset[dataset][m][key].append(r['pre_weighted'])
                    elif key == 'Recall':
                        all_dic_dataset[dataset][m][key].append(r['rec_weighted'])
                    elif key == 'F1-score':
                        all_dic_dataset[dataset][m][key].append(r['f1_weighted'])
            except EOFError:
                break      
    return all_dic_dataset

def calibration_errors(c_cal, y):
    c_mse = np.mean((np.array(c_cal)-y)**2, axis=-1) 
    c_mae = np.mean(np.abs(np.array(c_cal)-y), axis=-1)
    return c_mse, c_mae

def get_dataset_types(dataset):
    dataset_marked = ['lastfm_filtered', 'mooc_filtered', 'wikipedia_filtered', 'github_filtered', 'mimic2_filtered', 'hawkes_exponential_mutual', 'retweets_filtered', 'stack_overflow_filtered']
    datasets_unmarked = ['taxi', 'twitter', 'reddit_politics_submissions', 'reddit_askscience_comments', 'pubg', 'yelp_toronto', 'yelp_mississauga', 'yelp_airport']
    if dataset in dataset_marked:
        return 'marked'
    elif dataset in datasets_unmarked:
        return 'unmarked'
    else:
        raise ValueError('Unknown dataset')

def initialize_results_dic(datasets, models, marks=True, barplot=False):
    all_dic_dataset = dict.fromkeys(datasets)
    models = [map_model(model, barplot) for model in models]
    for dataset in all_dic_dataset.keys():       
        if get_dataset_types(dataset) == 'marked':
            if marks:
                model_dic = {model:{
                            'NLL Mark':[],                                       
                            'Mark Cal.':[],
                            'F1-score':[]
                            } for model in models}
            else:
                model_dic = {model:{
                        'NLL Time':[],                                       
                        'Time Cal.':[]
                        } for model in models}
            all_dic_dataset[dataset] = model_dic
        elif get_dataset_types(dataset) == 'unmarked':
            model_dic = {model:{'NLL Time':[], 
                        'Time Cal.':[]} for model in models}
            all_dic_dataset[dataset] = model_dic    
    return all_dic_dataset


def get_history_results(models, dataset, results_dir, num_split, histories=None):
    all_models = {}
    for model in models:
        if dataset=='wikipedia_filtered' and model == 'selfattention-fixed_selfattention-cm_log_concatenate_base':
            model = 'gru-fixed_selfattention-cm_log_concatenate_base' 
        all_models[model] = {h:{s:[] for s in range(num_split)} for h in histories}
    file_dir = os.path.join(results_dir, dataset)
    file_dir_his = os.path.join(file_dir, 'history') 
    file_dir_zero_hist = os.path.join(file_dir, 'best')
    file_names = os.listdir(file_dir_his)
    file_names_zero_hist = os.listdir(file_dir_zero_hist)
    for model in models:
        if dataset=='wikipedia_filtered' and model == 'selfattention-fixed_selfattention-cm_log_concatenate_base':
            model = 'gru-fixed_selfattention-cm_log_concatenate_base' 
        for hist in histories:
            if hist != '0':
                for split in range(num_split):
                    if hist == 'full':
                        if 'base' in model:
                            file_to_find = 'poisson_' + dataset + '_' + model.replace('-fixed','').replace('_base', '') + '_split' + str(split) + '_history' + str(hist)
                        else:
                            file_to_find = dataset + '_' + model.replace('-fixed','') + '_split' + str(split) + '_history' + str(hist)
                    else:
                        if 'base' in model:
                            file_to_find = 'poisson_' + dataset + '_' + model.replace('_base', '') + '_split' + str(split) + '_history' + str(hist)
                        else:
                            file_to_find = dataset + '_' + model + '_split' + str(split) + '_history' + str(hist)
                    path = None
                    for file_name in file_names:
                        if file_name.startswith(file_to_find):
                            path = os.path.join(file_dir_his, file_name)
                            with open(path, 'rb') as fp:
                                while True:
                                    try:
                                        e = pickle.load(fp)
                                        #hist = find_history(path)
                                        #print(his)
                                        all_models[model][hist][split] = e['test']
                                    except EOFError:
                                        break 
                            break 
                    if path is None:
                        print(file_to_find)
                        raise ValueError('File not found')
            else:
                if 'gru-fixed' in model:
                    decoder_encoding = model.split('gru-fixed_')[1].split('split')[0]
                elif 'selfattention-fixed' in model:
                    decoder_encoding = model.split('selfattention-fixed_')[1].split('split')[0]
                if 'log-normal' in model:
                    decoder_encoding = 'log-normal-mixture_temporal' 
                elif 'conditional' in model:
                    decoder_encoding = 'conditional-poisson_temporal' 
                elif 'rmtpp' in model:
                    decoder_encoding = 'rmtpp_temporal' 
                for split in range(num_split):
                    if 'base' in model:
                        file_zero_hist_to_find = 'poisson_' + dataset + '_constant_' + decoder_encoding.replace('_base','') + '_split' + str(split) + '_config'
                    else:
                        file_zero_hist_to_find = dataset + '_constant_' + decoder_encoding + '_split' + str(split) + '_config'
                    path = None 
                    for file_name in file_names_zero_hist:
                        if file_name.startswith(file_zero_hist_to_find):
                            path = os.path.join(file_dir_zero_hist, file_name)
                            with open(path, 'rb') as fp:
                                while True:
                                    try:
                                        e = pickle.load(fp)
                                        all_models[model][hist][split] = e['test']
                                    except EOFError:
                                        break 
                            break
                    if path is None:
                        print(file_zero_hist_to_find)
                        raise ValueError('File not found')
    return all_models



def plot_history_multi_datasets(models, dataset, results_dir, num_split, histories, marks=True):
    #fig, ax = plt.subplots(1,3, figsize=(40,10), sharex=True)
    #font = {#'family' : 'normal',
    #    'weight' : 'bold',
    #    'size'   : 20}
    #matplotlib.rc('font', **font)
    #matplotlib.rc('xtick', labelsize=20) 
    #matplotlib.rc('ytick', labelsize=20) 
    all_models = get_history_results(models=models, dataset=dataset, 
                                    results_dir=results_dir,
                                    num_split=num_split, histories=histories)
    df_all = pd.DataFrame()
    data = dict.fromkeys(['history', 'loss mark', 'loss time', 'model'])
    for model, model_dic in all_models.items():
        histories = []
        model_losses_time, model_losses_mark, ste_model_losses_time, ste_model_losses_mark = [], [], [], []
        for hist, hist_dic in model_dic.items():
            if hist != 'full':
                    histories.append(int(hist))
            else: 
                list_keys = list(model_dic.keys())
                list_keys.remove('full')
                list_keys = [int(key) for key in list_keys]
                histories.append(max(list_keys) +1)
            loss_time = [-hist_dic[split]['log ground density'] for split in hist_dic.keys()]
            model_losses_time.append(np.mean(loss_time))
            ste_model_losses_time.append(np.std(loss_time)/np.sqrt(len(loss_time)))
            if marks:
                loss_mark = [-hist_dic[split]['log mark density'] for split in hist_dic.keys()]
                model_losses_mark.append(np.mean(loss_mark))
                ste_model_losses_mark.append(np.std(loss_mark)/np.sqrt(len(loss_mark)))
        sort_index = np.argsort(histories)
        histories = np.array(histories)[sort_index]
        histories = [str(history) for history in histories]
        histories[-1] = 'F'
        model_losses_time = np.array(model_losses_time)[sort_index]
        ste_model_losses_time = np.array(ste_model_losses_time)[sort_index]
        model = get_acronym([model])[0]
        data["history"] = histories
        data["loss time"] = model_losses_time
        data["model"] = [model]*len(histories)
        if marks:
            model_losses_mark = np.array(model_losses_mark)[sort_index]
            ste_model_losses_mark = np.array(ste_model_losses_mark)[sort_index]
            data["loss mark"] = model_losses_mark
        df = pd.DataFrame(data=data)
        df_all = pd.concat([df_all, df], axis=0, ignore_index=True)
    sns.set_style("darkgrid")
    if marks:
        fig, ax = plt.subplots(1,2, figsize=(14,7))
        palette = sns.color_palette("Set2", 12)    
        sns.lineplot(data=df_all, x ='history', y='loss time', hue='model', linewidth=2, marker='o', ax=ax[0])
        sns.lineplot(data=df_all, x='history', y='loss mark', hue='model' , linewidth=2, marker='o', ax=ax[1]) 
        ax[0].set_xlabel('History Size', fontsize=24)
        ax[1].set_xlabel('History Size', fontsize=24)
        ax[0].set_ylabel('Time NLL', fontsize=20)  
        ax[1].set_ylabel('Mark NLL', fontsize=20)
        ax[0].tick_params(axis='both', which='major', labelsize=16 )
        ax[1].tick_params(axis='both', which='major', labelsize=16)
        ax[0].legend().set_title('')
        ax[1].legend().set_title('')
        ax[1].get_legend().remove()
        ax[0].get_legend().remove()

        #ax.flatten()[0].legend(fontsize=16,loc='upper center', bbox_to_anchor=(1.06, -0.12), ncol=4)
        fig.suptitle(map_dataset(dataset), y=.95, fontsize=16)
        plt.savefig('figures/history/history2_{}'.format(dataset) , bbox_inches='tight')
    else:
        fig, ax = plt.subplots(1,1, figsize=(7,4))
        palette = sns.color_palette("Set2", 12)    
        sns.lineplot(data=df_all, x ='history', y='loss time', hue='model', linewidth=2, marker='o', ax=ax)
        ax.set_xlabel('History Size', fontsize=20)
        ax.set_ylabel('Time NLL', fontsize=20)  
        ax.tick_params(axis='both', which='major', labelsize=14  )
        ax.get_legend().remove()
        #ax.legend(loc='upper center', bbox_to_anchor=(1.06, -0.12), ncol=4)
        fig.suptitle(map_dataset(dataset), y=.95, fontsize=16)
        plt.savefig('figures/history/history2_{}'.format(dataset) , bbox_inches='tight')
    
    plt.plot()
    
   

def mapping_coefficients(model):
    print(model)
    if 'conditional-poisson' in model:
        if 'gru' in model:
            if 'learnable_with_labels' in model:
                mapping = 'EC-GRU-LEWL'
    elif 'log-normal-mixture' in model:
        if 'gru' in model:
            if 'learnable_with_labels' in model:
                mapping = 'LNM-GRU-LEWL'
    elif 'mlp-cm' in model:
        if 'gru' in model:
            if 'learnable_with_labels' in model:
                mapping = 'FNN-GRU-LEWL'
    elif 'mlp-mc' in model:
        if 'gru' in model:
            if 'learnable_with_labels' in model:
                mapping = 'MLP/MC-GRU-LEWL'
    elif 'rmtpp' in model:
        if 'gru' in model:
            if 'learnable_with_labels' in model:
                mapping = 'RMTPP-GRU-LEWL'
    elif 'selfattention-cm' in model:
        if 'gru' in model:
            if 'learnable_with_labels' in model:
                mapping = 'SA/CM-GRU-LEWL'
    elif 'selfattention-mc' in model:
        if 'gru' in model:
            if 'learnable_with_labels' in model:
                mapping = 'SA/MC-GRU-LEWL'
    return mapping

def map_dataset(dataset):
    mapping = {
        'lastfm_filtered': 'LastFM',
        'mooc_filtered':'MOOC',
        'wikipedia_filtered':'Wikipedia',
        'github_filtered':'Github',
        'mimic2_filtered':'MIMIC2',
        'retweets_filtered':'Retweets',
        'stack_overflow_filtered':'Stack Overflow',
        'hawkes_exponential_mutual':'Hawkes',
        'taxi':'Taxi',
        'twitter':'Twitter',
        'reddit_politics_submissions':'Reddit Subs.',
        'reddit_askscience_comments':'Reddit Ask.',
        'pubg':'PUBG',
        'yelp_toronto':'Yelp Toronto',
        'yelp_mississauga':'Yelp Mississauga',
        'yelp_airport':'Yelp Airport'
    }
    return mapping[dataset]


def coefficients_plots(models, dataset, results_dir, num_split=5, split_to_show=0):
    all_models = dict.fromkeys(models)
    for model in models:
        all_models[model] = {key:{split:[] for split in range(num_split)} for key in ['val']}
        file_dir = os.path.join(results_dir, dataset)
        file_dir = os.path.join(file_dir, 'best')
        file_names = os.listdir(file_dir)
        for split in range(num_split):
            file_to_find = 'poisson_' + dataset + '_' + model + '_split' + str(split) + '_config'
            for file_name in file_names: 
                if file_name.startswith(file_to_find):
                    path = os.path.join(file_dir, file_name)
                    with open(path, 'rb') as fp:
                        while True:
                            try:
                                e = pickle.load(fp)
                                all_models[model]['val'][split] = e['val']
                            except EOFError:
                                break
    df_all = pd.DataFrame()
    keys = ['epoch', 'alpha0', 'alpha1', 'lambda_ratio', 'model']
    data = dict.fromkeys(keys)
    for model, model_dic in all_models.items():
        alpha_0 = [epoch_results['alpha'][1] for epoch_results in model_dic['val'][split_to_show]]
        alpha_1 = [epoch_results['alpha'][0] for epoch_results in model_dic['val'][split_to_show]]
        lambda_0 = np.array([epoch_results['intensity_0'] for epoch_results in model_dic['val'][split_to_show]])
        lambda_1 = np.array([epoch_results['intensity_1'] for epoch_results in model_dic['val'][split_to_show]])
        print('lambda_0', lambda_0[-1])
        print('alpha_0', alpha_0[-1])
        print('lambda_1', lambda_1[-1])
        print('alpha_1', alpha_1[-1])
        lambda_ratio = np.log((lambda_1*alpha_1)/(lambda_0*alpha_0))
        data['alpha0'] = alpha_0[:-1]
        data['alpha1'] = alpha_1[:-1]
        data['lambda_ratio'] = lambda_ratio[:-1]
        data['epoch'] = np.arange(len(alpha_0[:-1]))
        model = mapping_coefficients(model)
        data['model'] = [model]*len(alpha_0[:-1])
        df = pd.DataFrame(data=data )
        df_all = pd.concat([df_all, df], axis=0, ignore_index=True)
    #plt.rcParams["figure.figsize"] = (14,7)
    sns.set_style("whitegrid")
    palette = sns.color_palette("Set2", 12)    
    fig, ax = plt.subplots(1,1, figsize=(14,7))
    #ax = sns.lineplot(data=df_all, x ='epoch', y='alpha1', hue='model', linewidth=2, ax=ax)
    #ax1 = sns.lineplot(data=df_all, x='epoch', y='alpha0', hue='model' , linewidth=2)
    ax = sns.lineplot(data=df_all, x='epoch', y='lambda_ratio', hue='model', linewidth=2, ax=ax)
    ax.set_title(map_dataset(dataset), fontsize=20)
    #ax2.set_xlabel()
    ax.set_xlabel('Epoch', fontsize=24)
    #ax.set_ylabel(r'$\alpha_1$', fontsize=24)
    ax.set_ylabel('log(R)', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=16)
    #ax2.tick_params(axis='both', which='major', labelsize=16)
    ax.legend().set_title('')
    #ax2.legend().set_title('')
    ax.get_legend().remove()
    #ax1.get_legend().remove()
    #plt.legend([],[], frameon=False)
    #plt.gca().legend().set_title('')
    ax.legend(fontsize=16, loc='upper center', bbox_to_anchor=(1.06, -0.16), ncol=4)
    plt.savefig('figures/coefficients/alpha_plot_{}'.format(dataset) , bbox_inches='tight')
    plt.show()
    



def get_all_results_per_dataset(encoders, decoders, dataset, results_dir, num_split, metric ,save_dir=None, mark=True, base=None, result_set='test', histories=None, percentage=False, show_legend=False, show_variance=False):
    assert(metric in ['c_calibration', 'd_calibration', 'loss', 'intensity', 'history']), 'wrong metric passed.'
    assert(base in [None, 'base', 'base_fc']), 'Wrong type of model passed.'
    all_models = {}
    exp_name = dataset
    file_dir = os.path.join(results_dir, dataset)
    if histories is None:
        file_dir = os.path.join(file_dir, 'best')
    else:
        file_dir = os.path.join(file_dir, 'history') 
        file_dir_zero_hist = os.path.join(file_dir, 'best')
    file_names = os.listdir(file_dir)
    for encoder in encoders:
        for decoder in decoders:
            if encoder == 'identity':
                model = 'identity_poisson'
            elif encoder == 'stub':
                model = 'stub_hawkes'
            else:
                model = encoder + '_' + decoder
            if metric == 'history':
                all_models[model] = {history:{split:[] for split in range(num_split)} for history in histories}
            else:
                all_models[model] = {key:{split:[] for split in range(num_split)} for key in ['train', 'val', 'test']}
            for split in range(num_split):
                for file_name in file_names:
                    if base == 'base_fc':
                        if (model in file_name) and ('split' + str(split) in file_name) and ('coefficients_' + dataset in file_name):
                            file_path = os.path.join(file_dir, file_name)
                            fill_results_dic(all_models, model, split, file_path)
                    elif base == 'base':    
                        if (model in file_name) and ('split' + str(split) in file_name) and ('poisson_' + dataset in file_name):
                            file_path = os.path.join(file_dir, file_name)
                            fill_results_dic(all_models, model, split, file_path)
                    else:
                        if metric == 'history':
                            if (decoder in file_name) and ('split' + str(split) in file_name) and not ('poisson_' + dataset in file_name) and not ('coefficients_' + dataset in file_name):
                                file_path = os.path.join(file_dir, file_name)
                                fill_results_dic(all_models, model, split, file_path, histories)
                        else:
                            if (model in file_name) and ('split' + str(split) in file_name) and not ('poisson_' + dataset in file_name) and not ('coefficients_' + dataset in file_name):
                                file_path = os.path.join(file_dir, file_name)
                                fill_results_dic(all_models, model, split, file_path, histories)
    exp_name = dataset + '_' + encoder
    if base == 'base':
        exp_name = exp_name + '_base'
    elif base == 'base_fc':
        exp_name = exp_name + '_base_fc'
    if metric == 'c_calibration':
        continuous_calibration_plot(all_models, exp_name=exp_name, save_dir=save_dir, result_set=result_set)
    elif metric == 'd_calibration':
        exp_name = dataset       
        d_mse, d_mae = discrete_calibration(all_models,exp_name=exp_name,save_dir=save_dir, result_set=result_set, base=base)
    elif metric == 'loss':
        plot_loss_mesh(all_models, exp_name, split_to_show=0, mark=mark, save_dir=save_dir, result_set=result_set)
    elif metric == 'intensity':
        plot_intensities(all_models, exp_name, split_to_show=0, save_dir=save_dir, result_set=result_set)
    elif metric == 'history':
        exp_name = dataset + '_' + encoder
        plot_history(all_models, exp_name, save_dir=save_dir, mark=mark, percentage=percentage, show_legend=show_legend)


def map_model(model, barplot=False):
    if 'conditional-poisson' in model:
        mapping = 'EC'
    elif 'log-normal-mixture' in model:
        mapping = 'LNM'
    elif 'mlp-cm' in model:
        mapping = 'FNN'
    elif 'mlp-mc' in model:
        mapping = 'MLP/MC'
    elif 'rmtpp' in model:
        mapping = 'RMTPP'
    elif 'selfattention-cm' in model:
        mapping = 'SA/CM'
    elif 'selfattention-mc' in model:
        mapping = 'SA/MC'
    elif 'identity_poisson' in model:
        mapping = 'Poisson'
    elif 'stub_hawkes' in model:
        if barplot:
            mapping = 'H'
        else:
            mapping = 'Hawkes'
    elif 'neural-hawkes' in model:
        mapping = 'NH'
    return mapping

def map_models_time_cal(model):
    if 'conditional-poisson' in model:
        if 'log_times_only' in model:
            mapping = 'CP-LTO'
        elif 'learnable' in model:
            mapping = 'CP-LE'
        elif 'times_only' in model:
            mapping = 'CP-TO'
    elif 'log-normal-mixture' in model:
        if 'log_times_only' in model:
            mapping = 'LNM-LTO'
        elif 'learnable' in model:
            mapping = 'LNM-LE'
        elif 'times_only' in model:
            mapping = 'LNM-TO'
    elif 'mlp-cm' in model:
        if 'log_times_only' in model:
            mapping = 'MLP/CM-LTO'
        elif 'learnable' in model:
            mapping = 'MLP/CM-LE'
        elif 'times_only' in model:
            mapping = 'MLP/CM-TO'
    elif 'mlp-mc' in model:
        if 'log_times_only' in model:
            mapping = 'MLP/MC-LTO'
        elif 'learnable' in model:
            mapping = 'MLP/MC-LE'
        elif 'times_only' in model:
            mapping = 'MLP/MC-TO'
    elif 'rmtpp' in model:
        if 'log_times_only' in model:
            mapping = 'RMTPP-LTO'
        elif 'learnable' in model:
            mapping = 'RMTPP-LE'
        elif 'times_only' in model:
            mapping = 'RMTPP-TO'
    elif 'selfattention-cm' in model:
        if 'log_times_only' in model:
            mapping = 'SA/CM-LTO'
        elif 'learnable' in model:
            mapping = 'SA/CM-LE'
        elif 'times_only' in model:
            mapping = 'SA/CM-TO'
    elif 'selfattention-mc' in model:
        if 'log_times_only' in model:
            mapping = 'SA/MC-LTO'
        elif 'learnable' in model:
            mapping = 'SA/MC-LE'
        elif 'times_only' in model:
            mapping = 'SA/MC-TO'
    elif 'identity_poisson' in model:
        mapping = 'Poisson'
    elif 'hawkes' in model:
        mapping = 'Hawkes'
    return mapping



def calibration_plot_all_datasets(models, datasets, results_dir, num_split, save_dir, discrete=False, title=None, show_labels=True, marked=True):
    all_models = {}
    for model in models:
        all_models[model] = dict.fromkeys(datasets)
        for dataset in datasets:
            all_models[model][dataset] = {'test':{split:[] for split in range(num_split)}}
            file_dir = os.path.join(results_dir, dataset)
            file_dir = os.path.join(file_dir, 'best')
            file_names = os.listdir(file_dir)
            for split in range(num_split):
                file_to_find = dataset + '_' + model + '_split' + str(split)
                if 'base' in model:
                    if 'base_fc' in model:
                        file_to_find = 'poisson_coefficients_' +  dataset + '_' + model.replace('_base_fc', '') + '_split' + str(split)
                    elif 'lnmk1' in model:
                        model_to_find = model.split('_lnmk1')[0]
                        file_to_find = 'poisson_' + dataset + '_' + model_to_find + '_split' + str(split) + '_lnmk1'
                    else:
                        file_to_find = 'poisson_' +  dataset + '_' + model.replace('_base', '') + '_split' + str(split)
                elif 'lnmk1' in model:
                    model_to_find = model.split('_lnmk1')[0]
                    file_to_find = dataset + '_' + model_to_find + '_split' + str(split) + '_lnmk1'
                for file_name in file_names: 
                    if file_name.startswith(file_to_find):
                        path = os.path.join(file_dir, file_name)
                        with open(path, 'rb') as fp:
                            while True:
                                try:
                                    e = pickle.load(fp)
                                    all_models[model][dataset]['test'][split] = e['test']
                                except EOFError:
                                    break
    if discrete:
        df_all = pd.DataFrame()
        for model, model_dic in all_models.items():
            keys = ['accuracy', 'samples', 'bins', 'model']
            data = dict.fromkeys(keys)
            mean_dataset_calibration, mean_dataset_samples = [], []
            for dataset, dataset_dic in model_dic.items():
                all_splits_calibration, all_splits_samples = [], []
                for split, split_dic in dataset_dic['test'].items():
                    all_splits_calibration.append(np.array(split_dic['calibration']))
                    samples = np.array(split_dic['samples per bin'])
                    sum_samples = sum(samples)
                    prop_per_bin = [s/sum_samples for s in samples]
                    all_splits_samples.append(np.array(prop_per_bin))
                mean_split_calibration = np.mean(all_splits_calibration, axis=0)
                mean_dataset_calibration.append(mean_split_calibration)
                mean_split_samples = np.mean(all_splits_samples, axis=0)
                mean_dataset_samples.append(mean_split_samples)
            mean_dataset_calibration = np.mean(mean_dataset_calibration, axis=0)
            mean_dataset_samples = np.mean(mean_dataset_samples, axis=0)
            bins = [round(1/(2*len(mean_dataset_calibration)) + i/len(mean_dataset_calibration),2) for i in range(len
            (mean_dataset_calibration))]
            data['Accuracy'] = mean_dataset_calibration
            data['Samples'] = mean_dataset_samples
            data['bins'] = bins
            data['model'] = [model]*len(mean_dataset_calibration)
            df = pd.DataFrame(data=data)
            df_all = pd.concat([df_all, df], axis=0)
        def label(x, color, label):
            ax = plt.gca()
            ax.text(.1, .4, label, color='black', fontsize=40,
                ha="left", va="center", transform=ax.transAxes)
        sns.set_theme(style="white", rc={"font.size":20, "axes.facecolor": (0, 0, 0, 0), 'axes.linewidth':2})
        sns.set_style("darkgrid")
        palette = sns.color_palette("Set2", 12)
        g2 = sns.FacetGrid(df_all, palette=palette, col="model", hue="model", aspect=0.7)
        g1 = sns.FacetGrid(df_all, palette=palette, col="model", hue="model", aspect=.7)
        g2.map_dataframe(sns.pointplot, x='bins', y='bins', color='black', markers='')
        g2.map_dataframe(sns.barplot, x='bins', y='Accuracy', fill=True, alpha=0.7)
        g1.map_dataframe(sns.barplot, x='bins', y='Samples', fill=True, alpha=0.7)
        g2.fig.subplots_adjust(hspace=-.4)
        g2.fig.subplots_adjust(wspace=.1)
        g1.fig.subplots_adjust(hspace=-.4)
        g1.fig.subplots_adjust(wspace=.1)
        titles = list(map(map_model, models))
        g2.set_titles(list(titles))
        g1.set_titles("")
        #g2
        ax = g2.fig.axes[0]
        new_ticks = np.append(ax.get_xticks(),10) - 0.5
        xxx = np.round(np.arange(0,1.1,.1),1)
        #g2.set(xticks=new_ticks)
        g2.set(xticks=[])
        #g2.set_xticklabels(xxx, fontsize=12)
        g2.set_yticklabels(np.round(ax.get_yticks(),1),fontsize=14)
        g2.set_ylabels('Accuracy', fontsize=20)
        g2.set(xlabel=" ")
        for i, ax in enumerate(g2.fig.axes):
            ax.set_title(titles[i], fontsize=20)
        #g1
        ax = g1.fig.axes[0]
        axes = g1.fig.axes
        mid_ax = int((len(axes)-1)/2)
        for i, ax in enumerate(axes):
            if  i == mid_ax and show_labels:
                ax.set_xlabel('Confidence', fontsize=20)
                ax.xaxis.labelpad = 20
            else:
                ax.set_xlabel(" ")
        new_ticks = np.append(ax.get_xticks(),10) - 0.5
        #g1.set(xticks=new_ticks)
        g1.set(xticks=[])
        #g1.set_xticklabels(xxx, fontsize=12)
        ax.set_ylim(0,0.6)
        g1.set_yticklabels(np.round(ax.get_yticks(),1),fontsize=14)
        g1.set_ylabels('p% Samples', fontsize=20)
        
        
        #g2.despine( left=True)
        if title is not None:
            g1.savefig('figures/calibration/discrete_{}_samples.png'.format(title) , bbox_inches='tight')
            g2.savefig('figures/calibration/discrete_{}_calibration.png'.format(title) , bbox_inches='tight') 
        else:
            if 'gru_conditional-poisson_times_only' in models:
                g1.savefig('figures/calibration/discrete_calibrations_samples1.png' , bbox_inches='tight')
                g2.savefig('figures/calibration/discrete_calibration1.png' , bbox_inches='tight') 
            else:
                g1.savefig('figures/calibration/discrete_calibrations_samples2.png' , bbox_inches='tight')
                g2.savefig('figures/calibration/discrete_calibration2.png' , bbox_inches='tight') 
        
    else:
        df_all = pd.DataFrame()
        keys = ['True quantiles', 'Predicted quantiles', 'model']
        data = dict.fromkeys(keys)
        plt.rcParams["figure.figsize"] = (8,8)
        for model, model_dic in all_models.items():
            model_results_for_all_datasets = []
            for dataset, dataset_dic in model_dic.items(): 
                model_dataset_results = dataset_dic['test']
                bins, mean_calibration, std_error_calibration = calibration(model_dataset_results, num_bins=50, result_set='test')
                model_results_for_all_datasets.append(mean_calibration)
            mean_model_results = np.mean(np.array(model_results_for_all_datasets), axis=0)
            data['Observed frequency'] = mean_model_results
            data['Predicted probability'] = bins
            model = map_model(model)
            data['model'] = [model]*len(mean_model_results)
            df = pd.DataFrame(data=data)
            df_all = pd.concat([df_all, df], axis=0, ignore_index=True)
        #sns.set_theme(style="darkgrid", rc={"font.size":20, "axes.facecolor": (0, 0, 0, 0), 'axes.linewidth':2})
        sns.set_style("whitegrid")
        palette = sns.color_palette("Set2", 12)
        
        g1 = sns.FacetGrid(df_all, palette=palette, col="model", hue="model", aspect=1.1)
        #g2.map_dataframe(sns.pointplot, x='bins', y='bins', color='black', markers='')
        #g2.map_dataframe(sns.barplot, x='bins', y='Accuracy', fill=True, alpha=0.7)
        g1.map_dataframe(sns.lineplot, x='Predicted probability', y='Observed frequency', linewidth=1, marker='o')
        g1.map_dataframe(sns.lineplot, x='Predicted probability', y='Predicted probability', color='black')
        #g2.fig.subplots_adjust(hspace=-.4)
        g1.fig.subplots_adjust(wspace=.2)
        #sns.lineplot(data=df_all, x ='True quantiles', y='Predicted quantiles', hue='model', linewidth=1, marker="o")
        #sns.lineplot(data=df_all, x='True quantiles', y='True quantiles', color='black')
        #plt.xlabel('True quantiles', fontsize=24)
        #plt.ylabel('Predicted quantiles', fontsize=24)   
        #plt.tick_params(axis='both', which='major', labelsize=16)
        #plt.gca().legend().set_title('')
        #plt.legend(fontsize=16, loc='upper center', bbox_to_anchor=(1.06, -0.12), ncol=5)
        #plt.legend([],[], frameon=False)
        g1.set(xlim=[0,1], ylim=[0,1])
        g1.set_xlabels('Predicted probability', fontsize=20)
        g1.set_ylabels('Observed frequency', fontsize=20)
        titles = list(map(map_model, models))
        for i, ax in enumerate(g1.fig.axes):
            ax.set_title(titles[i], fontsize=20)
            if show_labels is True:
                if i != 2:
                    ax.set_xlabel(" ")
            else:
                ax.set_xlabel(" ")
        g1.set_yticklabels(np.round(ax.get_yticks(),1),fontsize=14)
        g1.set_xticklabels(np.round(ax.get_yticks(),1),fontsize=14)
        if title is None:
            if 'gru_conditional-poisson_log_times_only' in models:
                plt.savefig('figures/calibration/time_calibration_log_times_only.png' , bbox_inches='tight')
            elif 'gru_conditional-poisson_times_only' in models:
                plt.savefig('figures/calibration/time_calibration_times_only.png' , bbox_inches='tight')
            else:
                plt.savefig('figures/calibration/time_calibration_learnable.png' , bbox_inches='tight')
        else:
            plt.savefig('figures/calibration/{}.png'.format(title) , bbox_inches='tight')

    plt.show()


def plot_history(all_models, exp_name,  save_dir=None, mark=True, percentage=False, show_legend=True, show_variance=True):
    if mark:
        fig, ax = plt.subplots(1,3, figsize=(40,10), sharex=True)
    else:
        fig, ax = plt.subplots(figsize=(9,6))
    font = {#'family' : 'normal',
        'weight' : 'bold',
        'size'   : 25}

    matplotlib.rc('font', **font)
    for model in all_models:
        histories, losses_per_hist = [], []
        loss_time_per_hist, loss_mark_per_hist = [], []
        ste_loss_per_hist, ste_loss_time_per_hist, ste_loss_mark_per_hist = [], [], []
        if not percentage:
            for hist in all_models[model].keys():
                if hist != 'full':
                    histories.append(int(hist))
                else: 
                    list_keys = list(all_models[model].keys())
                    list_keys.remove('full')
                    list_keys = [int(key) for key in list_keys]
                    histories.append(max(list_keys) +1)
    
                losses = [all_models[model][hist][split]['loss'] for split in all_models[model][hist].keys()]
                losses_per_hist.append(np.mean(losses))
                ste_loss_per_hist.append(np.std(losses)/np.sqrt(len(losses)))
                if mark:
                    loss_time = [-all_models[model][hist][split]['log ground density'] for split in all_models[model][hist].keys()]
                    loss_mark = [-all_models[model][hist][split]['log mark density'] for split in all_models[model][hist].keys()]
                    loss_time_per_hist.append(np.mean(loss_time))
                    ste_loss_time_per_hist.append(np.std(loss_time)/np.sqrt(len(loss_time)))
                    loss_mark_per_hist.append(np.mean(loss_mark))
                    ste_loss_mark_per_hist.append(np.std(loss_mark)/np.sqrt(len(loss_mark)))
        else:
            loss_full = np.mean([all_models[model]['full'][split]['loss'] for split in all_models[model]['full'].keys()])
            if mark:
                loss_time_full = np.mean([-all_models[model]['full'][split]['log ground density'] for split in all_models[model]['full'].keys()])
                loss_mark_full = np.mean([-all_models[model]['full'][split]['log mark density'] for split in all_models[model]['full'].keys()])
            hists = list(all_models[model].keys())
            hists.remove('full')
            for hist in hists:
                histories.append(int(hist))
                losses = [all_models[model][hist][split]['loss'] for split in all_models[model][hist].keys()]
                losses_per_hist.append((np.mean(losses)-loss_full)/loss_full)
                if mark:
                    loss_time = [-all_models[model][hist][split]['log ground density'] for split in all_models[model][hist].keys()]
                    loss_mark = [-all_models[model][hist][split]['log mark density'] for split in all_models[model][hist].keys()]
                    loss_time_per_hist.append((np.mean(loss_time)-loss_time_full)/loss_time_full)
                    loss_mark_per_hist.append((np.mean(loss_mark)-loss_mark_full)/loss_mark_full)
        sort_index = np.argsort(histories)
        histories = np.array(histories)[sort_index]
        if not percentage:
            histories = [str(history) if i != len(histories)-1 else 'F' for i, history in enumerate(histories)]
        else:
            histories = [str(history) for history in histories]
        losses_per_hist = np.array(losses_per_hist)[sort_index]
        if not percentage:
            ste_loss_per_hist = np.array(ste_loss_per_hist)[sort_index]
        if mark:
            loss_time_per_hist = np.array(loss_time_per_hist)[sort_index]
            loss_mark_per_hist = np.array(loss_mark_per_hist)[sort_index]
            if not percentage:
                ste_loss_time_per_hist = np.array(ste_loss_time_per_hist)[sort_index]
                ste_loss_mark_per_hist = np.array(ste_loss_mark_per_hist)[sort_index]
        if mark:
            ax[0].plot(histories, losses_per_hist, label=model)
            ax[1].plot(histories, loss_time_per_hist, label=model)
            ax[2].plot(histories, loss_mark_per_hist, label=model)
            if not percentage and show_variance:
                ax[0].fill_between(histories, losses_per_hist - ste_loss_per_hist, losses_per_hist + ste_loss_per_hist, alpha=0.1)
                ax[1].fill_between(histories, loss_time_per_hist -ste_loss_time_per_hist, loss_time_per_hist + ste_loss_time_per_hist, alpha=0.1)
                ax[2].fill_between(histories, loss_mark_per_hist -ste_loss_mark_per_hist, loss_mark_per_hist + ste_loss_mark_per_hist, alpha=0.1)
        else:
            ax.plot(histories, losses_per_hist, label=model)
            if not percentage and show_variance:
                ax.fill_between(histories, losses_per_hist - ste_loss_per_hist, losses_per_hist + ste_loss_per_hist, alpha=0.1)
    if mark:
        ax[0].set_title(exp_name + ' - NLL Total' )
        ax[1].set_title(exp_name + ' - NLL Time' )
        ax[2].set_title(exp_name + ' - NLL Mark' )
        ax[2].legend(bbox_to_anchor=(0.95, 1.05), loc='upper left')
        ax[1].set_xlabel('History Size')
        ax[0].set_ylabel('NLL Total')
        ax[1].set_ylabel('NLL Time')
        ax[2].set_ylabel('NLL Mark')
        for a in ax:
            a.grid(True, linestyle='--')
    else:
        ax.set_title(exp_name + ' - NLL Total' )
        ax.set_xlabel('History Size')
        ax.set_ylabel('NLL')
        ax.grid(True, linestyle='--')
        if show_legend:
            ax.legend(bbox_to_anchor=(1.3, 1.05), loc='upper left')
    if save_dir is not None:      
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'history_{}.png'.format(exp_name))
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_intensities(all_models, exp_name, split_to_show=0, save_dir=None, result_set='test'):
    fig, ax = plt.subplots(figsize=(14,7))
    for model in all_models:
        alpha_0 = [epoch_results['alpha'][1] for epoch_results in all_models[model][result_set][split_to_show]]
        alpha_1 = [epoch_results['alpha'][0] for epoch_results in all_models[model][result_set][split_to_show]]
        color = next(ax._get_lines.prop_cycler)['color']
        ax.plot(alpha_0, label='Base alpha - {}'.format(model), linestyle='dashed', color=color)
        ax.plot(alpha_1, label='Model alpha - {}'.format(model), color=color)
    ax.legend()
    if save_dir is not None:      
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'coefficients_{}.png'.format(exp_name))
        fig.savefig(save_path, bbox_inches='tight')
    ax.set_title(exp_name)
    ax.grid(True, linestyle='--')
    plt.show()


def fill_results_dic(all_models, model, split, file_path, history=None):
    with open(file_path, 'rb') as fp:
        while True:
            try:
                e = pickle.load(fp)
                if history is None:
                    all_models[model]['train'][split] = e['train']
                    all_models[model]['val'][split] = e['val']
                    all_models[model]['test'][split] = e['test']
                else:
                    hist = find_history(file_path)
                    all_models[model][hist][split] = e['test']
            except EOFError:
                break

def find_history(file_name):
    hist = re.findall('_history(.*).txt', file_name)[0]
    return hist


def discrete_calibration(all_models, exp_name=None, plot=True, result_set='test', save_dir=None, base=None):
    assert(result_set in ['train', 'val', 'test']), 'result_set must either be train, val or test'
    for model in all_models:
        all_calibration, all_samples = [], []
        for split in all_models[model][result_set]:
            if result_set == 'test':
                all_calibration.append(np.array(all_models[model][result_set][split]['calibration']))
                samples = all_models[model][result_set][split]['samples per bin']
            else:
                all_calibration.append(np.array(all_models[model][result_set][split][-1]['calibration']))
                samples = all_models[model][result_set][split][-1]['samples per bin']
            sum_samples = sum(samples)
            prop_per_bin = [s/sum_samples for s in samples]
            all_samples.append(np.array(prop_per_bin))
        all_calibration = np.array(all_calibration)
        all_samples = np.array(all_samples)
        mean_calibration = np.mean(all_calibration, axis=0)
        std_calibration = np.std(all_calibration, axis=0)
        error_calibration = std_calibration/np.sqrt(all_calibration.shape[0])
        mean_samples = np.mean(all_samples, axis=0)
        std_samples = np.std(all_samples, axis=0)
        error_samples = std_samples/np.sqrt(all_samples.shape[0])
        bins = [1/(2*len(mean_calibration)) + i/len(mean_calibration) for i in range(len(mean_calibration))]
        if plot:
            fig, ax = plt.subplots(2, figsize=(7,7))
            ax[0].grid(True, linestyle='--')
            ax[1].grid(True, linestyle='--')
            xx = yy = np.around(np.linspace(0,1 ,10),1)
            ax[0].plot(xx, yy, color='red', label='Ideal')
            width = (bins[1] - bins[0])*0.9
            ax[0].bar(x=bins, width=width, height=mean_calibration,yerr=error_calibration ,align='center', label='Model')
            ax[1].bar(x=bins, width=width, height= mean_samples,yerr=error_samples,align='center')
            title = exp_name + '_' + model
            if base == 'base':
                title = title + '_base'
            elif base == 'base_fc':
                title = title + '_base_fc'
            ax[0].set_title(title)
            ax[0].set_ylabel('Accuracy')
            ax[1].set_ylabel('p% of samples per bin')
            ax[0].legend()
            xxx = np.arange(0,1.1,0.1)
            ax[0].set_xticks(xxx)
            ax[1].set_xticks(xxx)
            plt.show()
            if save_dir is not None:      
                save_path = os.path.join(save_dir, 'calD_{}.png'.format(title))
                fig.savefig(save_path, bbox_inches='tight')
    cal_mse, cal_mae = None, None #Necessary to return calibration errors ? 
    return cal_mse, cal_mae

def plot_loss_mesh(all_model_results, exp_name, split_to_show=0, start_epoch=0, save_dir=None, mark=True, result_set='val'):
    assert(result_set in ['train', 'val']), 'result_set must either be train or val'
    if mark:
        fig, ax = plt.subplots(1, 3, figsize=(16, 8), sharex=True)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for model in all_model_results:    
        losses = [epoch_results['loss'] for epoch_results in all_model_results[model][result_set][split_to_show]]
        if mark:
            losses_density = [epoch_results['log ground density'] for epoch_results in all_model_results[model][result_set][split_to_show]]
            losses_marks = [epoch_results['log mark density'] for epoch_results in all_model_results[model][result_set][split_to_show]]
            losses_window = [epoch_results['window integral'] for epoch_results in all_model_results[model][result_set][split_to_show]]
            #check_val = -np.array(losses_densityl) - np.array(losses_marks) + np.array(losses_window)
            ax[0].plot(-np.array(losses_density[start_epoch:-1]), label='{}'.format(model))
            ax[1].plot(-np.array(losses_marks[start_epoch:-1]), label='{}'.format(model))
            ax[2].plot(np.array(losses[start_epoch:-1]),label='{}'.format(model))
        else:
            ax.plot(np.array(losses[start_epoch:-1]),label='{}'.format(model))
    if mark:
        ax[0].set_ylabel('Density NLL')
        ax[1].set_ylabel('Mark NLL')
        ax[2].set_ylabel('Total NLL')
        ax[2].legend(bbox_to_anchor=(1.1, 1.05))
        for axis in ax:
            axis.set_xlabel('Epochs')
    else:
        ax.legend(bbox_to_anchor=(1.1, 1.05))
        ax.set_ylabel('Total NLL')
        ax.set_xlabel('Epochs')
    if save_dir is not None:
        save_path = os.path.join(save_dir, 'loss_{}.png'.format(exp_name))
        print(exp_name)
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


def continuous_calibration_plot(all_model_results, exp_name, num_bins=50, save_dir=None, result_set='test', seq_num=None) : 
    assert(result_set in ['train', 'val', 'test']), 'result_set must either be train, val or test'
    fig, ax = plt.subplots(figsize=(8,8))
    ax.grid(True, linestyle='--')
    for model in all_model_results:
        model_results = all_model_results[model][result_set]
        bins, mean_calibration, std_error_calibration = calibration(model_results, num_bins=num_bins, result_set=result_set, seq_num=seq_num)
        ax.errorbar(bins,mean_calibration, yerr=std_error_calibration, marker='o', label=model, markersize=4)
    xx = yy = np.around(np.linspace(0,1,10),1)
    ax.plot(xx, yy, color='black', marker = '_', label='Ideal')
    ax.set_xlabel('True quantiles')
    ax.set_ylabel('Predicted quantiles')
    ax.set_title(exp_name)
    ax.legend()
    if save_dir is not None:
        print(exp_name)
        save_path = os.path.join(save_dir, 'calC_{}.png'.format(exp_name))
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()

def calibration(model_results, num_bins, result_set='test', seq_num=None, tabular=False):
    if tabular:
        cdf = model_results['cdf']
        cdf = np.array(cdf, dtype=object)
        cdf = np.array([item for seq in cdf for item in seq])
        bins, x = compute_calibration_bins(cdf, num_bins)
        return bins, x     
    else:
        all_split_x = []
        for split in model_results:
            if result_set == 'test':
                cdf = model_results[split]['cdf']
            else:
                cdf = model_results[split][-1]['cdf']
            cdf = np.array(cdf, dtype=object)
            if seq_num is None:
                cdf = np.array([item for seq in cdf for item in seq])
            else:
                cdf = np.array(cdf[seq_num])
            bins, x  = compute_calibration_bins(cdf, num_bins)
            all_split_x.append(x)
        all_split_x = np.array(all_split_x)
        mean_calibration = np.mean(all_split_x, axis=0)
        std_calibration = np.std(all_split_x, axis=0)
        std_error_calibration = std_calibration/np.sqrt(all_split_x.shape[0])
        return bins, mean_calibration, std_error_calibration


def compute_calibration_bins(cdf, num_bins):
    bins = [i / num_bins for i in range(1, num_bins + 1)]
    counts_cdf = []
    for i, bin in enumerate(bins):
        cond = cdf <= bin
        counts_cdf.append(cond.sum())
    x = np.array([count / len(cdf) for count in counts_cdf])
    return bins, x

#TO BE TESTED
def plot_calibration_per_seq(results, exp_name, num_bins=50):
    fig, ax = plt.subplots(figsize=(15,15))
    all_split_x = []
    bins = [i / num_bins for i in range(1, num_bins + 1)]
    all_counts_cdf = []
    cdf = results['cdf']
    #print(cdf)
    cdf = np.array(cdf, dtype=object)
    for seq in cdf:
        if len(seq) >= 0:
            seq = np.array(seq)
            print(seq)
            counts_cdf = []
            for i, bin in enumerate(bins):
                cond =  seq <= bin
                counts_cdf.append(cond.sum())
            x = np.array([count / len(seq) for count in counts_cdf])
            ax.plot(bins,x)
    xx = yy = np.around(np.linspace(0,1,10),1)
    ax.plot(xx, yy, color='black', marker = '_', label='Ideal')
    ax.set_xlabel('True quantiles')
    ax.set_ylabel('Predicted quantiles')
    #ax.set_title('Test calibration for {}'.format(dataset))
    ax.legend()
    #if save:
    #    fig.savefig('results/figures/calibration/calC_{}.png'.format(exp_name[:-2]), bbox_inches='tight')
    plt.show()

def discrete_calibration_error(d_cal):
    bins = [1/(2*len(d_cal)) + i/len(d_cal) for i in range(len(d_cal))]
    c_mse = np.mean((np.array(d_cal)-bins)**2, axis=-1) 
    c_mae = np.mean(np.abs(np.array(d_cal)-bins), axis=-1)
    return c_mse, c_mae


def history_sequences_histograms(data_dir, dataset, max_bins=None):
    fig, ax = plt.subplots()
    data_path = os.path.join(data_dir, dataset)
    data_path = os.path.join(data_path, dataset + '.json')
    counts = []
    with open(data_path, 'r') as f:
        data = json.load(f)
    seq_lengths = np.array([len(seq) for seq in data])
    if max_bins is None:
        max_bins = np.max(seq_lengths)
        bins = np.arange(1,max_bins)
    else:
        bins = np.arange(1,max_bins+1)
    for bin in bins:
        complete_histories = seq_lengths - bin 
        complete_histories[complete_histories < 0] = 0
        counts.append(np.sum(complete_histories))
    width = (bins[1] - bins[0])*0.9
    bins = [str(bin) for bin in bins]
    ax.bar(x=bins, width=width, height=counts,align='center')
    ax.set_title(dataset)
    plt.show()

def seq_histogram(data_dir, dataset):
    fig, ax = plt.subplots()
    data_path = os.path.join(data_dir, dataset)
    data_path = os.path.join(data_path, dataset + '.json')
    with open(data_path, 'r') as f:
        data = json.load(f)
    seq_lenghts = [len(seq) for seq in data]
    seq_mask = np.array(seq_lenghts) <= 1000
    seq_lenghts = np.array(seq_lenghts)[seq_mask]
    print(sum(np.array(seq_lenghts) > 1000))
    num_bins = 100
    ax.hist(x=seq_lenghts, bins=num_bins, align='mid')
    ax.set_title(dataset)
    plt.show()


def plot_sequences_distribution(datasets, data_dir, mark=True, log=True):
    matplotlib.rcParams['figure.figsize'] = (20, 10)
    bins = 50
    for dataset in datasets:
        keys_time = ['times', 'dataset', 'seq']
        keys_mark = ['mark', 'prop', 'seq']
        data_time = dict.fromkeys(keys_time)
        data_mark = dict.fromkeys(keys_mark)
        data_path = os.path.join(data_dir, dataset)
        data_path = os.path.join(data_path, dataset + '.json')
        with open(data_path, 'r') as f:
            data = json.load(f)
            seq_lens = np.array([len(seq) for seq in data])
            seq_mask = seq_lens >= 5
            sequences = np.array(data)[seq_mask]
            sequences = sequences[:10]
        len_seq = [len(seq) for seq in sequences]
        print(len_seq)
        seq_times = [[0] + [event["time"] for event in seq] for seq in sequences]
        inter_seq_times = [[seq[i+1] - seq[i] for i in range(len(seq)-1)] for seq in seq_times]
        seq_index = [[i]*len(seq) for i, seq in enumerate(inter_seq_times)]
        inter_seq_times = np.concatenate(inter_seq_times).ravel()
        if log:
            inter_seq_times = np.log(inter_seq_times)
        seq_index = np.concatenate(seq_index).ravel()
        data_time['times'] = inter_seq_times
        data_time['seq'] = seq_index
        data_time['dataset'] = [dataset] * len(seq_index)
        if mark:
            mark_seqs = np.array([[str(event['labels'][0]) for event in seq] for seq in sequences])
            bins = np.array([str(i) for i in range(50)])
            all_marks, all_props, all_idxs = np.array([]), np.array([]), np.array([])
            for i, mark_seq in enumerate(mark_seqs):
                unique, counts = np.unique(mark_seq, return_counts=True)
                prop = counts/len(mark_seq)
                missing_marks = np.setdiff1d(bins, unique)
                missing_props = np.zeros(len(missing_marks))
                all_seq_marks = np.append(unique, missing_marks)
                all_seq_props = np.append(prop, missing_props)
                seq_idx = np.array([i] * len(bins))
                all_marks = np.append(all_marks, all_seq_marks)
                all_props = np.append(all_props, all_seq_props)
                all_idxs = np.append(all_idxs, seq_idx)
            data_mark['mark'] = all_marks
            data_mark['prop'] = all_props
            data_mark['seq'] = all_idxs
            df_all_mark = pd.DataFrame(data=data_mark)
        dataset = map_datasets_name(dataset)
        df_all_time = pd.DataFrame(data=data_time)
        sns.set_theme(style="white", rc={"font.size":20, "axes.facecolor": (0, 0, 0, 0), 'axes.linewidth':2})
        palette = sns.color_palette("Set2", 12)
        g1 = sns.FacetGrid(df_all_time, palette=palette, row="seq", hue="seq", aspect=20, height=0.7)
        g1.map_dataframe(sns.kdeplot, x="times", fill=True, alpha=0.7)
        #g1.map(label, "dataset")
        g1.fig.subplots_adjust(hspace=-.9)
        g1.despine(left=True)
        g1.set_titles("")
        g1.fig.suptitle(dataset, fontsize=30)
        g1.set_xticklabels(fontsize = 40)
        g1.set(yticks=[], xlabel=" " , ylabel=" ")
        if dataset in ['Wikipedia', 'Yelp Toronto', 'Retweets']:
            ax = plt.gca()
            ax.set_xticks(ax.get_xticks()[::2])
        g1.savefig('figures/dataset/time_seq_{}.png'.format(dataset) , bbox_inches='tight') 
        if mark:
            bins = [str(i) for i in range(50)]
            g2 = sns.FacetGrid(df_all_mark, palette=palette, row="seq", hue="seq", aspect=20, height=0.7)
            g2.map_dataframe(sns.barplot, x='mark', y='prop', fill=True, alpha=0.7)
            g2.fig.subplots_adjust(hspace=-.9)
            g2.set_titles("")
            g1.fig.suptitle(dataset, fontsize=30)
            g2.set_xticklabels(fontsize = 40, color='white')
            g2.set(yticks=[], xlabel=" " , ylabel=" ")
            g2.despine( left=True)
            g2.savefig('figures/dataset/mark_seq_{}.png'.format(dataset) , bbox_inches='tight') 
    plt.show()

def plot_times_distributions(datasets, data_dir, x_lim=None, log=True, mark=True):
    df_all = pd.DataFrame()
    plt.figure(figsize=(30,30))
    for dataset in datasets:
        keys = ['times', 'dataset']
        if mark:
            keys.append('mark')
        data = dict.fromkeys(keys)
        data_path = os.path.join(data_dir, dataset)
        data_path = os.path.join(data_path, dataset + '.json')
        with open(data_path, 'r') as f:
            sequences = json.load(f)
        seq_times = [[0] + [event["time"] for event in seq] for seq in sequences]
        inter_seq_times = [[seq[i+1] - seq[i] for i in range(len(seq)-1)] for seq in seq_times]
        
        inter_seq_times = np.concatenate(inter_seq_times).ravel()
        if log:
            inter_seq_times = np.log(inter_seq_times)
        data['times'] = inter_seq_times
        if mark:
            marks = np.array([str(event['labels'][0]) for seq in sequences for event in seq])
            data['mark'] = marks
        dataset = map_datasets_name(dataset)
        data['dataset'] = dataset
        df = pd.DataFrame(data=data)
        df_all = pd.concat([df_all, df], axis=0)
    def label(x, color, label):
        ax = plt.gca()
        ax.text(.1, .4, label, color='black', fontsize=40,
            ha="left", va="center", transform=ax.transAxes)
       
    sns.set_theme(style="white", rc={"font.size":20, "axes.facecolor": (0, 0, 0, 0), 'axes.linewidth':2})

    palette = sns.color_palette("Set2", 12)
    g1 = sns.FacetGrid(df_all, palette=palette, row="dataset", hue="dataset", aspect=9)
    g1.map_dataframe(sns.kdeplot, x="times", fill=True, alpha=0.7)
    g1.map(label, "dataset")
    g1.fig.subplots_adjust(hspace=-.4)
    g1.set_titles("")
    g1.set_xticklabels(fontsize = 40)
    g1.set(yticks=[], xlabel=" " , ylabel=" ")
    g1.despine( left=True)
    if mark:
        g1.savefig('figures/dataset/continuous_dis_mark.png' , bbox_inches='tight')
        g2 = sns.FacetGrid(df_all, palette=palette, row="dataset", hue="dataset", aspect=9)
        g2.map_dataframe(sns.histplot, x='mark', stat='probability', fill=True, alpha=0.7)
        g2.fig.subplots_adjust(hspace=-.4)
        g2.set_titles("")
        g2.set_xticklabels(fontsize = 40, color='white')
        g2.set(yticks=[], xlabel=" " , ylabel=" ")
        g2.despine( left=True)
        g2.savefig('figures/dataset/discrete_dis.png' , bbox_inches='tight') 
    else:
        title = 'figures/dataset/continuous_dis_unmark_' + datasets[0] + '.png'
        g1.savefig(title , bbox_inches='tight') 
    plt.show()

def map_datasets_name(dataset):
    mapping = {'lastfm_filtered': 'LastFM',
               'mooc_filtered': 'MOOC',
               'wikipedia_filtered': 'Wikipedia',
               'github_filtered': 'Github',
               'mimic2_filtered': 'MIMIC2',
               'stack_overflow_filtered': 'Stack O.',
               'retweets_filtered':'Retweets',
               'taxi': 'Taxi',
               'twitter': 'Twitter',
               'reddit_askscience_comments': 'Reddit Comments',
               'reddit_politics_submissions': 'Reddit Subs.',
               'pubg': 'PUBG',
               'yelp_toronto': 'Yelp Toronto',
               'yelp_mississauga': 'Yelp Mississauga',
               'yelp_airport': 'Yelp Airport', 
               'hawkes_exponential_mutual': 'Hawkes',
               'hawkes_exponential_mutual2': 'Hawkes2',
               'hawkes_exponential_mutual3': 'Hawkes3'}
    return mapping[dataset]

def show_nll_weights(result_dir, model, dataset, split):
    model_file = find_model_file(result_dir, model, dataset, split)
    with open(model_file, 'rb') as f:
        results = pickle.load(f)
    time_weights = [results['train'][i]['nll_weights'].cpu().numpy()[0] for i in range(len(results['train'])-1)]
    mark_weights = [results['train'][i]['nll_weights'].cpu().numpy()[1] for i in range(len(results['train'])-1)]
    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(time_weights, label='Time')
    #ax.plot(mark_weights, label='Mark')
    dataset_name = map_datasets_name(dataset)
    model_name = get_acronym([model])[0]
    fig.suptitle('{} {}'.format(dataset_name, model_name), fontsize=30, y=0.95)
    ax.set_xlabel('Epoch', fontsize=30)
    ax.set_ylabel(r"$\alpha_T$", fontsize=30)
    ax.legend()
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.tick_params(axis='both', which='minor', labelsize=30)
    fig.savefig('figures/weights/WDA_{}_{}'.format(dataset_name, model_name), bbox_inches='tight')


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

def var_mixtures(result_dir, model, dataset, n_mix, split=0):
    losses_time, losses_mark = [], []
    for mix in n_mix:
        model_mix = model + f'_mixture{mix}' 
        file = find_model_file(result_dir, model_mix, dataset, split)
        with open(file, 'rb') as f:
            results = pickle.load(f)
        loss_time = -results['test']['log ground density']
        loss_mark = -results['test']['log mark density']
        losses_time.append(loss_time)
        losses_mark.append(loss_mark)
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].plot(n_mix, losses_time)
    ax[0].set_title('NLL-T')
    ax[1].plot(n_mix, losses_mark)
    ax[1].set_title('NLL-M')
    model = get_acronym([model])[0]
    fig.suptitle(f'{model}, {dataset}')
    plt.show()

