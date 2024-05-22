import os 
import pickle 
import numpy as np
import pandas as pd
import re 
import json
from collections import defaultdict

from plot.acronyms import get_acronym
from plot.plots import calibration, discrete_calibration_error
from tpp.utils.metrics import weighted_metrics

def f1_most_present_class(dataset, split, data_dir='data/baseline3'):
    dataset_path = os.path.join(data_dir, dataset)
    split_path = os.path.join(dataset_path, 'split_{}'.format(split))
    train_path = os.path.join(split_path, 'train.json')
    test_path = os.path.join(split_path, 'test.json')
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    marks = np.array([event["labels"][0] for seq in train_data for event in seq])
    cat, counts = np.unique(marks, return_counts=True)
    n_class = len(cat)
    idx_most_present_mark = counts.argsort()[::-1][0]
    most_present_class = cat[idx_most_present_mark]
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    test_marks = np.array([event["labels"][0] for seq in test_data for event in seq])
    pred = np.ones(len(test_marks)) * most_present_class
    pre_w, rec_w, f1_w, _ = weighted_metrics(pred, test_marks, n_class)                
    return pre_w, rec_w, f1_w


def calibration_errors(c_cal, y):
    c_mse = np.mean((np.array(c_cal)-y)**2, axis=-1) 
    c_mae = np.mean(np.abs(np.array(c_cal)-y), axis=-1)
    return c_mse, c_mae

 
def rank(df, models, per_dataset=False, result_type='unmarked', std_vals=False):
    df_dic = {}
    for i, column in enumerate(df.columns):
        if not 'U.' in column[-1]:
            if std_vals is False:
                res = np.array([float(re.sub('\(([^\)]+)\)', '', val)) for val in df[column].values]) 
            else:
                res = np.array([val for val in df[column].values]) 
            idx = res.argsort()
            if not per_dataset:
                if not ('NLL' in column[-1]) and not ('Cal.' in column[-1]):
                    idx = idx[::-1]
                res_sorted = df[column].values[idx]
                models_sorted = np.array(models)[idx]
                df_dic.update({(column[0],str(i)):models_sorted, (column[0], column[1]):res_sorted})
            else:
                if not ('NLL' in column[0]) and not ('Cal.' in column[0]):
                    idx = idx[::-1]
                ranks = idx.argsort() + 1
                df_dic.update({(column[0], column[1], 'R'):ranks, (column[0], column[1], 'Val'):df[column].values})
    df_ranked = pd.DataFrame(df_dic, columns=pd.MultiIndex.from_tuples(df_dic.keys()))
    return df_ranked

def average_ranks_tabular(models, datasets_time, datasets_mark,  results_dir):
    df_av = pd.DataFrame()
    df_time = get_test_loss_results_v4(results_dir=results_dir, datasets=datasets_time, models=models, n_s=5, return_ranked_dic=True, per_dataset=True, result_type='unmarked')
    df_mark = get_test_loss_results_v4(results_dir=results_dir, datasets=datasets_mark, models=models, n_s=5, return_ranked_dic=True, per_dataset=True, result_type='marked')
    dfs = [df_time, df_mark]
    for df in dfs:
        columns_to_drop = [column for column in df.columns if 'R' not in column]
        df.drop(columns_to_drop, inplace=True, axis=1)
        metrics = [column[0] for column in df.columns]
        metrics = list(set([column[0] for column in df.columns]))
        for metric in metrics:
            cols = [column for column in df.columns if metric in column]
            av = pd.DataFrame(df[cols].mean(axis=1))
            av = av.rename(columns={0:metric})
            df_av = pd.concat([df_av, av], axis=1) 
    df_av = df_av.round(decimals=2)
    build_df_tex(df=df_av, models=models, to_rank=False)

def highlight(df):
    for i, column in enumerate(df.columns):
        res = np.array([float(re.sub('\(([^\)]+)\)', '', val)) for val in df[column].values]) 
        if 'NLL' in column[-2] or 'Cal.' in column[-2]:
            idx_max = np.argmin(res)
            idx_sec = res.argsort()[1]
            idx_th = res.argsort()[2]
        else:
            idx_max = np.argmax(res)
            idx_sec = res.argsort()[-2]
            idx_th = res.argsort()[-3]

        df[column][idx_max] = '\\textbf{{{0}}}'.format(df[column][idx_max])
        df[column][idx_sec] = '$\\text{{{}}}^*$'.format(df[column][idx_sec]) 
        df[column][idx_th] = '$\\text{{{}}}^{{**}}$'.format(df[column][idx_th]) 
    #high_df = pd.concat([high_df, df_components], axis=0)
    return df


def highlight_v2(df, components_group): 
    high_df = pd.DataFrame()
    for components in components_group:
        print(components)
        df_components = df[df["models"].isin(components)]
        df_components.reset_index(drop=True, inplace=True)
        for i, column in enumerate(df_components.columns[1:]):
            if str(i) not in column:
                res = df_components[column]
                #res = np.array([float(re.sub('\(([^\)]+)\)', '', val)) for val in df_components[column].values]) 
                if 'NLL' in column[-2] or 'Cal.' in column[-2]:
                    idx_max = np.argmin(res)
                    #idx_sec = res.argsort()[1]
                    #idx_th = res.argsort()[2]
                else:
                    idx_max = np.argmax(res)
                    #idx_sec = res.argsort()[-2]
                    #idx_th = res.argsort()[-3]
                #print(column)
                #print(idx_max)
                #print(df_components[column][idx_max])
                df_components[column][idx_max] = '\\textbf{{{0}}}'.format(df_components[column][idx_max])
                #df[column][idx_sec] = '$\\text{{{}}}^*$'.format(df[column][idx_sec]) 
                #df[column][idx_th] = '$\\text{{{}}}^{{**}}$'.format(df[column][idx_th]) 
        high_df = pd.concat([high_df, df_components], axis=0)
    return high_df


def get_dividing_NLL_constant(results_dic):
    constant_per_dataset = dict.fromkeys(results_dic.keys())
    for dataset, metrics_dic in results_dic.items():
        nlls = np.array(metrics_dic['NLL Total'])
        print(nlls)
        nll_max = np.max(np.max(nlls))
        print(nll_max)
        constant_per_dataset[dataset] = nll_max
    return constant_per_dataset

def get_mean_std(results_dic, int_to_str=True, std=True):
    for k1, v1 in results_dic.items():
        for k2, v2 in v1.items():
            means = []
            for i, model_res in enumerate(v2):
                model_res = np.array(model_res)
                mean = np.mean(model_res)
                std_val = np.std(model_res)
                ste = std_val/np.sqrt(len(model_res))
                if int_to_str:
                    if 'NLL' in k2:
                        if std:
                            means.append('{} ({})'.format(round(mean, 1), round(ste, 1)))
                        else:
                            means.append('{}'.format(round(mean, 2)))
                    elif 'C.' in k2:
                        if std: 
                            means.append('{} ({})'.format(format(mean,'.1E'), format(std, '.1E')))
                        else:
                            means.append('{}'.format(format(mean,'.1E')))
                    elif 'D.' in k2:
                        if std:
                            means.append('{} ({})'.format(format(mean,'.1E'), format(std, '.1E')))
                        else:
                            means.append('{}'.format(format(mean,'.1E')))
                    else:
                        if std:
                            means.append('{} ({})'.format(round(mean, 2), round(ste, 2)))
                        else:
                            means.append('{}'.format(round(mean, 3)))
                else:
                    means.append(mean)
            results_dic[k1][k2] = means
    return results_dic

def initialize_results_dic_v2(n_m, datasets):
    
    all_dic_dataset = dict.fromkeys(datasets)
    for key in all_dic_dataset.keys():
        if get_dataset_types(key) == 'marked':
            all_dic_dataset[key] = {'NLL-T':[[] for s in range(n_m)], 
                                    'PCE':[[] for s in range(n_m)],
                                    'NLL-M':[[] for s in range(n_m)],                                       
                                    'ECE':[[] for s in range(n_m)],
                                    'Acc': [[] for s in range(n_m)], 
                                    'Acc@3':[[] for s in range(n_m)],
                                    'Acc@5':[[] for s in range(n_m)],
                                    'Acc@10':[[] for s in range(n_m)],
                                    'MRR':[[] for s in range(n_m)],
                                    'F1-score':[[] for s in range(n_m)]
                                    }
        elif get_dataset_types(key) == 'unmarked':
            all_dic_dataset[key] = {'NLL Time':[[] for s in range(n_m)], 
                                    'Time Cal.':[[] for s in range(n_m)]
                                    }    
    return all_dic_dataset

def initialize_results_dic(result_type, n_m, datasets):
    all_dic_dataset = dict.fromkeys(datasets)
    for key in all_dic_dataset.keys():
        if result_type == 'marked':
            all_dic_dataset[key] = {'NLL Time':[[] for s in range(n_m)], 
                                    #'NLL Time U.':[[] for s in range(n_m)],
                                    'NLL Mark':[[] for s in range(n_m)],
                                    'NLL Total':[[] for s in range(n_m)],                                        
                                    'Time Cal.':[[] for s in range(n_m)],
                                    #'Time Cal. U.':[[] for s in range(n_m)],
                                    'Mark Cal.':[[] for s in range(n_m)],
                                    'F1-score':[[] for s in range(n_m)]
                                    }
            
        elif result_type == 'unmarked':
            all_dic_dataset[key] = {'NLL Time':[[] for s in range(n_m)], 
                                    'Time Cal.':[[] for s in range(n_m)]}    
        else:
            all_dic_dataset[key] = {'NLL Time':[[] for s in range(n_m)], 
                                    'NLL Mark':[[] for s in range(n_m)],
                                    'NLL Window':[[] for s in range(n_m)],
                                    'NLL Total':[[] for s in range(n_m)], 
                                    'Time Cal.': [[] for s in range(n_m)],
                                    'Mark Cal.': [[] for s in range(n_m)], 
                                    'F1-score':[[] for s in range(n_m)], 
                                    } 
    return all_dic_dataset

def get_test_loss_results_v4(results_dir, datasets, models, result_type='marked', n_s=5, to_rank=False, return_dic=False, per_dataset=False, return_ranked_dic=False, std=True):
    assert(result_type in ['marked', 'unmarked', 'simu']), 'result_ must be marked, unmarked or simu'
    simu = True if result_type == 'simu' else False
    n_m = len(models) 
    all_dic_dataset = initialize_results_dic_v2(n_m, datasets)
    for m, model in enumerate(models):
        for dataset in datasets:
            if 'mark_only' in model or 'time_only' in model:
                file_dir = os.path.join('results/mark_training', dataset)
            else:
                file_dir = os.path.join(results_dir, dataset)
            file_names = os.listdir(file_dir)
            for s in range(n_s):
                if 'base' in model:
                    if 'base_fc' in model:
                        file_to_find = 'poisson_coefficients_' +  dataset + '_' + model.replace('_base_fc', '') + '_split' + str(s)
                    elif 'lnmk1' in model:
                        model_to_find = model.split('_lnmk1')[0]
                        file_to_find = 'poisson_' + dataset + '_' + model_to_find + '_split' + str(s) + '_lnmk1'
                    elif 'evaluation' in model:
                        file_to_find = 'poisson_' +  dataset + '_' + model.replace('_base_evaluation', '') + '_split' + str(s) + '_evaluation'
                    else:
                        file_to_find = 'poisson_' +  dataset + '_' + model.replace('_base', '') + '_split' + str(s) 
                elif 'lnmk1' in model:
                    model_to_find = model.split('_lnmk1')[0]
                    file_to_find = dataset + '_' + model_to_find + '_split' + str(s) + '_lnmk1'  
                else:
                    if 'evaluation' in model:
                        file_to_find =  dataset + '_' + model.replace('_evaluation', '') + '_split' + str(s) + '_evaluation'
                    else:
                        file_to_find = dataset + '_' + model + '_split' + str(s) 
                path = None
                for file_name in file_names: 
                    if file_name.startswith(file_to_find):
                        path = os.path.join(file_dir, file_name)
                        all_dic_dataset = fill_result_dict(path, all_dic_dataset, dataset, m, file_result_='original', simu=simu)
                        break
                if path is None:
                    #print(model)
                    #print(file_dir)
                    #print(file_names)
                    print(file_to_find)
                    raise ValueError('File not found')
                if result_type == 'marked' and return_dic is False and per_dataset is False:
                    for file_name in file_names_un:
                        if file_name.startswith(file_to_find):
                            print(file_name)
                            path = os.path.join(file_dir_un, file_name)
                            all_dic_dataset = fill_result_dict(path, all_dic_dataset, dataset, m, file_result_='unmarked', simu=simu)
    if return_dic:
        all_dic_dataset = get_mean_std(all_dic_dataset, int_to_str=False, std=std)
        return all_dic_dataset
    else:
        all_dic_dataset = get_mean_std(all_dic_dataset, int_to_str=True, std=std)
        #all_dic_dataset = std_values(all_dic_dataset, robust=True)
    if not per_dataset:
        for dataset, metrics_dic in all_dic_dataset.items():
            for metric, val_list in metrics_dic.items():
                cols.append((dataset, metric))
                data.append(val_list)
        df = pd.DataFrame(list(zip(*data)), columns=pd.MultiIndex.from_tuples(cols))
        build_df_tex(df, models, to_rank, result_type)
    else:
        metrics = all_dic_dataset[list(all_dic_dataset.keys())[0]].keys()
        if return_ranked_dic:
            df_ranked = pd.DataFrame()
        for metric in metrics:
            #if result_type == 'marked' and metric not in ['NLL Mark', 'F1-score']:
            #    continue
            cols, data = [], []
            for dataset, metrics_dic in all_dic_dataset.items():
                dataset = map_datasets(dataset)
                cols.append((metric, dataset))
                data.append(metrics_dic[metric])
            df = pd.DataFrame(list(zip(*data)), columns=pd.MultiIndex.from_tuples(cols))
            #df = rank(df, model, per_dataset=True, result_type=result_type, std_vals=True)
            if return_ranked_dic:
                df_ranked = pd.concat([df_ranked, df], axis=1)  
            else:
                build_df_tex(df, models, to_rank)
        if return_ranked_dic:
            return df_ranked

def map_datasets(dataset):
    mapping = {
        'lastfm_filtered': 'LastFM',
        'lastfm': 'LastFM',
        'mooc_filtered':'MOOC',
        'mooc':'MOOC',
        'wikipedia_filtered':'Wikipedia',
        'reddit_filtered_short':'Reddit',
        'reddit':'Reddit',
        'mimic2_filtered':'MIMIC2',
        'github_filtered':'Github',
        'stack_overflow_filtered':'Stack Overflow',
        'stack_overflow':'Stack Overflow',
        'retweets_filtered_short': 'Retweets',
        'hawkes_exponential_mutual_large': 'Hawkes',
        'hawkes_exponential_mutual_bis': 'Hawkes',
        'amazon_toys': 'Amazon Toys',
        'amazon_movies': 'Amazon Movies'
    }
    return mapping[dataset]

def get_dataset_types(dataset):
    print(dataset)
    dataset_marked_filtered = ['lastfm_filtered', 'mooc_filtered', 'wikipedia_filtered', 'github_filtered', 'mimic2_filtered', 'hawkes_exponential_mutual_large', 'stack_overflow_filtered', 'retweets_filtered_short', 'reddit_filtered_short', 'amazon_toys', 'amazon_movies', 'hawkes_exponential_mutual_bis']
    dataset_marked_unfiltered = ['lastfm', 'mooc', 'reddit', 'stack_overflow']
    dataset_marked = dataset_marked_filtered + dataset_marked_unfiltered
    
    datasets_unmarked = ['taxi', 'twitter', 'reddit_politics_submissions', 'reddit_askscience_comments', 'pubg', 'yelp_toronto', 'yelp_mississauga', 'yelp_airport']
    if dataset in dataset_marked:
        return 'marked'
    elif dataset in datasets_unmarked:
        return 'unmarked'
    else:
        raise ValueError('Unknown dataset')


def best_model_tabular(results_dir, datasets_groups, models, components=None, tabular=True, separate_marked_unmarked=False, saved_dic_path=None, robust=False):
    datasets_marked, datasets_unmarked = datasets_groups
    models_marked, models_unmarked = models
    components_marked, components_unmarked = components
    if saved_dic_path is not None:
        saved_marked = os.path.join(saved_dic_path, 'results_dic_marked.pkl')
        saved_unmarked = os.path.join(saved_dic_path, 'results_dic_unmarked.pkl')
        df_marked_time, df_marked_mark = summary_tabular(results_dir, datasets_marked, models_marked, n_s=5, 
                                                         components=components_marked, tabular=tabular, marks=True, best_methods=True, saved_dic_path=saved_marked, robust=robust)
        df_unmarked, _ = summary_tabular(results_dir, datasets_unmarked, models_unmarked, n_s=5, components=components_unmarked, tabular=tabular, marks=False, best_methods=True, saved_dic_path=saved_unmarked, robust=robust)
    else:
        df_marked_time, df_marked_mark = summary_tabular(results_dir, datasets_marked, models_marked, n_s=5, 
        components=components_marked, tabular=tabular, marks=True, best_methods=True, robust=robust)
        df_unmarked, _ = summary_tabular(results_dir, datasets_unmarked, models_unmarked, n_s=5, components=components_unmarked, tabular=tabular, marks=False, best_methods=True, robust=robust)
    if tabular:
        df_marked_time, tab_models_time = df_marked_time
        df_marked_mark, tab_models_mark = df_marked_mark
        df_unmarked, tab_models_unmarked = df_unmarked

        df_marked_time = reorder_columns(df_marked_time, marks=True)
        df_marked_mark = reorder_columns(df_marked_mark, marks=True)
        df_unmarked = reorder_columns(df_unmarked, marks=False)    
        if separate_marked_unmarked:
            cols_marked_f, data_marked_time, data_marked_mark, cols_unmarked_f, data_unmarked = [], [], [], [], []
            cols_marked, cols_unmarked = list(df_marked_time.columns), list(df_unmarked.columns)
            for col in cols_marked:
                cols_marked_f.append(('Marked Datasets', col[0], col[1]))
                data_marked_time.append(df_marked_time[col])
                data_marked_mark.append(df_marked_mark[col])
            for col in cols_unmarked:
                cols_unmarked_f.append(('Unmarked Datasets', col[0], col[1]))
                data_unmarked.append(df_unmarked[col])
            df_marked_time = pd.DataFrame(list(zip(*data_marked_time)), columns=pd.MultiIndex.from_tuples(cols_marked_f))
            df_marked_mark = pd.DataFrame(list(zip(*data_marked_mark)), columns=pd.MultiIndex.from_tuples(cols_marked_f))
            df_unmarked = pd.DataFrame(list(zip(*data_unmarked)), columns=pd.MultiIndex.from_tuples(cols_unmarked_f))
            #df_marked_time, df_marked_mark,  df_unmarked = df_marked_time.round(decimals=2),df_marked_mark.round(decimals=2), df_unmarked.round(decimals=2)
            df_marked_time = df_marked_time.astype('float64').round(decimals=2)
            df_marked_mark  = df_marked_mark.astype('float64').round(decimals=2)
            df_unmarked = df_unmarked.astype('float64').round(decimals=2)
            #df_marked_time.style.set_precision(2)
            #df_marked_mark.style.set_precision(2)
            #df_unmarked.style.set_precision(2)
            build_df_tex(df_marked_time, tab_models_time, component=True, to_rank=False, components_group=[tab_models_time])
            build_df_tex(df_marked_mark, tab_models_mark, component=True, to_rank=False, components_group=[tab_models_mark])
            build_df_tex(df_unmarked, tab_models_unmarked, component=True, to_rank=False, components_group=[tab_models_unmarked])
    else:
        return df_marked_time, df_marked_mark, df_unmarked

def summary_tab_per_type(results_dir, datasets_groups, models, components=None, tabular=True, separate_marked_unmarked=False, saved_dic_path=None, robust=False):
    datasets_marked, datasets_unmarked = datasets_groups
    components_marked, components_unmarked = components
    models_marked, models_unmarked = models
    if saved_dic_path is not None:
        saved_dic_path_marked = os.path.join(saved_dic_path, 'results_dic_marked.pkl')
        saved_dic_path_unmarked = os.path.join(saved_dic_path, 'results_dic_unmarked.pkl')
    else:
        saved_dic_path_marked = None
        saved_dic_path_unmarked = None
    df_marked = summary_tabular(results_dir, datasets_marked, models_marked, n_s=5, components=components_marked, tabular=tabular, marks=True, saved_dic_path=saved_dic_path_marked, robust=robust)
    df_unmarked = summary_tabular(results_dir, datasets_unmarked, models_unmarked, n_s=5, components=components_unmarked, tabular=tabular, marks=False, saved_dic_path=saved_dic_path_unmarked, robust=robust)
    if tabular: 
        df_marked = reorder_columns(df_marked, marks=True)
        df_unmarked = reorder_columns(df_unmarked, marks=False)
        if separate_marked_unmarked:
            cols_marked_f, data_marked, cols_unmarked_f, data_unmarked = [], [], [], []
            cols_marked, cols_unmarked = list(df_marked.columns), list(df_unmarked.columns)
            for col in cols_marked:
                cols_marked_f.append(('Marked Datasets', col[0], col[1]))
                data_marked.append(df_marked[col])
            for col in cols_unmarked:
                cols_unmarked_f.append(('Unmarked Datasets', col[0], col[1]))
                data_unmarked.append(df_unmarked[col])
            df_marked = pd.DataFrame(list(zip(*data_marked)), columns=pd.MultiIndex.from_tuples(cols_marked_f))
            df_unmarked = pd.DataFrame(list(zip(*data_unmarked)), columns=pd.MultiIndex.from_tuples(cols_unmarked_f))
            df_marked, df_unmarked = df_marked.round(decimals=2), df_unmarked.round(decimals=2)
            components_marked_flat = [comp for comp_group in components_marked for comp in comp_group]
            components_unmarked_flat = [comp for comp_group in components_unmarked for comp in comp_group]
            build_df_tex(df_marked, components_marked_flat, component=True, to_rank=False, components_group=components_marked)
            build_df_tex(df_unmarked, components_unmarked_flat, component=True, to_rank=False, components_group=components_unmarked)
        else:
            cols, data = [], []
            cols_marked, cols_unmarked = list(df_marked.columns), list(df_unmarked.columns)
            for col in cols_marked:
                cols.append(('Marked Datasets', col[0], col[1]))
                data.append(df_marked[col])
            for col in cols_unmarked:
                cols.append(('Unmarked Datasets', col[0], col[1]))
                data.append(df_unmarked[col])
            df = pd.DataFrame(list(zip(*data)), columns=pd.MultiIndex.from_tuples(cols))
            df = df.round(decimals=2)
            components = [comp for comp_group in components for comp in comp_group]
            build_df_tex(df, components, component=True, to_rank=False)
    else:
        return df_marked, df_unmarked

def reorder_columns(df, marks=True):
    order_marked = ['NLL Time', 'Time Cal.', 'NLL Mark', 'Mark Cal.', 'F1-score']
    order_unmarked = ['NLL Time', 'Time Cal.']
    new_cols = []
    if marks:
        for col in order_marked:
            new_cols.append((col, 'Mean'))
            new_cols.append((col, 'Median'))
            #new_cols.append((col, 'Worst'))
            new_cols.append((col, 'Rank'))
    else:
        for col in order_unmarked:
            new_cols.append((col, 'Mean'))
            new_cols.append((col, 'Median'))
            #new_cols.append((col, 'Worst'))
            new_cols.append((col, 'Rank'))
    df = df[new_cols]
    return df

def best_models_dataframe(models, all_metrics, all_dic_dataset, components, selection_metric='NLL Time', tabular=False):
    all_dic_comp = {component:{metric:None for metric in all_metrics} for component in components}
    models = get_acronym(models)
    for component in components:
        mask = get_comparison_mask(component, models)
        filtered_models = np.array(models)[mask]
        for metric in all_metrics:
            all_dic_comp[component][metric] = {model:[] for model in filtered_models}
        for dataset, dataset_scores in all_dic_dataset.items():
            for metric, metrics_scores in dataset_scores.items():
                values = np.array(metrics_scores)[mask]
                for i, model_value in enumerate(values):
                    all_dic_comp[component][metric][filtered_models[i]].append(model_value)
    for component, component_dic in all_dic_comp.items():
        for metric, metric_dic in component_dic.items():
            for model, model_results in metric_dic.items():
                if model != 'MRC' and np.isnan(model_results).any():
                    print('NAN FOUND', flush=True)
                    print(model, flush=True)
                    print(model_results, flush=True)
                    raise ValueError()                
    best_methods_dic = {metric:{} for metric in all_metrics}
    for component in components:
        selection_metric_dic = all_dic_comp[component][selection_metric]
        model_means = []
        component_models = list(selection_metric_dic.keys())
        for model, model_results in selection_metric_dic.items():
            model_means.append(np.mean(model_results))
        idx = np.argmin(model_means)
        best_model = component_models[idx]
        for metric in all_metrics: 
            best_methods_dic[metric][best_model] = {'vals':None, 'Ranks':None}
            best_methods_dic[metric][best_model]['vals'] = all_dic_comp[component][metric][best_model]
    if tabular is False:
        return best_methods_dic
    else:
        for metric, metrics_dic in best_methods_dic.items():
            all_models_results = []
            for model, model_results in metrics_dic.items():
                all_models_results.append(np.array(model_results['vals']))
            if metric == 'F1-score':
                idx = np.argsort(-np.array(all_models_results), axis=0) 
            else:
                idx = np.argsort(np.array(all_models_results), axis=0) 
            ranks = np.argsort(idx, axis=0) + 1
            models = list(metrics_dic.keys())
            for i, rank in enumerate(ranks):             
                best_methods_dic[metric][models[i]]['Ranks'] = np.mean(rank)
                best_methods_dic[metric][models[i]]['Mean'] = np.mean(best_methods_dic[metric][models[i]]['vals'])
                best_methods_dic[metric][models[i]]['Median'] = np.median(best_methods_dic[metric][models[i]]['vals'])
                if metric == 'F1-score':
                    best_methods_dic[metric][models[i]]['Worst'] = np.min(best_methods_dic[metric][models[i]]['vals'])
                else:
                    best_methods_dic[metric][models[i]]['Worst'] = np.max(best_methods_dic[metric][models[i]]['vals'])
        #tabular_dic = {metric:{'Mean':[], 'Median':[], 'Rank':[]} for metric in all_metrics}
        tabular_dic = {metric:{'Mean':[], 'Median':[], 'Worst':[], 'Rank':[]} for metric in all_metrics}
        for metric, metrics_dic in best_methods_dic.items():
            for model, model_dic in metrics_dic.items():
                tabular_dic[metric]['Mean'].append(best_methods_dic[metric][model]['Mean'])
                tabular_dic[metric]['Median'].append(best_methods_dic[metric][model]['Median'])
                tabular_dic[metric]['Worst'].append(best_methods_dic[metric][model]['Worst'])
                tabular_dic[metric]['Rank'].append(best_methods_dic[metric][model]['Ranks'])
        cols, data = [], []
        for metric, submetrics_dic in tabular_dic.items():
            for submetric, values in submetrics_dic.items():
                cols.append((metric, submetric))
                data.append(values)
        df = pd.DataFrame(list(zip(*data)), columns=pd.MultiIndex.from_tuples(cols))
        return df, models


    

def component_tabular(models, all_metrics, all_dic_dataset, components):
    all_dic_sum = dict.fromkeys(all_metrics)
    models = get_acronym(models)
    for metric in all_metrics:
        av_metrics_list, median_metrics_list, min_metrics_list, av_ranks_list = [], [], [], []
        all_dic_sum[metric] = dict.fromkeys(["Mean", "Median", "Worst", "Rank"])
        for component_group in components:
            mean_metrics_list, ranks_list = [], []
            for dataset, metrics_dic in all_dic_dataset.items(): 
                if metric in metrics_dic:
                    mean_vals = []
                    for component in component_group:
                        mask = get_comparison_mask(component, models)
                        values = np.array(metrics_dic[metric])[mask]
                        mean_vals_per_component = np.nanmean(values)
                        mean_vals.append(mean_vals_per_component)
                    idx = np.argsort(mean_vals)
                    if 'NLL' not in metric and 'Cal' not in metric:
                        idx = idx[::-1] 
                    ranks_per_component_group = idx.argsort() + 1        
                    ranks_list.append(ranks_per_component_group)
                    mean_metrics_list.append(mean_vals)
            #if metric == 'NLL Time':
            #    print(component_group)
            #    print(ranks_list)
            #    print(mean_metrics_list)
            av_ranks_list.extend(np.mean(np.array(ranks_list), axis=0))
            av_metrics_list.extend(np.mean(np.array(mean_metrics_list), axis=0))
            median_metrics_list.extend(np.median(np.array(mean_metrics_list), axis=0))
            if 'NLL' in metric or 'Cal' in metric:
                min_metrics_list.extend(np.max(np.array(mean_metrics_list), axis=0))
            else:
                min_metrics_list.extend(np.min(np.array(mean_metrics_list), axis=0))
        all_dic_sum[metric]["Mean"] = av_metrics_list
        all_dic_sum[metric]["Median"] = median_metrics_list
        all_dic_sum[metric]["Worst"] = min_metrics_list
        all_dic_sum[metric]["Rank"] = av_ranks_list

    cols, data = [], []
    for metric, submetrics_dic in all_dic_sum.items():
        for submetric, values in submetrics_dic.items():
            cols.append((metric, submetric))
            data.append(values)
    df = pd.DataFrame(list(zip(*data)), columns=pd.MultiIndex.from_tuples(cols))
    return df


def stat_dic(models, all_metrics, all_dic_dataset, components):
    all_dic_sum = dict.fromkeys(all_metrics)
    models = get_acronym(models)
    for metric in all_metrics:
        av_metrics_list, av_ranks_list = [], []
        all_components = [comp for component_group in components for comp in component_group]
        all_dic_sum[metric] = dict.fromkeys(all_components)
        for component in all_components:
            mask = get_comparison_mask(component, models)
            mean_comp_value_per_dataset = []
            for dataset, metrics_dic in all_dic_dataset.items():
                if metric in metrics_dic:
                    values = np.array(metrics_dic[metric])[mask]
                    mean_comp_value_per_dataset.append(np.nanmean(values))
            all_dic_sum[metric][component] = mean_comp_value_per_dataset
    return all_dic_sum            
    

def load_results(results_dir, datasets, models, marks, n_s=5):
    n_m = len(models) 
    all_dic_dataset = initialize_results_dic_v2(n_m, datasets)
    for m, model in enumerate(models):
        for dataset in datasets:
            file_dir = os.path.join(results_dir, dataset + '/best')
            file_names = os.listdir(file_dir)
            for s in range(n_s):
                if model == 'MRC' and marks is True:
                    path = None
                    all_dic_dataset = fill_result_dict_v2(path, all_dic_dataset, dataset, m, mrc=True, split=s)
                else:
                    file_to_find = dataset + '_' + model + '_split' + str(s) + '_config'
                    if 'base' in model:
                        if 'base_fc' in model:
                            file_to_find = 'poisson_coefficients_' +  dataset + '_' + model.replace('_base_fc', '') + '_split' + str(s)
                        elif 'lnmk1' in model:
                            model_to_find = model.split('_lnmk1')[0]
                            file_to_find = 'poisson_' + dataset + '_' + model_to_find + '_split' + str(s) + '_lnmk1'
                        else:
                            file_to_find = 'poisson_' +  dataset + '_' + model.replace('_base', '') + '_split' + str(s) + '_config'
                    elif 'lnmk1' in model:
                        model_to_find = model.split('_lnmk1')[0]
                        file_to_find = dataset + '_' + model_to_find + '_split' + str(s) + '_lnmk1'
                    for file_name in file_names: 
                        if file_name.startswith(file_to_find):
                            path = os.path.join(file_dir, file_name)
                            all_dic_dataset = fill_result_dict_v2(path, all_dic_dataset, dataset, m)
    print("RESULTS LOADED")
    if marks:
        with open('results/results_dic/results_dic_marked.pkl', 'wb') as f:
            print('Saved marked results')
            pickle.dump(all_dic_dataset, f)
    else:
        with open('results/results_dic/results_dic_unmarked.pkl', 'wb') as f:
            print('Saved unmarked results')
            pickle.dump(all_dic_dataset, f)
        

def summary_tabular(results_dir, datasets, models, n_s=5, components=None, tabular=True, marks=True, best_methods=False, saved_dic_path=None, robust=False):
    if saved_dic_path is None:
        load_results(results_dir, datasets, models, marks, n_s)
        if marks:
            saved_dic_path = 'results/results_dic/results_dic_marked.pkl'
        else:
            saved_dic_path = 'results/results_dic/results_dic_unmarked.pkl'
    with open(saved_dic_path, 'rb') as f:
        all_dic_dataset = pickle.load(f)
    all_dic_dataset = get_mean_std(all_dic_dataset, int_to_str=False)
    all_dic_dataset = std_values(all_dic_dataset, robust=robust)
    all_metrics = []
    for dataset in datasets:
        metrics = list(all_dic_dataset[dataset].keys())
        all_metrics.extend(metrics)
    all_metrics = list(set(all_metrics))
    if components is not None:
        if best_methods:
            df_time = best_models_dataframe(models, all_metrics, all_dic_dataset, components, tabular=tabular, selection_metric='NLL Time')
            if marks:
                df_mark = best_models_dataframe(models, all_metrics, all_dic_dataset, components, tabular=tabular, selection_metric='NLL Mark')
            else:
                df_mark = None
            return df_time, df_mark
        if tabular:
            return component_tabular(models, all_metrics, all_dic_dataset, components)
        else:
            return stat_dic(models, all_metrics, all_dic_dataset, components)
    else:
        all_dic_sum = dict.fromkeys(all_metrics)
        for metric in all_metrics:
            all_dic_sum[metric] = dict.fromkeys(["Val.", "Rank"])
            metrics_list, ranks_list = [], []
            for dataset, metrics_dic in all_dic_dataset.items():
                if metric in metrics_dic:
                    values = np.array(metrics_dic[metric])
                    idx = np.argsort(values)
                    if 'NLL' not in metric and 'Cal' not in metric:
                        idx = idx[::-1] 
                    ranks = idx.argsort() + 1
                    ranks_list.append(ranks)
                    metrics_list.append(metrics_dic[metric])
            av_ranks_list = np.mean(np.array(ranks_list), axis=0)
            av_metrics_list = np.mean(np.array(metrics_list), axis=0)
            all_dic_sum[metric]["Val."] = av_metrics_list
            all_dic_sum[metric]["Rank"] = av_ranks_list
    if components is None:
        cols, data = [], []
        for metric, submetrics_dic in all_dic_sum.items():
            for submetric, values in submetrics_dic.items():
                cols.append((metric, submetric))
                data.append(values)
        df = pd.DataFrame(list(zip(*data)), columns=pd.MultiIndex.from_tuples(cols))
        build_df_tex(df, models, to_rank=False)
    """
    else:
        dic_component = dict.fromkeys(all_metrics)
        models = get_acronym(models)
        for metric in all_metrics:
            dic_component[metric] = dict.fromkeys(["Val.", "Rank"])
            av_rank_comp, av_val_comp = [], []
            for component in components:
                mask = get_comparison_mask(component, models)
                vals = np.array(all_dic_sum[metric]["Val."])[mask]
                mask_vals = ~np.isnan(vals)
                av_val = np.mean(vals[mask_vals])
                av_val_comp.append(av_val)
                ranks = np.array(all_dic_sum[metric]["Rank"])[mask]
                av_rank = np.mean(ranks[mask_vals])
                av_rank_comp.append(av_rank)
            dic_component[metric]['Val.'] = av_val_comp
            dic_component[metric]['Rank'] = av_rank_comp
        cols, data = [], []
        for metric, submetrics_dic in dic_component.items():
            for submetric, values in submetrics_dic.items():
                cols.append((metric, submetric))
                data.append(values)
        df = pd.DataFrame(list(zip(*data)), columns=pd.MultiIndex.from_tuples(cols))
    return df  
    """    
        #print(df.shape)
        #build_df_tex(df, components, to_rank=False, component=True)


def get_comparison_mask(comparison, models):
    encoding_comparisons = ['-TO', '-LTO', '-CONCAT', '-LCONCAT', '-TEM', '-TEMWL', '-LE', '-LEWL']
    encoder_comparisons = ['GRU-', 'SA-', 'CONS-']
    base_comparisons = ['-B', '-NB']
    decoder_comparisons = ['EC', 'LNM', 'LN', 'FNN', 'MLP/MC', 'RMTPP', 'SA/MC', 'SA/CM', 'Poisson', 'Hawkes', 'MRC', 'NH']
    if any(comp in comparison for comp in base_comparisons):
        if '-NB' in comparison:
            base_model = comparison.rstrip('-NB')
            mask1 = np.array(['+ B' not in model for model in models])
            mask2 = np.array([base_model in model for model in models])
            mask = mask1 * mask2 
        else:
            base_model = comparison.rstrip('-B')
            mask1 = np.array(['+ B' in model for model in models])
            mask2 = np.array([base_model in model for model in models])
            mask = mask1 * mask2 
    elif any(comp in comparison for comp in encoder_comparisons):
        mask = np.array([comparison in model for model in models])
    elif any(comp in comparison for comp in encoding_comparisons):
        mask1 = np.array(['CONS' not in model for model in models])
        #mask_bis = np.array(['SA-' not in model for model in models]) #!!
        mask2 = np.array(['Hawkes' not in model for model in models])
        mask3 = np.array(['Poisson' not in model for model in models])
        mask4 = np.array([comparison in model for model in models])
        mask = mask1 * mask2 * mask3 * mask4 
        if 'TEM' in comparison and 'TEMWL' not in comparison:
            mask1 = np.array(['TEMWL' not in model for model in models])
            mask = mask1 * mask
        elif 'LE' in comparison and 'LEWL' not in comparison:
            mask1 = np.array(['LEWL' not in model for model in models])
            mask = mask1 * mask
    elif any(comp in comparison for comp in decoder_comparisons):
        mask = np.array([comparison in model for model in models])
        if 'LN' in comparison and 'LNM' not in comparison:
            mask1 = np.array(['LNM' not in model for model in models])
            mask = mask1 * mask
    else:
        raise ValueError("Comparison {} not understood".format(comparison))
    return mask  
    """
    if comparison in ['TO', 'LTO', 'TEM', 'TEMWL', 'LE', 'LEWL', 'CONCAT', 'LCONCAT']:
        mask1 = np.array(['CM' not in model for model in models])
        mask2 = np.array(['Hawkes' not in model for model in models])
        mask3 = np.array(['Poisson' not in model for model in models])
        mask4 = np.array([' + B' not in model for model in models])
        mask5 = np.array(['C-RMTPP' not in model for model in models])
        mask6 = np.array(['C-CP' not in model for model in models])
        mask7 = np.array(['C-LNM' not in model for model in models])
        mask8 = np.array(['C-RMTPP' not in model for model in models])
        base_mask = mask1 * mask2 * mask3 * mask4 * mask5 * mask6 * mask7 * mask8
        if comparison == 'TO': #Marked/unmarked
            mask1 = np.array(['TO' in model for model in models])
            mask2 = np.array(['LTO' not in model for model in models])
            mask = mask1 * mask2 * base_mask
        elif comparison == 'LTO': #Marked/Unmarked
            mask1 = np.array(['LTO' in model for model in models])
            mask = mask1 * base_mask
        elif comparison == 'TEM': #Marked/Unmarked
            mask1 = np.array(['TEM' in model for model in models])
            mask0 = np.array(['TEMWL' not in model for model in models])
            mask = mask0 * mask1 * base_mask
        elif comparison == 'LE': #Marked/Unmarked
            mask1 = np.array(['LE' in model for model in models])
            mask0 = np.array(['LEWL' not in model for model in models])
            mask = mask0* mask1 * base_mask
        elif comparison == 'TEMWL': #Marked
            mask1 = np.array(['TEMWL' in model for model in models])
            mask = mask1 * base_mask
        elif comparison == 'LEWL': #Marked
            mask1 = np.array(['LEWL' in model for model in models])
            mask = mask1 * base_mask
        elif comparison == 'CONCAT': #Marked
            mask1 = np.array(['CONCAT' in model for model in models])
            mask2 = np.array(['LCONCAT' not in model for model in models])
            mask = mask1 * mask2 * base_mask
        elif comparison == 'LCONCAT': #Marked
            mask1 = np.array(['LCONCAT' in model for model in models])
            mask = mask1 * base_mask
    elif comparison in ['SA', 'GRU', 'C-']:
        base_mask = np.array([' + B' not in model for model in models])
        if comparison == 'SA':
            mask1 = np.array(['SA' in model for model in models])
            mask2 = np.array(['SA/' not in model for model in models])
            mask = mask1 * mask2 * base_mask
        elif comparison in ['GRU', 'C-']:
            mask = np.array([comparison in model for model in models])
            mask = base_mask * mask
    elif comparison == 'RMTPP':
        mask1 = np.array([comparison in model for model in models])
        mask2 = np.array(['LTO' not in model for model in models])
        mask3 = np.array(['LCONCAT' not in model for model in models])
        mask = mask1 * mask2 * mask3 
    elif comparison in ['CP', 'LNM', 'MLP/MC', 'MLP/CM', 'RMTPP', 'SA/CM', 'SA/MC','Hawkes', 'Poisson', '+ B', 'MRC']:
        mask = np.array([comparison in model for model in models])
    elif comparison == 'No B':
        mask = np.array([' + B' not in model for model in models])
    """
    

def std_values(all_dic_dataset, robust=False):
    for dataset, metrics_dic in all_dic_dataset.items():
        for metric, values in metrics_dic.items():
            if 'NLL' in metric:
                assert(sum(np.isnan(values))==0), 'Nan encountered.'
                values_mask = np.array(values) < 1e9
                values_to_std = np.array(values)[values_mask]
                if robust:
                    median = np.median(values_to_std)
                    quartile_025 = np.quantile(values_to_std, q=0.25)
                    quartile_075 = np.quantile(values_to_std, q=0.75)
                    inter_q_range = np.abs(quartile_075-quartile_025)
                    std_values = (np.array(values)-median)/inter_q_range
                else:
                    mean = np.mean(values_to_std)
                    std = np.std(values_to_std)
                    std_values = (np.array(values)-mean)/std
                all_dic_dataset[dataset][metric] = std_values
    return all_dic_dataset

def build_df_tex(df, models, to_rank, component=False, components_group=None):
    if component is False:
        models = get_acronym(models)
    if to_rank:
        df_rank = rank(df, models)
        columns = 'c' * df_rank.shape[1]
        df_rank_tex = df_rank.to_latex(index=False, escape=False, multicolumn_format='c', column_format=columns)
        print(df_rank_tex)
    else:
        #df = highlight(df)
        df.insert(0,'models', models)
        columns = 'c' * df.shape[1]
        df = df.astype('str')
        df_tex = df.to_latex(index=False, escape=False, multicolumn_format='c', column_format=columns)
        print(df_tex)


def fill_result_dict(path, all_dic_dataset, dataset, m, file_result_, simu=False):
    with open(path, 'rb') as fp:
        while True:
            try:
                e = pickle.load(fp)
                r = e['test']
                for key in all_dic_dataset[dataset].keys():
                    if file_result_ == 'original':
                        if key == 'NLL-T':
                            all_dic_dataset[dataset][key][m].append(r['loss_t'] + r['loss_w'])
                        elif key == 'NLL-M':
                            all_dic_dataset[dataset][key][m].append(r['loss_m'])
                        elif key in ['NLL Total', 'NLL']:
                            all_dic_dataset[dataset][key][m].append(r['loss'])
                        elif key == 'PCE':
                            bins, cal = calibration(r, num_bins=50, tabular=True)
                            _ , c_cal_mae = calibration_errors(bins, cal)
                            all_dic_dataset[dataset][key][m].append(c_cal_mae)
                        elif key == 'ECE':
                            d_cal = r['calibration']
                            _ , d_cal_mae = discrete_calibration_error(d_cal)
                            all_dic_dataset[dataset][key][m].append(d_cal_mae)
                        elif key == 'Acc':
                            all_dic_dataset[dataset][key][m].append(r['acc at 1'])
                        elif key == 'Acc@3':
                            if dataset in ['retweets_filtered_short']:
                                all_dic_dataset[dataset][key][m].append(1)
                            else:
                                all_dic_dataset[dataset][key][m].append(r['acc at 3'])
                        elif key == 'Acc@5':
                            if dataset in ['retweets_filtered_short', 'hawkes_exponential_mutual_bis', 'amazon_toys', 'amazon_movies']:
                                all_dic_dataset[dataset][key][m].append(1)
                            else:
                                all_dic_dataset[dataset][key][m].append(r['acc at 5'])
                        elif key == 'Acc@10':
                            if dataset in ['github_filtered', 'retweets_filtered_short', 'hawkes_exponential_mutual_bis', 'amazon_toys', 'amazon_movies']:
                                all_dic_dataset[dataset][key][m].append(1)
                            else:
                                all_dic_dataset[dataset][key][m].append(r['acc at 10'])
                        elif key == 'MRR':
                            all_dic_dataset[dataset][key][m].append(r['mean reciprocal rank'])
                        elif key == 'Precision':
                            all_dic_dataset[dataset][key][m].append(r['pre_weighted'])
                        elif key == 'Recall':
                            all_dic_dataset[dataset][key][m].append(r['rec_weighted'])
                        elif key == 'F1-score':
                            all_dic_dataset[dataset][key][m].append(r['f1_weighted'])
                        if simu:
                            all_dic_dataset[dataset][key][m].append(r['window integral'])
                    elif file_result_ == 'unmarked':
                        if key == 'NLL Time U.':
                           all_dic_dataset[dataset][key][m].append(r['loss'])
                        elif key == 'Time Cal. U.':
                            bins, cal = calibration(r, num_bins=50, tabular=True)
                            _ , c_cal_mae = calibration_errors(bins, cal)
                            all_dic_dataset[dataset][key][m].append(c_cal_mae)
                    elif file_result_ == 'poisson':
                        if key == 'NLL Time P.':
                            all_dic_dataset[dataset][key][m].append(-r['log ground density'])
                        elif key == 'NLL Mark P.':
                            all_dic_dataset[dataset][key][m].append(-r['log mark density'])
                        elif key in ['NLL Total P.', 'NLL P.']:
                            all_dic_dataset[dataset][key][m].append(r['loss'])
                        elif key == 'C. MAE P.':
                            bins, cal = calibration(r, num_bins=50, tabular=True)
                            _ , c_cal_mae = calibration_errors(bins, cal)
                            all_dic_dataset[dataset][key][m].append(c_cal_mae)   
                        elif key == 'D. MAE P.':
                            d_cal = r['calibration']
                            _ , d_cal_mae = discrete_calibration_error(d_cal)
                            all_dic_dataset[dataset][key][m].append(d_cal_mae)
                        elif key == 'Precision P.':
                            all_dic_dataset[dataset][key][m].append(r['pre_weighted'])
                        elif key == 'Recall P.':
                            all_dic_dataset[dataset][key][m].append(r['rec_weighted'])
                        elif key == 'F1-score P.':
                            all_dic_dataset[dataset][key][m].append(r['f1_weighted'])
                    elif file_result_ == 'unmarked_poisson':
                        if key == 'C. MAE Un. P.':
                            bins, cal = calibration(r, num_bins=50, tabular=True)
                            _ , c_cal_mae = calibration_errors(bins, cal)
                            all_dic_dataset[dataset][key][m].append(c_cal_mae)
                        elif key == 'NLL Time Un. P.':
                           all_dic_dataset[dataset][key][m].append(r['loss'])
                        
            except EOFError:
                break      
    return all_dic_dataset

def fill_result_dict_v2(path, all_dic_dataset, dataset, m, mrc=False, split=None):
    if mrc is True:
        assert(split is not None)
        for key in all_dic_dataset[dataset].keys():
            if key == 'F1-score':
                pre, rec, f1 = f1_most_present_class(dataset, split=split)
                all_dic_dataset[dataset][key][m].append(f1)
            else:
                all_dic_dataset[dataset][key][m].append(1e10)
    else:
        with open(path, 'rb') as fp:
            while True:
                try:
                    e = pickle.load(fp)
                    r = e['test']
                    for key in all_dic_dataset[dataset].keys():
                        if key == 'NLL Time':
                            all_dic_dataset[dataset][key][m].append(-r['log ground density'])
                        elif key == 'NLL Mark':
                            all_dic_dataset[dataset][key][m].append(-r['log mark density'])
                        elif key == 'Time Cal.':
                            bins, cal = calibration(r, num_bins=50, tabular=True)
                            _ , c_cal_mae = calibration_errors(bins, cal)
                            all_dic_dataset[dataset][key][m].append(c_cal_mae)
                        elif key == 'Mark Cal.':
                            d_cal = r['calibration']
                            _ , d_cal_mae = discrete_calibration_error(d_cal)
                            all_dic_dataset[dataset][key][m].append(d_cal_mae)
                        elif key == 'Precision':
                            all_dic_dataset[dataset][key][m].append(r['pre_weighted'])
                        elif key == 'Recall':
                            all_dic_dataset[dataset][key][m].append(r['rec_weighted'])
                        elif key == 'F1-score':
                            all_dic_dataset[dataset][key][m].append(r['f1_weighted'])
                except EOFError:
                    break      
    return all_dic_dataset


def num_events_df(models, datasets, results_dir):
    all_results_dic = {model:None for model in models}
    for model in models:    
        model_list = []
        for dataset in datasets:
            model_dir = os.path.join(results_dir, dataset)
            model_dir = os.path.join(model_dir, 'best')
            all_models_files = os.listdir(model_dir)
            split_list = []
            for split in range(5):
                if 'base' in model:
                    if 'lnmk1' in model:
                        file_to_find = 'poisson_' + dataset + '_' + model.split('_lnmk1')[0] + '_split{}_lnmk1_config'.format(str(split))    
                    else:    
                        file_to_find = 'poisson_' + dataset + '_' + model.split('_base')[0] + '_split{}_config'.format(str(split))
                else:
                    if 'lnmk1' in model:
                        file_to_find = dataset + '_' + model.split('_lnmk1')[0] + '_split{}_lnmk1_config'.format(str(split))
                    else:
                        file_to_find = dataset + '_' + model + '_split{}_config'.format(str(split))
                model_file = None
                for file in all_models_files:
                    if file.startswith(file_to_find):
                        model_file = os.path.join(model_dir, file)
                        break
                if model_file is None:
                    raise AttributeError('File not found')
                with open(model_file, 'rb') as f:
                    data = pickle.load(f)
                    split_list.append(data['test']['num_events_squared'])
            model_list.append(np.mean(split_list))
        all_results_dic[model] = [np.mean(model_list)]
    df = pd.DataFrame(data=all_results_dic)
    print(df)
                
            
def merge_file_results(results_dir, models, dataset, n_splits):
    results_dir = os.path.join(results_dir, dataset)
    files = os.listdir(results_dir)
    for model in models:
        for split in range(n_splits):
            file_to_find_time = dataset + '_' + model + f'_time_only_split{split}'
            file_to_find_mark = dataset + '_' + model + f'_mark_only_split{split}'
            model_file_time, model_file_mark = None, None 
            for file in files:
                if file.startswith(file_to_find_time):
                    model_file_time = file 
                if file.startswith(file_to_find_mark):
                    model_file_mark = file 
                if model_file_time is not None and model_file_mark is not None:
                    break
            if model_file_time is None:
                print(file_to_find_time)
            if model_file_mark is None:
                print(file_to_find_mark)
            file_time = os.path.join(results_dir, model_file_time)
            file_mark = os.path.join(results_dir, model_file_mark)
            with open(file_time, 'rb') as f:
                result_time = pickle.load(f)
            with open(file_mark, 'rb') as f:
                result_mark = pickle.load(f)
            keys = ['log ground density', 'cdf', 'log mark density', 'f1_weighted', 'calibration', 'samples per bin']
            results = {'test':{key:None for key in keys}} 
            for key in result_time['test'].keys():
                if key in ['log ground density', 'cdf']:
                    results['test'][key]  = result_time['test'][key]
                elif key in ['log mark density', 'f1_weighted', 'calibration', 'samples per bin']:
                    results['test'][key]  = result_mark['test'][key]
            save_file = dataset + '_' + model + f'_merged_split{split}.txt'
            save_file = os.path.join(results_dir, save_file)
            with open(save_file, 'wb') as f:
                pickle.dump(results, f)
        print('DONE')