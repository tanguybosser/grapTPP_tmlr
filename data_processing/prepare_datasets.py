from cProfile import label
from enum import unique
import json
from logging.config import DEFAULT_LOGGING_CONFIG_PORT
import pickle as pkl
from posixpath import split
import torch
import numpy as np
import os 
import pandas as pd
import pickle 
import sys
#sys.path.remove(os.path.abspath(os.path.join('..', 'neuralTPPs')))
#sys.path.append(os.path.abspath(os.path.join('..', 'cntpp')))


def combine_reweets(path):
    all_sequences = []
    for split in ['train', 'dev', 'test']:
        print(split)
        file_path = os.path.join(path, split + '.pkl')
        with open(file_path, 'rb') as fp:
            while True:
                try:
                    e = pickle.load(fp, encoding="latin1")
                except EOFError:
                    break
        sequences = [[{"time":float(event["time_since_start"]), "labels":[event["type_event"]]} for event in seq] for seq in e[split]]
        all_sequences.extend(sequences)
    print(len(all_sequences))
    save_path = 'data/baseline3/retweets/retweets.json'
    with open(save_path, 'w') as f:
        json.dump(all_sequences, f)


def count_marks(data):
    seq_marks = data["marks"]
    unique_marks = list(set().union(*seq_marks))
    return len(unique_marks)


def to_format_neuralTPPs(dataset_names, marked=True, dir_path='data/baseline3/'):
    for i, dataset_name in enumerate(dataset_names):
        file_path = dir_path  + dataset_name + '/' + dataset_name + '.pkl'
        save_path = dir_path  + dataset_name + '/' + dataset_name + '.json'
        all_seq = []      
        data_old = torch.load(file_path, map_location=torch.device('cpu'))
        sequences = data_old["sequences"]
        if marked:
            all_seq = [[{'time': time, 'labels':[seq['marks'][j]]} for j, time in enumerate(seq['arrival_times'])]
                        for i, seq in enumerate(sequences)]
        else:
            all_seq = [[{'time': time, 'labels':[0]} for j, time in enumerate(seq['arrival_times'])]
                        for i, seq in enumerate(sequences)]
        with open(save_path, 'w') as f:
            json.dump(all_seq, f)
            print('Succesfully saved json file')


def github_to_format_neuralTPPs():
    all_seq = []
    file = 'data/baseline3/github/github2013.csv'
    save_path = 'data/baseline3/github/github.json'
    data_old = pd.read_csv(file)
    srcs = np.unique(data_old['sr'])
    for src in srcs:
        df_src = data_old[data_old['sr'] == src]
        tgt = list(df_src.tr) 
        ts = list(np.array(df_src.ts))
        rel = list(df_src.rel)
        seq = [{'time':ts[i], "labels":[rel[i]]} for i in range(len(ts))]
        if len(seq) > 2:
            all_seq.append(seq)
    
    with open(save_path, 'w') as f:
            json.dump(all_seq, f)
            print('Succesfully saved json file')




def split_datasets(dataset_name, load_path, save_path, window_end,shuffle=True, train_prop=0.6, val_prop=0.2):
    file_path = os.path.join(load_path, dataset_name + '.json')
    save_path_train = os.path.join(save_path,'train.json')
    save_path_val = os.path.join(save_path,'val.json')
    save_path_test = os.path.join(save_path, 'test.json')
    args_path = os.path.join(save_path, 'args.json')
    with open(file_path, "r") as f:
        data = json.load(f)
    n_train = int(train_prop * len(data)) 
    n_val = int(val_prop * len(data)) 
    if shuffle:
        all_idxs = np.arange(0,len(data))
        train_idxs = np.random.choice(all_idxs, n_train, replace=False)
        remaining_idxs = np.setdiff1d(all_idxs, train_idxs)
        val_idxs  = np.random.choice(remaining_idxs, n_val, replace=False)
        test_idxs = np.setdiff1d(remaining_idxs, val_idxs)
        train_data = list(np.array(data)[train_idxs])
        train_data = [list(seq) for seq in train_data]
        val_data = list(np.array(data)[val_idxs])
        val_data = [list(seq) for seq in val_data]
        test_data = list(np.array(data)[test_idxs])
        test_data = [list(seq) for seq in test_data]
    else:
        train_data = data[0:n_train]
        val_data = data[n_train:n_train+n_val]
        test_data = data[n_train+ n_val:]
    #Write args#
    assert(len(train_data) > 0 and len(val_data) > 0 and len(test_data) > 0)
    print(dataset_name)
    set_args(args_path, load_path, train_data, val_data, test_data, data, window_end)       

    with open(save_path_train, 'w') as f:
        json.dump(train_data, f)
    with open(save_path_val, 'w') as f:
        json.dump(val_data, f)
    with open(save_path_test, 'w') as f:
        json.dump(test_data, f)
    print('Succesfully split datasets')

#TO DO: load and preprocess before looping over splits. 
def split_datasets_calibration(dataset_name, load_path, save_path,shuffle=True, 
                               train_prop=0.65, val_prop=0.1, cal_prop=0.15, 
                               window_end=True, rescaling=None, max_len=None):
    save_path_train = os.path.join(save_path,'train.json')
    save_path_val = os.path.join(save_path,'val.json')
    save_path_cal = os.path.join(save_path,'cal.json')
    save_path_test = os.path.join(save_path, 'test.json')
    args_path = os.path.join(save_path, 'args.json')
    with open(load_path, "r") as f:
        data = json.load(f)
    if rescaling is not None:
        data = rescale_dataset(data, rescaling)
    if max_len is not None:
        data = filter_long_sequences(data, max_len)
    n_train = int(train_prop * len(data)) 
    n_val = int(val_prop * len(data)) 
    n_cal = int(cal_prop * len(data)) 
    if shuffle:
        all_idxs = np.arange(0,len(data))
        train_idxs = np.random.choice(all_idxs, n_train, replace=False)
        remaining_idxs = np.setdiff1d(all_idxs, train_idxs)
        val_idxs  = np.random.choice(remaining_idxs, n_val, replace=False)
        remaining_idxs = np.setdiff1d(remaining_idxs, val_idxs)
        cal_idxs  = np.random.choice(remaining_idxs, n_cal, replace=False)
        test_idxs = np.setdiff1d(remaining_idxs, cal_idxs)
        np_data = np.array(data, dtype=object)
        train_data = list(np_data[train_idxs])
        train_data = [list(seq) for seq in train_data]
        val_data = list(np_data[val_idxs])
        val_data = [list(seq) for seq in val_data]
        cal_data = list(np_data[cal_idxs])
        cal_data = [list(seq) for seq in cal_data]
        test_data = list(np_data[test_idxs])
        test_data = [list(seq) for seq in test_data]
    else:
        train_data = data[0:n_train]
        val_data = data[n_train:n_train+n_val]
        cal_data = data[n_train+n_val:n_train+n_val+n_cal]
        test_data = data[n_train+ n_val+n_cal:]
    
    #Write args#
    assert(len(train_data) > 0 and len(val_data) > 0 and len(test_data) > 0 and len(test_data) > 0)
    set_args_calibration(args_path, load_path, train_data, val_data, cal_data, test_data, data, window_end)       
    with open(save_path_train, 'w') as f:
        json.dump(train_data, f)
    with open(save_path_val, 'w') as f:
        json.dump(val_data, f)
    with open(save_path_cal, 'w') as f:
        json.dump(cal_data, f)
    with open(save_path_test, 'w') as f:
        json.dump(test_data, f)
    print('Succesfully split datasets')

def rescale_dataset(data, t_lim=10):
    max_time = np.max([event['time'] for seq in data for event in seq])
    rescaled_data = [[{'time':t_lim*(event['time']/max_time), 'labels':event['labels']} for event in seq] for seq in data]
    return rescaled_data


def set_args(args_path, load_path, train_data, val_data, test_data, data, window_end):
    args = {}
    args['seed'] = 0
    args['eval_metrics'] = True
    marks = [event['labels'][0] for seq in data for event in seq]
    marks_train = [event['labels'][0] for seq in train_data for event in seq]
    nm_train = np.unique(marks_train)
    marks_val = [event['labels'][0] for seq in val_data for event in seq]
    nm_val = np.unique(marks_val)
    marks_test = [event['labels'][0] for seq in test_data for event in seq]
    nm_test = np.unique(marks_test)
   #assert(np.array_equal(nm_train, nm_val) & np.array_equal(nm_train, nm_test) & np.array_equal(nm_val, nm_test))
    events = [event['time'] for seq in data for event in seq]
    args['window'] = window_end
    num_marks = len(np.unique(marks))
    args['marks'] = num_marks
    args['train_size'] = len(train_data)
    args['val_size'] = len(val_data)
    args['test_size'] = len(test_data)
    load_args_path = load_path + 'args.json'
    if os.path.exists(load_args_path):
        with open(load_args_path, 'r') as f:
            data_args_dict = json.load(f)
        args.update(data_args_dict)
    with open(args_path, 'w') as f:
        json.dump(args,f)
    print('Succesfully updated args file')

def set_args_calibration(args_path, load_path, train_data, val_data, cal_data, test_data, data, window_end):
    args = {}
    marks = [event['labels'][0] for seq in data for event in seq]
    delta_t = 0.001
    max_time = np.max([event['time'] for seq in data for event in seq]) + delta_t    
    args['window'] = max_time if window_end else None 
    num_marks = len(np.unique(marks))
    args['marks'] = num_marks
    args['train_size'] = len(train_data)
    args['val_size'] = len(val_data)
    args['cal_size'] = len(cal_data) 
    args['test_size'] = len(test_data)
    with open(args_path, 'w') as f:
        json.dump(args,f)
    print('Succesfully updated args file')


def filter_datasets_per_class(datasets_name, n_classes):
    dir_path = 'data/baseline3/'
    for dataset in datasets_name:
        file_path = dir_path + dataset + '/' + dataset + '.json'
        save_dir = dir_path + dataset + '_filtered'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = save_dir + '/' + dataset + '_filtered.json'
        with open(file_path, 'r') as f:
            sequences = json.load(f)
        #data = torch.load(file_path, map_location=torch.device('cpu')) 
        bins = np.arange(n_classes)
        mf_classes, filter_counts_prop_ori = get_most_frequent_classes(sequences, n_classes)
        print('Proportion of original events kept in dataset {} : {}'.format(dataset, filter_counts_prop_ori))
        #filtered_sequences_1 = [[event for event in seq if event['labels'][0] in mf_classes]for seq in data_json] 
        filtered_sequences = []
        for seq in sequences:
            filtered_seq = []
            for event in seq:
                if event['labels'][0] in mf_classes:
                    filtered_seq.append(event)
            filtered_sequences.append(filtered_seq)
        filtered_sequences = clean(filtered_sequences)
        filtered_marks = np.unique([event['labels'][0] for seq in filtered_sequences for event in seq])
        new_marks = np.arange(len(filtered_marks))
        mapping = {k:new_marks[i] for i, k in enumerate(filtered_marks)}
        filtered_sequences  = map_marks(filtered_sequences, mapping)
        print("Proportion of original sequences kept in dataset {} : {}".format(dataset, len(filtered_sequences)/len(sequences)))
        ##CHECKS##
        marks = np.array([event["labels"][0] for seq in filtered_sequences for event in seq])
        cat = np.unique(marks)
        #assert(len(cat) == n_classes)
        with open(save_path, 'w') as f:
            json.dump(filtered_sequences, f)
            print("Succesfully saved dataset")

def map_marks(sequences, mapping):
    for seq in sequences:
        for event in seq:
            event['labels'] = [int(mapping[event['labels'][0]])]
    return sequences

def get_most_frequent_classes(sequences, n_classes):
    marks = []
    marks = np.array([event["labels"][0] for seq in sequences for event in seq])
    cat, counts = np.unique(marks, return_counts=True)
    idx_sorted_counts = counts.argsort()[::-1]
    sorted_counts = counts[idx_sorted_counts]
    sorted_cats = cat[idx_sorted_counts]
    if len(sorted_cats) <= n_classes:
        filter_cats = sorted_cats
        filter_counts = sorted_counts
    else:
        filter_counts = sorted_counts[:n_classes]
        filter_cats = sorted_cats[:n_classes]
    filter_counts_prop  = filter_counts/sum(filter_counts)
    filter_counts_prop_ori = sum(filter_counts)/len(marks)
    print(sum(filter_counts_prop))
    return filter_cats, filter_counts_prop_ori

def shorten_dataset(datasets, prop=0.3):
    dir_path = 'data/baseline3/'
    for dataset in datasets:
        path = dir_path + dataset + "/" + dataset + ".json"
        save_dir = dir_path + dataset + "_short" 
        if not os.path.exists(save_dir):
            os.makedirs(save_dir) 
        save_path = save_dir + "/" + dataset + "_short.json"
        with open(path, 'r') as f:
            data = json.load(f)
        num_seq = len(data)
        n_new = int(num_seq * prop)
        if num_seq <= 400:
            new_seq = data
        else:    
            new_seq = data[:n_new]
        new_seq = clean(new_seq)
        with open(save_path, 'w') as f:
            json.dump(new_seq, f)
            print('Succesfully reduced dataset')

def make_splits(datasets, n_split=5, window_end=None):
    for i in range(n_split):
        for dataset in datasets:
            load_path = 'data/baseline3/{}/'.format(dataset)
            save_path = 'data/baseline3/{}/split_{}'.format(dataset, i)
            if os.path.exists(save_path) is False:
                os.makedirs(save_path)
            split_datasets(dataset, load_path, save_path, window_end)
            print('Succesfully created split {} for dataset {}'.format(i, dataset))

def make_splits_calibration(datasets, load_dir, n_split=5, train_prop=0.65, val_prop=0.1, cal_prop=0.15, window_end=True, rescaling=None, max_len=None, save_name=None, shuffle=True):
    assert(train_prop + val_prop + cal_prop < 1), 'Invalid split proportions.'
    for i in range(n_split):
        for dataset in datasets:
            load_path = os.path.join(load_dir, f'{dataset}/{dataset}.json')
            if save_name is None:
                save_path = f'../data/baseline3/{dataset}/split_{i}'
            else:
                save_path = f'../data/baseline3/{dataset}_{save_name}/split_{i}'
            if os.path.exists(save_path) is False:
                os.makedirs(save_path)
            split_datasets_calibration(dataset, load_path, save_path, train_prop=train_prop, 
                                    val_prop=val_prop, cal_prop=cal_prop, window_end=window_end, rescaling=rescaling,
                                    max_len=max_len, shuffle=shuffle)
            print('Succesfully created calibration split {} for dataset {}'.format(i, dataset))

def rescale(datasets, t_lim=10):
    file_dir = '../neuralTPPs/data/baseline3/'
    for dataset in datasets:
        save_dir = f'data/baseline3/{dataset}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_path = file_dir + dataset + '/' + dataset + '.json'
        with open(file_path, 'r') as f:
            sequences = json.load(f)
        max_time = np.max([event['time'] for seq in sequences for event in seq])
        new_sequences = [[{'time':t_lim*(event['time']/max_time), 'labels':event['labels']} for event in seq] for seq in sequences]
        new_sequences = clean(new_sequences)
        save_path = f'data/baseline3/{dataset}/{dataset}.json'
        with open(save_path, 'w') as f:
            json.dump(new_sequences, f)


def combine(path, save_path):
    all_seqs = []
    for split in ['train', 'val', 'test']:
        file = path + split + '.json'
        with open(file, 'r') as f:
            sequences = json.load(f)
            all_seqs.extend(sequences)
    with open(save_path, 'w') as f:
        json.dump(all_seqs,f)

def marked_to_unmarked(datasets):
    dir = 'data/baseline3/'
    for dataset in datasets:
        file  = dir + dataset + '/' + dataset + '.json'
        save_dir  = dir + dataset + '_unmarked'
        with open(file, 'r') as f:
            data = json.load(f)
        new_data = [[{"time":event["time"], "labels":[0]} for event in seq] for seq in data]
        new_data = clean(new_data) #Remove sequences smaller than 2 events. 
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_file = save_dir + '/' + dataset + '_unmarked.json'
        with open(save_file, 'w') as f:
            json.dump(new_data,f)
        print('Successfully created {}'.format(dataset + '_unmarked'))

def datastats(datasets,dir,ori_datasets =None, prop=True):
    if prop:
        metrics = {'\\#Seq.':[], 'p\\%Seq.':[], '\\#Events':[], 'p\\%Events':[], 'MSL':[], 'Max Len.':[], 'Min Len.':[], '\\#Marks':[]}
    else:
        metrics = {'\\#Seq.':[], '\\#Events':[], 'MSL':[], 'Max Len.':[], 'Min Len.':[], '\\#Marks':[]}
    for i, dataset in enumerate(datasets):
        if 'hawkes' in dataset:
            data = combine_hawkes(dir, dataset)
        else:
            path = dir + dataset + '/' + dataset + '.json'
            with open(path, 'r') as f:
                data = json.load(f)
        metrics["\\#Seq."].append(len(data))
        seq_length = [len(seq) for seq in data]
        metrics["\\#Events"].append(np.sum(seq_length))
        metrics['MSL'].append(np.round(np.mean(seq_length),1))
        metrics['Max Len.'].append(np.max(seq_length))
        metrics['Min Len.'].append(np.min(seq_length))
        num_marks = len(np.unique([event["labels"][0] for seq in data for event in seq]))
        if num_marks == 1:
            metrics["\\#Marks"].append("\\textbackslash")
        else:
            metrics["\\#Marks"].append(num_marks)
        if prop:
            ori_path = dir + ori_datasets[i] + '/' + ori_datasets[i] + '.json'
            with open(ori_path, 'r') as f:
                data_ori = json.load(f)
            metrics["p\\%Seq."].append(np.round(len(data)/len(data_ori),2))
            ori_events = [len(seq) for seq in data_ori]
            metrics["p\\%Events"].append(np.round(np.sum(seq_length)/np.sum(ori_events),2))
    df = pd.DataFrame(data=metrics)
    df.insert(0,'datasets', datasets)
    columns = 'c|' * df.shape[1]
    df_tex = df.to_latex(index=False, escape=False, multicolumn_format='c|', column_format=columns)
    print(df_tex)

def combine_hawkes(dir, dataset):
    dir = os.path.join(dir, dataset)
    dir = os.path.join(dir, 'split_0')
    splits = ['train', 'val', 'cal', 'test']
    all_seq = []
    for split in splits:
        file = os.path.join(dir, split + '.json')
        with open(file, 'r') as f:
            data = json.load(f)
        all_seq.extend(data)
    return all_seq

def clean(sequences):
    new_sequences = [[event for event in seq if event['time'] != 0] for seq in sequences]
    new_sequences = [seq for seq in new_sequences if len(seq) >= 2]
    return new_sequences

def clean_and_save(datasets):
    dir = 'data/baseline3/'
    for dataset in datasets:
        file_path = dir + dataset + '/' + dataset + '.json'
        with open(file_path, 'r') as f:
            data = json.load(f)
        new_seq = clean(data)
        with open(file_path, 'w') as f:
            json.dump(new_seq, f)

def shorten_sequences(datasets, max_length=128):
    file_dir = 'data/baseline3/'
    for dataset in datasets:
        file_path = file_dir + dataset + '/' + dataset + '.json'
        save_dir = file_dir + dataset + '_seqshort'
        save_path = save_dir + '/' + dataset + '_seqshort.json'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(file_path, 'r') as f:
            data = json.load(f)
        new_seqs = [seq if len(seq) <= 128 else seq[:128] for seq in data]
        new_seqs = clean(new_seqs)
        with open(save_path, 'w') as f:
            json.dump(new_seqs, f)


def shorten_train_val_sequences(datasets, max_length=200, n_splits=5):
    for i in range(n_splits):
        for dataset in datasets:
            path = 'data/baseline3/' + dataset + '/' + split


def set_max_seq_lenght(data_dir, dataset, max_length=1000):
    data_path = os.path.join(data_dir, dataset)
    data_path = os.path.join(data_path, dataset + '.json')
    with open(data_path, 'r') as f:
        data = json.load(f)
    seq_lenghts = np.array([len(seq) for seq in data])
    seq_mask =  seq_lenghts <= max_length
    data_short = list(np.array(data)[seq_mask])
    with open(data_path, 'w') as f:
        json.dump(data_short, f)

def split_in_distinct_datasets(data_dir, dataset, num_split=2):
    data_path = os.path.join(data_dir, dataset)
    data_path = os.path.join(data_path, dataset + '.json')
    with open(data_path, 'r') as f:
        data = json.load(f)
    inc = int(len(data)/num_split)
    lims = np.arange(0, len(data), inc)
    data_splits = [data[lims[i-1]:lims[i]] for i in range(1,len(lims))]
    for i, split in enumerate(data_splits):
        data_name = dataset + str(i+1)
        save_path = os.path.join(data_dir, data_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, data_name + '.json')
        with open(save_path, 'w') as f:
            json.dump(split, f)

def filter_long_sequences(data, max_len=10000):
    lenghts = np.array([len(seq) for seq in data])
    mask = lenghts < max_len
    new_data = np.array(data, dtype='object')[mask]
    #Marks must be remapped#
    new_data = [list(seq) for seq in new_data]
    marks = np.sort(np.unique([event['labels'][0] for seq in new_data for event in seq]))
    new_mapping = {marks[i]:i for i in range((len(marks)))}
    new_data = [[{'time':event['time'], 'labels':[new_mapping[event['labels'][0]]]} for event in seq] for seq in new_data]
    return new_data

if __name__ == '__main__':
    data_dir = 'data/baseline3'
    
    #split_in_distinct_datasets(data_dir, dataset)
    
    #set_max_seq_lenght(data_dir, dataset)    
    #datasets = ['reddit_filtered1', 'reddit_filtered2']
    
    #make_splits(datasets, n_split=5)




    ########## SIMULATE HAWKES DATASETS ############

    """ process_decays = [1.0, 1.5, 1.2, .9, 1.1]
    process_baselines = [1, .8, 1.1, .7, .9]
    process_adjacency = [0.7, 0.5, 0.6, 0.8, 0.75]
    end_time = 10
    n_seq = 1000
    n_processes = 5
    save_dir = 'data/baseline3/' """

    #hawkes_dataset(end_time, n_seq, n_processes, process_decays, process_baselines, process_adjacency, save_dir)
    #dataset = ['hawkes']
    #make_splits(dataset, n_split=1)

    ########### INITIAL PRE-PROCESSING ############

    """ datasets = ['reddit', 'mooc', 'wikipedia', 'lastfm']
    mimic_path = 'data/baseline/mimic2/split_1/'
    to_format_neuralTPPs(datasets)
    github_to_format_neuralTPPs()
    combine_mimic(mimic_path) """
    so_path = 'data/baseline3/stack_overflow/'
    so_save_path = 'data/baseline3/stack_overflow/stack_overflow.json'
    mimic_path = 'data/baseline3/mimic2/split_0/'
    mimic_save_path = 'data/baseline3/mimic2/mimic2.json'
    retweets_path = 'data/baseline3/retweets'
    retweets_save_path = 'data/baseline3/retweets/retweets.json'
    #combine(retweets_path, retweets_save_path)
    #combine_reweets(retweets_path)

    
    ######## MARKED FILTERED #########

    '''
    marked_datasets = ['reddit', 'wikipedia', 'mooc', 'lastfm', 'mimic2', 'github']
    marked_datasets = ['retweets']
    filter_datasets_per_class(marked_datasets, n_classes=50) #Github is not filtered.
    marked_datasets_filtered = ['reddit_filtered', 'wikipedia_filtered', 'mooc_filtered', 'lastfm_filtered', 'mimic2_filtered', 'github_filtered']
    marked_datasets_filtered = ['stack_overflow_filtered']
    marked_datasets_filtered = ['retweets_filtered']
    rescale(marked_datasets_filtered, t_lim=10)
    make_splits(marked_datasets_filtered, n_split=5)
    '''


    """ ########## MARKED SHORT FILTERED ########
    marked_datasets = ['reddit', 'wikipedia', 'mooc', 'lastfm', 'mimic2', 'github']
    shorten_dataset(marked_datasets, prop=0.5)
    marked_datasets_short = ['reddit_short', 'wikipedia_short', 'mooc_short', 'lastfm_short', 'mimic2_short', 'github_short'] #Github is not shortened.
    filter_datasets_per_class(marked_datasets_short, n_classes=50) #Github is not filtered.
    marked_datasets_short_filtered = ['reddit_short_filtered', 'wikipedia_short_filtered', 'mooc_short_filtered', 'lastfm_short_filtered', 'mimic2_short_filtered',                 'github_short_filtered']
    make_splits(marked_datasets_short_filtered, n_split=5)
    """

    ######### MARKED UNMARKED ###############

    """ marked_datasets = ['reddit_filtered', 'wikipedia_filtered', 'mooc_filtered', 'lastfm_filtered', 'mimic2_filtered', 'github_filtered']
    #marked_to_unmarked(marked_datasets)
    marked_unmarked_datasets = ['reddit_filtered_unmarked', 'wikipedia_filtered_unmarked', 'mooc_filtered_unmarked', 'lastfm_filtered_unmarked', 'mimic2_filtered_unmarked', 'github_filtered_unmarked']
    #rescale(marked_unmarked_datasets, t_lim=10)
    make_splits(marked_unmarked_datasets, n_split=5) """


    ########### UNMARKED FULL ##############
    """ unmarked_datasets = ['reddit_politics_submissions', 'reddit_askscience_comments', 'taxi', 'twitter', 'yelp_toronto', 'yelp_airport', 'yelp_mississauga', 'pubg']
    rescale(unmarked_datasets, t_lim=10)
    make_splits(unmarked_datasets, n_split=5)
    """

    """ ########### UNMARKED SHORT #############
    unmarked_datasets = ['reddit_politics_submissions', 'reddit_askscience_comments', 'taxi', 'twitter', 'yelp_toronto', 'yelp_airport', 'yelp_mississauga', 'pubg']
    shorten_dataset(unmarked_datasets, prop=0.5)
    unmarked_datasets_short = ['reddit_politics_submissions_short', 'reddit_askscience_comments_short', 'taxi_short', 'twitter_short', 'yelp_toronto_short', 'yelp_airport_short', 'yelp_mississauga_short', 'pubg_short']
    make_splits(unmarked_datasets_short, n_split=5)
    """
    """ 
    dataset_names = ['reddit', 'mooc', 'lastfm', 'wikipedia']
    scales = [1000000, 1000000, 100, 1000000]
    #to_format_neuralTPPs(dataset_names, scales)
    """

    ########### MARKED FILTERED SHORT SEQUENCES ############


    """ datasets = ['reddit_filtered', 'mooc_filtered', 'lastfm_filtered', 'wikipedia_filtered', 'mimic2_filtered', 'github_filtered']
    #shorten_sequences(datasets, max_length=128)
    datasets_short_seq = ['reddit_filtered_seqshort', 'mooc_filtered_seqshort', 'wikipedia_filtered_seqshort', 'lastfm_filtered_seqshort', 'mimic2_filtered_seqshort', 'github_filtered_seqshort']
    make_splits(datasets_short_seq, n_split=5)  """

    """ dataset = ['mimic2_filtered']
    #filter_datasets_per_class(dataset, n_classes=50)
    shorten_sequences(dataset, max_length=128)
    datasets_short_seq = ['mimic2_filtered_seqshort']
    make_splits(datasets_short_seq, n_split=5)
    """
    ########## MARKED UNMARKED SHORT SEQUENCES ############

    """ datasets = ['reddit_unmarked', 'wikipedia_unmarked', 'lastfm_unmarked', 'mooc_unmarked', 'mimic2_unmarked', 'github_unmarked']
    shorten_sequences(datasets, max_length=128)
    datasets_un_seqshort = ['reddit_unmarked_seqshort', 'wikipedia_unmarked_seqshort', 'lastfm_unmarked_seqshort', 'mooc_unmarked_seqshort', 'mimic2_unmarked_seqshort', 'github_unmarked_seqshort']
    make_splits(datasets_un_seqshort, n_split=5)
    """
    #dataset_names = ['reddit_askscience_comments', 'reddit_politics_submissions', 'taxi', 'twitter', 'pubg', 'yelp_toronto', 'yelp_mississauga', 'yelp_airport']
    #scales = [10, 10, 10, 10, 10, 100000000, 10, 10]
    #to_format_neuralTPPs(dataset_names, scales, marked=False)


    ########### UNMARKED SHORT SEQUENCES ###############

    """ datasets_un = ['reddit_politics_submissions','reddit_askscience_comments', 'taxi','twitter','yelp_toronto','yelp_airport','yelp_mississauga', 'pubg']
    #shorten_sequences(datasets_un, max_length=128)
    datasets_un_shortseq = ['reddit_politics_submissions_seqshort','reddit_askscience_comments_seqshort', 'taxi_seqshort','twitter_seqshort','yelp_toronto_seqshort','yelp_airport_seqshort','yelp_mississauga_seqshort', 'pubg_seqshort']
    make_splits(datasets_un_shortseq, n_split=5)
    """
    
    datasets_m = ['reddit_filtered',
                'wikipedia_filtered',
                'mooc_filtered',
                'lastfm_filtered',
                'mimic2_filtered',
                'github_filetered',]


    datasets_ori = ['reddit',  
                    'wikipedia',
                    'mooc',
                    'lastfm',
                    'mimic2',
                    'github']

    datasets_u = ['reddit_politics_submissions',
                'reddit_askscience_comments',
                'taxi',
                'twitter',
                'yelp_toronto',
                'yelp_airport',
                'yelp_mississauga',
                'pubg']

    datasets_ori_u = ['reddit_politics_submissions',
                    'reddit_askscience_comments',
                    'taxi',
                    'twitter',
                    'yelp_toronto', 
                    'yelp_airport', 
                    'yelp_mississauga', 
                    'pubg']


    datasets_m = [
                'lastfm_filtered',
                'mooc_filtered',
                'github_filtered',
                'reddit_filtered_short',
                'retweets_filtered_short',
                'stack_overflow_filtered'
                #'hawkes_exponential_mutual'
                ]


    datasets_ori = [ 
                    #'wikipedia',
                    'lastfm'
                    #'mooc',
                    #'github',
                    #'reddit',
                    #'retweets',
                    #'stack_overflow'
                    #'mimic2'
                    ]


    #datastats(datasets_ori, dir='../neuralTPPs/data/baseline3/', ori_datasets=datasets_ori, prop=False)   

'''
######## CALIBRATION ############

    load_path = '../neuralTPPs/data/baseline3/'
    datasets = ['mooc_filtered']
    rescale(datasets, t_lim=10)
    make_splits_calibration(datasets, load_path=load_path, n_split=5, window_end=None, train_prop=0.65, val_prop=0.1, cal_prop=0.15)
'''


####### NEURIPS ##########

datasets = [
    'lastfm',
    #'mooc',
    #'reddit',
    #'retweets',
    #'stack_overflow'
]
load_path = '../neuralTPPs/data/baseline3'

#make_splits_calibration(datasets, load_path=load_path, n_split=5, train_prop=0.65, val_prop=0.1, cal_prop=0.15, window_end=True, rescaling=10)
