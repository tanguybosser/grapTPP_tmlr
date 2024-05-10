import matplotlib.pyplot as plt 
import seaborn as sns
import os
import json

def sequence_len_dis(data_dir, dataset):
    file_path = os.path.join(data_dir, f'{dataset}.json')
    with open(file_path, 'r') as f:
        sequences = json.load(f)
    seq_lens = [len(seq) for seq in sequences]
    fig, ax = plt.subplots(figsize=(10,10))
    ax.hist(seq_lens, bins=1000)
    ax.set_ylabel('Count', fontsize=20)
    ax.set_xlabel('Sequence Lenghts', fontsize=20)
    #fig.savefig(f'../figures/datasets/{dataset}.png' , bbox_inches='tight')


def inter_arrival_time_dis(data_dir, dataset):
    sequences = combine_splits(data_dir, dataset)
    times = [[event['time'] for event in seq] for seq in sequences]
    arrival_times_till_last = [seq[i+1]-seq[i] for seq in times for i in range(len(seq)-2)]
    last_arrival_times = [seq[-1]-seq[-2] for seq in times]
    #last_event = [10.001 - seq[-1] for seq in times]
    fig, ax = plt.subplots()
    print('Sequences made')
    ax.hist(arrival_times_till_last, color='blue', alpha=0.7, density=True, label='tau_i')
    ax.hist(last_arrival_times, color='red', alpha=0.7, density=True, label='tau_n')
    ax.legend()
    #print(last_event)
    #sns.displot(data=arrival_times, color='blue', alpha=0.7, ax=ax)
    #sns.displot(data=last_event, color='red', alpha=0.7, ax=ax)
    plt.show()


def combine_splits(data_dir, dataset):
    train = os.path.join(data_dir, f'{dataset}/split_0/train.json')
    val = os.path.join(data_dir, f'{dataset}/split_0/val.json')
    cal = os.path.join(data_dir, f'{dataset}/split_0/cal.json')
    test = os.path.join(data_dir, f'{dataset}/split_0/test.json')
    splits = [train, val, cal, test]
    all_sequences = []
    for split in splits:
        with open(split, 'r') as f:
            sequences = json.load(f)
        all_sequences.extend(sequences)
    print('Splits combined')
    return all_sequences