import matplotlib.pyplot as plt 
import os 
import pickle
from acronyms import get_acronym

def find_model_file(result_dir, model, dataset ,split):
    result_dir = os.path.join(result_dir, dataset)
    files = os.listdir(result_dir)
    model_file = None
    file_to_find = f'{dataset}_{model}_split{str(split)}'
    for file in files:
        if file.startswith(file_to_find):
            model_file = file
            break
    if model_file is None:
        print(file_to_find)
        raise ValueError('model not found')    
    return os.path.join(result_dir, model_file)


def plot_training_losses(result_dir, model, dataset ,split, save_dir=None):
    fig, axes = plt.subplots(1,2,figsize=(10,5), constrained_layout=True)
    model_file = find_model_file(result_dir, model, dataset, split)
    with open(model_file, 'rb') as f:
        results = pickle.load(f)
    #time_loss_train = [-results['train'][i]['log ground density'] for i in range(len(results['train'])-1)]
    #mark_loss_train = [-results['train'][i]['log mark density'] for i in range(len(results['train'])-1)]
    time_loss_val = [-results['val'][i]['log ground density'] for i in range(len(results['val'])-1)]
    mark_loss_val = [-results['val'][i]['log mark density'] for i in range(len(results['val'])-1)]
    #model_name = map_model_name_cal(model)
    axes[0].plot(time_loss_val)
    axes[1].plot(mark_loss_val)
    #axes[0].plot(time_loss_val, label='Time val')
    #axes[1].plot(mark_loss_val, label='Val')
    #axes[0].legend(fontsize=20) 
    for ax in axes:
        ax.set_xlabel('Epoch', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        #ax.set_yticklabels([])
    axes[0].set_ylabel('NLL-T', fontsize=20)
    axes[1].set_ylabel('NLL-M', fontsize=20)
    fig.suptitle(f'{dataset}, {get_acronym([model])[0]}, split{split}', fontsize=16)
    if save_dir is not None:
        save_file = f'{model}_{dataset}'
        save_file = os.path.join(save_dir, save_file)
        fig.savefig(save_file, bbox_inches='tight')