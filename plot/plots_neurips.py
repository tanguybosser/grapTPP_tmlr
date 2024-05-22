import os 
import matplotlib.pyplot as plt 
import pickle 
import seaborn as sns
import numpy as np 

def find_model_file(result_dir, model, dataset ,split):
    result_dir = os.path.join(result_dir, dataset)
    files = os.listdir(result_dir)
    model_file = None
    file_to_find = dataset + '_' + model + '_split' + str(split)
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
    time_loss_train = [results['train'][i]['loss_t'] for i in range(len(results['train'])-3)]
    mark_loss_train = [results['train'][i]['loss_m'] for i in range(len(results['train'])-3)]
    print(time_loss_train)
    #swindow_loss_train = [results['train'][i]['window integral'] for i in range(len(results['train'])-1)]
    
    #print('Time loss train', -results['train'][-1]['log ground density'])
    #print('Mark loss train', -results['train'][-1]['log mark density'])
    #print('Window loss train', results['train'][-1]['window integral'])

    #time_loss_val = [-results['val'][i]['log ground density'] for i in range(len(results['val'])-1)]
    #mark_loss_val = [-results['val'][i]['log mark density'] for i in range(len(results['val'])-1)]
    #window_loss_val = [results['val'][i]['window integral'] for i in range(len(results['val'])-1)]

    axes[0].plot(time_loss_train)
    axes[1].plot(mark_loss_train)
    #axes[2].plot(window_loss_train)
    #axes[0].plot(time_loss_val, label='Time val')
    #axes[1].plot(mark_loss_val, label='Val')
    axes[0].legend(fontsize=20) 
    for ax in axes:
        ax.set_xlabel('Epoch', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        #ax.set_yticklabels([])
    axes[0].set_ylabel('NLL-T', fontsize=20)
    axes[1].set_ylabel('NLL-M', fontsize=20)
    #axes[2].set_ylabel('NLL-W', fontsize=20)
    if save_dir is not None:
        save_file = f'{model}_{dataset}'
        save_file = os.path.join(save_dir, save_file)
        fig.savefig(save_file, bbox_inches='tight')


def plot_val_losses(result_dir, model, dataset ,split, save_dir=None):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1,2,figsize=(10,5), constrained_layout=True)
    model_file = find_model_file(result_dir, model, dataset, split)
    with open(model_file, 'rb') as f:
        results = pickle.load(f)        
    time_loss_val = [results['val'][i]['loss_t'] for i in range(len(results['val'])-3)]
    mark_loss_val = [results['val'][i]['loss_m'] for i in range(len(results['val'])-3)]
    epochs = np.arange(len(time_loss_val))
    #swindow_loss_train = [results['train'][i]['window integral'] for i in range(len(results['train'])-1)]
    
    #print('Time loss train', -results['train'][-1]['log ground density'])
    #print('Mark loss train', -results['train'][-1]['log mark density'])
    #print('Window loss train', results['train'][-1]['window integral'])

    #time_loss_val = [-results['val'][i]['log ground density'] for i in range(len(results['val'])-1)]
    #mark_loss_val = [-results['val'][i]['log mark density'] for i in range(len(results['val'])-1)]
    #window_loss_val = [results['val'][i]['window integral'] for i in range(len(results['val'])-1)]

    sns.lineplot(x=epochs, y=time_loss_val, ax=axes[0], color='orange')
    sns.lineplot(x=epochs, y=mark_loss_val, ax=axes[1], color='orange')
    
    #axes[0].plot(time_loss_train)
    #axes[1].plot(mark_loss_train)
    #axes[2].plot(window_loss_train)
    #axes[0].plot(time_loss_val, label='Time val')
    #axes[1].plot(mark_loss_val, label='Val')
    axes[0].legend(fontsize=20) 
    for ax in axes:
        ax.set_xlabel('Epoch', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        #ax.set_yticklabels([])
    axes[0].set_ylabel(r'$\mathcal{L}_T$', fontsize=20)
    axes[1].set_ylabel(r'$\mathcal{L}_M$', fontsize=20)
    #axes[2].set_ylabel('NLL-W', fontsize=20)
    if save_dir is not None:
        save_file = f'{save_dir}/{model}_{dataset}.png'
        fig.savefig(save_file, bbox_inches='tight')