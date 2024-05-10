import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_modelled_intensity(times, intensity, axes, args, model_name, label_idxs=None):
    if args.ground_intensity is False:
        for i, ax in enumerate(axes):
            ax.plot(times, intensity[:,i], label=model_name, color='darkorange')
    else:
        if label_idxs is None:
            axes.plot(times, intensity, label=model_name, color='darkorange')
        else:
            label_unique = np.unique(label_idxs)
            for label in label_unique:
                axes.plot(times, intensity[:,label], label=f'mark_{label}')


def plot_modelled_density(times, density, axes, args, model_name):
    if args.ground_intensity is False:
        for i, ax in enumerate(axes):
            ax.plot(times, density[:,i], label=model_name, color='forestgreen')
    else:
        axes.plot(times, density, label=model_name, color='forestgreen')

def plot_modelled_ground_density(times, density, ax):
    #density = density
    ax.plot(times, density, color='forestgreen', linewidth=5)


def plot_modelled_pmf(times, pmf, axes, labels_idxs=None):
    if labels_idxs is None:
        num_marks = int(pmf.shape[-1])
        for label in range(num_marks):
            axes.plot(times, pmf[:,label], label=f'mark {label}')
    else:
        label_unique = np.unique(labels_idxs)
        filtered_pmfs = pmf[:, label_unique]
        cum_filtered_pmfs = np.cumsum(filtered_pmfs, axis=-1)    
        zeros = np.zeros_like(cum_filtered_pmfs[:,0])
        axes.set_prop_cycle('color', sns.color_palette("Set2", len(label_unique)))
        axes.fill_between(times, cum_filtered_pmfs[:,0], zeros)
        for i in range(len(label_unique)-1):
            axes.fill_between(times, cum_filtered_pmfs[:,i+1], cum_filtered_pmfs[:,i])
        

def plot_entropy(times, pmf, axes):
    entropy = pmf * np.log(pmf)
    entropy = -np.sum(entropy, axis=-1)
    unif_entropy = -np.log(1/pmf.shape[-1])
    axes.axhline(y=unif_entropy, color='black', label='Uniform')
    axes.plot(times, entropy)