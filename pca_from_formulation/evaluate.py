import numpy as np
import torch
import os
import sys
sys.path.append('../utils')
import numpy as np
from load_dataset import build_nu, load_dataset
import matplotlib.pyplot as plt

def load_data(begin, end, train_data_dir, nu_list):
    x_dataset = []
    u_dataset = []
    nu_dataset = []
    window_size = end - begin

    for i in range(len(train_data_dir)):
        data_dir = train_data_dir[i]
        nu = nu_list[i]
        nu = torch.tensor(build_nu(nu_list, window_size, nu), dtype=torch.int16)
        for item in os.listdir(data_dir):
            data_file_path = os.path.join(data_dir, item)

            # Check if the file exists before trying to load it
            if os.path.exists(data_file_path) and item.endswith('.npy'):
                data_dict = np.load(data_file_path, allow_pickle=True).item()
                x_data, _, u_data, _ = load_dataset(data_dict)
                x_dataset.append(x_data[begin:end, :])
                u_dataset.append(u_data[begin:end, :])
                nu_dataset.append(nu)
            else:
                print(f"File not found: {data_file_path}")
    
    return x_dataset, u_dataset, nu_dataset

def generate_trajectories(x_dataset, u_dataset, nu_dataset, model, device):
    x_data_pred_traj = []
    x_data_pca_traj = []
    x_data_pca_pred_traj = []
    window_size = len(x_dataset[0])

    for x_data, u_data, nu_data in zip(x_dataset, u_dataset, nu_dataset):
        steps = window_size    

        x_data = torch.tensor(x_data, dtype=torch.float32).to(device)
        u_data = torch.tensor(u_data, dtype=torch.float32).to(device)
        x_pred = torch.zeros_like(x_data).to(device)
        x_pred[0, :] = x_data[0, :]
        nu = nu_data[0:1, :]
        x0 = model.encode(x_data[0, :].reshape(1, -1), nu)
        
        for step in range(1, steps):
            u = u_data[step - 1, :].reshape(1, -1)
            x1 = model.latent_to_latent_forward(x0, u, nu)
            x_pred[step, :] = model.decode(x1, nu)
            x0 = x1
        
        x_data_pred_traj.append(x_pred.detach().cpu().numpy())

        x_pca_true = model.std_layer_1.transform(x_data.detach(), nu_data)
        x_pca_true = model.pca_transformer.transform(x_pca_true)
        x_pca_pred = model.std_layer_1.transform(x_pred.detach(), nu_data)
        x_pca_pred = model.pca_transformer.transform(x_pca_pred)
        
        x_data_pca_traj.append(x_pca_true)
        x_data_pca_pred_traj.append(x_pca_pred)

    return x_data_pred_traj, x_data_pca_traj, x_data_pca_pred_traj

def calculate_relative_diff(x_true, x_pred):
    row_norm_diff = np.linalg.norm(x_true - x_pred, axis=1, ord=2)
    max_norm = np.max(np.linalg.norm(x_true, axis=1, ord=2))
    relative_diff = row_norm_diff / max_norm
    return relative_diff

def calculate_mean_relative_diff_set(x_true_traj, x_pred_traj):
    relative_diffs = [calculate_relative_diff(x_true, x_pred) for x_true, x_pred in zip(x_true_traj, x_pred_traj)]
    mean_relative_diffs = np.mean(relative_diffs, axis=0)
    return mean_relative_diffs

def plot_mean_relative_diff(x_true_traj_train, x_pred_traj_train, x_true_traj_test = None, x_pred_traj_test = None):
    mean_relative_diffs_train = calculate_mean_relative_diff_set(x_true_traj_train, x_pred_traj_train)
    if x_true_traj_test is not None and x_pred_traj_test is not None:
        mean_relative_diffs_test = calculate_mean_relative_diff_set(x_true_traj_test, x_pred_traj_test)

    plt.figure(figsize=(10, 4))
    
    plt.plot(mean_relative_diffs_train, label = 'Training Set', color='mediumseagreen')
    if x_true_traj_test is not None and x_pred_traj_test is not None:
        plt.plot(mean_relative_diffs_test, label='Validation Set', color='pink')
    
    plt.xlabel('Prediction Step')
    plt.ylabel('Mean Relative Diff')
    plt.title('Mean Relative Diff on Training Set')
    plt.legend()
    plt.show()

def calculate_relative_error(x_true, x_pred):
    return np.linalg.norm(x_true - x_pred, ord = 'fro') / np.linalg.norm(x_true, ord = 'fro')

def calculate_mean_relative_error(x_true_traj, x_pred_traj):
    return np.mean([calculate_relative_error(x_true, x_pred) for x_true, x_pred in zip(x_true_traj, x_pred_traj)])

def plot_relative_error(x_true_traj_train, x_pred_traj_train, x_true_traj_test=None, x_pred_traj_test=None):
    mean_relative_errors_train = []
    mean_relative_errors_test = []
    
    for i in range(len(x_true_traj_train)):
        mean_relative_errors_train.append(calculate_relative_error(x_true_traj_train[i], x_pred_traj_train[i]))
    
    if x_true_traj_test is not None and x_pred_traj_test is not None:
        for i in range(len(x_true_traj_test)):
            mean_relative_errors_test.append(calculate_relative_error(x_true_traj_test[i], x_pred_traj_test[i]))

        fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(9, 7))
    else:
        fig, ax0 = plt.subplots(nrows=1, figsize=(9, 7))

    ax0.hist(mean_relative_errors_train, bins=50, density=True, histtype='bar', facecolor='yellowgreen', alpha=0.75)
    ax0.set_xlabel('Relative Error')
    ax0.set_ylabel('Frequency')
    ax0.set_title('Training Set')
    
    if x_true_traj_test is not None and x_pred_traj_test is not None:
        ax1.hist(mean_relative_errors_test, bins=50, density=True, histtype='bar', facecolor='pink', alpha=0.75)
        ax1.set_title('Validation Set')
        ax1.set_xlabel('Relative Error')
        ax1.set_ylabel('Frequency')
    
    plt.subplots_adjust(hspace=0.4)
    plt.show()


def calculate_relative_error_set(x_true, x_pred):
    row_norm_diff = np.linalg.norm(x_true - x_pred, axis=1, ord=2)
    total_norm_true = np.linalg.norm(x_true, ord='fro')
    return row_norm_diff / total_norm_true


def plot_relative_error_boxplot(x_true_traj_train, x_pred_traj_train, x_true_traj_test=None, x_pred_traj_test=None):
    mean_relative_errors_train = []
    mean_relative_errors_test = []
    
    for i in range(len(x_true_traj_train)):
        mean_relative_errors_train.append(calculate_relative_error(x_true_traj_train[i], x_pred_traj_train[i]))
    
    if x_true_traj_test is not None and x_pred_traj_test is not None:
        for i in range(len(x_true_traj_test)):
            mean_relative_errors_test.append(calculate_relative_error(x_true_traj_test[i], x_pred_traj_test[i]))

    fig, ax = plt.subplots(figsize=(8, 4))

    data = [mean_relative_errors_train]
    labels = ['Training Set']

    if x_true_traj_test is not None and x_pred_traj_test is not None:
        data.append(mean_relative_errors_test)
        labels.append('Validation Set')

    boxprops_train = dict(facecolor='yellowgreen', color='black')
    boxprops_test = dict(facecolor='pink', color='black')

    boxplots = ax.boxplot(data, patch_artist=True, labels=labels)

    for patch, props in zip(boxplots['boxes'], [boxprops_train, boxprops_test] if len(data) > 1 else [boxprops_train]):
        patch.set_facecolor(props['facecolor'])
        patch.set_edgecolor(props['color'])

    ax.set_xlabel('Relative Error')
    ax.set_title('Relative Error Distribution')
    
    plt.show()
