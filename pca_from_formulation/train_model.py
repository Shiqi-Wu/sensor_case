from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
sys.path.append('/home/shiqi/code/Project2-sensor-case/model_combination_Argos/utils')
print(sys.path)
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import pca_koopman_dir as km
from load_dataset import *
import argparse
import os
import yaml
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import copy

scaler_x = StandardScaler()
scaler_u = StandardScaler()
scaler_pca = StandardScaler()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Reaction-diffusion model')
    parser.add_argument('--config', type=str, help='configuration file path')
    args = parser.parse_args()
    return args

def read_config_file(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def koopman_loss(model, x, u, nu):
    loss_fn = nn.MSELoss()
    N = x.shape[1]
    if N <= 1:
        raise ValueError('Number of time steps should be greater than 1')
    
    x_latent = model.encode(x)
    x0 = model.state_dic(x_latent[:, 0, :])
    x_pred = torch.zeros((x_latent.shape[0], x_latent.shape[1], x0.shape[1]), dtype=torch.float32, device=x.device)
    x_true = torch.zeros((x_latent.shape[0], x_latent.shape[1], x0.shape[1]), dtype=torch.float32, device=x.device)
    
    x_pred[:, 0, :] = x0
    x_true[:, 0, :] = x0
    for i in range(1, N):
        x_pred_cur = model.latent_to_latent_forward(x0, u[:, i - 1, :], nu[:, i - 1, :])
        x_pred[:, i, :] = x_pred_cur
        x_true[:, i, :] = model.state_dic(x_latent[:, i, :])
    loss = loss_fn(x_pred, x_true)
    return loss

def koopman_loss_extract(model, x, u, nu):
    loss_fn = nn.MSELoss()
    N = x.shape[1]
    if N <= 1:
        raise ValueError('Number of time steps should be greater than 1')
    
    x_latent = model.encode(x)
    x0 = x_latent[:, 0, :]
    x_pred = torch.zeros_like(x_latent, dtype=torch.float32, device=x.device)

    x_pred[:, 0, :] = x0
    for i in range(1, N):
        x_pred_cur = model.pca_forward(x0, u[:, i - 1, :], nu[:, i - 1, :])
        x_pred[:, i, :] = x_pred_cur
    loss = loss_fn(x_pred, x_latent)
    return loss

def koopman_loss_DicWithInputs(model, x, u, nu):
    loss_fn = nn.MSELoss()
    N = x.shape[1]
    if N <= 1:
        raise ValueError('Number of time steps should be greater than 1')
    

    x_pca = model.encode(x)
    x0 = x_pca[:, 0, :]
    x_pred = torch.zeros_like(x_pca, dtype=torch.float32, device=x.device)
    x_pred[:, 0, :] = x0
    for i in range(1, N):
        x1 = model.latent_to_latent_forward(x0, u[:, i-1, :], nu[:, i-1, :])
        x_pred[:, i, :] = x1
        x0 = x1
    
    loss = loss_fn(x_pred, x_pca)
    return loss

def koopman_loss_DicWithInputs_pca_ver(model, x_pca, u, nu):
    loss_fn = nn.MSELoss()
    N = x_pca.shape[1]
    if N <= 1:
        raise ValueError('Number of time steps should be greater than 1')
    
    x0 = x_pca[:, 0, :]
    x_pred = torch.zeros_like(x_pca, dtype=torch.float32, device=x_pca.device)
    x_pred[:, 0, :] = x0
    for i in range(1, N):
        x1 = model.latent_to_latent_forward(x0, u[:, i-1, :], nu[:, i-1, :])
        x_pred[:, i, :] = x1
        x0 = x1
    
    loss = loss_fn(x_pred, x_pca)
    return loss

def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    i = 0
    for x, u, nu in train_loader:
        x, u, nu = x.to(device), u.to(device), nu.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model, x, u, nu)
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"Batch {i}: Loss is nan or inf, stopping training.")
            break
        i += 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(train_loader)

def test_one_epoch(model, test_loader,loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, u, nu in test_loader:
            x, u, nu = x.to(device), u.to(device), nu.to(device)
            loss = loss_fn(model, x, u, nu)
            total_loss += loss.item()
    return total_loss/len(test_loader)



def main(config):

    # Save dir
    save_dir = config['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # todo: multi nu
    nu = config['nu']
    nu_list = [nu]

    # Data loader
    x_data, u_data, nu_data, n_features, n_inputs = data_preparation_xu(config, nu_list, nu)
        
    # Params
    params = km.Params(n_features, n_inputs, config)

    # Model
    if config['experiment'] == 'linear':
        model = km.build_model(params, x_data, u_data)
        loss_fn = koopman_loss
    elif config['experiment'] == 'DicWithInputs':
        model = km.build_model_DicWithInputs(params, x_data, u_data)
        loss_fn = koopman_loss_DicWithInputs
    elif config['experiment'] == 'MatrixWithInputs':
        model = km.build_model_MatrixWithInputs(params, x_data, u_data)
        if config.get('loss', 'non_extract') == 'extract':
            loss_fn = koopman_loss_extract
        else:
            loss_fn = koopman_loss
    else:
        raise ValueError('Experiment not supported')
    
    # model_path = os.path.join(config['save_dir'], 'model.pth')
    model = model.to(device)
    # model.load_state_dict(torch.load(model_path))

    # slices
    window_size = config['window_size']
    predict_num = config['predict_num']
    x_data_slices = cut_slides(x_data, window_size - 1, predict_num)
    u_data_slices = cut_slides(u_data, window_size - 1, predict_num)
    nu_data_slices = cut_slides(nu_data, window_size - 1, predict_num)

    x_data = np.concatenate(x_data_slices, axis=0)
    u_data = np.concatenate(u_data_slices, axis=0)
    nu_data = np.concatenate(nu_data_slices, axis=0)
    

    shuffled_indices = np.arange(len(x_data))
    np.random.shuffle(shuffled_indices)

    x_data = x_data[shuffled_indices]
    u_data = u_data[shuffled_indices]
    nu_data = nu_data[shuffled_indices]


    x_train, x_test, u_train, u_test, nu_train, nu_test = train_test_split(x_data, u_data, nu_data, test_size=0.2, random_state=42)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    u_train = torch.tensor(u_train, dtype=torch.float32)
    nu_train = torch.tensor(nu_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    u_test = torch.tensor(u_test, dtype=torch.float32)
    nu_test = torch.tensor(nu_test, dtype=torch.float32)

    train_dataset = TensorDataset(x_train, u_train, nu_train)
    test_dataset = TensorDataset(x_test, u_test, nu_test)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=config['lr'])
    scheduler = StepLR(optimizer, step_size=100, gamma=0.8)

    # Train
    train_losses = []
    test_losses = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    for epoch in range(config['num_epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        test_loss = test_one_epoch(model, test_loader, loss_fn, device)
        print(f'Epoch {epoch+1}/{config["num_epochs"]}, Train Loss: {train_loss}, Test Loss: {test_loss}')
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        scheduler.step()

        if test_loss < best_loss:
            best_loss = test_loss
            best_model_wts = copy.deepcopy(model.state_dict())
    
    torch.save(best_model_wts, os.path.join(save_dir, 'model.pth'))

    # Plot
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    np.save(os.path.join(save_dir, 'train_losses.npy'), train_losses)
    np.save(os.path.join(save_dir, 'test_losses.npy'), test_losses)

    return


def main_multinu(config):

    # Save dir
    save_dir = config['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # todo: multi nu
    nu_list = config['nu_list']

    # Data loader
    x_dataset, u_dataset, nu_dataset = [], [], []
    for i in range(len(nu_list)):
        nu = nu_list[i]
        config['data_dir'] = config['data_dir_list'][i]
        x_data, u_data, nu_data, n_features, n_inputs = data_preparation_xu(config, nu_list, nu)
        x_dataset.append(x_data)
        u_dataset.append(u_data)
        nu_dataset.append(nu_data)
        # print(np.max(x_data))
        # print(np.min(x_data))
        # print(nu_data.shape)
        # print(nu_data[1, :])
    
    # Params
    params = km.Params(n_features, n_inputs, config)

    x_data = np.concatenate(x_dataset, axis=0)
    u_data = np.concatenate(u_dataset, axis=0)
    nu_data = np.concatenate(nu_dataset, axis=0)

    # Model
    if config['experiment'] == 'linear':
        model = km.build_model(params, x_data, u_data)
        loss_fn = koopman_loss
    elif config['experiment'] == 'DicWithInputs':
        model = km.build_model_DicWithInputs(params, x_data, u_data)
        loss_fn = koopman_loss_DicWithInputs
    elif config['experiment'] == 'MatrixWithInputs':
        model = km.build_model_MatrixWithInputs(params, x_data, u_data)
        if config.get('loss', 'non_extract') == 'extract':
            loss_fn = koopman_loss_extract
        else:
            loss_fn = koopman_loss
    else:
        raise ValueError('Experiment not supported')
    
    # model_path = os.path.join(config['save_dir'], 'model.pth')
    model = model.to(device)
    # model.load_state_dict(torch.load(model_path))

    # slices
    window_size = config['window_size']
    predict_num = config['predict_num']
    x_data_slices = cut_slides(x_data, window_size - 1, predict_num)
    u_data_slices = cut_slides(u_data, window_size - 1, predict_num)
    nu_data_slices = cut_slides(nu_data, window_size - 1, predict_num)

    x_data = np.concatenate(x_data_slices, axis=0)
    u_data = np.concatenate(u_data_slices, axis=0)
    nu_data = np.concatenate(nu_data_slices, axis=0)
    

    shuffled_indices = np.arange(len(x_data))
    np.random.shuffle(shuffled_indices)

    x_data = x_data[shuffled_indices]
    u_data = u_data[shuffled_indices]
    nu_data = nu_data[shuffled_indices]


    x_train, x_test, u_train, u_test, nu_train, nu_test = train_test_split(x_data, u_data, nu_data, test_size=0.2, random_state=42)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    u_train = torch.tensor(u_train, dtype=torch.float32)
    nu_train = torch.tensor(nu_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    u_test = torch.tensor(u_test, dtype=torch.float32)
    nu_test = torch.tensor(nu_test, dtype=torch.float32)

    train_dataset = TensorDataset(x_train, u_train, nu_train)
    test_dataset = TensorDataset(x_test, u_test, nu_test)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=config['lr'])
    scheduler = StepLR(optimizer, step_size=100, gamma=0.8)

    # Train
    train_losses = []
    test_losses = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    for epoch in range(config['num_epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        test_loss = test_one_epoch(model, test_loader, loss_fn, device)
        print(f'Epoch {epoch+1}/{config["num_epochs"]}, Train Loss: {train_loss}, Test Loss: {test_loss}')
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        scheduler.step()

        if test_loss < best_loss:
            best_loss = test_loss
            best_model_wts = copy.deepcopy(model.state_dict())
    
    torch.save(best_model_wts, os.path.join(save_dir, 'model.pth'))

    # Plot
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    np.save(os.path.join(save_dir, 'train_losses.npy'), train_losses)
    np.save(os.path.join(save_dir, 'test_losses.npy'), test_losses)

    return

def main_multinu_ver2(config):
    # Save dir
    save_dir = config['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # todo: multi nu
    nu_list = config['nu_list']

    # -------------REAL DATA----------------
    # Data loader
    x_dataset, u_dataset, nu_dataset = [], [], []
    for i in range(len(nu_list)):
        nu = nu_list[i]
        config['data_dir'] = config['data_dir_list'][i]
        x_data, u_data, nu_data, n_features, n_inputs = data_preparation_xu(config, nu_list, nu)
        x_dataset.append(x_data)
        u_dataset.append(u_data)
        nu_dataset.append(nu_data)
    # -------------REAL DATA----------------

    # Params
    params = km.Params(n_features, n_inputs, config)

    # Model
    if config['experiment'] == 'linear':
        model, x_pca_scaled = km.build_model_linear_multi_nu(params, x_dataset, u_dataset)
    if config['experiment'] == 'DicWithInputs':
        model, x_pca_scaled = km.build_model_DicWithInputs_multi_nu(params, x_dataset, u_dataset)
    if config['experiment'] == 'MatrixWithInputs':
        model, x_pca_scaled = km.build_model_MatrixWithInputs_multi_nu(params, x_dataset, u_dataset)

    model = model.to(device)
    
    # Rescale and Slices
    x_data = x_pca_scaled.cpu().numpy()
    u_data = np.concatenate(u_dataset, axis=0)
    nu_data = np.concatenate(nu_dataset, axis=0)

    window_size = config['window_size']
    predict_num = config['predict_num']
    x_data_slices = cut_slides(x_data, window_size - 1, predict_num)
    u_data_slices = cut_slides(u_data, window_size - 1, predict_num)
    nu_data_slices = cut_slides(nu_data, window_size - 1, predict_num)

    x_data = np.concatenate(x_data_slices, axis=0)
    u_data = np.concatenate(u_data_slices, axis=0)
    nu_data = np.concatenate(nu_data_slices, axis=0)
    

    shuffled_indices = np.arange(len(x_data))
    np.random.shuffle(shuffled_indices)

    x_data = x_data[shuffled_indices]
    u_data = u_data[shuffled_indices]
    nu_data = nu_data[shuffled_indices]
    print(x_data.shape)


    x_train, x_test, u_train, u_test, nu_train, nu_test = train_test_split(x_data, u_data, nu_data, test_size=0.2, random_state=42)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    u_train = torch.tensor(u_train, dtype=torch.float32)
    nu_train = torch.tensor(nu_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    u_test = torch.tensor(u_test, dtype=torch.float32)
    nu_test = torch.tensor(nu_test, dtype=torch.float32)

    train_dataset = TensorDataset(x_train, u_train, nu_train)
    test_dataset = TensorDataset(x_test, u_test, nu_test)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=config['lr'])
    scheduler = StepLR(optimizer, step_size=100, gamma=0.8)

    # Loss function
    loss_fn = koopman_loss_DicWithInputs_pca_ver

    # Train
    train_losses = []
    test_losses = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    for epoch in range(config['num_epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        test_loss = test_one_epoch(model, test_loader, loss_fn, device)
        print(f'Epoch {epoch+1}/{config["num_epochs"]}, Train Loss: {train_loss}, Test Loss: {test_loss}')
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        scheduler.step()

        if test_loss < best_loss:
            best_loss = test_loss
            best_model_wts = copy.deepcopy(model.state_dict())
    
    torch.save(best_model_wts, os.path.join(save_dir, 'model.pth'))

    # Plot
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    np.save(os.path.join(save_dir, 'train_losses.npy'), train_losses)
    np.save(os.path.join(save_dir, 'test_losses.npy'), test_losses)



def main_cookup_data(config):
    # Save dir
    save_dir = config['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # nu_list = config['nu_list']

     ## -------------Cookup Data----------------
    # Set the random seed for reproducibility
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Define the dimensions
    state_dim = 4  # Dimension of the state vector x_n
    control_dim = 2  # Dimension of the control vector u_n
    num_trajectories = 5000  # Number of different initial states x_0
    trajectory_length = 2  # Number of steps for each trajectory

    # Initialize the matrices A and B
    A = torch.randn(state_dim, state_dim)
    B = torch.randn(state_dim, control_dim)

    # Initialize the datasets
    x_dataset = []
    u_dataset = []
    nu_dataset = []

    # Generate the trajectories
    for i in range(num_trajectories):
        x_n = torch.randn(state_dim)  # Initial state x_0
        x_traj = []
        u_traj = []
        nu_traj = np.ones((trajectory_length, 1))  # nu_dataset with all ones
        for t in range(trajectory_length):
            x_traj.append(x_n.numpy())
            u_n = torch.randn(control_dim)  # Random control input u_n
            x_n = A @ x_n + B @ u_n  # Compute the next state x_{n+1}
            u_traj.append(u_n.numpy())
        x_dataset.append(np.array(x_traj))
        u_dataset.append(np.array(u_traj))
        nu_dataset.append(nu_traj)
    
    x_data = np.concatenate(x_dataset, axis=0)
    u_data = np.concatenate(u_dataset, axis=0)
    nu_data = np.concatenate(nu_dataset, axis=0)

    # Params
    params = km.Params(state_dim, control_dim, config)

    # Model
    if config['experiment'] == 'linear':
        model, x_pca_scaled = km.build_model_linear_multi_nu(params, [x_data], [u_data])
    if config['experiment'] == 'DicWithInputs':
        model, x_pca_scaled = km.build_model_DicWithInputs_multi_nu(params,[x_data], [u_data])
    if config['experiment'] == 'MatrixWithInputs':
        model, x_pca_scaled = km.build_model_MatrixWithInputs_multi_nu(params, [x_data], [u_data])
    model = model.to(device)

    print(model)

    # Rescale and Slices
    x_data = x_pca_scaled.cpu().numpy()
    u_data = np.concatenate(u_dataset, axis=0)
    nu_data = np.concatenate(nu_dataset, axis=0)
    print(x_data.shape)
    print(u_data.shape)
    print(nu_data.shape)

    window_size = config['window_size']
    predict_num = config['predict_num']
    x_data_slices = cut_slides(x_data, window_size, predict_num)
    u_data_slices = cut_slides(u_data, window_size, predict_num)
    nu_data_slices = cut_slides(nu_data, window_size, predict_num)
    

    x_data = np.concatenate(x_data_slices, axis=0)
    u_data = np.concatenate(u_data_slices, axis=0)
    nu_data = np.concatenate(nu_data_slices, axis=0)
    

    shuffled_indices = np.arange(len(x_data))
    np.random.shuffle(shuffled_indices)

    x_data = x_data[shuffled_indices]
    u_data = u_data[shuffled_indices]
    nu_data = nu_data[shuffled_indices]
    print(x_data.shape)


    x_train, x_test, u_train, u_test, nu_train, nu_test = train_test_split(x_data, u_data, nu_data, test_size=0.2, random_state=42)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    u_train = torch.tensor(u_train, dtype=torch.float32)
    nu_train = torch.tensor(nu_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    u_test = torch.tensor(u_test, dtype=torch.float32)
    nu_test = torch.tensor(nu_test, dtype=torch.float32)

    train_dataset = TensorDataset(x_train, u_train, nu_train)
    test_dataset = TensorDataset(x_test, u_test, nu_test)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=config['lr'])
    scheduler = StepLR(optimizer, step_size=100, gamma=0.8)

    # Loss function
    loss_fn = koopman_loss_DicWithInputs_pca_ver

    # Train
    train_losses = []
    test_losses = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    for epoch in range(config['num_epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        test_loss = test_one_epoch(model, test_loader, loss_fn, device)
        print(f'Epoch {epoch+1}/{config["num_epochs"]}, Train Loss: {train_loss}, Test Loss: {test_loss}')
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        scheduler.step()

        if test_loss < best_loss:
            best_loss = test_loss
            best_model_wts = copy.deepcopy(model.state_dict())
    
    torch.save(best_model_wts, os.path.join(save_dir, 'model.pth'))

    # Plot
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    np.save(os.path.join(save_dir, 'train_losses.npy'), train_losses)
    np.save(os.path.join(save_dir, 'test_losses.npy'), test_losses)

if __name__ == '__main__':
    args = parse_arguments()
    config = read_config_file(args.config)
    main_multinu_ver2(config)
    # main_cookup_data(config)   







