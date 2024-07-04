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
from joblib import dump, load

def parse_arguments():
    parser = argparse.ArgumentParser(description='Reaction-diffusion model')
    parser.add_argument('--config', type=str, help='configuration file path')
    args = parser.parse_args()
    return args

def read_config_file(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def build_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    # Params
    params = km.Params(n_features, n_inputs, config)

    # Model
    if config['experiment'] == 'linear':
        model, _ = km.build_model_linear_multi_nu(params, x_dataset, u_dataset)
    if config['experiment'] == 'DicWithInputs':
        model, _ = km.build_model_DicWithInputs_multi_nu(params, x_dataset, u_dataset)
    if config['experiment'] == 'MatrixWithInputs':
        model, _ = km.build_model_MatrixWithInputs_multi_nu(params, x_dataset, u_dataset)
    model = model.to(device)

    model_dir = config['model_dir']
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth'), map_location=device))
    model.eval()
    return model

def find_optimal_nu(model, data_loader_1, nu_dim=3, learning_rate=0.01, num_epochs=100, device='cpu'):
    nu_initial = torch.rand(nu_dim, requires_grad=True, dtype=torch.float32, device=device)
    optimizer = Adam([nu_initial], lr=learning_rate)
    model.to(device)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for x0, x1, u in data_loader_1:
            x0 = x0.to(device, dtype=torch.float32)
            x1 = x1.to(device, dtype=torch.float32)
            u = u.to(device, dtype=torch.float32)
            
            batch_size = x0.shape[0]
            
            nu_expanded = nu_initial.unsqueeze(0).expand(batch_size, -1)
            
            x1_pred = model.latent_to_latent_forward(x0, u, nu_expanded)
            
            loss = torch.nn.functional.mse_loss(x1_pred, x1)
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_loader_1)}')
    
    return nu_initial.detach().cpu().numpy()


def data_preparation(config, data_dir):
    window_size = config['window_size']

    # Load data
    for item in os.listdir(data_dir):
        data_file_path = os.path.join(data_dir, item)

        # Check if the file exists before trying to load it
        x_dataset, y_dataset, u_dataset = [], [], []
        if os.path.exists(data_file_path) and data_file_path.endswith('.npy'):
            
            data_dict = np.load(data_file_path, allow_pickle=True).item()
            x_data, y_data, u_data, _ = load_dataset(data_dict)
            x_dataset.append(x_data[1:window_size])
            y_dataset.append(y_data[1:window_size])
            u_dataset.append(u_data[1:window_size])
        else:
            print(f"File not found: {data_file_path}")

    # Concatenate data
    x_data = np.concatenate(x_dataset, axis=0)
    y_data = np.concatenate(y_dataset, axis=0)
    u_data = np.concatenate(u_dataset, axis=0)

    return x_data, y_data, u_data


def test_linear_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nu_list = config['nu_list']

    # Linear model
    model = build_model(config)

    # Data loader
    x_data, y_data, u_data = data_preparation(config, config['fit_data_dir'])
    x_data = torch.tensor(x_data, dtype=torch.float32).to(device)
    y_data = torch.tensor(y_data, dtype=torch.float32).to(device)
    u_data = torch.tensor(u_data, dtype=torch.float32).to(device)
    x_mean = torch.mean(x_data, dim=0)
    x_std = torch.std(x_data, dim=0)
    std_layer_x = km.StdScalerLayer(x_mean, x_std)
    x_data_scaled = std_layer_x.transform(x_data)
    y_data_scaled = std_layer_x.transform(y_data)
    x_data_scaled_pca = model.pca_transformer.transform(x_data_scaled)
    y_data_scaled_pca = model.pca_transformer.transform(y_data_scaled)
    
    # Data loader
    data = TensorDataset(x_data_scaled_pca, y_data_scaled_pca, u_data)
    data_loader = DataLoader(data, batch_size=1024, shuffle=True)

    # Find optimal nu
    nu_optimal = find_optimal_nu(model, data_loader, nu_dim=3, learning_rate=0.001, num_epochs=10000, device=device)
    print(f'Optimal nu: {nu_optimal}')

    # Save optimal nu
    nu_optimal_path = os.path.join(config['save_dir'], 'nu_optimal.npy')
    np.save(nu_optimal_path, nu_optimal)

    return

def main():
    args = parse_arguments()
    config = read_config_file(args.config)
    test_linear_model(config)

if __name__ == '__main__':
    main()
    



    