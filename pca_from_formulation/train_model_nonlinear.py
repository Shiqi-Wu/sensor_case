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

def find_optimal_nu(model, data_loader, std_scaler_x, std_scaler_u, nu_dim=3, learning_rate=0.01, num_epochs=100, device='cpu'):
    nu_initial = torch.rand(nu_dim, requires_grad=True, dtype=torch.float32, device=device)
    optimizer = Adam([nu_initial], lr=learning_rate)
    model.to(device)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for x0, x1, u in data_loader:
            x0 = x0.to(device, dtype=torch.float32)
            x0 = std_scaler_x.transform(x0)
            x1 = x1.to(device, dtype=torch.float32)
            x1 = std_scaler_x.transform(x1)
            u = u.to(device, dtype=torch.float32)
            u = std_scaler_u.transform(u)
            
            batch_size = x0.shape[0]
            
            nu_expanded = nu_initial.unsqueeze(0).expand(batch_size, -1)
            
            x1_pred = model.latent_to_latent_forward(x0, u, nu_expanded)
            
            loss = torch.nn.functional.mse_loss(x1_pred, x1)
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_loader)}')
    
    return nu_initial.detach().cpu().numpy()


def data_preparation(config, data_dir):
    window_size = config['window_size']
    print(f'window_size: {window_size}')
    x_dataset, y_dataset, u_dataset = [], [], []
    # Load data
    for item in os.listdir(data_dir):
        data_file_path = os.path.join(data_dir, item)

        # Check if the file exists before trying to load it
        
        if os.path.exists(data_file_path) and data_file_path.endswith('.npy'):
            
            data_dict = np.load(data_file_path, allow_pickle=True).item()
            x_data, y_data, u_data, _ = load_dataset(data_dict)
            # print(x_data.shape, y_data.shape, u_data.shape)
            x_dataset.append(x_data[1:window_size])
            y_dataset.append(y_data[1:window_size])
            u_dataset.append(u_data[1:window_size])
            # print(f"Loaded data from {data_file_path}")
        else:
            print(f"File not found: {data_file_path}")

    print(x_dataset[0].shape, y_dataset[0].shape, u_dataset[0].shape)
    # Concatenate data
    x_data = np.concatenate(x_dataset, axis=0)
    y_data = np.concatenate(y_dataset, axis=0)
    u_data = np.concatenate(u_dataset, axis=0)
    print(f'x_data shape: {x_data.shape}, y_data shape: {y_data.shape}, u_data shape: {u_data.shape}')

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

def prepare_data_residual(x_data, y_data, u_data, model, device, batch_size=128, shuffle=True):
    x_data = torch.tensor(x_data, dtype=torch.float32).to(device)
    y_data = torch.tensor(y_data, dtype=torch.float32).to(device)
    u_data = torch.tensor(u_data, dtype=torch.float32).to(device)

    if model is not None:
        model.to(device)
        x_data_reshape = x_data.view(-1, x_data.shape[-1])
        u_data_reshape = u_data.view(-1, u_data.shape[-1])

        
        # Calculate err_data
        pred_data = model(x_data_reshape, u_data_reshape)
        pred_data = pred_data.detach()
        
        # Reshape err_data to match y_data shape
        pred_data = pred_data.view(y_data.shape)
        err_data = y_data - pred_data
    else:
        err_data = y_data

     # Create a TensorDataset and DataLoader
    dataset = TensorDataset(x_data, err_data, u_data)
    data_loader = DataLoader(dataset,batch_size=batch_size, shuffle=shuffle)

    return data_loader, err_data

class LinearModel(nn.Module):
    def __init__(self, model, std_layer_x, nu):
        super(LinearModel, self).__init__()
        self.model = model
        self.std_layer_x = std_layer_x
        self.nu = nn.Parameter(torch.tensor(nu, dtype=torch.float32).to(std_layer_x.mean.device))


    def forward(self, x, u):
        x_scaled = self.std_layer_x.transform(x)
        x_pred = self.model.scaled_to_scaled_forward(x_scaled, u, self.nu)
        x_pred = self.std_layer_x.inverse_transform(x_pred)
        return x_pred

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=1):
        super(SimpleNN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        
        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_size)

        # Additional hidden layers
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_layers.append(nn.ReLU())

        # Output layer
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x, u):
        x = torch.cat((x, u), dim=1)
        x = self.input_layer(x)
        for i in range(0, len(self.hidden_layers), 2):
            linear_layer = self.hidden_layers[i]
            relu_layer = self.hidden_layers[i + 1]
            x = x + relu_layer(linear_layer(x))  # Residual connection
        x = self.output(x)
        return x

class ResidualModel(nn.Module):
    def __init__(self, model, std_layer_x, std_layer_u, std_layer_err):
        super(ResidualModel, self).__init__()
        self.model = model
        self.std_layer_x = std_layer_x
        self.std_layer_u = std_layer_u
        self.std_layer_err = std_layer_err

    def forward(self, x, u):
        x_scaled = self.std_layer_x.transform(x)
        u_scaled = self.std_layer_u.transform(u)
        x_pred = self.model(x_scaled, u_scaled)
        x_pred = self.std_layer_err.inverse_transform(x_pred)
        return x_pred

class ResidualModelWithPCA(nn.Module):
    def __init__(self, model, std_layer_x1, std_layer_x2, std_layer_u, std_layer_err, pca_transformer):
        super(ResidualModelWithPCA, self).__init__()
        self.model = model
        self.std_layer_x1= std_layer_x1
        self.std_layer_x2 = std_layer_x2
        self.std_layer_u = std_layer_u
        self.std_layer_err = std_layer_err
        self.pca_transformer = pca_transformer

    def forward(self, x, u):
        x_scaled = self.std_layer_x1.transform(x)
        u_scaled = self.std_layer_u.transform(u)
        x_pca = self.pca_transformer.transform(x_scaled)
        x_pca = self.std_layer_x2.transform(x_pca)
        x_pred = self.model(x_pca, u_scaled)
        # x_pred = self.std_layer_x2.inverse_transform(x_pred)
        # x_pred = self.pca_transformer.inverse_transform(x_pred)
        x_pred = self.std_layer_err.inverse_transform(x_pred)
        return x_pred

def train_residual_model(pre_trained_model, residual_NN, std_layer_x, std_layer_u, dataset, num_epochs=100, learning_rate=0.001, device='cpu'):

    pre_trained_model.to(device)
    residual_NN.to(device)
    pre_trained_model.eval()

    # Data loader
    x_data, y_data, u_data = dataset
    data_loader, err_data = prepare_data_residual(x_data, y_data, u_data, pre_trained_model, device)
    print(err_data.shape)

    # Model
    std_layer_err = km.StdScalerLayer(torch.mean(err_data, dim=0), torch.std(err_data, dim=0))
    residual_model = ResidualModel(residual_NN, std_layer_x, std_layer_u, std_layer_err)
    residual_model.to(device)
    residual_model.train()

    # Optimizer
    optimizer = Adam(residual_model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.8)

    # Training
    for epoch in range(num_epochs):
        total_loss = 0.0
        for x, err, u in data_loader:
            x = x.to(device, dtype=torch.float32)
            err = err.to(device, dtype=torch.float32)
            u = u.to(device, dtype=torch.float32)
            
            x_pred = residual_model(x, u)
            loss = torch.nn.functional.mse_loss(x_pred, err)
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_loader)}')
    
    return residual_model


def train_residual_model_multi_steps(linear_model, residual_NN, std_layer_x, std_layer_u, dataset, num_epochs=100, learning_rate=0.001, device='cpu'):
    residual_NN.to(device)

    if linear_model is not None:
        linear_model.to(device)
        linear_model.eval()

    # Data loader
    x_data, y_data, u_data = dataset
    data_loader, _ = prepare_data_residual(x_data, y_data, u_data, None, device)

    _, err_data = prepare_data_residual(x_data, y_data, u_data, linear_model, device)

    err_data = err_data.detach()
    err_data = err_data.reshape(-1, err_data.shape[-1])
    # Model
    std_layer_err = km.StdScalerLayer(torch.mean(err_data, dim=0), torch.std(err_data, dim=0))
    # print(std_layer_err.mean, std_layer_err.std)
    residual_model = ResidualModel(residual_NN, std_layer_x, std_layer_u, std_layer_err)
    residual_model.to(device)
    residual_model.train()

    # Optimizer
    optimizer = Adam(residual_model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.8)

    # Training
    for epoch in range(num_epochs):
        i = 0
        for x, _, u in data_loader:
            x = x.to(device, dtype=torch.float32)
            u = u.to(device, dtype=torch.float32)
            loss = loss_fn(linear_model, residual_model, x, u)
            optimizer.zero_grad()
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"Batch {i}: Loss is nan or inf, stopping training.")
                break
            i += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(residual_model.parameters(), max_norm=1)
            optimizer.step()
        
        scheduler.step()
    
    print(f'Training Residual Model: Epoch {epoch+1}/{num_epochs}, Loss: {loss}')
    return residual_model

def train_residual_model_multi_steps_PCA(linear_model, residual_NN, std_layer_x1, std_layer_x2, std_layer_u, pac_transformer, dataset, num_epochs=100, learning_rate=0.001, device='cpu'):
    residual_NN.to(device)

    if linear_model is not None:
        linear_model.to(device)
        linear_model.eval()

    # Data loader
    x_data, y_data, u_data = dataset
    data_loader, _ = prepare_data_residual(x_data, y_data, u_data, None, device)

    _, err_data = prepare_data_residual(x_data, y_data, u_data, linear_model, device)

    err_data = err_data.detach()
    err_data = err_data.reshape(-1, err_data.shape[-1])
    # Model
    std_layer_err = km.StdScalerLayer(torch.mean(err_data, dim=0), torch.std(err_data, dim=0))
    # print(std_layer_err.mean, std_layer_err.std)
    residual_model = ResidualModelWithPCA(residual_NN, std_layer_x1, std_layer_x2, std_layer_u, std_layer_err, pac_transformer)
    residual_model.to(device)
    residual_model.train()

    # Optimizer
    optimizer = Adam(residual_model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.8)

    # Training
    for epoch in range(num_epochs):
        i = 0
        for x, _, u in data_loader:
            x = x.to(device, dtype=torch.float32)
            u = u.to(device, dtype=torch.float32)
            loss = loss_fn(linear_model, residual_model, x, u)
            optimizer.zero_grad()
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"Batch {i}: Loss is nan or inf, stopping training.")
                break
            i += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(residual_model.parameters(), max_norm=1)
            optimizer.step()
        
        scheduler.step()
    
    print(f'Training Residual Model: Epoch {epoch+1}/{num_epochs}, Loss: {loss}')
    return residual_model

def train_linear_model(linear_model, residual_model, dataset, num_epochs=100, learning_rate=0.001, device='cpu'):
    linear_model.to(device)
    linear_model.train()
    if residual_model is not None:
        residual_model.to(device)
        residual_model.eval()

    # Data loader
    x_data, y_data, u_data = dataset
    data_loader, err_data = prepare_data_residual(x_data, y_data, u_data, residual_model, device)

    # Optimizer
    optimizer = Adam(linear_model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.8)

    # Training
    for epoch in range(num_epochs):
        total_loss = 0.0
        for x, err, u in data_loader:
            x = x.to(device, dtype=torch.float32)
            err = err.to(device, dtype=torch.float32)
            u = u.to(device, dtype=torch.float32)
            
            x_pred = linear_model(x, u)
            loss = torch.nn.functional.mse_loss(x_pred, err)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        # print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_loader)}')

    return linear_model

def train_linear_model_multi_steps(linear_model, residual_model, dataset, num_epochs=100, learning_rate=0.0001, device='cpu'):
    linear_model.to(device)
    linear_model.train()
    if residual_model is not None:
        residual_model.to(device)
        residual_model.eval()

    # Data loader
    x_data, y_data, u_data = dataset
    data_loader, _ = prepare_data_residual(x_data, y_data, u_data, None, device)

    # Optimizer
    optimizer = Adam(linear_model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.8)

    # Training
    for epoch in range(num_epochs):
        i = 0
        for x, _, u in data_loader:
            x = x.to(device, dtype=torch.float32)
            u = u.to(device, dtype=torch.float32)
            # print(x.shape, u.shape)
            loss = loss_fn(linear_model, residual_model, x, u)
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"Batch {i}: Loss is nan or inf, stopping training.")
                break
            i += 1
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(linear_model.parameters(), max_norm=1)
            optimizer.step()
            
        
        scheduler.step()
        # print(f'Training Linear Model: Epoch {epoch+1}/{num_epochs}, Loss: {loss}')
    return linear_model



def loss_fn(linear_model, residual_model, x_data, u_data):
    mse = nn.MSELoss()
    N = x_data.shape[1]
    if N <= 1:
        raise ValueError('Number of time steps should be greater than 1')
    
    # print(N)
    x0 = x_data[:, 0, :]
    x_pred = torch.zeros_like(x_data)
    x_pred[:, 0, :] = x0
    
    if residual_model is not None:
        for i in range(1, N):
            x_pred_cur = linear_model(x_pred[:, i-1, :], u_data[:, i-1, :]) + residual_model(x_pred[:, i-1, :], u_data[:, i-1, :])
            x_pred[:, i, :] = x_pred_cur

    else:
        for i in range(1, N):
            x_pred_cur = linear_model(x_pred[:, i-1, :], u_data[:, i-1, :])
            x_pred[:, i, :] = x_pred_cur


    loss =  mse(x_pred, x_data)
    return loss
    

                    


def iterative_training(config, pre_trained_model, dataset):
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    iterations = config['iterations']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loader
    x_data, y_data, u_data = dataset
    x_mean = torch.mean(x_data, dim=0)
    x_std = torch.std(x_data, dim=0)
    std_layer_x = km.StdScalerLayer(x_mean, x_std)
    u_mean = torch.mean(u_data, dim=0)
    u_std = torch.std(u_data, dim=0)
    std_layer_u = km.StdScalerLayer(u_mean, u_std)
    linear_model = LinearModel(pre_trained_model, std_layer_x, np.zeros(len(config['nu_list'])))

    # Model
    n_features = x_data.shape[1]
    n_inputs = u_data.shape[1]
    if config['PCA'] == True:
        x_data_scaled = std_layer_x.transform(x_data)
        pca = PCA(n_components=config['PCA_dim'])
        x_data_scaled = x_data_scaled.detach().cpu().numpy()
        x_data_pca = pca.fit_transform(x_data_scaled)
        components = pca.components_
        pca_matrix = torch.tensor(components, dtype=torch.float32).to(device)
        pca_layer = km.PCALayer(n_features, config['PCA_dim'], pca_matrix)
        residual_NN = SimpleNN(config['PCA_dim'] + n_inputs, config['NN_nodes_num'], n_features, num_hidden_layers=config['num_hidden_layers'])
        x_data_pca = torch.tensor(x_data_pca, dtype=torch.float32).to(device)
        std_layer_x2 = km.StdScalerLayer(torch.mean(x_data_pca, dim=0), torch.std(x_data_pca, dim=0))
    else:
        residual_NN = SimpleNN(n_features + n_inputs, config['NN_nodes_num'], n_features, num_hidden_layers=config['num_hidden_layers'])

    residual_model = None
        
    # Iterative training
    iteration_errors = []
    for i in range(iterations):
        print(f'Iteration {i+1}/{iterations}')
        if config['predict_num'] == 1:
            linear_model = train_linear_model(linear_model, residual_model, dataset, num_epochs, learning_rate, device)
            if config['PCA'] == True:
                raise ValueError('PCA is not supported for single step prediction (to be implemented)')
            else:
                residual_model = train_residual_model(linear_model, residual_NN, std_layer_x, std_layer_u, dataset, num_epochs, learning_rate, device)
        else:
            x_data_slices = cut_slices(x_data, config['window_size'] - 1, config['predict_num'])
            y_data_slices = cut_slices(y_data, config['window_size'] - 1, config['predict_num'])
            u_data_slices = cut_slices(u_data, config['window_size'] - 1, config['predict_num'])

            x_data_slices = torch.cat(x_data_slices, dim=0)
            y_data_slices = torch.cat(y_data_slices, dim=0)
            u_data_slices = torch.cat(u_data_slices, dim=0)
            dataset = [x_data_slices, y_data_slices, u_data_slices]
            linear_model = train_linear_model_multi_steps(linear_model, residual_model, dataset, num_epochs, learning_rate, device)
            torch.cuda.empty_cache()
            if config['PCA'] == True:
                residual_model = train_residual_model_multi_steps_PCA(linear_model, residual_NN, std_layer_x, std_layer_x2, std_layer_u, pca_layer, dataset, num_epochs, learning_rate, device)
            else:
                residual_model = train_residual_model_multi_steps(linear_model, residual_NN, std_layer_x, std_layer_u, dataset, num_epochs, learning_rate, device)

        residual_NN = copy.deepcopy(residual_model.model)

        linear_pred = linear_model(x_data, u_data).detach()
        residual_pred = residual_model(x_data, u_data).detach()

        combined_pred = linear_pred + residual_pred

        combined_pred = combined_pred.cpu().numpy()
        y_data_cpu = y_data.cpu().numpy()
    
        error = mean_squared_error(y_data_cpu, combined_pred)
        iteration_errors.append(error)
        print(f'Error: {error}')
    return linear_model, residual_model, iteration_errors

def train_together(config, pre_trained_model, dataset):
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    iterations = config['iterations']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loader
    x_data, y_data, u_data = dataset
    x_mean = torch.mean(x_data, dim=0)
    x_std = torch.std(x_data, dim=0)
    std_layer_x = km.StdScalerLayer(x_mean, x_std)
    u_mean = torch.mean(u_data, dim=0)
    u_std = torch.std(u_data, dim=0)
    std_layer_u = km.StdScalerLayer(u_mean, u_std)
    linear_model = LinearModel(pre_trained_model, std_layer_x, np.zeros(len(config['nu_list'])))
    
    x_data_slices = cut_slides(x_data, config['window_size'] - 1, config['predict_num'])
    y_data_slices = cut_slides(y_data, config['window_size'] - 1, config['predict_num'])
    u_data_slices = cut_slides(u_data, config['window_size'] - 1, config['predict_num'])

    x_data_slices = torch.cat(x_data_slices, dim=0)
    y_data_slices = torch.cat(y_data_slices, dim=0)
    u_data_slices = torch.cat(u_data_slices, dim=0)
    
    n_features = x_data.shape[1]
    n_inputs = u_data.shape[1]
    residual_NN = SimpleNN(n_features + n_inputs, config['NN_nodes_num'], n_features, num_hidden_layers=config['num_hidden_layers'])
    x_data, y_data, u_data = dataset
    data_loader, _ = prepare_data_residual(x_data_slices, y_data_slices, u_data_slices, None, device)

    _, err_data = prepare_data_residual(x_data, y_data, u_data, linear_model, device)
    err_data = err_data.detach()
    err_data = err_data.reshape(-1, err_data.shape[-1])
    # Model
    std_layer_err = km.StdScalerLayer(torch.mean(err_data, dim=0), torch.std(err_data, dim=0))
    # print(std_layer_err.mean, std_layer_err.std)
    residual_model = ResidualModel(residual_NN, std_layer_x, std_layer_u, std_layer_err)
    residual_model.to(device)
    residual_model.train()
    linear_model.to(device)
    linear_model.train()
    linear_model.load_state_dict(torch.load('/home/shiqi/code/Project2-sensor-case/model_combination_Argos/pca_from_formulation/output_hybrid/experiment_1/linear_model.pth'))
    residual_model.load_state_dict(torch.load('/home/shiqi/code/Project2-sensor-case/model_combination_Argos/pca_from_formulation/output_hybrid/experiment_1/residual_model.pth'))

    # Optimizer
    optimizer = Adam(list(linear_model.parameters()) + list(residual_model.parameters()), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.8)

    # Training
    losses = []
    for epoch in range(num_epochs):
        i = 0
        total_loss = 0.0
        for x, _, u in data_loader:
            x = x.to(device, dtype=torch.float32)
            u = u.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            loss = loss_fn(linear_model, residual_model, x, u)
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"Batch {i}: Loss is nan or inf, stopping training.")
                break
            i += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(linear_model.parameters(), max_norm=1)
            optimizer.step()
            total_loss += loss.item()
        total_loss = total_loss / len(data_loader)
            
        
        scheduler.step()
        
        print(f'Training Together: Epoch {epoch+1}/{num_epochs}, Loss: {total_loss}')
        losses.append(total_loss)
    return linear_model, residual_model, losses
    
    


def main():
    args = parse_arguments()
    config = read_config_file(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Linear model
    pre_trained_model = build_model(config)

    # Data loader
    x_data, y_data, u_data = data_preparation(config, config['fit_data_dir'])
    x_data = torch.tensor(x_data, dtype=torch.float32).to(device)
    y_data = torch.tensor(y_data, dtype=torch.float32).to(device)
    u_data = torch.tensor(u_data, dtype=torch.float32).to(device)
    dataset = [x_data, y_data, u_data]

    linear_model, residual_model, iteration_errors = iterative_training(config, pre_trained_model, dataset)

    # Save models
    model_dir = config['save_dir']
    torch.save(linear_model.state_dict(), os.path.join(model_dir, 'linear_model.pth'))
    torch.save(residual_model.state_dict(), os.path.join(model_dir, 'residual_model.pth'))
    np.save(os.path.join(model_dir, 'iteration_errors.npy'), iteration_errors)
    return

def main_test():
    args = parse_arguments()
    config = read_config_file(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Linear model
    pre_trained_model = build_model(config)

    # Data loader
    x_data, y_data, u_data = data_preparation(config, config['fit_data_dir'])
    x_data = torch.tensor(x_data, dtype=torch.float32).to(device)
    y_data = torch.tensor(y_data, dtype=torch.float32).to(device)
    u_data = torch.tensor(u_data, dtype=torch.float32).to(device)
    
    dataset = [x_data, y_data, u_data]

    linear_model, residual_model, losses = train_together(config, pre_trained_model, dataset)

    # Save models
    model_dir = config['save_dir']
    torch.save(linear_model.state_dict(), os.path.join(model_dir, 'linear_model.pth'))
    torch.save(residual_model.state_dict(), os.path.join(model_dir, 'residual_model.pth'))
    np.save(os.path.join(model_dir, 'losses.npy'), losses)
    return

if __name__ == '__main__':
    main()
    



    