from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
sys.path.append('/home/shiqi/code/Project2-sensor-case/model_combination_Argos/utils')
import numpy as np
import torch
import torch.nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import pca_koopman_dir as km
import argparse
import os
import yaml
from load_dataset import cut_slices

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    args = parser.parse_args()
    return args

def read_config_file(config_file):
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

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

def train_residual_model(linear_model, residual_model, dataset, config, num_epoches = 100, learning_rate = 0.001, device = 'cpu'):

    linear_model.to(device)
    residual_model.to(device)
    linear_model.to(device)

    # Data Loader
    x_data, y_data, u_data = dataset

    x_data_slices = cut_slices(x_data, config['window_size'] - 1, config['predict_num'])
    y_data_slices = cut_slices(y_data, config['window_size'] - 1, config['predict_num'])
    u_data_slices = cut_slices(u_data, config['window_size'] - 1, config['predict_num'])

    data_loader, _ = prepare_data_residual(x_data_slices, y_data_slices, u_data_slices, None, device)

    _, err_data = prepare_data_residual(x_data, y_data, u_data, linear_model, device)

    err_data = err_data.detach()

    # scale the err_data
    std_layer_err = km.StdScalerLayer(torch.mean(err_data, dim=0), torch.std(err_data, dim=0))


    # Optimizer
    optimizer = Adam(residual_model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # Training
    for epoch in range(num_epoches):
        for x_batch, _, u_batch in data_loader:
            optimizer.zero_grad()
            x_batch = x_batch.to(device)
            u_batch = u_batch.to(device)
            optimizer.zero_grad()

            loss = hybrid_loss(linear_model, residual_model, std_layer_err, x_batch, u_batch)
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(residual_model.parameters(), max_norm=1)
        scheduler.step()
    
    print(f'Training Residual Model: Epoch {epoch+1}/{num_epoches}, Loss: {loss}')
    return residual_model, std_layer_err

def hybrid_loss(linear_model, residual_model, std_layer_err, x_data, u_data):
    mse = torch.nn.MSELoss()
    N = x_data.shape[1]
    if N <= 1:
        raise ValueError('Number of time steps should be greater than 1')
    
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
            x_pred_cur = std_layer_err.inverse_transform(x_pred_cur)
            x_pred[:, i, :] = x_pred_cur
    
    loss = mse(x_pred, x_data)
    return loss

def linear_regression(data_loader, linear_model, device='cpu'):
    for x, y, u in data_loader:
        x = x.to(device)
        y = y.to(device)
        u = u.to(device)
        
        x_dic = linear_model.state_dict(x)
        y_dic = linear_model.state_dict(y)
        u_dic = linear_model.control_dict(u)

        z = torch.cat((x_dic, u_dic), dim=1)
        z_pseudo_inv = torch.pinverse(z)

        param_pseudo_inv = torch.matmul(z_pseudo_inv, y_dic)
        A = param_pseudo_inv[:x_dic.shape[1], :]. T
        B = param_pseudo_inv[x_dic.shape[1]:, :].T

        with torch.no_grad():
            linear_model.A = A
            linear_model.B = B
    
    return linear_model

class linear_model(torch.nn.Module):
    def __init__(self, state_dim, output_dim):
        super(linear_model, self).__init__()
        self.A = torch.nn.Parameter(state_dim, state_dim)
        self.B = torch.nn.Parameter(state_dim, output_dim)
    
    def state_dict(self, x):
        ones = torch.ones(x.shape[0], 1).to(x.device)
        return torch.cat((x, ones), dim=1)
    
    def control_dict(self, u):
        return u
    
    def forward(self, x, u):
        x = self.state_dict(x)
        u = self.control_dict(u)
        y = torch.matmul(x, self.A) + torch.matmul(u, self.B)
        return y[:, 1:]
    
def iterative_training(dataset, linear_model, residual_model, num_iter=10, num_epoches=100, learning_rate=0.001, device='cpu'):
    # Initialize the linear model
    x_data, y_data, u_data = dataset
    data_loader, _ = prepare_data_residual(x_data, y_data, u_data, None, device, batch_size = torch.shape(x_data)[0], shuffle=True)
    linear_model = linear_regression(data_loader, linear_model, device)

    # Iterative training
    iterative_losses = []
    for i in range(num_iter):
        residual_model, std_layer_err = train_residual_model(linear_model, residual_model, dataset, num_epoches, learning_rate, device)
        linear_model = linear_regression(data_loader, linear_model, device)
        linear_pred = linear_model(x_data, u_data).detach()
        residual_pred_scaled = residual_model(x_data, u_data).detach()
        residual_pred = std_layer_err.inverse_transform(residual_pred_scaled)
        y_pred = linear_pred + residual_pred
        y_data = y_data.detach()
        mse = torch.nn.MSELoss()
        loss = mse(y_pred, y_data)
        print(f'Iterative Training: Iteration {i+1}/{num_iter}, Loss: {loss}')
        iterative_losses.append(loss)
    return linear_model, residual_model, std_layer_err, iterative_losses

def main():
    # Parse arguments
    args = parse_arguments()
    config = read_config_file(args.config)

    # Load data
    
        

        
    



