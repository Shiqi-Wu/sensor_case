from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
sys.path.append('../utils')
print(sys.path)
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import koopman_dir as km
from load_dataset import *
import argparse
import os
import yaml
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description='Reaction-diffusion model')
    parser.add_argument('--config', type=str, help='configuration file path')
    args = parser.parse_args()
    return args

def read_config_file(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def koopman_loss_function_multi_step(model, x, u, nu, lambda_1 = 1, lambda_2 = 1, lambda_3 = 1, relative_error = False):
    loss_fn = nn.MSELoss()

    N = x.shape[1]
    # print(N)
    if N <= 1:
        raise ValueError('Number of time steps should be greater than 1')
    
    # autoencoder loss
    x_reconstructed = model.auto_state(x)
    loss_2 = loss_fn(x_reconstructed, x)

    # Koopman loss
    x0 = x[:, 0:1, :]
    x_latent = model.encode_state(x)
    x0_latent = model.encode_state(x0)
    x_pred_latent = torch.zeros_like(x_latent)
    x_pred_latent[:, 0:1, :] = x0_latent
    for i in range(1, N):
        x0_latent = model.forward_latent_to_latent(x0_latent, u[:, i-1:i, :], nu[:, i-1:i, :])
        x_pred_latent[:, i:i+1, :] = x0_latent
    loss_3 = loss_fn(x_latent[:,1:,:], x_pred_latent[:,1:,:])

    # prediction loss
    x_pred = model.decode_state(x_pred_latent)
    loss_1 = loss_fn(x_pred[:,1:,:], x[:,1:,:])

    return lambda_1 * loss_1 + lambda_2 * loss_2 + lambda_3 * loss_3, loss_1, loss_2, loss_3
    
def dmd_loss_function(model, x, u, nu, relative_error = False):
    loss_fn = nn.MSELoss()

    N = x.shape[1]
    if N <= 1:
        raise ValueError('Number of time steps should be greater than 1')
    
    # Koopman loss
    x0 = x[:, 0:1, :]
    x_pred = torch.zeros_like(x)
    x_pred[:, 0:1, :] = x0
    for i in range(1, N):
        x0 = model(x0, u[:, i-1:i, :], nu[:, i-1:i, :])
        x_pred[:, i:i+1, :] = x0
    loss = loss_fn(x[:,1:,:], x_pred[:,1:,:])

    return loss

def train_one_epoch(model, optim, loss_fn, train_loader, epoch, device, lambda_1 = 1, lambda_2 = 1, lambda_3 = 1):
    print(f'Epoch {epoch}')
    train_loss, train_loss1, train_loss2, train_loss3 = 0, 0, 0, 0
    model.train()
    for i, data in enumerate(train_loader):
        x, u, nu = data
        x, u, nu = x.to(device), u.to(device), nu.to(device)
        optim.zero_grad()
        loss, loss_1, loss_2, loss_3 = loss_fn(model, x, u, nu, lambda_1, lambda_2, lambda_3)
        loss.backward()
        optim.step()
        train_loss += loss.item()
        train_loss1 += loss_1.item()
        train_loss2 += loss_2.item()
        train_loss3 += loss_3.item()

    train_loss /= len(train_loader)
    train_loss1 /= len(train_loader)
    train_loss2 /= len(train_loader)
    train_loss3 /= len(train_loader)

    print(f'Epoch {epoch}, Loss: {train_loss}, Loss1: {train_loss1}, Loss2: {train_loss2}, Loss3: {train_loss3}')
    return train_loss, train_loss1, train_loss2, train_loss3

def test_one_epoch(model, loss_fn, test_loader, epoch, device, lambda_1 = 1, lambda_2 = 1, lambda_3 = 1):
    test_loss, test_loss1, test_loss2, test_loss3 = 0, 0, 0, 0
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            x, u, nu = data
            x, u, nu = x.to(device), u.to(device), nu.to(device)
            loss, loss_1, loss_2, loss_3 = loss_fn(model, x, u, nu, lambda_1, lambda_2, lambda_3)
            test_loss += loss.item()
            test_loss1 += loss_1.item()
            test_loss2 += loss_2.item()
            test_loss3 += loss_3.item()
    
    test_loss /= len(test_loader)
    test_loss1 /= len(test_loader)
    test_loss2 /= len(test_loader)
    test_loss3 /= len(test_loader)
    
    print(f'Epoch {epoch}, Loss: {test_loss}, Loss1: {test_loss1}, Loss2: {test_loss2}, Loss3: {test_loss3}')
    return test_loss, test_loss1, test_loss2, test_loss3

def train_one_epoch_DMD(model, optim, loss_fn, train_loader, epoch, device, max_norm = 0.01):
    print(f'Epoch {epoch}')
    train_loss = 0
    model.train()
    for i, data in enumerate(train_loader):
        x, u, nu = data
        x, u, nu = x.to(device), u.to(device), nu.to(device)
        optim.zero_grad()
        loss = loss_fn(model, x, u, nu)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optim.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    print(f'Epoch {epoch}, Loss: {train_loss}')
    return train_loss

def test_one_epoch_DMD(model, loss_fn, test_loader, epoch, device):
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            x, u, nu = data
            x, u, nu = x.to(device), u.to(device), nu.to(device)
            loss = loss_fn(model, x, u, nu)
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    
    print(f'Epoch {epoch}, Loss: {test_loss}')
    return test_loss

def main(config):
    # Save dir
    save_dir = config['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # todo: multi nu
    nu = config['nu']
    nu_list = [nu]

    # Data loader
    train_dataset, test_dataset, x_dim, u_dim = data_preparation_koopman(config, nu_list, nu)
    train_loader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = config['batch_size'], shuffle = False)

    # Set params
    params = km.Params(x_dim = x_dim, u_dim = u_dim, config = config)


    # Model
    model = km.BuildModelFromParams(params).to(device)
    optim = Adam(model.parameters(), lr = config['lr'])
    loss_fn = koopman_loss_function_multi_step
    scheduler = StepLR(optim, step_size = 100, gamma = 0.9)
    train_losses, test_losses = [], []
    train_losses1, test_losses1 = [], []
    train_losses2, test_losses2 = [], []
    train_losses3, test_losses3 = [], []
    for epoch in range(config['num_epoches']):
        train_loss, train_loss1, train_loss2, train_loss3 = train_one_epoch(model, optim, loss_fn, train_loader, epoch, device, config['lambda_1'], config['lambda_2'], config['lambda_3'])
        test_loss, test_loss1, test_loss2, test_loss3 = test_one_epoch(model, loss_fn, test_loader, epoch, device, config['lambda_1'], config['lambda_2'], config['lambda_3'])
        scheduler.step()
        torch.save(model, os.path.join(save_dir, 'model.pth'))
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_losses1.append(train_loss1)
        test_losses1.append(test_loss1)
        train_losses2.append(train_loss2)
        test_losses2.append(test_loss2)
        train_losses3.append(train_loss3)
        test_losses3.append(test_loss3)
    
    np.save(os.path.join(save_dir, 'train_losses.npy'), train_losses)
    np.save(os.path.join(save_dir, 'test_losses.npy'), test_losses)
    np.save(os.path.join(save_dir, 'train_losses1.npy'), train_losses1)
    np.save(os.path.join(save_dir, 'test_losses1.npy'), test_losses1)
    np.save(os.path.join(save_dir, 'train_losses2.npy'), train_losses2)
    np.save(os.path.join(save_dir, 'test_losses2.npy'), test_losses2)
    np.save(os.path.join(save_dir, 'train_losses3.npy'), train_losses3)
    np.save(os.path.join(save_dir, 'test_losses3.npy'), test_losses3)

    plt.figure()
    plt.plot(train_losses, label = 'train_loss')
    plt.plot(test_losses, label = 'test_loss')
    plt.plot(train_losses1, label = 'train_prediction_loss')
    plt.plot(test_losses1, label = 'test_prediction_loss')
    plt.plot(train_losses2, label = 'train_autoencoder_loss')
    plt.plot(test_losses2, label = 'test_autoencoder_loss')
    plt.plot(train_losses3, label = 'train_koopman_loss')
    plt.plot(test_losses3, label = 'test_koopman_loss')

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.savefig(os.path.join(save_dir, 'train_test_loss.png'))

    return

def main_DMD(config):
    # Save dir
    save_dir = config['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # todo: multi nu
    nu = config['nu']
    nu_list = [nu]

    # Data loader
    train_dataset, test_dataset, x_dim, u_dim = data_preparation_koopman(config, nu_list, nu)
    train_loader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = config['batch_size'], shuffle = False)

    # Set params
    params = km.Params(x_dim = x_dim, u_dim = u_dim, config = config)

    # Model
    model = km.BuildModelFromParams_DMD(params).to(device)
    optim = Adam(model.parameters(), lr = config['lr'])
    loss_fn = dmd_loss_function
    scheduler = StepLR(optim, step_size = 100, gamma = 0.9)
    train_losses, test_losses = [], []
    for epoch in range(config['num_epoches']):
        train_loss = train_one_epoch_DMD(model, optim, loss_fn, train_loader, epoch, device)
        test_loss = test_one_epoch_DMD(model, loss_fn, test_loader, epoch, device)
        scheduler.step()
        torch.save(model, os.path.join(save_dir, 'model.pth'))
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    
    np.save(os.path.join(save_dir, 'train_losses.npy'), train_losses)
    np.save(os.path.join(save_dir, 'test_losses.npy'), test_losses)

    plt.figure()
    plt.plot(train_losses, label = 'train_loss')
    plt.plot(test_losses, label = 'test_loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.savefig(os.path.join(save_dir, 'train_test_loss.png'))

    return
    

if __name__ == '__main__':

    """
    Main function
    """
    args = parse_arguments()
    config = read_config_file(args.config)
    main(config)
    # main_DMD(config)


    """
    Test for shape
    
    """
    # args = parse_arguments()
    # config = read_config_file(args.config)

    # # Save dir
    # save_dir = config['save_dir']
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    # # Device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)

    # # Data loader
    # nu = config['nu']
    # nu_list = [nu]
    # train_dataset, test_dataset, x_dim, u_dim = data_preparation_koopman(config, nu_list, nu)
    # train_loader = DataLoader(train_dataset, batch_size = config['batch_size'], shuffle = True)
    # test_loader = DataLoader(test_dataset, batch_size = config['batch_size'], shuffle = False)

    # # Set params
    # params = km.Params(x_dim = x_dim, u_dim = u_dim, config = config)


    # # Model
    # model = km.BuildModelFromParams(params).to(device)
    # for x, u, nu in train_loader:
    #     x, u, nu = x.to(device), u.to(device), nu.to(device)
    #     print(x.shape)
    #     print(model.encode_state(x).shape)
    #     print(model.auto_state(x).shape)
    #     print(model.control_encoder(u).shape)
    #     print(model.forward(x[:, 0:1, :], u[:, 0:1, :], nu[:, 0:1, :]).shape)
    #     # print(model(x, u, nu).shape)
    #     break





        

