from sklearn.model_selection import train_test_split
import numpy as np
import torch
from model_dir import *
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import argparse
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

def cut_slides(data, window_size, predict_num):
    data_slices = []
    for i in range(0, data.shape[0], window_size):
        for j in range(window_size - predict_num + 1):
            slice = data[i+j:i+j+predict_num,:].reshape((1, predict_num, -1))
            data_slices.append(slice)
    return data_slices

def koopman_loss_test(model, x, y, u):
    loss_fn = nn.MSELoss()
    x_psi = model.dic(x)
    y_psi = model.dic(y)
    y_psi_pred = model.latent_to_latent_forward(x_psi, u)
    loss = loss_fn(y_psi_pred, y_psi)
    return loss

def koopman_loss(model, x, u):
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
        x_pred_cur = model.latent_to_latent_forward(x0, u[:, i, :])
        x_pred[:, i, :] = x_pred_cur
        x_true[:, i, :] = model.state_dic(x_latent[:, i, :])
    loss = loss_fn(x_pred, x_true)
    return loss

def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for x, y, u in train_loader:
        x, y, u = x.to(device), y.to(device), u.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model, x, y, u)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(train_loader)

def test_one_epoch(model, test_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y, u in test_loader:
            x, y, u = x.to(device), y.to(device), u.to(device)
            loss = loss_fn(model, x, y, u)
            total_loss += loss.item()
    return total_loss/len(test_loader)

def train_one_epoch_xu(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for x, u in train_loader:
        x, u = x.to(device), u.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model, x, u)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(train_loader)

def test_one_epoch(model, test_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, u in test_loader:
            x, u = x.to(device), u.to(device)
            loss = loss_fn(model, x, u)
            total_loss += loss.item()
    return total_loss/len(test_loader)

def main(config):

    # save dir
    save_dir = config['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # load data and model
    model, x_data, y_data, u_data = build_model_with_data(config)
    model = model.to(device)
    shuffled_indices = np.random.permutation(x_data.shape[0])
    x_data = x_data[shuffled_indices]
    y_data = y_data[shuffled_indices]
    u_data = u_data[shuffled_indices]

    # split data
    x_train, x_test, y_train, y_test, u_train, u_test = train_test_split(x_data, y_data, u_data, test_size=0.2, random_state=49)

    # data loader
    train_data = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32), torch.tensor(u_train, dtype=torch.float32))
    test_data = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32), torch.tensor(u_test, dtype=torch.float32))
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=config['lr'])
    scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])

    # Loss function
    loss_fn = koopman_loss_test

    # Train
    train_losses = []
    test_losses = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    for epoch in range(config['num_epochs']):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        test_loss = test_one_epoch(model, test_loader, loss_fn, device)
        scheduler.step()
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Train Loss: {train_loss}, Test Loss: {test_loss}")

        if test_loss < best_loss:
            best_loss = test_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(save_dir, 'best_model.pth'))
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))

    # Save losses
    np.save(os.path.join(save_dir, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(save_dir, 'test_losses.npy'), np.array(test_losses))

    # Plot
    plt.figure()
    plt.plot(range(1, config['num_epochs'] + 1), train_losses, label='Train Loss')
    plt.plot(range(1, config['num_epochs'] + 1), test_losses, label='Test Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))

    return

def twice_train(config):
    # save dir
    save_dir = config['save_dir']
    if not os.path.exists(save_dir):
        raise ValueError(f"Directory {save_dir} does not exist")

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # load data and model
    model, x_data, y_data, u_data = build_model_with_data(config)
    model = model.to(device)

    # build dataset
    x_data_slices, u_data_slices = [], []
    window_size = config['window_size']
    slice_len = config['slice_len']
    x_data_slices = cut_slides(x_data, window_size, slice_len)
    u_data_slices = cut_slides(u_data, window_size, slice_len)
    x_data = np.concatenate(x_data_slices, axis=0)
    u_data = np.concatenate(u_data_slices, axis=0)
    
    shuffled_indices = np.arange(len(x_data))
    np.random.shuffle(shuffled_indices)

    x_data = x_data[shuffled_indices]
    u_data = u_data[shuffled_indices]

    # split data
    x_train, x_test, u_train, u_test = train_test_split(x_data, u_data, test_size=0.2, random_state=49)

    # data loader
    train_data = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(u_train, dtype=torch.float32))
    test_data = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(u_test, dtype=torch.float32))
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)

    # Load pretrained dictionary
    state_dict = torch.load(os.path.join(config['save_dir'], 'best_model.pth'))
    state_dict_model = model.state_dict()
    # Loop through the state_dict items
    for key, value in state_dict.items():
        if key.startswith('state_dic.'):
            state_dict_model[key] = value
    model.load_state_dict(state_dict_model)

    for param in model.state_dic.parameters():
        param.requires_grad = False


    # Optimizer
    optimizer = Adam(model.parameters(), lr=config['lr'])
    scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])

    # Loss function
    loss_fn = koopman_loss

    # Train
    train_losses = []
    test_losses = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    for epoch in range(config['num_epochs']):
        train_loss = train_one_epoch_xu(model, train_loader, optimizer, loss_fn, device)
        test_loss = test_one_epoch(model, test_loader, loss_fn, device)
        scheduler.step()
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Train Loss: {train_loss}, Test Loss: {test_loss}")
        if test_loss < best_loss:
            best_loss = test_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(save_dir, 'twice_train_best_model.pth'))
    torch.save(model.state_dict(), os.path.join(save_dir, 'twice_train_final_model.pth'))

    # Save losses
    np.save(os.path.join(save_dir, 'twice_train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(save_dir, 'twice_test_losses.npy'), np.array(test_losses))

    # Plot
    plt.figure()
    plt.plot(range(1, config['num_epochs'] + 1), train_losses, label='Train Loss')
    plt.plot(range(1, config['num_epochs'] + 1), test_losses, label='Test Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'twice_loss_plot.png'))

    return



if __name__ == '__main__':
    args = parse_arguments()
    config = read_config_file(args.config)
    twice_train(config)