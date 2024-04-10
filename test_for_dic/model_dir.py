import numpy as np
import torch
import os
from sklearn.decomposition import PCA
import torch.nn.functional as F
import torch.nn as nn
import copy

def load_dataset(data_dict, predict_num = 1):
    time = []
    data = []
    I_p = []
    for _, contents in data_dict.items():
        time.append(contents['time'])
        data.append(contents['data'])
        I_p.append(contents['I_p'])
    
    data = np.array(data)
    x_data = data[:-predict_num,:]
    y_data = data[predict_num:,:]
    # I_p = np.reshape(np.array(I_p)[:-1], (-1,1))
    u_data = np.concatenate((np.reshape(np.array(I_p)[:-1], (-1,1)), np.reshape(np.array(I_p)[1:], (-1,1))), axis = 1)

    u_data_slices = []
    for k in range(len(x_data)):
        if k + predict_num > len(u_data):
            raise ValueError("The requested prediction number exceeds the bounds of u_data.")
        u_data_slice = u_data[k:k+predict_num, :].reshape(1, -1)
        u_data_slices.append(u_data_slice)
    u_data = np.concatenate(u_data_slices, axis= 0)
    return x_data, y_data, u_data

# Dataset
def build_dataset_test(config):
    x_dataset = []
    y_dataset = []
    u_dataset = []

    data_dir = config['data_dir']
    for item in os.listdir(data_dir):
        data_file_path = os.path.join(data_dir, item)
        # Check if the file exists before trying to load it
        if data_file_path.endswith('.npy') and os.path.exists(data_file_path):
            data_dict = np.load(data_file_path, allow_pickle=True).item()
            x_data, y_data, u_data = load_dataset(data_dict, config['predict_num'])
            x_dataset.append(x_data[:config['window_size'], :])
            y_dataset.append(y_data[:config['window_size'], :])
            u_dataset.append(u_data[:config['window_size'], :])
        else:
            print(f"File not found: {data_file_path}")

    x_data = np.concatenate(x_dataset, axis=0)
    y_data = np.concatenate(y_dataset, axis=0)
    u_data = np.concatenate(u_dataset, axis=0)

    return x_data, y_data, u_data

class PCAKoopmanWithInputsInMatrix(nn.Module):
    def __init__(self, params, std_layer_1, pca_transformer, std_layer_2, std_layer_u, state_dic, state_matrix):
        super(PCAKoopmanWithInputsInMatrix, self).__init__()
        self.params = params
        self.std_layer_1 = std_layer_1
        self.std_layer_u = std_layer_u
        self.pca_transformer = pca_transformer
        self.std_layer_2 = std_layer_2
        self.state_dic = state_dic
        self.state_matrix = state_matrix
    
    def forward(self, x, u):
        x_psi = self.dic(x)
        y_psi = self.latent_to_latent_forward(x_psi, u)
        y = self.decode(y_psi)
        return y
    
    def latent_to_latent_forward(self, x_psi, u):
        u = self.std_layer_u.transform(u)
        K = self.state_matrix(u)
        x_psi_extended = x_psi.unsqueeze(1)
        y_psi = torch.matmul(x_psi_extended, K).squeeze(1)
        return y_psi
    
    def dic(self, x):
        x = self.encode(x)
        x = self.state_dic(x)
        return x

    def encode(self, x):
        x = self.std_layer_1.transform(x)
        x = self.pca_transformer.transform(x)
        x = self.std_layer_2.transform(x)
        return x
    
    def decode(self, psi_x):
        x = psi_x[:, 1:self.params.pca_dim+1]
        x = self.std_layer_2.inverse_transform(x)
        x = self.pca_transformer.inverse_transform(x)
        x = self.std_layer_1.inverse_transform(x)
        return x
    
class Matrix_NN(nn.Module):
    def __init__(self, params):
        super(Matrix_NN, self).__init__()
        self.params = params
        self.input_layer = nn.Linear(params.u_dim, params.Ku_ff)
        self.layers = nn.ModuleList([ResNetBlock(params.Ku_ff) for _ in range(params.N_Matrix)])
        self.output_layer = nn.Linear(params.Ku_ff, params.d_model ** 2)
    
    def forward(self, u):
        x = self.input_layer(u)
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        K = x.view(-1, self.params.d_model, self.params.d_model)
        return K
    
class ResNetBlock(nn.Module):
    def __init__(self, dim, dropout=0):
        super(ResNetBlock, self).__init__()
        self.layer1 = nn.Linear(dim, dim)
        self.layer2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        identity = x
        out = F.tanh(self.layer1(x))
        out = self.dropout(out)
        out = self.layer2(out)
        out += identity  # Skip connection
        return out

class FeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout = 0.2): 
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class State_Encoder(nn.Module):
    "Implements State dictionary"
    def __init__(self, params):
        super(State_Encoder, self).__init__()
        self.dic_model = params.dic_model
        if params.dic_model != 0:
            self.input_layer = nn.Linear(params.pca_dim, params.dd_model)
            self.layers = nn.ModuleList([ResNetBlock(params.dd_model) for _ in range(params.N_State)])
            self.output_layer = nn.Linear(params.dd_model, params.dic_model)

    def forward(self, x):
        if self.dic_model == 0:
            ones = torch.ones(x.shape[0], 1).to(x.device)
            return torch.cat((ones, x), dim = 1)
        else:
            y = self.input_layer(x)
            y = F.relu(y)
            for layer in self.layers:
                y = layer(y)
            y = self.output_layer(y)
            ones = torch.ones(x.shape[0], 1).to(x.device)
            y = torch.cat((ones, x, y), dim = 1)
            return y

class PCALayer(nn.Module):
    def __init__(self, input_dim, output_dim, pca_matrix = None):
        super(PCALayer, self).__init__()
        self.pca_matrix = pca_matrix
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.transform = nn.Linear(input_dim, output_dim, bias = False)
        self.inverse_transform = nn.Linear(output_dim, input_dim, bias = False)
        if pca_matrix is not None:
            self.transform.weight = nn.Parameter(pca_matrix.clone().detach(), requires_grad=False)
            self.inverse_transform.weight = nn.Parameter(pca_matrix.T.clone().detach(), requires_grad=False)
        else:
            self.transform.weight = nn.Parameter(torch.zeros(output_dim, input_dim), requires_grad=False)
            self.inverse_transform.weight = nn.Parameter(torch.zeros(input_dim, output_dim), requires_grad=False)

class StdScalerLayer(nn.Module):
    def __init__(self, mean = 0, std = 1, feature_length = None):
        super(StdScalerLayer, self).__init__()
        if feature_length is None:
            self.mean = nn.Parameter(torch.tensor(mean).clone().detach(), requires_grad=False)
            self.std = nn.Parameter(torch.tensor(std).clone().detach(), requires_grad=False)
        else:
            self.mean = nn.Parameter(torch.zeros(feature_length), requires_grad=False)
            self.std = nn.Parameter(torch.ones(feature_length), requires_grad=False)

    def transform(self, x):
        return (x - self.mean) / self.std
    
    def inverse_transform(self, input):
        return input * self.std + self.mean

class Params:
    def __init__(self, x_dim, u_dim, config):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.pca_dim = config.get('pca_dim', 4)
        self.dic_model = config.get('dic_model', 20)
        self.d_model = 1+ self.pca_dim + self.dic_model
        self.dd_model = config.get('dd_model', 64)
        self.dropout = config.get('dropout', 0.1)
        self.N_State = config.get('N_State', 6)
        self.Ku_ff = config.get('Ku_ff', 256)
        self.N_Matrix = config.get('N_Matrix', 6)

def build_model_with_data(config):
    # Load the dataset
    x_data, y_data, u_data = build_dataset_test(config)

    # set params
    x_dim = x_data.shape[1]
    u_dim = u_data.shape[1]
    params = Params(x_dim, u_dim, config)

    # build the model
    x_data = torch.tensor(x_data, dtype=torch.float32)
    y_data = torch.tensor(y_data, dtype=torch.float32)
    u_data = torch.tensor(u_data, dtype=torch.float32)
    
    ## Std Layer 1
    mean_1 = torch.mean(x_data, dim=0)
    std_1 = torch.std(x_data, dim=0)
    std_layer_1 = StdScalerLayer(mean_1, std_1)

    ## Rescale x_data
    x_data_scaled = std_layer_1.transform(x_data)

    ## Std Layer u
    mean_u = torch.mean(u_data, dim=0)
    std_u = torch.std(u_data, dim=0)
    std_layer_u = StdScalerLayer(mean_u, std_u)

    ## PCA
    pca_transformer = PCA(n_components=params.pca_dim)
    x_pca = pca_transformer.fit_transform(x_data_scaled.cpu().detach().numpy())
    components = pca_transformer.components_
    pca_matrix = torch.tensor(components, dtype=torch.float32)
    pca_layer = PCALayer(params.x_dim, params.pca_dim, pca_matrix)
    x_pca_tensor = torch.tensor(x_pca, dtype=torch.float32)

    ## Std Layer 2
    mean_2 = torch.mean(x_pca_tensor, dim=0)
    std_2 = torch.std(x_pca_tensor, dim=0)
    std_layer_2 = StdScalerLayer(mean_2, std_2)

    ## State Dictionary
    state_dic = State_Encoder(params)

    ## State Matrix
    state_matrix = Matrix_NN(params)

    ## Load the model
    model = PCAKoopmanWithInputsInMatrix(params, std_layer_1, pca_layer, std_layer_2, std_layer_u, state_dic, state_matrix)
    return model, x_data, y_data, u_data

def build_model(params):

    std_layer_1 = StdScalerLayer(feature_length=params.x_dim)

    std_layer_u = StdScalerLayer(feature_length=params.u_dim)

    ## PCA
    pca_layer = PCALayer(params.x_dim, params.pca_dim)

    ## Std Layer 2
    std_layer_2 = StdScalerLayer(feature_length=params.pca_dim)

    ## State Dictionary
    state_dic = State_Encoder(params)

    ## State Matrix
    state_matrix = Matrix_NN(params)

    ## Load the model
    model = PCAKoopmanWithInputsInMatrix(params, std_layer_1, pca_layer, std_layer_2, std_layer_u, state_dic, state_matrix)
    return model