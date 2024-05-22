import math, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")


from sklearn.decomposition import PCA


class PCAKoopman(nn.Module):
    def __init__(self, params, std_layer_1, pca_transformer, std_layer_2, std_layer_u, state_dic, control_encoder, state_matrix, control_matrix):
        super(PCAKoopman, self).__init__()
        self.params = params
        self.std_layer_1 = std_layer_1
        self.std_layer_u = std_layer_u
        self.pca_transformer = pca_transformer
        self.std_layer_2 = std_layer_2
        self.state_dic = state_dic
        self.control_encoder = control_encoder
        self.state_matrix = state_matrix
        self.control_matrix = control_matrix
    
    def forward(self, x, u, nu):
        x = self.std_layer_1.transform(x)
        x = self.pca_transformer.transform(x)
        # x = self.std_layer_2.transform(x)
        x = self.state_dic(x)
        u = self.std_layer_u.transform(u)
        u = self.control_encoder(u)
        y = self.state_matrix(x, nu) + self.control_matrix(u, nu)
        y = self.std_layer_2.inverse_transform(y)
        y = self.pca_transformer.inverse_transform(y)
        y = self.std_layer_1.inverse_transform(y)
        return y
    
    def latent_to_latent_forward(self, x, u, nu):
        u = self.std_layer_u.transform(u)
        u = self.control_encoder(u)
        y = self.state_matrix(x, nu) + self.control_matrix(u, nu)
        return y
    
    def dic(self, x):
        x = self.encode(x)
        x = self.state_dic(x)
        return x

    def encode(self, x):
        x = self.std_layer_1.transform(x)
        x = self.pca_transformer.transform(x)
        x = self.std_layer_2.transform(x)
        return x
    
    def decode(self, x):
        x = x[:, 1:self.params.pca_dim+1]
        x = self.std_layer_2.inverse_transform(x)
        x = self.pca_transformer.inverse_transform(x)
        x = self.std_layer_1.inverse_transform(x)
        return x
    

class PCAKoopmanWithInputsInDic(nn.Module):
    def __init__(self, params, std_layer_1, pca_transformer, std_layer_2, std_layer_u, state_dic, state_matrix):
        super(PCAKoopmanWithInputsInDic, self).__init__()
        self.params = params
        self.std_layer_1 = std_layer_1
        self.std_layer_u = std_layer_u
        self.pca_transformer = pca_transformer
        self.std_layer_2 = std_layer_2
        self.state_dic = state_dic
        self.state_matrix = state_matrix
    
    def forward(self, x, u, nu):
        x = self.std_layer_1.transform(x)
        x = self.pca_transformer.transform(x)
        x = self.std_layer_2.transform(x)
        u = self.std_layer_u.transform(u)
        x = self.state_dic(x, u)
        x = self.state_matrix(x, nu)
        x = self.std_layer.inverse_transform(x)
        x = self.pca_transformer.inverse_transform(x)
        x = self.std_layer_1.inverse_transform(x)
        return x
    
    def latent_to_latent_forward(self, x, u, nu):
        u = self.std_layer_u.transform(u)
        x = self.state_dic(x, u)
        x = self.state_matrix(x, nu)
        return x[:, 1:self.params.pca_dim+1]
    
    def dic(self, x, u):
        u = self.std_layer_u.transform(u)
        x = self.encode(x)
        x = self.state_dic(x, u)
        return x

    def encode(self, x):
        x = self.std_layer_1.transform(x)
        x = self.pca_transformer.transform(x)
        x = self.std_layer_2.transform(x)
        return x
    
    def decode(self, x):
        x = self.std_layer_2.inverse_transform(x)
        x = self.pca_transformer.inverse_transform(x)
        x = self.std_layer_1.inverse_transform(x)
        return x
    
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
    
    def forward(self, x, u, nu):
        x = self.std_layer_1.transform(x)
        x = self.pca_transformer.transform(x)
        x = self.std_layer_2.transform(x)
        u = self.std_layer_u.transform(u)
        x = self.state_dic(x)
        K = self.state_matrix(u, nu)
        x = torch.matmul(x, K)
        x = self.std_layer.inverse_transform(x)
        x = self.pca_transformer.inverse_transform(x)
        x = self.std_layer_1.inverse_transform(x)
        return x
    
    def pca_forward(self, x_pca, u, nu):
        x_psi = self.state_dic(x_pca)
        u = self.std_layer_u.transform(u)
        K = self.state_matrix(u, nu)
        x_psi_extended = x_psi.unsqueeze(1)
        y_psi = torch.matmul(x_psi_extended, K).squeeze(1)
        y_pca = y_psi[:, 1:self.params.pca_dim+1]
        return y_pca

    
    def latent_to_latent_forward(self, x_psi, u, nu):
        u = self.std_layer_u.transform(u)
        K = self.state_matrix(u, nu)
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
    
    def pca_decode(self, x_pca):
        x = self.std_layer_2.inverse_transform(x_pca)
        x = self.pca_transformer.inverse_transform(x)
        x = self.std_layer_1.inverse_transform(x)
        return x

class PCAKoopmanWithInputsInStd(nn.Module):
    def __init__(self, params, std_layer_1, pca_transformer, std_layer_2, std_layer_u, state_dic, control_encoder, state_matrix, control_matrix):
        super(PCAKoopmanWithInputsInStd, self).__init__()
        self.params = params
        self.std_layer_1 = std_layer_1
        self.std_layer_u = std_layer_u
        self.pca_transformer = pca_transformer
        self.std_layer_2 = std_layer_2
        self.state_dic = state_dic
        self.control_encoder = control_encoder
        self.state_matrix = state_matrix
        self.control_matrix = control_matrix
    
    def forward(self, x, u, nu):
        x = self.encode(x, nu)
        x = self.latent_to_latent_forward(x, u, nu)
        x = self.decode(x)
        return x
    
    def latent_to_latent_forward(self, x, u, nu):
        x = self.state_dic(x)
        u = self.std_layer_u.transform(u)
        u = self.control_encoder(u)
        y = self.state_matrix(x, nu) + self.control_matrix(u, nu)
        return y[:, 1:self.params.pca_dim+1]
    
    def dic(self, x, nu):
        x = self.encode(x, nu)
        x = self.state_dic(x)
        return x

    def encode(self, x, nu):
        x = self.std_layer_1.transform(x, nu)
        x = self.pca_transformer.transform(x)
        x = self.std_layer_2.transform(x)
        return x
    
    def decode(self, x, nu):
        x = self.std_layer_2.inverse_transform(x)
        x = self.pca_transformer.inverse_transform(x)
        x = self.std_layer_1.inverse_transform(x, nu)
        return x
class PCAKoopmanWithInputsInDicAndStd(nn.Module):
    def __init__(self, params, std_layer_1, pca_transformer, std_layer_2, std_layer_u, state_dic, state_matrix):
        super(PCAKoopmanWithInputsInDicAndStd, self).__init__()
        self.params = params
        self.std_layer_1 = std_layer_1
        self.std_layer_u = std_layer_u
        self.pca_transformer = pca_transformer
        self.std_layer_2 = std_layer_2
        self.state_dic = state_dic
        self.state_matrix = state_matrix
    
    def forward(self, x, u, nu):
        x = self.encode(x, nu)
        x = self.latent_to_latent_forward(x, u, nu)
        x = self.decode(x)
        return x
    
    def latent_to_latent_forward(self, x, u, nu):
        u = self.std_layer_u.transform(u)
        x = self.state_dic(x, u)
        x = self.state_matrix(x, nu)
        return x[:, 1:self.params.pca_dim+1]
    
    def dic(self, x, u, nu):
        u = self.std_layer_u.transform(u)
        x = self.encode(x, nu)
        x = self.state_dic(x, u)
        return x

    def encode(self, x, nu):
        x = self.std_layer_1.transform(x, nu)
        x = self.pca_transformer.transform(x)
        x = self.std_layer_2.transform(x)
        return x
    
    def decode(self, x, nu):
        x = self.std_layer_2.inverse_transform(x)
        x = self.pca_transformer.inverse_transform(x)
        x = self.std_layer_1.inverse_transform(x, nu)
        return x

class PCAKoopmanWithInputsInMatrixAndStd(nn.Module):
    def __init__(self, params, std_layer_1, pca_transformer, std_layer_2, std_layer_u, state_dic, state_matrix):
        super(PCAKoopmanWithInputsInMatrixAndStd, self).__init__()
        self.params = params
        self.std_layer_1 = std_layer_1
        self.std_layer_u = std_layer_u
        self.pca_transformer = pca_transformer
        self.std_layer_2 = std_layer_2
        self.state_dic = state_dic
        self.state_matrix = state_matrix
    
    def forward(self, x, u, nu):
        x = self.encode(x, nu)
        x = self.latent_to_latent_forward(x, u, nu)
        x = self.decode(x)
        return x
    
    def latent_to_latent_forward(self, x, u, nu):
        u = self.std_layer_u.transform(u)
        x_psi = self.state_dic(x)
        K = self.state_matrix(u, nu)
        x_psi_extended = x_psi.unsqueeze(1)
        y_psi = torch.matmul(x_psi_extended, K).squeeze(1)
        return y_psi[:, 1:self.params.pca_dim+1]
    
    def dic(self, x, u, nu):
        u = self.std_layer_u.transform(u)
        x = self.encode(x, nu)
        x = self.state_dic(x)
        return x

    def encode(self, x, nu):
        x = self.std_layer_1.transform(x, nu)
        x = self.pca_transformer.transform(x)
        x = self.std_layer_2.transform(x)
        return x
    
    def decode(self, x, nu):
        x = self.std_layer_2.inverse_transform(x)
        x = self.pca_transformer.inverse_transform(x)
        x = self.std_layer_1.inverse_transform(x, nu)
        return x

class StateMatrix_With_Input(nn.Module):
    def __init__(self, params, Matrix_NN):
        # nu_list should be a list without repeated values
        super(StateMatrix_With_Input, self).__init__()
        self.Ku_size = params.Ku_size
        self.k_size = params.d_model
        self.output_size = params.d_model ** 2
        self.length = len(params.nu_list)
        self.k_matrices = nn.ModuleList()
        for _ in range(self.length):
            self.k_matrices.append(nn.Linear(self.Ku_size, self.output_size, bias=True))
        self.Matrix_NN = Matrix_NN
        

    def forward(self, u, nu):
        if nu.shape[1] != self.length:
            raise ValueError("Length of nu does not match number of matrices in k_matrices.")
        
        NN_output = self.Matrix_NN(u)

        out_sum = torch.zeros((NN_output.shape[0], self.output_size), device=NN_output.device)
        
        for i in range(self.length):
            out = self.k_matrices[i](NN_output)
            out_sum = out_sum + nu[:, i:i+1] * out
        
        return out_sum.reshape(-1, self.k_size, self.k_size)

class StateMatrix_sum(nn.Module):
    def __init__(self, params):
        # nu_list should be a list without repeated values
        super(StateMatrix_sum, self).__init__()
        self.k_size = params.d_model
        self.length = len(params.nu_list)
        self.k_matrices = nn.ModuleList()
        for _ in range(self.length):
            self.k_matrices.append(nn.Linear(self.k_size, self.k_size))        
        

    def forward(self, x, nu):
        if nu.shape[1] != self.length:
            raise ValueError("Length of nu does not match number of matrices in k_matrices.")
        
        out_sum = torch.zeros_like(x)
        
        for i in range(self.length):
            out = self.k_matrices[i](x)
            out_sum = out_sum + nu[:, i:i+1] * out
        
        return out_sum

class ControlMatrix_sum(nn.Module):
    def __init__(self, params):
        # nu_list should be a list without repeated values
        super(ControlMatrix_sum, self).__init__()
        self.k_size = params.d_model
        self.u_size = params.u_dim + 1 + params.u_dic_model
        self.length = len(params.nu_list)
        self.k_matrices = nn.ModuleList()
        for _ in range(self.length):
            self.k_matrices.append(nn.Linear(self.u_size, self.k_size))        
        

    def forward(self, x, nu):
        if nu.shape[1] != self.length:
            raise ValueError("Length of nu does not match number of matrices in k_matrices.")
        
        out_sum = torch.zeros((x.shape[0], self.k_size), device=x.device)
        
        for i in range(self.length):
            out = self.k_matrices[i](x)
            # print(out.shape)
            out_sum = out_sum + nu[:, i:i+1] * out
        
        return out_sum

class State_Encoder(nn.Module):
    "Implements State dictionary"
    def __init__(self, params):
        super(State_Encoder, self).__init__()
        self.dic_model = params.dic_model
        if params.dic_model != 0:
            self.input_layer = nn.Linear(params.pca_dim, params.dd_model)
            self.Layer = FeedForwardLayerConnection(params.dd_model, FeedForward(params.dd_model, params.dd_ff), params.dropout)
            self.layers = clones(self.Layer, params.N_State)
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
class Control_Encoder(nn.Module):
    "Implements State dictionary"
    def __init__(self, params):
        super(Control_Encoder, self).__init__()
        self.dic_model = params.u_dic_model
        if params.u_dic_model != 0:
            self.input_layer = nn.Linear(params.u_dim, params.u_model)
            self.Layer = FeedForwardLayerConnection(params.u_model, FeedForward(params.u_model, params.u_ff), params.dropout)
            self.layers = clones(self.Layer, params.N_Control)
            self.output_layer = nn.Linear(params.u_model, params.u_dic_model)

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

class State_Encoder_With_Inputs(nn.Module):
    def __init__(self, params):
        super(State_Encoder_With_Inputs, self).__init__()
        self.dic_model = params.dic_model
        if params.dic_model != 0:
            self.input_layer = nn.Linear(params.pca_dim + params.u_dim, params.dd_model)
            self.layers = nn.ModuleList([ResNetBlock(params.dd_model) for _ in range(params.N_State)])
            self.output_layer = nn.Linear(params.dd_model, params.dic_model)
        
    def forward(self, x, u):
        if self.dic_model == 0:
            ones = torch.ones(x.shape[0], 1).to(x.device)
            return torch.cat((ones, x), dim = 1)
        else:
            y = torch.cat((x, u), dim = 1)
            y = self.input_layer(y)
            y = F.relu(y)
            for layer in self.layers:
                y = layer(y)
            y = self.output_layer(y)
            ones = torch.ones(x.shape[0], 1).to(x.device)
            y = torch.cat((ones, x, y), dim = 1)
            return y

class Matrix_NN(nn.Module):
    def __init__(self, params):
        super(Matrix_NN, self).__init__()
        self.input_layer = nn.Linear(params.u_dim, params.Ku_ff)
        self.layers = nn.ModuleList([ResNetBlock(params.Ku_ff) for _ in range(params.N_Matrix)])
        self.output_layer = nn.Linear(params.Ku_ff, params.Ku_size)
    
    def forward(self, u):
        x = self.input_layer(u)
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


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

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6): 
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class FeedForwardLayerConnection(nn.Module):
    def __init__(self, size, feed_forward, dropout):
        super(FeedForwardLayerConnection, self).__init__()
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 1)
        self.size = size
    
    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.feed_forward(x))
        return x
        
class Control_Encoder_FullyNonlinear(nn.Module):
    def __init__(self, params):
        super(Control_Encoder_FullyNonlinear, self).__init__()
        self.params = params
        self.input_layer = nn.Linear(params.u_dim, params.u_model)
        self.Layer = FeedForwardLayerConnection(params.u_model, FeedForward(params.u_model, params.u_ff), params.dropout)
        self.layers = clones(self.Layer, params.N_Control)
        self.norm = LayerNorm(params.u_model)
        self.output_layer = nn.Linear(params.u_model, params.pca_dim + params.dic_model)
    
    def forward(self, control):
        x = F.relu(self.input_layer(control))
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.output_layer(x)
        ones = torch.ones(x.shape[0], 1).to(x.device)
        return torch.cat((ones, x), dim = 1)

class PCALayer(nn.Module):
    def __init__(self, input_dim, output_dim, pca_matrix):
        super(PCALayer, self).__init__()
        self.pca_matrix = pca_matrix
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.transform = nn.Linear(input_dim, output_dim, bias = False)
        self.transform.weight = nn.Parameter(pca_matrix, requires_grad=False)
        self.inverse_transform = nn.Linear(output_dim, input_dim, bias = False)
        self.inverse_transform.weight = nn.Parameter(pca_matrix.T, requires_grad=False)

class StdScalerLayer(nn.Module):
    def __init__(self, mean, std):
        super(StdScalerLayer, self).__init__()
        self.mean = nn.Parameter(torch.tensor(mean, dtype=torch.float32), requires_grad=False)
        self.std = nn.Parameter(torch.tensor(std, dtype=torch.float32), requires_grad=False)

    def transform(self, x):
        return (x - self.mean) / self.std
    
    def inverse_transform(self, input):
        return input * self.std + self.mean
    
class StdScalerLayerSet(nn.Module):
    def __init__(self, StdScalerLayers):
        super(StdScalerLayerSet, self).__init__()
        self.StdScalerLayers = StdScalerLayers
    
    def transform(self, x, nu):
        y = torch.zeros_like(x).to(x.device)
        for i in range(len(self.StdScalerLayers)):
            y += nu[:, i:i+1] * self.StdScalerLayers[i].transform(x)
        return y
    
    def inverse_transform(self, x, nu):
        y = torch.zeros_like(x).to(x.device)
        for i in range(len(self.StdScalerLayers)):
            y += nu[:, i:i+1] * self.StdScalerLayers[i].inverse_transform(x)
        return y

def build_model(params, x_data, u_data):
    x_data = torch.tensor(x_data, dtype=torch.float32)
    u_data = torch.tensor(u_data, dtype=torch.float32)  # Ensure u_data is also a Tensor

    # Std Layer 1
    mean_1 = torch.mean(x_data, dim=0)
    std_1 = torch.std(x_data, dim=0)
    std_layer_1 = StdScalerLayer(mean_1, std_1)

    # rescale x_data
    x_data_scaled = std_layer_1.transform(x_data)

    # Std Layer u
    mean_u = torch.mean(u_data, dim=0)
    std_u = torch.std(u_data, dim=0)
    std_layer_u = StdScalerLayer(mean_u, std_u)

    # PCA layer
    pca = PCA(n_components=params.pca_dim)
    # Ensure x_data_scaled is converted back to a NumPy array for PCA, if necessary
    x_pca = pca.fit_transform(x_data_scaled.cpu().detach().numpy())
    components = pca.components_
    pca_matrix = torch.tensor(components, dtype=torch.float32)
    pca_layer = PCALayer(params.x_dim, params.pca_dim, pca_matrix)

    x_pca_tensor = torch.tensor(x_pca, dtype=torch.float32)

    # Std Layer 2
    mean_2 = torch.mean(x_pca_tensor, dim=0)
    std_2 = torch.std(x_pca_tensor, dim=0)
    std_layer_2 = StdScalerLayer(mean_2, std_2)

    # State dictionary
    state_dic = State_Encoder(params)

    # Control Encoder
    control_encoder = ControlEncoder(params)

    # State Matrix
    state_matrix = StateMatrix_sum(params)

    # Control Matrix
    control_matrix = StateMatrix_sum(params)

    model = PCAKoopman(params, std_layer_1, pca_layer, std_layer_2, std_layer_u, state_dic, control_encoder, state_matrix, control_matrix)
    return model

def build_model_DicWithInputs(params, x_data, u_data):
    x_data = torch.tensor(x_data, dtype=torch.float32)
    u_data = torch.tensor(u_data, dtype=torch.float32)  # Ensure u_data is also a Tensor

    # Std Layer 1
    mean_1 = torch.mean(x_data, dim=0)
    std_1 = torch.std(x_data, dim=0)
    std_layer_1 = StdScalerLayer(mean_1, std_1)

    # rescale x_data
    x_data_scaled = std_layer_1.transform(x_data)

    # Std Layer u
    mean_u = torch.mean(u_data, dim=0)
    std_u = torch.std(u_data, dim=0)
    std_layer_u = StdScalerLayer(mean_u, std_u)

    # PCA layer
    pca = PCA(n_components=params.pca_dim)
    # Ensure x_data_scaled is converted back to a NumPy array for PCA, if necessary
    x_pca = pca.fit_transform(x_data_scaled.cpu().detach().numpy())
    components = pca.components_
    pca_matrix = torch.tensor(components, dtype=torch.float32)
    pca_layer = PCALayer(params.x_dim, params.pca_dim, pca_matrix)

    x_pca_tensor = torch.tensor(x_pca, dtype=torch.float32)

    # Std Layer 2
    mean_2 = torch.mean(x_pca_tensor, dim=0)
    std_2 = torch.std(x_pca_tensor, dim=0)
    std_layer_2 = StdScalerLayer(mean_2, std_2)

    # State dictionary
    state_dic = State_Encoder_With_Inputs(params)

    # State Matrix
    state_matrix = StateMatrix_sum(params)

    model = PCAKoopmanWithInputsInDic(params, std_layer_1, pca_layer, std_layer_2, std_layer_u, state_dic, state_matrix)
    return model

def build_model_MatrixWithInputs(params, x_data, u_data):
    x_data = torch.tensor(x_data, dtype=torch.float32)
    u_data = torch.tensor(u_data, dtype=torch.float32)  # Ensure u_data is also a Tensor

    # Std Layer 1
    mean_1 = torch.mean(x_data, dim=0)
    std_1 = torch.std(x_data, dim=0)
    std_layer_1 = StdScalerLayer(mean_1, std_1)

    # rescale x_data
    x_data_scaled = std_layer_1.transform(x_data)

    # Std Layer u
    mean_u = torch.mean(u_data, dim=0)
    std_u = torch.std(u_data, dim=0)
    std_layer_u = StdScalerLayer(mean_u, std_u)

    # PCA layer
    pca = PCA(n_components=params.pca_dim)
    # Ensure x_data_scaled is converted back to a NumPy array for PCA, if necessary
    x_pca = pca.fit_transform(x_data_scaled.cpu().detach().numpy())
    components = pca.components_
    pca_matrix = torch.tensor(components, dtype=torch.float32)
    pca_layer = PCALayer(params.x_dim, params.pca_dim, pca_matrix)

    x_pca_tensor = torch.tensor(x_pca, dtype=torch.float32)

    # Std Layer 2
    mean_2 = torch.mean(x_pca_tensor, dim=0)
    std_2 = torch.std(x_pca_tensor, dim=0)
    std_layer_2 = StdScalerLayer(mean_2, std_2)

    # State dictionary
    state_dic = State_Encoder(params)

    # State Matrix
    matrix_NN = Matrix_NN(params)
    state_matrix = StateMatrix_With_Input(params, matrix_NN)

    model = PCAKoopmanWithInputsInMatrix(params, std_layer_1, pca_layer, std_layer_2, std_layer_u, state_dic, state_matrix)
    return model

class Params:
    def __init__(self, x_dim, u_dim, config):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.nu_list = config['nu_list']   # To-do
        self.pca_dim = config.get('pca_dim', 4)
        self.dic_model = config.get('dic_model', 20)
        self.d_model = 1+ self.pca_dim + self.dic_model
        self.dd_model = config.get('dd_model', 64)
        self.dd_ff = config.get('dd_ff', 256)
        self.u_model = config.get('u_model',64)
        self.u_ff = config.get('u_ff', 256)
        self.N_Control = config.get('N_Control', 6)
        self.dropout = config.get('dropout', 0.1)
        self.N_State = config.get('N_State', 6)
        self.Ku_size = config.get('Ku_size', 256)
        self.Ku_ff = config.get('Ku_ff', 256)
        self.N_Matrix = config.get('N_Matrix', 6)
        self.u_dic_model = config.get('u_dic_model', 0)


def build_model_DicWithInputs_multi_nu(params, x_dataset, u_dataset):
    # Std Layer 1
    std_layers_1_set = torch.nn.ModuleList()
    x_dataset_scaled = []
    for x_data in (x_dataset):
        x_data = torch.tensor(x_data, dtype=torch.float32)
        mean_1 = torch.mean(x_data, dim=0)
        std_1 = torch.std(x_data, dim=0)
        std_layer_1 = StdScalerLayer(mean_1, std_1)
        std_layers_1_set.append(std_layer_1)
        x_data_scaled = std_layer_1.transform(x_data)
        x_dataset_scaled.append(x_data_scaled)
    
    std_layer_1 = StdScalerLayerSet(std_layers_1_set)

    # Std Layer u
    u_data = np.concatenate(u_dataset, axis=0)
    u_data = torch.tensor(u_data, dtype=torch.float32)  # Ensure u_data is also a Tensor
    mean_u = torch.mean(u_data, dim=0)
    std_u = torch.std(u_data, dim=0)
    std_layer_u = StdScalerLayer(mean_u, std_u)

    # PCA layer
    x_data_scaled = torch.cat(x_dataset_scaled, dim=0)
    pca = PCA(n_components=params.pca_dim)
    # Ensure x_data_scaled is converted back to a NumPy array for PCA, if necessary
    x_pca = pca.fit_transform(x_data_scaled.cpu().detach().numpy())
    components = pca.components_
    pca_matrix = torch.tensor(components, dtype=torch.float32)
    pca_layer = PCALayer(params.x_dim, params.pca_dim, pca_matrix)

    x_pca_tensor = torch.tensor(x_pca, dtype=torch.float32)

    # Std Layer 2
    mean_2 = torch.mean(x_pca_tensor, dim=0)
    std_2 = torch.std(x_pca_tensor, dim=0)
    std_layer_2 = StdScalerLayer(mean_2, std_2)

    # State dictionary
    state_dic = State_Encoder_With_Inputs(params)

    # State Matrix
    state_matrix = StateMatrix_sum(params)

    model = PCAKoopmanWithInputsInDicAndStd(params, std_layer_1, pca_layer, std_layer_2, std_layer_u, state_dic, state_matrix)
    x_pca_scaled = std_layer_2.transform(x_pca_tensor)
    return model, x_pca_scaled

def build_model_linear_multi_nu(params, x_dataset, u_dataset):
    # Std Layer 1
    std_layers_1_set = torch.nn.ModuleList()
    x_dataset_scaled = []
    for x_data in (x_dataset):
        x_data = torch.tensor(x_data, dtype=torch.float32)
        mean_1 = torch.mean(x_data, dim=0)
        std_1 = torch.std(x_data, dim=0)
        std_layer_1 = StdScalerLayer(mean_1, std_1)
        std_layers_1_set.append(std_layer_1)
        x_data_scaled = std_layer_1.transform(x_data)
        x_dataset_scaled.append(x_data_scaled)
    
    std_layer_1 = StdScalerLayerSet(std_layers_1_set)

    # Std Layer u
    u_data = np.concatenate(u_dataset, axis=0)
    u_data = torch.tensor(u_data, dtype=torch.float32)  # Ensure u_data is also a Tensor
    mean_u = torch.mean(u_data, dim=0)
    std_u = torch.std(u_data, dim=0)
    std_layer_u = StdScalerLayer(mean_u, std_u)

    # PCA layer
    x_data_scaled = torch.cat(x_dataset_scaled, dim=0)
    pca = PCA(n_components=params.pca_dim)
    # Ensure x_data_scaled is converted back to a NumPy array for PCA, if necessary
    x_pca = pca.fit_transform(x_data_scaled.cpu().detach().numpy())
    components = pca.components_
    pca_matrix = torch.tensor(components, dtype=torch.float32)
    pca_layer = PCALayer(params.x_dim, params.pca_dim, pca_matrix)

    x_pca_tensor = torch.tensor(x_pca, dtype=torch.float32)

    # Std Layer 2
    mean_2 = torch.mean(x_pca_tensor, dim=0)
    std_2 = torch.std(x_pca_tensor, dim=0)
    std_layer_2 = StdScalerLayer(mean_2, std_2)

    # State dictionary
    state_dic = State_Encoder(params)

    # Control encoder
    control_encoder = Control_Encoder(params)

    # State Matrix
    state_matrix = StateMatrix_sum(params)

    # Control Matrix
    control_matrix = ControlMatrix_sum(params)

    model = PCAKoopmanWithInputsInStd(params, std_layer_1, pca_layer, std_layer_2, std_layer_u, state_dic, control_encoder, state_matrix, control_matrix)
    x_pca_scaled = std_layer_2.transform(x_pca_tensor)
    return model, x_pca_scaled

def build_model_MatrixWithInputs_multi_nu(params, x_dataset, u_dataset):
    # Std Layer 1
    std_layers_1_set = torch.nn.ModuleList()
    x_dataset_scaled = []
    for x_data in (x_dataset):
        x_data = torch.tensor(x_data, dtype=torch.float32)
        mean_1 = torch.mean(x_data, dim=0)
        std_1 = torch.std(x_data, dim=0)
        std_layer_1 = StdScalerLayer(mean_1, std_1)
        std_layers_1_set.append(std_layer_1)
        x_data_scaled = std_layer_1.transform(x_data)
        x_dataset_scaled.append(x_data_scaled)
    
    std_layer_1 = StdScalerLayerSet(std_layers_1_set)

    # Std Layer u
    u_data = np.concatenate(u_dataset, axis=0)
    u_data = torch.tensor(u_data, dtype=torch.float32)  # Ensure u_data is also a Tensor
    mean_u = torch.mean(u_data, dim=0)
    std_u = torch.std(u_data, dim=0)
    std_layer_u = StdScalerLayer(mean_u, std_u)

    # PCA layer
    x_data_scaled = torch.cat(x_dataset_scaled, dim=0)
    pca = PCA(n_components=params.pca_dim)
    # Ensure x_data_scaled is converted back to a NumPy array for PCA, if necessary
    x_pca = pca.fit_transform(x_data_scaled.cpu().detach().numpy())
    components = pca.components_
    pca_matrix = torch.tensor(components, dtype=torch.float32)
    pca_layer = PCALayer(params.x_dim, params.pca_dim, pca_matrix)

    x_pca_tensor = torch.tensor(x_pca, dtype=torch.float32)

    # Std Layer 2
    mean_2 = torch.mean(x_pca_tensor, dim=0)
    std_2 = torch.std(x_pca_tensor, dim=0)
    std_layer_2 = StdScalerLayer(mean_2, std_2)

    # State dictionary
    state_dic = State_Encoder(params)

    # Matrix
    matrix_NN = Matrix_NN(params)

    # State Matrix
    state_matrix = StateMatrix_With_Input(params, matrix_NN)

    model = PCAKoopmanWithInputsInMatrixAndStd(params, std_layer_1, pca_layer, std_layer_2, std_layer_u, state_dic, state_matrix)
    x_pca_scaled = std_layer_2.transform(x_pca_tensor)
    return model, x_pca_scaled