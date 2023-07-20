import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class FeaturesLinear(nn.Module):
    """
    A module to create the linear part of Factorization Machines and Field-aware Factorization Machines.

    Parameters
    ----------
    field_dims : List[int]
        A list of field dimensions.
    """
    def __init__(self, field_dims):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), 1)
        self.bias = torch.nn.Parameter(torch.zeros((1,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the linear part of the model.

        Parameters
        ----------
        x : Long tensor of size (batch_size, num_fields)

        Returns
        ----------
        y : Float tensor of size (batch_size)
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        y = torch.sum(self.fc(x), dim=1) + self.bias
        
        return y
    
    
class FeaturesEmbedding(nn.Module):
    """
    A module to create feature embeddings.
    
    Parameters
    ----------
    field_dims : List[int]
        A list of field dimensions.
    embedding_dim : int
        The dimension of dense embeddings.
    """

    def __init__(self, field_dims, embedding_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embedding_dim)
        self.offsets = torch.tensor((0, *np.cumsum(field_dims)[:-1]), dtype=torch.int32)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        Forward pass through the feature embeddings.

        Parameters
        ----------
        x : Long tensor of size (batch_size, num_fields)

        Returns
        -------
        y : Float tensor of size (batch_size, num_fields, embedding_dim)
        """
        x = x + self.offsets.unsqueeze(0).to(x.device)
        y = self.embedding(x)
        
        return y

    
class FieldAwareFeaturesEmbedding(nn.Module):
    """
    A module to create field-aware feature embeddings.

    Parameters
    ----------
    field_dims : List[int]
        A list of field dimensions.
    embedding_dim : int
        The dimension of dense embeddings.
    """

    def __init__(self, field_dims, embedding_dim):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(field_dim, embedding_dim) for field_dim in field_dims
        ])
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x):
        """
        Forward pass through the field-aware feature embeddings.

        Parameters
        ----------
        x : Long tensor of size (batch_size, num_fields)

        Returns
        ---------
        y : List of num_fields tensors of size (batch_size, embedding_dim)
        """
        y = [embedding(x[:, i]) for i, embedding in enumerate(self.embeddings)]
        
        return y
    
    
class PairwiseInteraction(nn.Module):
    """
    A module to compute the pairwise interactions.

    Parameters
    ----------
    embedding_dim : int
        The dimension of dense embeddings.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Forward pass through the pairwise interactions.

        Parameters
        ----------
        x : Float tensor of size (batch_size, num_fields, embedding_dim)

        Returns
        -------
        y : Float tensor of size (batch_size)
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        y = 0.5 * (square_of_sum - sum_of_square)
        y = torch.sum(y, dim=1).unsqueeze(-1)
        
        return y
    

class DNNLayer(nn.Module):
    '''The Multi Layer Percetron (MLP); Fully-Connected Layer (FC); Deep Neural Network (DNN) with 1-dimensional output
    Parameter
        input_dim : Input feature dimension (= num_fields * embed_dim)
        mlp_dims : List of positive integer, the layer number and units in each layer
        dropout_rate : Float value in [0,1). Fraction of the units to dropout
        use_bn : Boolean value to indicate usage of batch normalization.
    '''
    def __init__(self, input_dim, mlp_dims, dropout_rate, use_bn):
        super().__init__()
        self.input_dim = input_dim
        self.mlp_dims = [input_dim] + mlp_dims
        self.dropout = nn.Dropout(dropout_rate)
        self.use_bn = use_bn
        self.num_layers = len(mlp_dims)
        # mlp layers
        self.linears = nn.Sequential()
        for i in range(self.num_layers) :
            self.linears.add_module(f'linear_{i+1}', nn.Linear(self.mlp_dims[i], self.mlp_dims[i+1], bias=True))
            if self.use_bn :
                self.linears.add_module(f'batchnorm_{i+1}', nn.BatchNorm1d(self.mlp_dims[i+1]))
            self.linears.add_module(f'activation_{i+1}', nn.ReLU(inplace=True))
            self.linears.add_module(f'dropout_{i+1}', self.dropout)
        # 마지막 출력층
        self.output_linear = nn.Linear(self.mlp_dims[-1], 1, bias=False)
        # print(self.linears)
        # print(self.output_linear)
    
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        '''
        Parameter
            x : nD tensor of size "(batch_size, ..., input_dim)"
               The most common situation would be a 2D input with shape "(batch_size, input_dim)".
        
        Return
            y_dnn : nD tensor of size "(batch_size, ..., 1)"
               For instance, if input x is 2D tensor, the output y would have shape "(batch_size, 1)".
        '''
        x = self.linears(x)
        y_dnn = self.output_linear(x)
            
        return y_dnn

