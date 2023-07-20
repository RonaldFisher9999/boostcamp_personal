import torch
import torch.nn as nn
import numpy as np
from layers import FeaturesLinear, FeaturesEmbedding, FieldAwareFeaturesEmbedding, PairwiseInteraction, DNNLayer


class FM(nn.Module):
    """
    Parameters
    ----------
    field_dims : List[int]
        A list of field dimensions.
    embedding_dim : int
        The dimension of dense embeddings.
    """

    def __init__(self,
                 field_dims,
                 embed_dim,
                 device):
        super().__init__()
        self.field_dims = field_dims
        self.embedding_dim = embed_dim
        self.linear = FeaturesLinear(self.field_dims)
        self.embedding = FeaturesEmbedding(self.field_dims, self.embedding_dim)
        self.interaction = PairwiseInteraction()
        self.device = device

    def forward(self, x):
        """
        Parameters
        ----------
        x : Long tensor of size (batch_size, num_fields)

        Returns
        ----------
        y : Float tensor of size (batch_size)
        """
        y_linear = self.linear(x)
        embed_x = self.embedding(x)
        y_interaction = self.interaction(embed_x)
        y = y_linear + y_interaction
        
        return y.squeeze(1)


class FFM(nn.Module):
    """   
    Parameters
    ----------
    field_dims : List[int]
        A list of field dimensions.
    embedding_dim : int
        The dimension of dense embeddings.
    """

    def __init__(self,
                 field_dims,
                 embed_dim,
                 device):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FieldAwareFeaturesEmbedding(field_dims, embed_dim)
        self.interaction = PairwiseInteraction()
        self.device = device

    def forward(self, x):
        """
        Forward pass through the Factorization Machine model.

        Parameters
        ----------
        x : Long tensor of size (batch_size, num_fields)

        Returns
        ----------
        y : Float tensor of size (batch_size)
        """
        y_linear = self.linear(x)
        embed_x = self.embedding(x)
        y_interaction = self.interaction(torch.stack(embed_x, dim=1))
        y = y_linear + y_interaction
        
        return y.squeeze(1)


class DeepFM(nn.Module):
    '''
    Parameter
        field_dims : List of field dimensions
        args.embed_dim : Factorization dimension for dense embedding
        args.mlp_dims : List of positive integer, the layer number and units in each layer
        args.dropout : Float value in [0,1). Fraction of the units to dropout in DNN layer
        args.use_bn : Boolean value to indicate usage of batch normalization in DNN layer.
    '''
    def __init__(self,
                 field_dims,
                 embed_dim,
                 mlp_dims,
                 dropout_rate,
                 use_bn,
                 device):
        super().__init__()
        # 각 필드의 차원을 담은 배열
        self.field_dims = field_dims
        # 전체 입력 차원
        self.input_dim = sum(self.field_dims)
        # 필드의 개수
        self.num_fields = len(self.field_dims)
        # FFMLayer
        self.linear = FeaturesLinear(self.field_dims)
        self.embedding = FieldAwareFeaturesEmbedding(self.field_dims, embed_dim)
        self.interaction = PairwiseInteraction()
        # DNNLayer
        self.dnn = DNNLayer(self.num_fields*embed_dim, 
                            mlp_dims, 
                            dropout_rate,
                            use_bn)
        self.device = device
        
    def forward(self, x):
        '''
        Parameter
            x : Long tensor of size "(batch_size, num_fields)" is coverted to sparse_x and dense_x.
                sparse_x : Float tensor with size "(batch_size, input_dim)"
                dense_x  : List of "num_fields" float tensors of size "(batch_size, embed_dim)"
        Return
            y : Float tensor of size "(batch_size)"
        '''
        y_linear = self.linear(x)
        embed_x = self.embedding(x)
        y_interaction = self.interaction(torch.stack(embed_x, dim=1))
        y_ffm = y_linear + y_interaction
        
        y_dnn = self.dnn(torch.cat(embed_x, dim=1))
        
        y = y_ffm + y_dnn

        return y.squeeze(1)