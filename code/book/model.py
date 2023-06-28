from layers import FeaturesLinear, FeaturesEmbedding, FieldAwareFeaturesEmbedding, PairwiseInteraction
import torch.nn as nn


class FM(nn.Module):
    """
    A class implementing the Factorization Machine model.
    
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
        y_interaction = self.interaction(embed_x)
        y = y_linear + y_interaction
        return y.squeeze(1)



class FFM(nn.Module):
    """
    A class implementing the Factorization Machine model.
    
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
        self.embedding = FieldAwareFeaturesEmbedding(self.field_dims, self.embedding_dim)
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
        y_interaction = self.interaction(embed_x)
        y = y_linear + y_interaction
        return y.squeeze(1)
