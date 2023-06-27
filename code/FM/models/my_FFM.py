from ..layers.layers import FeaturesLinear, FieldAwareFeaturesEmbedding, PairwiseInteraction
import torch.nn as nn


class MyFieldAwareFactorizationMachine(nn.Module):
    """
    A class implementing the Factorization Machine model.
    
    Parameters
    ----------
    field_dims : List[int]
        A list of field dimensions.
    embedding_dim : int
        The dimension of dense embeddings.
    """

    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.embedding_dim = args.embed_dim
        self.linear = FeaturesLinear(self.field_dims)
        self.embedding = FieldAwareFeaturesEmbedding(self.field_dims, self.embedding_dim)
        self.interaction = PairwiseInteraction()

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
