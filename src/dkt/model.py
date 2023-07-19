import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn.conv import LGConv


# LightGCN 모델 정의
class LightGCN(nn.Module) :
    """
    Custom LightGCN model for recommendation system.
    
    Args:
        n_users : The number of users.
        n_items : The number of items.
        group : Dictionary containing group information.
        embedding_dim : The dimension of the node embedding vector.
        n_layers : The number of layers.
    """
    def __init__(self,
                 n_users: int,
                 n_items: int,
                 group: dict,
                 embedding_dim: int,
                 n_layers: int) :
        super().__init__()
        self.n_users = n_users
        self.n_new_users = 0
        self.n_items = n_items
        self.group = group
        
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        # 각 레이어의 가중치
        self.alpha = 1 / (self.n_layers + 1)
        
        # user, item 합친 임베딩
        self.embeddings = nn.Embedding(self.n_users + self.n_items, self.embedding_dim)
        # Graph Convolution Layer
        self.convs = nn.ModuleList([LGConv() for _ in range(self.n_layers)])
        # 마지막에 확률값 예측하기 위한 sigmoid 추가
        self.predict = nn.Sigmoid()
        
        # 파라미터 초기화
        self.reset_parameters()
        
        
    def reset_parameters(self) :
        """
        Reset the model parameters.
        """
        torch.nn.init.xavier_uniform_(self.embeddings.weight)
        for conv in self.convs :
            # conv에 붙는 reset_paramters는 LGConv가 상속한 MessagePasssing의 method임 
            conv.reset_parameters()
            
    # LGConv 레이어를 통과한 노드의 임베딩 벡터        
    def get_embeddings(self, edge_index: np.array) -> torch.tensor :
        """
        Get embeddings for the given edge index.
        
        Args:
            edge_index : Array of edge indices. Shape : (2, n_edges)

        Returns:
            x_embed : Tensor of embeddings. Shape : (2, n_edges, embedding_dim)
        """
        x = self.embeddings.weight
        for i in range(self.n_layers) :
            x_i = self.convs[i](x, edge_index)
            x = x + x_i
        x_embed = x * self.alpha
        
        return x_embed

    
    # 두 노드가 연결되어 있을 확률
    def forward(self, edge_index: torch.tensor) -> torch.tensor :
        """
        Forward pass through the network.
        
        Args:
            edge_index : Tensor of edge indices. Shape : (2, n_edges)

        Returns:
            pred : Predicted output. Shape : (n_edges)
        """
        src, dst = edge_index
        embed = self.get_embeddings(edge_index)
        src_embed = embed[src]
        dst_embed = embed[dst]
        out = (src_embed * dst_embed).sum(dim=1)
        pred = self.predict(out)
        
        return pred
            
    
    # loss 계산
    def link_pred_loss(self,
                       pred: torch.tensor,
                       edge_label:torch.tensor) -> torch.tensor :
        """
        Compute the binary cross entropy loss for link prediction.

        Args:
            pred : Predicted tensor. Shape : (n_edges)
            edge_label : Ground truth tensor. Shape : (n_edges)

        Returns:
            loss: Computed loss. Shape : (1)
        """
        loss_fn = torch.nn.BCELoss()
        loss = loss_fn(pred, edge_label)
        
        return loss
    
    
    # 각 그룹별 유저들의 임베딩 벡터 평균
    def get_group_mean(self) -> torch.tensor :
        """
        Get the mean of the group.

        Returns:
            group_mean_list : Tensor of group means. Shape : (n_groups, embedding_dim)
        """
        group_mean_list = list()
        for group, users in self.group.items() :
            group_mean = self.embeddings.weight[users].mean(dim=0)
            group_mean_list.append(group_mean)
        
        group_mean_list = torch.stack(group_mean_list)
        # print("Group Mean")
        # print(group_mean_list)
        return group_mean_list
    
    
    # 새 유저 추가
    # 새 유저의 초기 임베딩 벡터값은 해당 유저가 속한 그룹의 평균
    # 기존 유저+아이템 임베딩의 뒤에 새 유저 임베딩 추가
    def add_new_users(self, user_groups: np.array) :
        """
        Add new users to the model.
        
        Args:
            user_groups : Array of user group indices. Shape : (n_new_users)
        """
        old_embeddings = self.embeddings.weight.detach()
        group_mean = self.get_group_mean()
        
        self.n_new_users += len(user_groups)
        new_embeddings = group_mean[user_groups].to(old_embeddings.device)

        total_embeddings = torch.cat([old_embeddings, new_embeddings], dim=0)

        self.embeddings = nn.Embedding(
            self.n_users + self.n_items + self.n_new_users, self.embedding_dim).to(old_embeddings.device)
        
        with torch.no_grad():
            self.embeddings.weight.data = total_embeddings
            
    
    # 유저, 아이템 임베딩 추출
    def extract_embeddings(self) -> tuple[torch.tensor, torch.tensor] :
        """
        Extract embeddings for users and items.
        
        Returns:
            user_embeddings : Tensors of user embeddings. Shape : (n_users, embedding_dim)
            item_embeddings : Tensors of item embeddings. Shape : (n_items, embedding_dim)
        """
        user_embeddings = self.embeddings.weight.data[:self.n_users]
        item_embeddings = self.embeddings.weight.data[self.n_users : self.n_users + self.n_items]
        
        return user_embeddings, item_embeddings