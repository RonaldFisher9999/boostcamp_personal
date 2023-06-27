import numpy as np
import torch
import torch.nn as nn

class FFMLayer(nn.Module):
    def __init__(self, input_dim):
        '''
        Parameter
            input_dim : Entire dimension of input vector (sparse)
        '''
        super().__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(self.input_dim, 1, bias=True)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def square(self, x):
        return torch.pow(x,2)

    def forward(self, sparse_x, dense_x):
        '''
        Parameter
            Input data of size "(batch_size, num_fields)" is converted to sparse_x and dense_x in DeepFFM.
            sparse_x : Float tensor with size "(batch_size, input_dim)"
            dense_x  : Float tensors of size "(batch_size, num_fields, factor_dim)"
                       
        
        Return
            y: Float tensor of size "(batch_size)"
        '''
        y_linear = self.linear(sparse_x)
        square_of_sum = self.square(torch.sum(dense_x, dim=1))
        sum_of_square = torch.sum(self.square(dense_x), dim=1)
        y_pairwise = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1)
        y_ffm = y_linear.squeeze(1) + y_pairwise

        return y_ffm
    
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
        self.mlp_dims = [input_dim] + list(mlp_dims)
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

# 최종적인 DeepFFM 모델
class DeepFFM(nn.Module):
    '''The DeepFM architecture
    Parameter
        field_dims : List of field dimensions
        args.embed_dim : Factorization dimension for dense embedding
        args.mlp_dims : List of positive integer, the layer number and units in each layer
        args.dropout : Float value in [0,1). Fraction of the units to dropout in DNN layer
        args.use_bn : Boolean value to indicate usage of batch normalization in DNN layer.
    '''
    def __init__(self, args, data):
        super().__init__()
        # 각 필드의 차원을 담은 배열
        self.field_dims = data['field_dims']
        # 전체 입력 차원
        self.input_dim = sum(self.field_dims)
        # 필드의 개수
        self.num_fields = len(self.field_dims)
        # FFMLayer
        self.ffm = FFMLayer(self.input_dim)
        # DNNLayer
        self.dnn = DNNLayer(input_dim=(self.num_fields*args.embed_dim), 
                            mlp_dims=args.mlp_dims, 
                            dropout_rate=args.dropout,
                            use_bn=args.use_bn)
        
        # self.encoding_dims = np.concatenate([[0], np.cumsum(self.field_dims)[:-1]])
        self.encoding_dims = np.array((0, *np.cumsum(self.field_dims)[:-1]), dtype=np.int32)
        # print(f"encoding_dims : {self.encoding_dims}")
        
        # 각 feature를 필드 개수만큼의 embed_dim 차원의 벡터로 임베딩
        self.embedding = nn.ModuleList([
            nn.Embedding(feature_size, args.embed_dim) for feature_size in self.field_dims
        ])

        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
        
                
    def forward(self, x):
        '''
        Parameter
            x : Long tensor of size "(batch_size, num_fields)" is coverted to sparse_x and dense_x.
                sparse_x : Float tensor with size "(batch_size, input_dim)"
                dense_x  : List of "num_fields" float tensors of size "(batch_size, embed_dim)"
        Return
            y : Float tensor of size "(batch_size)"
        '''
        # sparse_x : FFMLayer의 linear 부분의 input
        sparse_x = x + x.new_tensor(self.encoding_dims).unsqueeze(0)
        sparse_x = torch.zeros(x.size(0), self.input_dim, device=x.device).scatter_(1, x, 1.)
        # dense_x
        dense_x = [self.embedding[f](x[..., f]) for f in range(self.num_fields)]
        
        y_ffm = self.ffm(sparse_x, torch.stack(dense_x, dim=1))
        y_dnn = self.dnn(torch.cat(dense_x, dim=1))
        y = y_ffm + y_dnn.squeeze(1)

        return y
