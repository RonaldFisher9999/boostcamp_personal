import os
import argparse
import torch
from .lightgcn.args import parse_args
from .lightgcn.utils import set_seeds, view_parameters
from .lightgcn.dataloader import prepare_data, view_data_info
from .lightgcn.trainer import run
from datetime import datetime
from pytz import timezone
# import wandb
    

def main(args: argparse.Namespace) :
    # wandb.login(key="")
    set_seeds(args.seed)
    use_cuda: bool = torch.cuda.is_available() and args.use_cuda_if_available
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # 데이터 전처리
    print("Processing Data...")
    data = prepare_data(data_dir=args.data_dir, valid_size=args.valid_size, device=device)
    view_data_info(data)
    print("Processing Done")
    
    # 파라미터 출력
    print("Paramters")
    view_parameters(args)
    
    # wandb logging
    # h_readable_ts = datetime.now(timezone("Asia/Seoul")).strftime("%d-%H:%M")
    # wandb.init(
    #     project="dkt",
    #     name=f"{args.user_name}-{'LightGCN'}-{h_readable_ts}",
    #     config=vars(args),
    # )
    
    # 모델 훈련
    run(data=data,
        embedding_dim=args.embedding_dim,
        n_layers=args.n_layers,
        train_n_epochs=args.train_n_epochs,
        valid_n_epochs=args.valid_n_epochs,
        train_lr=args.train_lr,
        valid_lr=args.valid_lr,
        valid_size=args.valid_size,
        use_best=args.use_best,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        device=device)
    
    print("Paramters")
    view_parameters(args)
    
    
if __name__ == "__main__" :
    args = parse_args()
    os.makedirs(name=args.model_dir, exist_ok=True)
    os.makedirs(name=args.output_dir, exist_ok=True)
    main(args=args)