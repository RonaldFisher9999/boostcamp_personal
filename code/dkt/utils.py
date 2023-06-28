import os
import random
import argparse
from datetime import datetime
from pytz import timezone
import torch
import numpy as np


# 현재 시간
def get_timestamp() -> str :
    return datetime.now(timezone("Asia/Seoul")).strftime("%m%d_%H%M%S")

# 랜덤시드 고정
def set_seeds(seed: int) :
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def view_parameters(args: argparse.Namespace) :
    print(f"--embedding_dim {args.embedding_dim} --n_layers {args.n_layers} "
      f"--train_n_epochs {args.train_n_epochs} --valid_n_epochs {args.valid_n_epochs} "
      f"--train_lr {args.train_lr} --valid_lr {args.valid_lr} --valid_size {args.valid_size} "
      f"--use_best {args.use_best}")

    