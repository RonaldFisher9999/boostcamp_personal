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

    