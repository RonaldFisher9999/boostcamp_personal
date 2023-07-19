import os
import pandas as pd
import numpy as np
import torch
from typing import Tuple
from tqdm import tqdm
from torch.utils.data import Dataset


def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(os.path.join(data_dir, 'train/train_ratings.csv')).drop('time', axis=1)
    sub_df = pd.read_csv(os.path.join(data_dir, 'eval/sample_submission.csv'))
    
    return train_df, sub_df


# 전체 sequence에서 max_len만큼 sampling
# tail_ratio: max_len의 tail_ratio만큼은 제일 마지막에서 샘플링, 나머지는 중간에서 랜덤 샘플링
def seq_sampling(total: pd.Series, max_len: int, tail_ratio: float) -> np.array:
    if total.size > max_len:
        tail_len = int(max_len * tail_ratio)
        sample_idx = np.random.choice(np.arange(0, total.size - tail_len), max_len - tail_len, replace=False)
        sample_idx = np.sort(sample_idx)
        sample_seq = total[sample_idx]
        if tail_len != 0:
            sample_seq = np.append(sample_seq, total[-tail_len:])
    else :
        sample_seq = total
        
    return sample_seq


# valid, test sequence에 k개 mask 섞어줌
def mix_mask(temp_seq: np.array, k: int, mask: int) -> np.array:
    seq = np.zeros(temp_seq.size+k, dtype=int)
    seq[-k//2:] = mask
    mask_idx = np.sort(np.random.choice(np.arange(0, temp_seq.size+(k//2)), k//2, replace=False))
    seq[mask_idx] = mask
    seq[seq == 0] = temp_seq
    
    return seq


def add_padding(seq: np.array, max_len: int) -> np.array:
    pad_len = max_len - seq.size
    seq = np.append([0] * pad_len, seq)
    
    return seq


def process_data(train_df: pd.DataFrame,
                 max_len: int,
                 k: int,
                 n_samples: int,
                 tail_ratio: float) -> Tuple[dict, int, int, dict]:
    item_idx = train_df['item'].unique()
    user_idx = train_df['user'].unique()
    
    user2idx = {user:idx for idx,user in enumerate(user_idx)}
    item2idx = {item:idx+1 for idx,item in enumerate(item_idx)}
    idx2item = {idx+1:item for idx,item in enumerate(item_idx)}
    n_items = len(item2idx)
    n_users = len(user2idx)
    
    train_df['user'] = train_df['user'].map(user2idx)
    train_df['item'] = train_df['item'].map(item2idx)
    
    total = train_df.groupby('user')['item'].apply(np.array)
    train_seq = list()
    valid_seq = list()
    valid_target = list()
    infer_seq = list()
    valid_cand = list()
    infer_cand = list()
    for user_idx, user_total in enumerate(tqdm(total)):
        # user_valid_target: 맨 뒤에서 절반, 중간에서 랜덤 절반 추출
        user_valid_target = np.random.choice(user_total[:-(k//2)], (k//2), replace=False)
        user_valid_target = np.append(user_valid_target, user_total[-(k-k//2):])
        valid_target.append(user_valid_target)
        
        # user_total_train: user_valid_target 제외
        user_total_train = user_total[~np.isin(user_total, user_valid_target)]
        
        # user_train_seq: user_total_train에서 max_len만큼 샘플링(n_samples 수만큼)
        for _ in range(n_samples):
            user_train_seq = seq_sampling(user_total_train, max_len, tail_ratio)
            train_seq.append(user_train_seq)

        # user_valid_seq: user_total_train에서 max_len-k만큼 샘플링. 이후 k개의 masking 섞어줌 (절반은 맨 뒤에, 나머지는 중간에 랜덤)
        temp_valid_seq = seq_sampling(user_total_train, max_len-k, tail_ratio)
        user_valid_seq = mix_mask(temp_valid_seq, k, n_items+1)
        # add padding
        if user_valid_seq.size < max_len :
            user_valid_seq = add_padding(user_valid_seq, max_len)
        valid_seq.append(torch.tensor(user_valid_seq).unsqueeze(0))
        
        temp_infer_seq = seq_sampling(user_total, max_len-k, tail_ratio)
        user_infer_seq = mix_mask(temp_infer_seq, k, n_items+1)
        if user_infer_seq.size < max_len :
            user_infer_seq = add_padding(user_infer_seq, max_len)
        infer_seq.append(torch.tensor(user_infer_seq).unsqueeze(0))
        
        # user_valid_cand: 전체 negative + user_valid_target
        user_infer_cand = np.setdiff1d(np.arange(1, n_items+1), user_total)
        # user_infer_candL 전체 negative
        user_valid_cand = np.append(user_valid_target, user_infer_cand)
        valid_cand.append(user_valid_cand)
        infer_cand.append(user_infer_cand)
        
    data = {'train': train_seq,
            'valid': valid_seq,
            'valid_target': valid_target,
            'infer': infer_seq,
            'valid_cand': valid_cand,
            'infer_cand': infer_cand}
    
    return data, n_items, n_users, idx2item


class BERT4RecDataset(Dataset):
    def __init__(self,
                 train_data: list,
                 n_users: int,
                 n_items: int,
                 max_len: int,
                 k: int,
                 mask_prob: float):
        self.train_data = train_data
        self.n_users = n_users
        self.n_items = n_items
        self.max_len = max_len
        self.k = k
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, user_idx: int) -> Tuple[torch.tensor, torch.tensor]: 
        seq = self.train_data[user_idx]
        masked_seq = seq.copy()
        labels = np.zeros_like(seq)
        
        # 중간 랜덤 5개 masking
        mask_idx = np.random.choice(np.arange(0, seq.size-(self.k//2)), int(seq.size*self.mask_prob))
        masked_seq[mask_idx] = self.n_items+1
        # 마지막 5개 masking
        masked_seq[-(self.k//2):] = self.n_items+1
        labels[mask_idx] = seq[mask_idx]
        labels[-(self.k//2):] = seq[-(self.k//2):]
                
        # zero padding
        if seq.size < self.max_len:
            pad_len = self.max_len - seq.size
            masked_seq = np.append([0] * pad_len, masked_seq)
            labels = np.append([0] * pad_len, labels)
        
        masked_seq = torch.LongTensor(masked_seq)
        labels = torch.LongTensor(labels)
        
        return masked_seq, labels