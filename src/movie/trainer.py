import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple
import wandb


def train(model: torch.nn.Module,
          data_loader: torch.utils.data.DataLoader,
          criterion: torch.nn,
          optimizer: torch.optim) -> torch.tensor:
    model.train()
    total_loss = 0 
    for seq, label in tqdm(data_loader):
        seq = seq.to(model.device)
        label = label.to(model.device)
        
        pred = model.forward(seq)
        pred = pred.view(-1, pred.size(-1))
        label = label.view(-1).to(model.device)
        
        loss = criterion(pred, label)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return total_loss
    
    
def validate(model: torch.nn.Module,
             valid_data: list,
             valid_cand: list,
             k: int) -> Tuple[float, float]:
    model.eval()
    recall = 0
    for user_idx, user_seq in enumerate(tqdm(valid_data)):
        user_seq = user_seq.to(model.device)
        mask = (user_seq==model.n_items+1)
        with torch.no_grad():
            pred = model.forward(user_seq)
        softmax_score = nn.Softmax(dim=1)(pred[mask])
        score_sum = -softmax_score.sum(dim=0)
        cand_score = score_sum[valid_cand[user_idx]]
            
        # rank for valid item
        rank_list = cand_score.argsort().argsort()[:k] 
        recall_cnt = 0
        for rank in rank_list: # @k
            if rank < k:
                recall_cnt += 1
        recall += recall_cnt/10

    return recall/len(valid_data)


def run(model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        valid_data: list,
        valid_cand: list,
        k: int,
        n_epochs: int,
        lr: float,
        max_patience: int,
        logging: bool,
        model_dir: str,
        timestamp: str):
    criterion = nn.CrossEntropyLoss(ignore_index=0) # label이 0인 경우 무시
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_recall, best_epoch = 0, -1
    patience = 0
    state_dict = dict()
    for epoch in tqdm(range(1, n_epochs+1)): 
        print(f"Epoch: {epoch}")
        print("Train")
        train_loss = train(model, data_loader, criterion, optimizer)
        print(f"\nTrain Loss: {train_loss:.4f}")
        print("Validate")
        recall = validate(model, valid_data, valid_cand, k)
        print(f"\nValid Recall@{k}: {recall:.4f}")
        
        if best_recall < recall :
            print(f"Recall@{k} Updated From {best_recall:.4f} -> {recall:.4f}")
            best_recall = recall
            best_epoch = epoch
            state_dict = model.state_dict()
            torch.save(obj={"model": state_dict, "recall": best_recall, "epoch": best_epoch},
               f=os.path.join(model_dir, f"bert4rec_{timestamp}.pt"))
            patience = 0
        else :
            patience += 1
            print(f"Current Best Recall@{k}: {best_recall:.4f}")
            print(f"Current Best Epoch: {best_epoch}")
            print(f"Patience Count: {patience}/{max_patience}")
            if patience == max_patience:
                print(f"No Score Improvement for {max_patience} epochs")
                print("Early Stopped Training")
                break
            
        if logging == True:
            wandb.log(
                dict(
                    train_loss=train_loss,
                    recall=recall,
                    best_recall=best_recall,
                )
            )
            
    print(f"Best Recall@{k}: {best_recall:.4f}")
    print(f"Best Recall@{k} Confirmed: {best_epoch}'th epoch")
    
    return best_recall, best_epoch
    
    
def inference(model: torch.nn.Module,
              infer_data: pd.Series,
              infer_cand: list,
              k: int,
              sub_df: pd.DataFrame,
              idx2item: dict,
              model_dir: str,
              output_dir: str,
              timestamp: str,
              model_name: str=None):
    model.eval()
    
    if model_name is not None:
        state_dict = torch.load(os.path.join(model_dir, f"{model_name}.pt"))['model']
        print(f"Load From Pretrained {model_name}.pt")
    else:
        state_dict = torch.load(os.path.join(model_dir, f'bert4rec_{timestamp}.pt'))['model']
    model.load_state_dict(state_dict)
    
    inference = np.array([])
    for user_idx, user_seq in enumerate(tqdm(infer_data)):
        user_seq = user_seq.to(model.device)
        mask = (user_seq==model.n_items+1)
        with torch.no_grad():
            pred = model.forward(user_seq)
        softmax_score = nn.Softmax(dim=1)(pred[mask])
        score_sum = softmax_score.sum(dim=0)
        cand_score = score_sum[infer_cand[user_idx]]

        top_k_idx = cand_score.argsort(descending=True)[:k].to('cpu')
        top_k = infer_cand[user_idx][top_k_idx]
        inference = np.append(inference, top_k)
    
    if model_name is not None:
        sub_df = sub_df.drop_duplicates(subset=['user'])
        sub_df = sub_df.loc[np.repeat(sub_df.index.values, k)]
    sub_df['item'] = inference
    sub_df['item'] = sub_df['item'].map(idx2item).astype(int)
    sub_df.to_csv(os.path.join(output_dir, f"sub_{timestamp}.csv"), index=False)
    
    
    