import os
import copy
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from typing import Tuple
import wandb


# 모델 훈련. auc, acc, loss 반환    
def train(model: nn.Module,
          train_graph: dict,
          optimizer: torch.optim) -> Tuple[float, float, float] :
    model.train()
    pred = model.forward(train_graph['edge'])
    
    label = train_graph['label'].float()
    loss = model.link_pred_loss(pred, label)
    auc = roc_auc_score(y_true=label.detach().cpu().numpy(),
                        y_score=pred.detach().cpu().numpy())
    acc = accuracy_score(y_true=label.detach().cpu().numpy(),
                         y_pred=(pred.detach().cpu().numpy() > 0.5))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return auc, acc, loss.item()


# valid 데이터에 대해 훈련(input_graph), 예측(target_graph), metric 계산
def validate(model: nn.Module,
             user_group: np.array,
             input_graph: dict,
             target_graph: dict,
             n_epochs: int,
             lr: float,
             use_best: bool) -> Tuple[float, float] :
    # 훈련 중인 모델은 건드리지 않고 copy해서 새 유저 추가
    val_model = copy.deepcopy(model)
    val_model.add_new_users(user_group)
    optimizer = torch.optim.Adam(params=val_model.parameters(), lr=lr)
    
    print(f"Training on Valid Data for {n_epochs} Epochs")
    best_auc = -1
    best_state_dict = dict()
    for epoch in range(n_epochs) :
        auc, acc, loss = train(val_model, input_graph, optimizer)
        
        if best_auc < auc :
            best_auc = auc
            best_state_dict = val_model.state_dict()
    print("Done")
    print("Making Prediction on Valid Data")
    
    if use_best == True :
        val_model.load_state_dict(best_state_dict)
    val_model.eval()
    with torch.no_grad() :
        pred = val_model.forward(target_graph['edge']).detach().cpu().numpy()
    label = target_graph['label'].detach().cpu().numpy()
    val_auc = roc_auc_score(label, pred)
    val_acc = accuracy_score(label, pred > 0.5)
        
    return val_auc, val_acc


def run(model: nn.Module,
        data: dict,
        train_n_epochs: int,
        valid_n_epochs: int,
        max_patience: int,
        train_lr: float,
        valid_lr: float,
        use_best: bool,
        model_dir: str,
        logging: bool,
        timestamp: str) -> Tuple[float, int]:
    # 최적화 함수 정의
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_lr)

    best_auc, best_epoch = 0, -1
    patience = 0
    for epoch in tqdm(range(train_n_epochs)) :
        print(f"\nEpoch : {epoch}")
        # train 결과 반환
        train_auc, train_acc, train_loss = train(model=model,
                                                 train_graph=data['train']['graph'],
                                                 optimizer=optimizer)
        # validate 결과 반환
        valid_auc, valid_acc = validate(model=model,
                                        user_group=data['valid']['user_group'],
                                        input_graph=data['valid']['input_graph'],
                                        target_graph=data['valid']['target_graph'],
                                        n_epochs=valid_n_epochs,
                                        lr=valid_lr,
                                        use_best=use_best)
    
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train ACC : {train_acc:.4f}")
        print(f"Train AUC : {train_auc:.4f}")
        print(f"Valid ACC : {valid_acc:.4f}")
        print(f"Valid AUC : {valid_auc:.4f}")
        
        # auc갱신했으면 모델 저장
        if best_auc < valid_auc:
            print(f"Best AUC updated from {best_auc:.4f} to {valid_auc:.4f}")
            best_auc, best_epoch = valid_auc, epoch
            torch.save(obj= {"model": model.state_dict(), "epoch": epoch},
                       f=os.path.join(model_dir, f"LightGCN_{timestamp}.pt"))
            patience = 0
        else :
            patience += 1
            print(f"Current Best AUC : {best_auc:.4f}")
            print(f"Current Best Epoch : {best_epoch}")
            print(f"Patience Count: {patience}/{max_patience}")
            if patience == max_patience:
                print(f"No Score Improvement for {max_patience} epochs")
                print("Early Stopped Training")
                break
        
        if logging == True:
            wandb.log(
                dict(
                    train_loss=train_loss,
                    train_auc=train_auc,
                    train_acc=train_acc,
                    valid_auc=valid_auc,
                    valid_acc=valid_acc,
                    best_auc=best_auc,
                    patience=patience
                )
            )
        
    # train 다 끝나면 최고 auc와 해당 epoch 출력
    print(f"Best AUC Score : {best_auc:.4f}")
    print(f"Best AUC Confirmed : {best_epoch}'th epoch")
    
    return best_auc, best_epoch
    

# test 데이터에 대해 훈련, 예측, 결과 저장
def inference(model: nn.Module,
              user_group: np.array,
              input_graph: dict,
              target_graph: dict,
              n_epochs: int,
              lr: float,
              use_best: bool,
              model_dir: str,
              output_dir: str,
              timestamp: str) :
    # valid data에서 가장 auc가 좋았던 모델의 파라미터 불러옴
    test_model = copy.deepcopy(model)
    state_dict = torch.load(os.path.join(model_dir, f'LightGCN_{timestamp}.pt'))['model']
    test_model.load_state_dict(state_dict)
    
    test_model.add_new_users(user_group)
                             
    print(f"Training on Test Data for {n_epochs} Epochs")
    optimizer = torch.optim.Adam(params=test_model.parameters(), lr=lr)
    best_auc = -1
    for epoch in range(n_epochs) :
        auc, acc, loss = train(test_model, input_graph, optimizer)
        
        if best_auc < auc :
            best_auc = auc
            best_state_dict = test_model.state_dict()
    print("Done")
    print("Making Prediction on Test Data")
    if use_best == True :
        test_model.load_state_dict(best_state_dict)
    test_model.eval()
    with torch.no_grad() :
        pred = test_model.forward(target_graph['edge']).detach().cpu().numpy()
        
    print("Saving Result ...")    
    write_path = os.path.join(output_dir, f"submission_{timestamp}.csv")
    pd.DataFrame({"prediction": pred}).to_csv(path_or_buf=write_path, index_label="id")
    print(f"Successfully saved submission as {write_path}")