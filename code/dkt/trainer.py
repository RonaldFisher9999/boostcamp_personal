import os
import copy
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from model import LightGCN
from utils import get_timestamp
# import wandb


# 모델 훈련. auc, acc, loss 반환    
def train(model: nn.Module,
          train_graph: dict,
          optimizer: torch.optim) -> tuple[float, float, float] :
    model.train()
    pred = model.forward(train_graph['edge'])
    
    label = train_graph['label'].float()
    loss = model.link_pred_loss(pred, label)
    auc = roc_auc_score(y_true=label.detach().cpu().numpy(), y_score=pred.detach().cpu().numpy())
    acc = accuracy_score(y_true=label.detach().cpu().numpy(), y_pred=(pred.detach().cpu().numpy() > 0.5))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return auc, acc, loss.item()


# test/valid 데이터에 대해 예측하기 위한 전처리
# target : 예측할 데이터 (유저가 푼 마지막 문제)
# input : 모델로 학습할 데이터 (target 아닌 나머지)
# user_groups : 각 유저의 group 정보 ex) [3, 4, 0, 0, 1 ...]
def process_valid_data(df: pd.DataFrame,
                       id2index: dict,
                       device: str) -> tuple[np.array, dict, dict] :
    valid_df = df.copy()
    val_id2index = id2index.copy()
    
    # 기존에 유저 id를 인덱싱했던 id2index에 새 유저의 인덱싱 정보 추가
    val_new_users = valid_df['user_id'].unique()
    for i, user_id in enumerate(val_new_users) :
        val_id2index[user_id] = i + len(id2index)
    valid_df['user_id'] = valid_df['user_id'].map(val_id2index)
    valid_df['item_id'] = valid_df['item_id'].map(val_id2index)
    
    # 각 유저에 대응되는 그룹 리스트
    user_groups = valid_df.drop_duplicates(subset=['user_id'], keep='last')['user_group'].values
    
    target_df = valid_df.drop_duplicates(subset=['user_id'], keep='last')
    input_df = valid_df[~valid_df.index.isin(target_df.index)]
    
    input_edge = torch.LongTensor(input_df.values.T[0:2]).to(device)
    input_label = torch.LongTensor(input_df.values.T[2]).to(device)
    
    target_edge = torch.LongTensor(target_df.values.T[0:2]).to(device)
    target_label = torch.LongTensor(target_df.values.T[2]).to(device)
    
    input_graph = {'edge' : input_edge, 'label' : input_label}
    target_graph = {'edge' : target_edge, 'label' : target_label}
    
    return user_groups, input_graph, target_graph


# valid 데이터에 대해 훈련(input_graph), 예측(target_graph), metric 계산
def validate(model: nn.Module,
             user_groups: np.array,
             input_graph: dict,
             target_graph: dict,
             n_epochs: int,
             lr: float,
             use_best: bool) -> tuple[float, float] :
    # 훈련 중인 모델은 건드리지 않고 copy해서 새 유저 추가
    val_model = copy.deepcopy(model)
    val_model.add_new_users(user_groups)
    optimizer = torch.optim.Adam(params=val_model.parameters(), lr=lr)
    
    print(f"Training on Valid Data for {n_epochs} Epochs")
    best_auc = -1
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


# test 데이터에 대해 훈련, 예측, 결과 저장
def inference(model: nn.Module,
              user_groups: np.array,
              input_graph: dict,
              target_graph: dict,
              n_epochs: int,
              lr: float,
              use_best: bool,
              timestamp: str,
              model_dir: str,
              output_dir: str) :
    # valid data에서 가장 auc가 좋았던 모델의 파라미터 불러옴
    test_model = copy.deepcopy(model)
    state_dict = torch.load(os.path.join(model_dir, f'LightGCN_{timestamp}.pt'))['model']
    test_model.load_state_dict(state_dict)
    
    test_model.add_new_users(user_groups)
                             
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


# 훈련부터 제출까지
def run(
    model: nn.Module,
    data : dict,
    embedding_dim : int,
    n_layers : int,
    train_n_epochs : int,
    valid_n_epochs : int,
    max_patience: int,
    train_lr : float,
    valid_lr : float,
    valid_size : float, # 여기서 사용하진 않지만 파라미터 출력용
    use_best : bool,
    model_dir : str,
    output_dir : str,
    device : str,
    timestamp: str
) :
    train_graph = data['train_graph']
    id2index = data['id2index']
    valid_df = data['valid_df']
    test_df = data['test_df']
    n_users = data['n_users']
    n_items = data['n_items']

    
    # 최적화 함수 정의, 모델 저장 폴더 생성
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_lr)
    
    
    print(f"Training Started : n_epochs={train_n_epochs}")
    # valid_df 전처리
    valid_user_groups, valid_input_graph, valid_target_graph = process_valid_data(df=valid_df, id2index=id2index, device=device)

    best_auc, best_epoch = 0, -1
    patience = 0
    state_dict = dict()
    for epoch in tqdm(range(train_n_epochs)) :
        print(f"\nEpoch : {epoch}")
        # train 결과 반환
        train_auc, train_acc, train_loss = train(model=model, train_graph=train_graph, optimizer=optimizer)
        # validate 결과 반환
        valid_auc, valid_acc = validate(model=model, user_groups=valid_user_groups, input_graph=valid_input_graph, target_graph=valid_target_graph,
                                        n_epochs=valid_n_epochs, lr=valid_lr, use_best=use_best)
        
        # wandb.log(
        #     dict(
        #         epoch=epoch,
        #         train_loss=train_loss,
        #         train_auc=train_auc,
        #         train_acc=train_acc,
        #         valid_auc=valid_auc,
        #         valid_acc=valid_acc,
        #     )
        # )
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
        
    # train 다 끝나면 최고 auc와 해당 epoch 출력
    print(f"Best AUC Score : {best_auc:.4f}")
    print(f"Best AUC Confirmed : {best_epoch}'th epoch")
    
    # 파라미터와 결과 json 파일로 저장
    report = {
        'params' : {
            "embedding_dim" : embedding_dim,
            "n_layers" : n_layers,
            "train_n_epochs" : train_n_epochs,
            "valid_n_epochs" : valid_n_epochs,
            "train_lr" : train_lr,
            "valid_lr" : valid_lr,
            "valid_size" : valid_size,
            "use_best" : use_best
        },
        'result' : {
            "best_epoch" : best_epoch,
            "best_auc" : best_auc
        }
    }
    
    with open(os.path.join(model_dir, f'LightGCN_{timestamp}.json'), 'w') as f:
        json.dump(report, f)
    
    # inference
    test_user_groups, test_input_graph, test_target_graph = process_valid_data(df=test_df, id2index=id2index, device=device)
    inference(model=model,
              user_groups=test_user_groups,
              input_graph=test_input_graph,
              target_graph=test_target_graph,
              n_epochs=valid_n_epochs,
              lr=valid_lr,
              use_best=use_best,
              timestamp=timestamp,
              model_dir=model_dir,
              output_dir=output_dir)