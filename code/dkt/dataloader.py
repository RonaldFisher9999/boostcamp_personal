import os
import pandas as pd
import numpy as np
import torch
from typing import Tuple

    
# 데이터 불러오기
def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dtype = {'userID': 'int16', 'answerCode': 'int8', 'KnowledgeTag': 'int16'}
    raw_train_df = pd.read_csv(os.path.join(data_dir, "train_data.csv"),
                               dtype=dtype, parse_dates=['Timestamp'])
    raw_test_df = pd.read_csv(os.path.join(data_dir, "test_data.csv"),
                              dtype=dtype, parse_dates=['Timestamp'])
    
    return raw_train_df, raw_test_df


# 유저가 마지막에 푼 'item_cat'으로 'user_group' feature 생성
def process_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df[['userID', 'assessmentItemID', 'answerCode']].copy()
    df.drop_duplicates(subset=["userID", "assessmentItemID"], keep="last", inplace=True)
    df.columns = ['user_id', 'item_id', 'answer']
    df['item_cat'] = df['item_id'].str[2].astype('int8')
    user_group = df.drop_duplicates(subset=['user_id'], keep='last').set_index('user_id')['item_cat']
    df['user_group'] = df['user_id'].map(user_group)
    df['user_group'] = df['user_group'] - 1
    df.drop('item_cat', axis=1, inplace=True)         

    return df

# 유저 평균 정답률로 그룹 생성
# def process_data(raw_df) :
#     df = raw_df[['userID', 'assessmentItemID', 'answerCode']].copy()
#     df.drop_duplicates(subset=["userID", "assessmentItemID"], keep="last", inplace=True)
#     df.columns = ['user_id', 'item_id', 'answer']
#     df['item_cat'] = df['item_id'].str[2].astype('int8')
    
#     last = df.drop_duplicates(subset=['user_id'], keep='last')
#     not_last = df[~df.index.isin(last.index)]
#     user_mean = not_last.groupby('user_id')['answer'].mean()
#     df['user_mean'] = df['user_id'].map(user_mean)
#     bins = [-1, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1]
#     df['user_group'] = pd.cut(x=df['user_mean'], bins=bins, right=True, labels=range(len(bins)-1))
#     df['user_group'] = df['user_group'].astype('int8')
    
#     df.drop(columns=['item_cat', 'user_mean'], axis=1, inplace=True)         

#     return df


# 전체 유저에서 일정 비율의 유저 valid로 분리
def split_data(train: pd.DataFrame,
               valid_size: float) -> Tuple[pd.DataFrame, pd.DataFrame] :
    n_valid_users = int(train['user_id'].nunique() * valid_size)
    valid_users = np.random.choice(train['user_id'].unique(), n_valid_users, replace=False)

    valid_data = train[train['user_id'].isin(valid_users)].copy()
    train_data = train[~train.index.isin(valid_data.index)].copy()
    
    return train_data, valid_data


# 유저, 아이템 인덱싱 딕셔너리 생성
def get_index(data: pd.DataFrame) -> dict :
    userid = data['user_id'].unique()
    itemid = data['item_id'].unique()
    n_user = len(userid)
    
    userid2index = {v: i for i, v in enumerate(userid)}
    itemid2index = {v: i + n_user for i, v in enumerate(itemid)}
    id2index = {**userid2index, **itemid2index}
    
    return id2index


# train 데이터 딕셔너리 형태의 그래프로 변환
# edge : 유저-아이템 엣지 정보
# label : 맞았으면 1, 틀렸으면 0
# group : 각 그룹에 어떤 유저가 속해있는지를 저장한 딕셔너리
def df_to_graph(data: pd.DataFrame, id2index: dict, device: str) -> dict :
    graph = dict()
    
    df = data.copy()
    df['user_id'] = df['user_id'].map(id2index)
    df['item_id'] = df['item_id'].map(id2index)
    
    edge = torch.LongTensor(df.values.T[0:2])
    label = torch.LongTensor(df.values.T[2])
    graph['edge'] = edge.to(device)
    graph['label'] = label.to(device)
    
    user_group = df.groupby('user_group')['user_id'].unique().to_dict()
    graph['group'] = user_group
    
    return graph


# 다 묶어서
def prepare_data(data_dir: str, valid_size:float, device: str) -> dict :
    raw_train_df, raw_test_df = load_data(data_dir)
    train_valid_df = process_data(raw_train_df)
    test_df = process_data(raw_test_df)
    train_df, valid_df = split_data(train_valid_df, valid_size)
    n_users = train_df['user_id'].nunique()
    n_items = train_df['item_id'].nunique()
    id2index = get_index(train_df)
    train_graph = df_to_graph(train_df, id2index, device)
    
    data = {'train_graph' : train_graph,
            'id2index' : id2index,
            'valid_df' : valid_df,
            'test_df' : test_df,
            'n_users' : n_users,
            'n_items' : n_items}
    
    return data


def view_data_info(data: dict) :
    print("Train Data Info")
    print(f"Number of Users : {data['n_users']}")
    print(f"Number of Items : {data['n_items']}")
    print(f"Number of Edges : {len(data['train_graph']['label'])}")
    print(f"Number of Groups : {len(data['train_graph']['group'])}")