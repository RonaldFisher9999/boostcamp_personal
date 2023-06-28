import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def load_data(data_dir):
    user_df = pd.read_csv(os.path.join(data_dir, 'preprocessed/users.csv'))
    book_df = pd.read_csv(os.path.join(data_dir, 'preprocessed/books.csv'))
    train_df = pd.read_csv(os.path.join(data_dir, 'preprocessed/train_ratings.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'preprocessed/test_ratings.csv'))
    sub_df = pd.read_csv(os.path.join(data_dir, 'preprocessed/sample_submission.csv'))
    
    return user_df, book_df, train_df, test_df, sub_df


def process_data(data_dir, valid_size, batch_size):
    """
    Parameters
    ----------
    user_df : pd.DataFrame
        user_df.csv를 인덱싱한 데이터
    book_df : pd.DataFrame
        book_df.csv를 인덱싱한 데이터
    train_df : pd.DataFrame
        train 데이터의 rating
    test_df : pd.DataFrame
        test 데이터의 rating
    ----------
    """
    
    user_df, book_df, train_df, test_df, sub_df = load_data(data_dir)
    train_test = pd.concat([train_df, test_df]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    context_df = train_test.merge(user_df, on='user_id', how='left')\
                    .merge(book_df, on='isbn', how='left')\
                    .drop(columns=['user_id', 'isbn'])
    train_df = train_df.merge(user_df, on='user_id', how='left')\
                    .merge(book_df, on='isbn', how='left')\
                    .drop(columns=['user_id', 'isbn'])
    test_df = test_df.merge(user_df, on='user_id', how='left')\
                    .merge(book_df, on='isbn', how='left')\
                    .drop(columns=['user_id', 'isbn', 'rating'])

    features = context_df.drop(columns=['rating']).columns
    # 모든 feature 인덱싱 처리
    field_dims = list()
    for feature in features :
        feature2idx = {v:k for k,v in enumerate(context_df[feature].unique())}
        field_dims.append(len(feature2idx))
        train_df[feature] = train_df[feature].map(feature2idx)
        test_df[feature] = test_df[feature].map(feature2idx)
    
    field_dims = np.array(field_dims, dtype=np.uint32)
    
    print(f"field_dims: {field_dims}")
    print(f"total input dim : {sum(field_dims)}")
    
    
    X_train, X_valid, y_train, y_valid = train_test_split(train_df.drop(['rating'], axis=1),
                                                          train_df['rating'],
                                                          test_size=valid_size,
                                                          shuffle=True,
                                                          stratify=train_df['rating'])
    
    train_dataset = TensorDataset(torch.tensor(X_train.values), torch.tensor(y_train.values, dtype=torch.float))
    valid_dataset = TensorDataset(torch.tensor(X_valid.values), torch.tensor(y_valid.values, dtype=torch.float))
    test_dataset = TensorDataset(torch.tensor(test_df.values))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    data = {'sub': sub_df,
            'field_dims': field_dims,
            'train_loader': train_loader,
            'valid_loader': valid_loader,
            'test_loader': test_loader
            }
    
    return data


def context_data_split(data,
                       valid_size):
    """
    Parameters
    ----------
    Args:
        test_size : float
            Train/Valid split 비율을 입력합니다.
        seed : int
            랜덤 seed 값
    ----------
    """

    X_train, X_valid, y_train, y_valid = train_test_split(
                                                    data['train'].drop(['rating'], axis=1),
                                                    data['train']['rating'],
                                                    test_size=valid_size,
                                                    shuffle=True,
                                                    stratify=data['train']['rating']
                                                    )

    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    
    return data

def context_data_loader(data,
                        batch_size):
    """
    Parameters
    ----------
    Args:
        batch_size : int
            데이터 batch에 사용할 데이터 사이즈
        data_shuffle : bool
            data shuffle 여부
    ----------
    """
    train_dataset = TensorDataset(torch.tensor(data['X_train'].values), torch.tensor(data['y_train'].values, dtype=torch.float))
    valid_dataset = TensorDataset(torch.tensor(data['X_valid'].values), torch.tensor(data['y_valid'].values, dtype=torch.float))
    test_dataset = TensorDataset(torch.tensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data