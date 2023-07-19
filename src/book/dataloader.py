import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def load_data(data_dir, processed):
    if processed:
        data_dir = os.path.join(data_dir, "processed")
    user_df = pd.read_csv(os.path.join(data_dir, 'users.csv'))
    book_df = pd.read_csv(os.path.join(data_dir, 'books.csv'))
    train_df = pd.read_csv(os.path.join(data_dir, 'train_ratings.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_ratings.csv'))
    sub_df = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
    
    return user_df, book_df, train_df, test_df, sub_df


def users_process(raw_users) :
    users = raw_users.copy()
    
    # location
    users['location'] = users['location'].str.lower().replace('[^0-9a-zA-Z:,]', '', regex=True)
    users['city'] = users['location'].apply(lambda x: x.split(',')[-3].strip())
    users['state'] = users['location'].apply(lambda x: x.split(',')[-2].strip())
    users['country'] = users['location'].apply(lambda x: x.split(',')[-1].strip())
    users = users.replace('na', np.nan)
    users = users.replace('', np.nan)
    users.drop(columns=['location'], inplace=True)
    
    city_state_map = dict(users.groupby('city')['state']
                          .value_counts().sort_values().index.tolist())
    city_country_map = dict(users.groupby('city')['country']
                            .value_counts().sort_values().index.tolist())
    users['state'] = users['city'].map(city_state_map)
    users['country'] = users['city'].map(city_country_map)
    
    # users['location'] = users['country'].copy()
    # users['location'] = np.where(users['location']=='usa',
    #                          users['state'],
    #                          users['location'])
    users['city'].fillna('na', inplace=True)
    users['state'].fillna('na', inplace=True)
    users['country'].fillna('na', inplace=True)
    
    # age
    users['age'].fillna(0, inplace=True)
    bins = [0, 1, 20, 30, 40, 50, 60, 70, 100]
    users['age_bin'] = pd.cut(x=users['age'], bins=bins, right=False, labels=range(8)).astype(int)

    users.drop(columns=['age'], inplace=True)
    
    return users


def isbn_area(isbn) :
    if isbn[0] in ('0', '1') :
        return '1'
    if isbn[0] in ('2', '3', '4', '5', '7') :
        return isbn[0]
    # 6으로 시작하는 경우 없음
    if isbn[0] == '8' :
        return isbn[:2]
    if isbn[0] == '9' :
        if int(isbn[:2]) < 95 :
            return isbn[:2]
        if int(isbn[:2]) < 99 :
            return isbn[:3]
        else :
            return isbn[:4]
    else :
        return 'others'

def books_ratings_process(raw_books, raw_train_ratings, raw_test_ratings) :
    books = raw_books.copy()
    train_ratings = raw_train_ratings.merge(raw_books[['isbn', 'img_url']], how='left', on='isbn')
    test_ratings = raw_test_ratings.merge(raw_books[['isbn', 'img_url']], how='left', on='isbn')
    
    # isbn
    train_ratings['isbn'] = train_ratings['img_url'].apply(lambda x: x.split('P/')[1][:10])
    test_ratings['isbn'] = test_ratings['img_url'].apply(lambda x: x.split('P/')[1][:10])
    books['isbn'] = books['img_url'].apply(lambda x: x.split('P/')[1][:10])
    
    # book_author
    books['book_author'] = books['book_author'].str\
                        .lower().replace('[^0-9a-zA-Z]', '', regex=True)
    
    # year_of_publication
    bins = [0, 1950, 1960, 1970, 1980, 1990, 2000, 2010]
    books['year_of_publication'] = pd.cut(x=books['year_of_publication'],
                                          bins=bins, right=False, labels=range(7)).astype(int)
    
    # publisher
    books['publisher'] = books['publisher'].str\
                        .lower().replace('[^0-9a-zA-Z]', '', regex=True)
    
    # category
    books['category'] = books['category'].str\
                        .lower().replace('[^0-9a-zA-Z]', '', regex=True)
    author_cat_map = dict(books.groupby('book_author')['category']
                      .value_counts().sort_values().index.tolist())
    books['category'] = books['book_author'].map(author_cat_map)
    publisher_cat_map = dict(books.groupby('publisher')['category']
                      .value_counts().sort_values().index.tolist())
    books['category'] = books['category'].fillna(
                        books['publisher'].map(publisher_cat_map))
    books['category'].fillna('na', inplace=True)
    major_cat = ['fiction', 'juvenilefiction', 'juvenilenonfiction', 'biography',
            'histor', 'religio', 'science', 'social', 'politic', 'humor',
            'spirit', 'business', 'cook', 'health', 'famil', 'computer',
            'travel', 'self', 'poet', 'language', 'art', 'language art',
            'literary', 'criticism', 'nature', 'philosoph', 'reference', 'drama',
            'sport', 'transportation', 'comic', 'craft', 'education', 'crime',
            'music', 'animal', 'garden', 'detective', 'house', 'tech', 'photograph',
            'adventure', 'game', 'architect', 'law', 'antique', 'friend',
            'sciencefiction', 'fantasy', 'mathematic', 'design', 'actor',
            'horror', 'adultery']
    books['major_cat'] = books['category'].copy()
    for category in major_cat :
        books['major_cat'] = np.where(books['category'].str.contains(category),
                                     category, books['major_cat'])
        
    # summary
    books['summary'] = np.where(books['summary'].notnull(), 1, 0)
    
    # isbn_area
    books['isbn_area'] = books['isbn'].apply(isbn_area)
    # 선택
    aut_cnt = books['book_author'].value_counts()
    low_cnt_aut = aut_cnt[aut_cnt < 2].index
    books.loc[books['book_author'].isin(low_cnt_aut), 'book_author'] = 'others'
    # 선택
    pub_cnt = books['publisher'].value_counts()
    low_cnt_pub = pub_cnt[pub_cnt < 2].index
    books.loc[books['publisher'].isin(low_cnt_pub), 'publisher'] = 'others'
    # 선택
    cat_cnt = books['major_cat'].value_counts()
    low_cnt_cat = cat_cnt[cat_cnt < 10].index
    books.loc[books['major_cat'].isin(low_cnt_cat), 'major_cat'] = 'others'
    # 선택
    area_cnt = books['isbn_area'].value_counts()
    low_cnt_area = area_cnt[area_cnt < 2].index
    books.loc[books['isbn_area'].isin(low_cnt_area), 'isbn_area'] = 'others'
    
    train_ratings.drop(columns=['img_url'], inplace=True)
    test_ratings.drop(columns=['img_url'], inplace=True)
    books.drop(columns=['book_title', 'img_url', 'language', 'category', 'img_path'],
               inplace=True)
    books.drop(columns=['book_author', 'publisher'], inplace=True)
    books.fillna('na', inplace=True)
    
    return books, train_ratings, test_ratings
    


def process_data(data_dir, processed, save):
    user_df, book_df, train_df, test_df, sub_df = load_data(data_dir, processed)
    if processed == False:
        print("Processing Data")
        user_df = users_process(user_df)
        book_df, train_df, test_df = books_ratings_process(book_df, train_df, test_df)
        if save == True:
            os.makedirs(name=os.path.join(data_dir, "processed/"), exist_ok=True)
            user_df.to_csv(os.path.join(data_dir, "processed/users.csv"), index=False)
            book_df.to_csv(os.path.join(data_dir, "processed/books.csv"), index=False)
            train_df.to_csv(os.path.join(data_dir, "processed/train_ratings.csv"), index=False)
            test_df.to_csv(os.path.join(data_dir, "processed/test_ratings.csv"), index=False)
            sub_df.to_csv(os.path.join(data_dir, "processed/sample_submission.csv"), index=False)
            
    train_test = pd.concat([train_df, test_df]).reset_index(drop=True)
    
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
    
    data = {'user': user_df,
            'book': book_df,
            'train': train_df,
            'test': test_df,
            'sub': sub_df}
    
    return data, field_dims
    

def create_dataloader(data, valid_size, batch_size):
    X_train, X_valid, y_train, y_valid = train_test_split(data['train'].drop(['rating'], axis=1),
                                                          data['train']['rating'],
                                                          test_size=valid_size,
                                                          shuffle=True,
                                                          stratify=data['train']['rating'])
    
    train_dataset = TensorDataset(torch.tensor(X_train.values), torch.tensor(y_train.values, dtype=torch.float))
    valid_dataset = TensorDataset(torch.tensor(X_valid.values), torch.tensor(y_valid.values, dtype=torch.float))
    test_dataset = TensorDataset(torch.tensor(data['test'].values))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    loader = {'train_loader': train_loader,
              'valid_loader': valid_loader,
              'test_loader': test_loader}
    
    return loader


def prepare_data(data_dir, processed, valid_size, batch_size, save):
    data, field_dims = process_data(data_dir, processed, save)
    loader = create_dataloader(data, valid_size, batch_size)
    
    return loader, field_dims, data['sub']