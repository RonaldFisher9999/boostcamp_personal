import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from .lightgcn.utils import set_seeds, get_timestamp
from datetime import datetime
from pytz import timezone
from tqdm import tqdm


def load_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dtype = {'userID': 'int16', 'answerCode': 'int8', 'KnowledgeTag': 'int16'}
    raw_train_df = pd.read_csv(os.path.join(data_dir, "train_data.csv"),
                               dtype=dtype, parse_dates=['Timestamp'])
    raw_test_df = pd.read_csv(os.path.join(data_dir, "test_data.csv"),
                               dtype=dtype, parse_dates=['Timestamp'])
    
    return raw_train_df, raw_test_df


def common_process(raw_df) :
    df = raw_df.copy()
    df.columns = ['user_id', 'item_id', 'test_id', 'answer', 'time', 'tag_id']
    diff = df.groupby('user_id')['time'].diff()
    df['time_diff'] = diff.apply(lambda x: x.total_seconds())
    df['user_test_cnt'] = df.groupby(['user_id', 'test_id']).cumcount()
    df['test_item_cnt'] = df['test_id'].map(df.groupby('test_id')['item_id'].nunique())
    df['user_test_cnt'] = df['user_test_cnt'] % df['test_item_cnt']
    df['time_diff'] = np.where(df['user_test_cnt']==0, -1, df['time_diff'])
    df['time_diff'] = df['time_diff'].apply(lambda x : 9999 if x > 320 else x)
    df.drop(columns=['user_test_cnt', 'test_item_cnt', 'time'], inplace=True)
    
    bins = [-1, 0, 10, 20, 40, 80, 160, 320, 100000]
    df['time_bin'] = pd.cut(x=df['time_diff'], bins=bins, right=False, labels=range(len(bins)-1))
    df['time_bin'] = df['time_bin'].astype('int8')
    df.drop(columns=['time_diff'], inplace=True)
    
    return df


def get_index(train_df, cat_features) :
    to_index = dict()
    for feat in cat_features :
        # For 0 padding
        feat_to_index = {v:(i+1) for i, v in enumerate(train_df[feat].unique())}
        to_index[feat] = feat_to_index
        
    return to_index


def get_train_valid_test(procssed_train_df, processed_test_df, cat_features, train_users) :
    train_df = processed_train_df[processed_train_df['user_id'].isin(train_users)].copy()
    valid_df = processed_train_df[~processed_train_df['user_id'].isin(train_users)].copy()
    test_df  = processed_test_df.copy()
    
    to_index = get_index(train_df, cat_features)
    for feat in cat_features :
        train_df[feat] = train_df[feat].map(to_index[feat])
        valid_df[feat] = valid_df[feat].map(to_index[feat])
        test_df[feat] = test_df[feat].map(to_index[feat])
        # for unknown values
        valid_df[feat].fillna(len(to_index[feat]), inplace=True)
        test_df[feat].fillna(len(to_index[feat]), inplace=True)
    
    train_group = train_df.groupby('user_id').apply(lambda x: {col:x[col].values for col in x.columns if col != 'user_id'})
    valid_group = valid_df.groupby('user_id').apply(lambda x: {col:x[col].values for col in x.columns if col != 'user_id'})
    test_group = test_df.groupby('user_id').apply(lambda x: {col:x[col].values for col in x.columns if col != 'user_id'})

    return train_group, valid_group, test_group


class SaintDataset(Dataset):
    def __init__(self, group: pd.Series, max_seq: int, train: bool=True) :
        super().__init__()
        self.group = group
        self.max_seq = max_seq
        self.data = []
        # Data Augumentation
        for id in self.group.index:
            item, test, answer, tag, time = self.group[id].values()
            if len(item) > max_seq :
                # Train
                if train == True :
                    for i in range((len(item)-1)//max_seq + 1):
                        self.data.append(
                            (item[i:i+max_seq], test[i:i+max_seq], tag[i:i+max_seq],
                            time[i:i+max_seq], answer[i:i+max_seq])
                            )
                # Valid/Test
                else :
                    self.data.append(
                        (item[-max_seq:], test[-max_seq:], tag[-max_seq:],
                        time[-max_seq:], answer[-max_seq:])
                    )
            else:
                self.data.append((item, test, tag, time, answer))

    def __len__(self):
        return len(self.data)
    
    def pad_seq(self, seq) :
        seq_len = len(seq)
        result = np.zeros(self.max_seq, dtype=int)
        result[-seq_len:] = seq
        
        return result
        
    def __getitem__(self, idx):
        item, test, tag, time, answer = self.data[idx]
        seq_len = len(answer)
        
        item_seq = self.pad_seq(item)
        test_seq = self.pad_seq(test)
        tag_seq = self.pad_seq(tag)
        time_seq = self.pad_seq(time)
        answer_seq = self.pad_seq(answer+1)
        
        input_features = {'item' : item_seq,
                          'test' : test_seq,
                          'tag' : tag_seq,
                          'time' : time_seq}
        
        start_point = self.max_seq - seq_len
        label = np.insert(answer_seq, start_point, 3)[:-1]
        answer_seq = answer_seq - 1
        pad = (label != 0)
        
        return input_features, label, answer_seq, pad
    
    
class FFN(nn.Module):
    def __init__(self, emb_dim, drop_out):
        super(FFN, self).__init__()
        self.linear_1 = nn.Linear(emb_dim, emb_dim)
        self.linear_2 = nn.Linear(emb_dim, emb_dim)
        
        # nn.init.xavier_normal_(self.linear_1.weight)
        # nn.init.xavier_normal_(self.linear_2.weight)
        # if self.linear_1.bias is not None:
        #     nn.init.zeros_(self.linear_1.bias)
        # if self.linear_2.bias is not None:
        #     nn.init.zeros_(self.linear_2.bias)
            
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        out = self.linear_1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear_2(out)
        out = self.dropout(out)

        return out
    
class EncoderEmbedding(nn.Module) :
    def __init__(self, emb_dim, n_items, n_tests, n_tags, max_seq, device) :
        super(EncoderEmbedding, self).__init__()
        self.max_seq = max_seq
        self.device = device
        self.item_embed = nn.Embedding(n_items+1, emb_dim, padding_idx=0)
        self.test_embed = nn.Embedding(n_tests+1, emb_dim, padding_idx=0)
        self.tag_embed = nn.Embedding(n_tags+1, emb_dim, padding_idx=0)
        self.position_embed = nn.Embedding(max_seq, emb_dim)
        
        # for embed in [self.item_embed, self.test_embed, self.tag_embed, self.position_embed] :
        #     nn.init.xavier_uniform_(embed.weight)
        #     embed._fill_padding_idx_with_zero()
        
    def forward(self, item_idx, test_idx, tag_idx) :
        item_emb = self.item_embed(item_idx)
        test_emb = self.test_embed(test_idx)
        tag_emb = self.tag_embed(tag_idx)
        pos = torch.arange(self.max_seq, device=self.device).unsqueeze(0)
        pos_emb = self.position_embed(pos)
        enc_emb = (item_emb + test_emb + tag_emb + pos_emb) / 4
    
        return enc_emb
        
    
class EncoderBlock(nn.Module) :
    def __init__(self, n_heads, emb_dim, drop_out, max_seq, device) :
        super(EncoderBlock, self).__init__()
        self.max_seq = max_seq
        self.device = device
        self.attn_mask = torch.triu(torch.full((max_seq, max_seq), float('-inf'), device=device), diagonal=1)
        # self.attn_mask = torch.triu(torch.ones((max_seq, max_seq), device=device), diagonal=1).bool()
        # print(self.attn_mask.shape)
        # print(self.attn_mask)
        # exit()

        self.mha = nn.MultiheadAttention(emb_dim, n_heads, batch_first=True)
        self.ffn = FFN(emb_dim, drop_out)
        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(emb_dim)
        
    def forward(self, enc_emb) :        
#         norm_emb = self.layer_norm1(enc_emb)
#         attn_out, _ = self.mha(norm_emb, norm_emb, norm_emb,
#                                attn_mask=self.attn_mask)
#         attn_out = attn_out + enc_emb
         
#         norm_attn_out = self.layer_norm2(attn_out)
#         lin_out = self.ffn(norm_attn_out)
#         enc_out = lin_out + attn_out
        
        attn_out, _ = self.mha(enc_emb, enc_emb, enc_emb,
                               attn_mask=self.attn_mask)
        attn_out = self.layer_norm1(attn_out + enc_emb)
        
        lin_out = self.ffn(attn_out)
        enc_out = self.layer_norm2(lin_out + attn_out)
         
        return enc_out
    
    
class DecoderEmbedding(nn.Module) :
    def __init__(self, emb_dim, n_times, max_seq, device) :
        super(DecoderEmbedding, self).__init__()
        self.max_seq = max_seq
        self.device = device
        self.time_embed = nn.Embedding(n_times+1, emb_dim, padding_idx=0)
        # padding=0, wrong=1, correct=2, start=3
        self.label_embed = nn.Embedding(4, emb_dim, padding_idx=0)
        self.position_embed = nn.Embedding(max_seq, emb_dim)
        
        # for embed in [self.time_embed, self.label_embed, self.position_embed] :
        #     nn.init.xavier_uniform_(embed.weight)
        #     embed._fill_padding_idx_with_zero()
        
    def forward(self, label, time_idx) :
        label_emb = self.label_embed(label)
        time_emb = self.time_embed(time_idx)
        pos = torch.arange(self.max_seq, device=self.device).unsqueeze(0)
        pos_emb = self.position_embed(pos)
        dec_emb = (label_emb + time_emb + pos_emb) / 3
        
        return dec_emb
    
    
class DecoderBlock(nn.Module) :
    def __init__(self, n_heads, emb_dim, drop_out, max_seq, device) :
        super(DecoderBlock, self).__init__()
        self.n_stacks = n_stacks
        self.max_seq = max_seq
        self.device = device
        self.attn_mask = torch.triu(torch.full((max_seq, max_seq), float('-inf'), device=device), diagonal=1)
        # self.attn_mask = torch.triu(torch.ones((max_seq, max_seq), device=device), diagonal=1).bool()
        
        self.mha1 = nn.MultiheadAttention(emb_dim, n_heads, batch_first=True)
        self.mha2 = nn.MultiheadAttention(emb_dim, n_heads, batch_first=True)
                 
        self.ffn = FFN(emb_dim, drop_out)
        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(emb_dim)
        self.layer_norm3 = nn.LayerNorm(emb_dim)
        self.layer_norm4 = nn.LayerNorm(emb_dim)
    
    def forward(self, dec_emb, enc_out) :        
#         norm_emb = self.layer_norm1(dec_emb)
#         attn_out1, _ = self.mha1(norm_emb, norm_emb, norm_emb,
#                             attn_mask=self.attn_mask)
#         attn_out1 = attn_out1 + dec_emb
         
#         norm_attn_out1 = self.layer_norm2(attn_out1)
#         norm_enc_out = self.layer_norm3(enc_out)
#         attn_out2, _ = self.mha2(norm_attn_out1, norm_enc_out, norm_enc_out)
#         attn_out2 = attn_out2 + attn_out1
        
#         norm_attn_out2 = self.layer_norm4(attn_out2)
#         lin_out = self.ffn(norm_attn_out2)
#         dec_out = lin_out + attn_out2
        
        attn_out1, _ = self.mha1(dec_emb, dec_emb, dec_emb,
                            attn_mask=self.attn_mask)
        attn_out1 = self.layer_norm1(attn_out1 + dec_emb)
         
        enc_out = self.layer_norm2(enc_out)
        attn_out2, _ = self.mha2(attn_out1, enc_out, enc_out)
        attn_out2 = self.layer_norm3(attn_out2 + attn_out1)
        
        lin_out = self.ffn(attn_out2)
        dec_out = self.layer_norm4(lin_out + attn_out2)
        
        return dec_out
    
    
class SaintPlus(nn.Module) :
    def __init__(self,
                 n_stacks,
                 n_enc_heads,
                 n_dec_heads,
                 emb_dim,
                 drop_out,
                 n_items, n_tests, n_tags, n_times,
                 max_seq,
                 device) :
        super(SaintPlus, self).__init__()
        self.enc_emb = EncoderEmbedding(emb_dim, n_items, n_tests, n_tags, max_seq, device)
        self.enc_block = EncoderBlock(n_enc_heads, emb_dim, drop_out, max_seq, device)
        self.dec_emb = DecoderEmbedding(emb_dim, n_times, max_seq, device)
        self.dec_block = DecoderBlock(n_dec_heads, emb_dim, drop_out, max_seq, device)
        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(max_seq)
        self.predict = nn.Linear(emb_dim, 1)
    
    
    def forward(self, input_features, label, valid=False) :
        item_idx = input_features['item'].to(device)
        test_idx = input_features['test'].to(device)
        tag_idx = input_features['tag'].to(device)
        time_idx = input_features['time'].to(device)
        label = label.to(device)
        
        enc_emb = self.enc_emb(item_idx, test_idx, tag_idx)
        enc_out = self.enc_block(enc_emb)
        dec_emb = self.dec_emb(label, time_idx)
        dec_out = self.dec_block(dec_emb, enc_out)
        dec_out = self.layer_norm1(dec_out)
        preds = torch.sigmoid(self.layer_norm2(self.predict(dec_out).squeeze(-1))).squeeze(-1)

        return preds
    
    
def train(model, loader, optim, loss_func, device) :
    model.train()
    
    total_loss = 0
    total_labels = list()
    total_preds = list()
    for batch in loader :
        input_features, label, answer_seq, pad = batch
        pad = pad.to(device)
        answer_seq = answer_seq.float().to(device)
        answer_seq = torch.masked_select(answer_seq, pad)
        
        preds = model.forward(input_features, label)
        preds = torch.masked_select(preds, pad)
        # print(f"preds : {preds.shape}")
        # print(preds)
        # print(torch.unique(preds))
        # exit()
        loss = loss_func(answer_seq, preds)
        
        total_loss += loss
        total_labels.extend(answer_seq.cpu().detach().numpy())
        total_preds.extend(preds.cpu().detach().numpy())
         
        optim.zero_grad()
        loss.backward()
        optim.step()
        
    loss = total_loss / len(loader)
    auc = roc_auc_score(total_labels, total_preds)
        
    return loss, auc
    

def validate(model, loader, optim, loss_func, device) :
    model.eval()
    
    total_loss = 0
    total_labels = list()
    total_preds = list()
    with torch.no_grad() :
        for batch in loader :
            input_features, label, answer_seq, pad = batch
            label = label.to(device)
            pad = pad.to(device)
            answer_seq = answer_seq.float().to(device)
            answer_seq = torch.masked_select(answer_seq, pad)

            preds = model.forward(input_features, label, valid=True)
            preds = torch.masked_select(preds, pad)
            # print(f"preds : {preds.shape}")
            # print(preds)
            # print(torch.unique(preds))
            # exit()
            loss = loss_func(answer_seq, preds)
            

            total_loss += loss
            total_labels.extend(answer_seq.cpu().detach().numpy())
            total_preds.extend(preds.cpu().detach().numpy())
            
        loss = total_loss / len(loader)
        auc = roc_auc_score(total_labels, total_preds)
    
    return loss, auc
            

def inference() :
    pass


def run(loaders, model, n_epochs, lr, model_dir, device) :
    model.to(device)
    timestamp = get_timestamp()
    optim = torch.optim.Adam(params=model.parameters(), lr=lr)
    loss_func = nn.BCEWithLogitsLoss()
    
    best_auc, best_epoch = 0, -1
    for epoch in tqdm(range(n_epochs)) :
        print(f"Epoch : {epoch}")
        train_loss, train_auc = train(model, loaders['train'], optim, loss_func, device)
        valid_loss, valid_auc = validate(model, loaders['valid'], optim, loss_func, device)
        
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train AUC : {train_auc:.4f}")
        print(f"Valid Loss : {valid_loss:.4f}")
        print(f"Valid AUC : {valid_auc:.4f}")
        
        if best_auc < valid_auc :
            print(f"Auc Updated {best_auc:.4f} -> {valid_auc:.4f}")
            best_auc, best_epoch = valid_auc, epoch
            # Save best model
            torch.save(obj= {"model": model.state_dict(), "epoch": epoch},
                       f=os.path.join(model_dir, f"best_model_{timestamp}.pt"))
    # Save last model
    torch.save(obj={"model": model.state_dict(), "epoch": epoch + 1},
               f=os.path.join(model_dir, f"last_model_{timestamp}.pt"))
    print(f"Best Weight Confirmed : {best_epoch}'th epoch")
    print(f"Best AUC Score : {best_auc:.4f}")
    
    exit()
    inference(model, loaders['test'], device, timestamp, model_dir, output_dir)
            
        

if __name__ == "__main__" :
    set_seeds(327)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    raw_train_df, raw_test_df = load_data("../../data")
    print(raw_train_df.shape)
    print(raw_test_df.shape)
    
    processed_train_df = common_process(raw_train_df)
    processed_test_df = common_process(raw_test_df)
    
    n_items = processed_train_df['item_id'].nunique()
    n_tests = processed_train_df['test_id'].nunique()
    n_tags = processed_train_df['tag_id'].nunique()
    n_times = processed_train_df['time_bin'].nunique()
    
    train_users, valid_users = train_test_split(processed_train_df['user_id'].unique(),
                                            test_size=0.1, shuffle=True, random_state=42)
    
    cat_features = ['item_id', 'test_id', 'tag_id', 'time_bin']
    train_group, valid_group, test_group = get_train_valid_test(processed_train_df,
                                                                processed_test_df,
                                                                cat_features,
                                                                train_users)
    n_stacks = 2
    n_enc_heads = 8
    n_dec_heads = 8
    emb_dim = 128
    drop_out = 0.2
    max_seq = 500
    n_epochs = 50
    lr = 0.005
    batch_size = 16
    model_dir = "./models"
    print("Parameters")
    print(n_stacks, n_enc_heads, n_dec_heads, emb_dim, drop_out, max_seq, n_epochs, lr, batch_size)
    
    train_dataset = SaintDataset(train_group, max_seq)
    valid_dataset = SaintDataset(valid_group, max_seq, train=False)
    test_dataset = SaintDataset(test_group, max_seq, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)   
    
    print('Data Processing Done')
    model = SaintPlus(n_stacks,
                      n_enc_heads,
                      n_dec_heads,
                      emb_dim,
                      drop_out,
                      n_items, n_tests, n_tags, n_times,
                      max_seq,
                      device).to(device)
    
    loaders = {'train' : train_loader,
               'valid' : valid_loader,
               'test' : test_loader}
    
    run(loaders, model, n_epochs, lr, model_dir, device)