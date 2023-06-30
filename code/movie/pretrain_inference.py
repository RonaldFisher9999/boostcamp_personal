import os
import json
import torch
from utils import get_timestamp, set_seeds
from dataloader import load_data, process_data
from model import BERT4Rec
from trainer import inference


def main():
    print("Load Pretrained Model Configuration.")
    with open(os.path.join(os.curdir, 'pretrain.json'), 'r') as f:
        PRETRAIN = json.load(f)
    for pre_config, value in PRETRAIN.items():
        print(f"{pre_config}: {value}")
        
    with open(os.path.join(os.curdir, PRETRAIN['model_dir'], f"{PRETRAIN['model_name']}.json"), 'r') as f:
        result_config = json.load(f)
        RESULT = result_config['result']
        CONFIG = result_config['config']
    for result, value in RESULT.items():
        print(f"{result}: {value}")
    for config, value in CONFIG.items():
        print(f"{config}: {value}")
    
    data_dir = PRETRAIN['data_dir']
    model_dir = PRETRAIN['model_dir']
    output_dir = PRETRAIN['output_dir']
    os.makedirs(name=model_dir, exist_ok=True)
    os.makedirs(name=output_dir, exist_ok=True)
    set_seeds(CONFIG['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    timestamp = get_timestamp()

    print("Load and Process Data.")
    train_df, sub_df = load_data(data_dir)
    data, n_items, n_users, idx2item = process_data(train_df,
                                                    CONFIG['max_len'],
                                                    CONFIG['k'],
                                                    CONFIG['n_samples'],
                                                    CONFIG['tail_ratio'])
    
    print("Create BERT4Rec Model.")
    model = BERT4Rec(n_items,
                     CONFIG['embed_dim'],
                     CONFIG['max_len'],
                     CONFIG['n_layers'],
                     CONFIG['n_heads'],
                     CONFIG['pffn_hidden_dim'],
                     CONFIG['unidirection'],
                     CONFIG['dropout_rate'],
                     device=device).to(device)
    
    print("Make Inference.")
    inference(model,
              data['infer'],
              data['infer_cand'],
              PRETRAIN['infer_k'],
              sub_df,
              idx2item,
              model_dir,
              output_dir,
              timestamp,
              PRETRAIN['model_name'])
    
    torch.save(obj={"model": model.state_dict(), "epoch": RESULT['best_epoch']},
               f=os.path.join(model_dir, f"bert4rec_{timestamp}.pt"))
    with open(f"{model_dir}/bert4rec_{timestamp}.json", "w") as f:
        json.dump(result_config, f)
    
if __name__ == "__main__" :
    main()