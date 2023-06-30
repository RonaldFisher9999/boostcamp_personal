import os
import json
import torch
import wandb

from args import parse_args
from utils import set_seeds, get_timestamp
from dataloader import prepare_data
from model import LightGCN
from trainer import run

    
def main() :
    print("Load Configuration File.")
    with open(os.path.join(os.curdir, 'config.json'), 'r') as f:
        CONFIG = json.load(f)
        
    os.makedirs(name=CONFIG['model_dir'], exist_ok=True)
    os.makedirs(name=CONFIG['output_dir'], exist_ok=True)
    
    timestamp = get_timestamp()
    set_seeds(CONFIG['seed'])
    
    if CONFIG['device'] == 'cuda':
        if torch.cuda.is_available() == True :
            device = torch.device('cuda')
        else:
            print("GPU Not Available.")
            device = torch.device('cpu')
            CONFIG['device'] = 'cpu'
    
    if CONFIG['logging'] == True:
        print("Start Logging with WandB")
        with open(os.path.join(os.curdir, 'key.txt'), 'r') as f:
            key = f.readline()
        try:
            wandb.login(key=key)
            wandb.init(
                project="book-rec",
                name=f"{CONFIG['model_name']}-{timestamp}",
                config=CONFIG,
            )
        except:
            print("WandB Login Failed.")
            CONFIG['logging'] = False
            
    for config, value in CONFIG.items():
        print(f"{config}: {value}")

    print("Load and Process Data.")
    data = prepare_data(CONFIG['data_dir'],
                        CONFIG['valid_size'],
                        device)

    print("Create LightGCN Model.")
    model = LightGCN(data['n_users'],
                     data['n_items'],
                     data['train_graph']['group'],
                     CONFIG['embed_dim'],
                     CONFIG['n_layers']).to(device)
    
    run(data=data,
        model=model,
        embedding_dim=CONFIG['embed_dim'],
        n_layers=CONFIG['n_layers'],
        train_n_epochs=CONFIG['train_n_epochs'],
        valid_n_epochs=CONFIG['valid_n_epochs'],
        train_lr=CONFIG['train_lr'],
        valid_lr=CONFIG['valid_lr'],
        max_patience=CONFIG['max_patience'],
        valid_size=CONFIG['valid_size'],
        use_best=CONFIG['use_best'],
        model_dir=CONFIG['model_dir'],
        output_dir=CONFIG['output_dir'],
        device=device,
        timestamp=timestamp)
    
    
if __name__ == "__main__" :
    main()