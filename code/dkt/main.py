import os
import json
import torch
import wandb

from args import parse_args
from utils import set_seeds, get_timestamp
from dataloader import prepare_data
from model import LightGCN
from trainer import run, inference

    
def main() :
    print("Load Configuration File.")
    with open(os.path.join(os.curdir, 'config.json'), 'r') as f:
        CONFIG = json.load(f)
        
    os.makedirs(name=CONFIG['model_dir'], exist_ok=True)
    os.makedirs(name=CONFIG['output_dir'], exist_ok=True)
    assert CONFIG['device'] in ['cuda', 'cpu'], "'device' must be 'cuda' or 'cpu'."
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
                project="dkt",
                name=f"LightGCN-{timestamp}",
                config=CONFIG,
            )
        except:
            print("WandB Login Failed.")
            CONFIG['logging'] = False
            
    for config, value in CONFIG.items():
        print(f"{config}: {value}")

    print("Load and Process Data.")
    data, n_users, n_items = prepare_data(CONFIG['data_dir'],
                                          CONFIG['valid_size'],
                                          device)

    print("Create LightGCN Model.")
    model = LightGCN(n_users,
                     n_items,
                     data['train']['user_group'],
                     CONFIG['embed_dim'],
                     CONFIG['n_layers']).to(device)
    
    print(f"Train For {CONFIG['valid_n_epochs']} Epochs.")
    best_auc, best_epoch = run(model,
                               data,
                               CONFIG['train_n_epochs'],
                               CONFIG['valid_n_epochs'],
                               CONFIG['max_patience'],
                               CONFIG['train_lr'],
                               CONFIG['valid_lr'],
                               CONFIG['use_best'],
                               CONFIG['model_dir'],
                               CONFIG['logging'],
                               timestamp)
    
    for config, value in CONFIG.items():
        print(f"{config}: {value}")
    result = {'best_recall': best_auc, 'best_epoch': best_epoch}
    result_config = {'result': result, 'config': CONFIG}
    with open(f"{CONFIG['model_dir']}/LightGCN_{timestamp}.json", "w") as f:
        json.dump(result_config, f)
    
    print("Make Inference.")
    inference(model,
              data['test']['user_group'],
              data['test']['input_graph'],
              data['test']['target_graph'],
              CONFIG['valid_n_epochs'],
              CONFIG['valid_lr'],
              CONFIG['use_best'],
              CONFIG['model_dir'],
              CONFIG['output_dir'],
              timestamp)
    
    wandb.finish()
    
if __name__ == "__main__" :
    main()