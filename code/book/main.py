import os
import json
import torch
import wandb

from utils import get_timestamp, set_seeds
from dataloader import prepare_data
from model import FM, FFM, DeepFM
from trainer import run, inference

def main():
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
    loader, field_dims, sub_df = prepare_data(CONFIG['data_dir'],
                                              CONFIG['processed'],
                                              CONFIG['valid_size'],
                                              CONFIG['batch_size'],
                                              CONFIG['save'])
    
    print(f"Create {CONFIG['model_name']} model.")
    model_name = CONFIG['model_name']
    assert model_name in ["FM", "FFM", "DeepFM"], "'model_name' must be 'FM', 'FFM', or 'DeepFM'."
    
    if model_name == "FM":
        model = FM(field_dims,
                   CONFIG['embed_dim'],
                   device)
    elif model_name == "FFM":
        model = FFM(field_dims,
                    CONFIG['embed_dim'],
                    device)
    else:    
        model = DeepFM(field_dims,
                    CONFIG['embed_dim'],
                    CONFIG['mlp_dims'],
                    CONFIG['dropout_rate'],
                    CONFIG['use_bn'],
                    device).to(device)
    
    print(f"Start Training for {CONFIG['n_epochs']} Epochs.")
    best_loss, best_epoch = run(model,
                                loader['train_loader'],
                                loader['valid_loader'],
                                CONFIG['lr'],
                                CONFIG['n_epochs'],
                                CONFIG['max_patience'],
                                CONFIG['model_dir'],
                                CONFIG['logging'],
                                timestamp)
    
    for config, value in CONFIG.items():
        print(f"{config}: {value}")
    result = {'best_loss': best_loss, 'best_epoch': best_epoch}
    result_config = {'result': result, 'config': CONFIG}
    with open(f"{CONFIG['model_dir']}/{CONFIG['model_name']}_{timestamp}.json", "w") as f:
        json.dump(result_config, f)
    
    print("Make Inference.")
    inference(model,
              loader['test_loader'],
              sub_df,
              CONFIG['model_dir'],
              CONFIG['output_dir'],
              timestamp)
    
    wandb.finish()


if __name__ == "__main__":
    main()