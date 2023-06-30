import os
import json
import torch


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
    
    if CONFIG['device'] == 'cuda' and torch.cuda.is_available() == True :
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        CONFIG['device'] = 'cpu'
        
    loader, field_dims, sub_df = prepare_data(CONFIG['data_dir'],
                                              CONFIG['processed'],
                                              CONFIG['valid_size'],
                                              CONFIG['batch_size'],
                                              CONFIG['save'])
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
    
    best_loss, best_epoch = run(model,
                                loader['train_loader'],
                                loader['valid_loader'],
                                CONFIG['lr'],
                                CONFIG['n_epochs'],
                                CONFIG['max_patience'],
                                CONFIG['model_dir'],
                                timestamp)
    
    inference(model,
              loader['test_loader'],
              sub_df,
              CONFIG['model_dir'],
              CONFIG['output_dir'],
              timestamp)


if __name__ == "__main__":
    main()