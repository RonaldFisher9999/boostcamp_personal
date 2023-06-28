import os
import torch


from utils import get_timestamp, set_seeds
from dataloader import process_data
from model import FM, FFM
from trainer import run, inference

def main():
    pass


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    data_dir = "../../data/book/"
    model_dir = "./model/"
    output_dir = "./output/"
    seed = 327
    model = "FM"
    valid_size = 0.2
    batch_size = 512
    embed_dim = 128
    lr = 0.001
    n_epochs = 50
    max_patience = 10
    timestamp = get_timestamp()
    
    os.makedirs(name=model_dir, exist_ok=True)
    os.makedirs(name=output_dir, exist_ok=True)
    set_seeds(seed)

    data = process_data(data_dir, valid_size, batch_size)
    print(data.keys())
    
    model = FFM(data['field_dims'], embed_dim, device).to(device)
    print(model)
    
    best_loss, best_epoch = run(model, data['train_loader'], data['valid_loader'],
                                lr, n_epochs, max_patience, model_dir, timestamp)
    
    inference(model, data['test_loader'], data['sub'], model_dir, timestamp, output_dir)