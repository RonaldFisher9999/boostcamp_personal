import os
import torch
import torch.nn as nn
from torch.nn import MSELoss
from tqdm import tqdm


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.eps = 1e-6
        
    def forward(self, x, y):
        criterion = MSELoss()
        loss = torch.sqrt(criterion(x, y) + self.eps)
        
        return loss
    

def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x = x.to(model.device)
        y = y.to(model.device)
        
        pred = model.forward(x)
        loss = criterion(y, pred)
        total_loss += loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return total_loss/len(train_loader)


def validate(model, valid_loader, criterion):
    model.eval()
    total_loss = 0
    for x, y in valid_loader:
        x = x.to(model.device)
        y = y.to(model.device)
        
        with torch.no_grad():
            pred = model.forward(x)
        loss = criterion(y, pred)
        total_loss += loss

    return total_loss/len(valid_loader)


def run(model,
        train_loader,
        valid_loader,
        lr,
        n_epochs,
        max_patience,
        model_dir,
        timestamp):
    criterion = RMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss, best_epoch = int(1e9), -1
    patience = 0
    state_dict = dict()
    for epoch in tqdm(range(1, n_epochs+1)): 
        print(f"Epoch: {epoch}")
        print("Train")
        train_loss = train(model, train_loader, criterion, optimizer)
        print(f"Loss: {train_loss:.4f}")
        print("Validate")
        valid_loss = validate(model, valid_loader, criterion)
        print(f"{valid_loss:.4f}")
        
        if best_loss > valid_loss :
            print(f"{best_loss:.4f} -> {valid_loss:.4f}")
            best_loss = valid_loss
            best_epoch = epoch
            state_dict = model.state_dict()
            torch.save(obj={"model": state_dict, "loss": best_loss, "epoch": best_epoch},
               f=os.path.join(model_dir, f"model_{timestamp}.pt"))
            patience = 0
        else :
            patience += 1
            print(f"Current Best Loss: {best_loss:.4f}")
            print(f"Current Best Epoch: {best_epoch}")
            print(f"Patience Count: {patience}/{max_patience}")
            if patience == max_patience:
                print(f"No Score Improvement for {max_patience} epochs")
                print("Early Stopped Training")
                break
            
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Best Loss Confirmed: {best_epoch}'th epoch")
    
    return best_loss, best_epoch


def inference(model, test_loader, sub_df, model_dir, timestamp, output_dir):
    model.eval()
    state_dict = torch.load(os.path.join(model_dir, f'model_{timestamp}.pt'))['model']
    model.load_state_dict(state_dict)
    
    infer = torch.tensor([]).to(model.device)
    for x in test_loader:
        x = x[0].to(model.device)
        with torch.no_grad():
            pred = model.forward(x)
        infer = torch.concat([infer, pred])
        
    sub_df['rating'] = infer.detach().cpu().numpy()
    sub_df.to_csv(os.path.join(output_dir, f"sub_{timestamp}.csv"), index=False)