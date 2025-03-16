import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
import sys
import os

# 添加父目录到路径，以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.LSTM import LSTMStockPredictor, StockDataset
from utils import plot_cv_results

def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            if isinstance(inputs, tuple) and len(inputs) == 2:
                price_inputs, text_inputs = inputs
                price_inputs = price_inputs.to(device)
                text_inputs = text_inputs.to(device)
                inputs = (price_inputs, text_inputs)
            else:
                inputs = inputs.to(device)
                
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.6f}')

# 使用时间序列交叉验证进行训练
def train_with_cross_validation(x_train_val, y_train_val, model_params, train_params, n_splits=5, save_path=None):
    from sklearn.model_selection import TimeSeriesSplit
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_results = []
    best_val_loss = float('inf')
    best_model = None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    fold = 1
    for train_idx, val_idx in tscv.split(x_train_val):
        print(f"\nStarting fold {fold}/{n_splits}")
        fold += 1
        
        x_train_fold, x_val_fold = x_train_val[train_idx], x_train_val[val_idx]
        y_train_fold, y_val_fold = y_train_val[train_idx], y_train_val[val_idx]
        
        train_dataset = StockDataset(x_train_fold, y_train_fold)
        val_dataset = StockDataset(x_val_fold, y_val_fold)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=train_params['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=train_params['batch_size'], 
            shuffle=False
        )
        
        model = LSTMStockPredictor(
            input_size=model_params['input_size'],
            hidden_size=model_params['hidden_size'],
            num_layers=model_params['num_layers'],
            output_size=model_params['output_size']
        ).to(device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=train_params['learning_rate'],
            weight_decay=train_params.get('weight_decay', 0)
        )
        
        model.train()
        best_fold_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(train_params['epochs']):
            model.train()
            train_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{train_params["epochs"]}], '
                      f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            if val_loss < best_fold_val_loss:
                best_fold_val_loss = val_loss
                patience_counter = 0
                fold_best_model = copy.deepcopy(model)
            else:
                patience_counter += 1
                if patience_counter >= train_params.get('patience', 20):
                    print(f"Early stopping: no improvement for {patience_counter} epochs")
                    break
        
        fold_result = {
            'train_loss': train_loss,
            'val_loss': best_fold_val_loss
        }
        cv_results.append(fold_result)
        
        if best_fold_val_loss < best_val_loss:
            best_val_loss = best_fold_val_loss
            best_model = copy.deepcopy(fold_best_model)
            print(f"Updated best model, validation loss: {best_val_loss:.6f}")
    
    print("\nCross-validation results:")
    val_losses = [result['val_loss'] for result in cv_results]
    print(f"Average validation loss: {np.mean(val_losses):.6f} ± {np.std(val_losses):.6f}")

    if save_path:
        import os
        os.makedirs('../Best_Model', exist_ok=True)

        save_dict = {
            'model_state_dict': best_model.state_dict(),
            'model_params': model_params,
            'cv_results': cv_results,
            'best_val_loss': best_val_loss,
            'train_params': train_params,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        torch.save(save_dict, save_path)
        print(f"Model saved to {save_path}")
    
    return best_model, cv_results 