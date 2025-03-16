import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred)/y_true)) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def plot_cv_results(cv_results):
    plt.figure(figsize=(10, 6))
    
    folds = range(1, len(cv_results) + 1)
    train_losses = [r['train_loss'] for r in cv_results]
    val_losses = [r['val_loss'] for r in cv_results]
    
    plt.plot(folds, train_losses, 'o-', label='Training Loss')
    plt.plot(folds, val_losses, 'o-', label='Validation Loss')
    
    plt.axhline(y=np.mean(val_losses), color='r', linestyle='--', 
                label=f'Avg Val Loss: {np.mean(val_losses):.6f}')
    
    plt.title('Cross-Validation Results')
    plt.xlabel('Fold')
    plt.ylabel('Loss')
    plt.xticks(folds)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_predictions(actual, raw, filtered=None):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual Price', linewidth=2)
    plt.plot(raw, label='LSTM Prediction', linestyle='--')
    
    if filtered is not None:
        plt.plot(filtered, label='Kalman Filtered', linewidth=1.5)
    
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_comparison(actual, raw, standard_filtered, improved_filtered):
    plt.figure(figsize=(14, 8))
    
    plt.plot(actual, label='actual price', linewidth=2, color='black')
    plt.plot(raw, label='LSTM Original Prediction', linestyle='--', color='blue', alpha=0.7)
    
    plt.plot(standard_filtered, label='Standard Kalman Filter', 
             linestyle='-', linewidth=1.5, color='green')
    
    plt.plot(improved_filtered, label='Improved Kalman Filter(reduce lag)', 
             linestyle='-', linewidth=1.5, color='red')
    
    length = len(actual)
    zoom_start = int(length * 0.6)
    zoom_end = int(length * 0.8)
    
    plt.axvspan(zoom_start, zoom_end, alpha=0.2, color='gray')
    
    plt.title('Stock Price Prediction - Kalman Filter Improvement Comparison')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.figure(figsize=(14, 6))
    plt.plot(actual[zoom_start:zoom_end], label='actual price', linewidth=2, color='black')
    plt.plot(raw[zoom_start:zoom_end], label='LSTM Original Prediction', 
             linestyle='--', color='blue', alpha=0.7)
    plt.plot(standard_filtered[zoom_start:zoom_end], label='Standard Kalman Filter', 
             linestyle='-', linewidth=1.5, color='green')
    plt.plot(improved_filtered[zoom_start:zoom_end], label='Improved Kalman Filter', 
             linestyle='-', linewidth=1.5, color='red')
    
    plt.title('Zoomed Area - Lag Comparison')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_lag(actual, predictions):
    corr = np.correlate(actual.flatten(), predictions, mode='full')
    lags = np.arange(-len(actual)+1, len(predictions))
    lag = lags[np.argmax(corr)]
    
    print(f"Detected lag: {lag} time steps")
    
    if lag != 0:
        if lag > 0:
            aligned_actual = actual[lag:]
            aligned_pred = predictions[:len(predictions)-lag]
        else:
            aligned_actual = actual[:len(actual)+lag]
            aligned_pred = predictions[-lag:]
        
        aligned_corr = np.corrcoef(aligned_actual.flatten(), aligned_pred)[0, 1]
        print(f"Aligned correlation: {aligned_corr:.4f}")
    
    return lag 