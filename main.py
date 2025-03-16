import numpy as np
import pandas as pd
import torch
import os
import sys
import argparse
from datetime import datetime

from models.LSTM import StockDataPreprocessor, LSTMStockPredictor, EnhancedStockPredictor
from train.train import train_with_cross_validation
from test.test import compare_kalman_filters
from utils import plot_cv_results

def main(data_path, text_data_path=None, window_size=60, test_ratio=0.2, n_splits=5, save_path=None):
    try:
        preprocessor = StockDataPreprocessor(window_size)
        (x_train_val, y_train_val), (x_test, y_test) = preprocessor.prepare_data(
            data_path, test_ratio=test_ratio, text_data_path=text_data_path)
        
        print(f"training and validation data shape: {x_train_val.shape}, {y_train_val.shape}")
        print(f"test data shape: {x_test.shape}, {y_test.shape}")
        
        model_params = {
            'input_size': 1,
            'hidden_size': 64,
            'num_layers': 2,
            'output_size': 1,
            'text_feature_size': 0
        }
        
        train_params = {
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 500,
            'patience': 15,
            'weight_decay': 1e-5
        }
        
        best_model, cv_results = train_with_cross_validation(
            x_train_val, y_train_val, model_params, train_params, n_splits=n_splits, save_path=save_path
        )
        
        # plot the cross-validation results
        plot_cv_results(cv_results)
        
        print("\nevaluate the best model on the test set:")
        
        # compare the standard Kalman filter and the improved Kalman filter
        metrics_standard, metrics_improved = compare_kalman_filters(
            best_model, x_test, y_test, preprocessor.scaler
        )
        
        return best_model, preprocessor, metrics_improved
        
    except Exception as e:
        print(f"error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM and Kalman filter for stock prediction')
    parser.add_argument('--data_path', type=str, default='./data/GOOG_1d_20250316.csv', help='data file path')
    parser.add_argument('--window_size', type=int, default=60, help='window size')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='test set ratio')
    parser.add_argument('--n_splits', type=int, default=5, help='number of cross-validation folds')
    parser.add_argument('--save_path', type=str, default=None, help='model save path')
    
    args = parser.parse_args()
    
    if args.save_path is None:
        os.makedirs('./Best_Model', exist_ok=True)
        args.save_path = './Best_Model/{}_model.pth'.format(args.data_path.split('/')[-1].split('.')[0])
    
    best_model, preprocessor, metrics = main(
        args.data_path, 
        window_size=args.window_size,
        test_ratio=args.test_ratio,
        n_splits=args.n_splits,
        save_path=args.save_path
    ) 