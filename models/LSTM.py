import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any
import copy

# Stock data preprocessor class - with extension interface for text features
class StockDataPreprocessor:
    def __init__(self, window_size=60):
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.window_size = window_size
        self.text_features = None

    def prepare_data(self, data_path, target_column='Close', test_ratio=0.2, 
                     text_data_path=None):
        dataset = pd.read_csv(data_path)
        prices = dataset[[target_column]].values
        
        scaled_data = self.scaler.fit_transform(prices)
        
        x, y = [], []
        for i in range(self.window_size, len(scaled_data)):
            x.append(scaled_data[i-self.window_size:i, 0])
            y.append(scaled_data[i, 0])
        
        x = np.array(x)
        y = np.array(y)
        
        test_size = int(len(x) * test_ratio)
        x_train_val = x[:-test_size]
        y_train_val = y[:-test_size]
        x_test = x[-test_size:]
        y_test = y[-test_size:]
        
        if text_data_path:
            self.text_features = self._process_text_data(text_data_path, dataset)
        
        return (x_train_val, y_train_val), (x_test, y_test)
    
    def _process_text_data(self, text_data_path, price_data):
        return None

class StockDataset(Dataset):
    def __init__(self, features, targets, text_features=None):
        self.features = features
        self.targets = targets
        self.text_features = text_features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature_tensor = torch.FloatTensor(self.features[idx]).unsqueeze(-1)
        target_tensor = torch.FloatTensor([self.targets[idx]])
        
        if self.text_features is not None:
            text_tensor = torch.FloatTensor(self.text_features[idx])
            return (feature_tensor, text_tensor), target_tensor
        
        return feature_tensor, target_tensor

# LSTM model for stock prediction - supports text feature input
class LSTMStockPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1,
                 text_feature_size=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.has_text_features = text_feature_size > 0
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        if self.has_text_features:
            self.text_encoder = nn.Sequential(
                nn.Linear(text_feature_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            self.fc = nn.Linear(hidden_size * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, text_features=None):
        if isinstance(x, tuple) and len(x) == 2:
            x, text_features = x
            
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        price_features = lstm_out[:, -1, :]
        
        if self.has_text_features and text_features is not None:
            text_encoded = self.text_encoder(text_features)
            combined = torch.cat((price_features, text_encoded), dim=1)
            return self.fc(combined)
        
        return self.fc(price_features)

# Kalman filter implementation - supports adjusting noise parameters through text analysis
class KalmanFilter:
    def __init__(self, process_noise=1e-4, measurement_noise=1e-1):
        self.F = np.array([[1, 1],
                          [0, 1]])
        self.H = np.array([[1, 0]])
        self.Q = np.eye(2) * process_noise
        self.base_process_noise = process_noise
        self.R = np.array([[measurement_noise]])
        self.base_measurement_noise = measurement_noise
        self.x = np.zeros((2, 1))
        self.P = np.eye(2)
        self.measurement_history = []

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0][0]

    def update(self, z, text_sentiment=None):
        self.measurement_history.append(z)
        if len(self.measurement_history) > 20:
            self.measurement_history.pop(0)
            
        if text_sentiment is not None:
            self._adjust_noise_from_text(text_sentiment)
        
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x[0][0]
    
    def _adjust_noise_from_text(self, text_info):
        pass

# Improved Kalman filter - reducing lag issues
class ImprovedKalmanFilter(KalmanFilter):
    def __init__(self, process_noise=1e-4, measurement_noise=1e-1, 
                 responsiveness=0.8, trend_weight=0.3):
        super().__init__(process_noise, measurement_noise)
        self.responsiveness = responsiveness
        self.trend_weight = trend_weight
        self.prev_measurements = []
        self.max_history = 5
        
    def update(self, z, text_sentiment=None):
        self.prev_measurements.append(z)
        if len(self.prev_measurements) > self.max_history:
            self.prev_measurements.pop(0)
            
        trend = 0
        if len(self.prev_measurements) >= 2:
            x = np.arange(len(self.prev_measurements))
            y = np.array(self.prev_measurements)
            A = np.vstack([x, np.ones(len(x))]).T
            try:
                slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
                trend = slope
            except:
                trend = 0
        
        if len(self.prev_measurements) >= 2:
            recent_change = abs(self.prev_measurements[-1] - self.prev_measurements[-2])
            relative_change = recent_change / (abs(self.prev_measurements[-2]) + 1e-10)
            
            adaptive_process_noise = self.base_process_noise * (1 + 10 * relative_change)
            self.Q = np.eye(2) * min(adaptive_process_noise, 1e-2)
        
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        K = K * self.responsiveness
        
        self.x += K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        
        self.x[1][0] = self.x[1][0] * (1 - self.trend_weight) + trend * self.trend_weight
        
        return self.x[0][0]

# Enhanced stock predictor - integrating LSTM and Kalman filter
class EnhancedStockPredictor:
    def __init__(self, model, scaler, kf_params=None, use_improved_kf=True):
        self.model = model
        self.scaler = scaler
        
        if use_improved_kf:
            self.kf = ImprovedKalmanFilter(**kf_params if kf_params else {})
        else:
            self.kf = KalmanFilter(**kf_params if kf_params else {})
        
    def predict_with_filter(self, inputs, text_features=None, look_ahead=1):
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            inputs_tensor = torch.FloatTensor(inputs).unsqueeze(0).to(device)
            
            current_pred = self.model(inputs_tensor).cpu().numpy()[0][0]
            
            future_inputs = inputs.copy()
            future_preds = []
            
            for _ in range(look_ahead):
                future_inputs = np.roll(future_inputs, -1)
                future_inputs[-1] = current_pred
                
                future_tensor = torch.FloatTensor(future_inputs).unsqueeze(0).to(device)
                next_pred = self.model(future_tensor).cpu().numpy()[0][0]
                future_preds.append(next_pred)
                current_pred = next_pred
        
        raw_price = self.scaler.inverse_transform([[current_pred]])[0][0]
        
        self.kf.predict()
        filtered_price = self.kf.update(raw_price)
        
        if look_ahead > 0 and future_preds:
            future_prices = [self.scaler.inverse_transform([[p]])[0][0] for p in future_preds]
            trend = (future_prices[-1] - filtered_price) / look_ahead
            
            compensated_price = filtered_price + trend * 0.5
            return raw_price, compensated_price
        
        return raw_price, filtered_price
    
    def _extract_sentiment(self, text_features):
        return None
