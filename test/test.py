import numpy as np
import torch
import sys
import os

# add the parent directory to the path, so that other modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.LSTM import EnhancedStockPredictor
from utils import calculate_metrics, plot_comparison

def evaluate_model(model, test_data, scaler, use_kalman=True, kf_params=None):
    x_test, y_test = test_data
    device = next(model.parameters()).device
    
    if use_kalman:
        predictor = EnhancedStockPredictor(model, scaler, kf_params)
    
    raw_predictions = []
    filtered_predictions = []
    
    for i in range(len(x_test)):
        current_window = x_test[i].reshape(-1, 1)
        
        if use_kalman:
            raw, filtered = predictor.predict_with_filter(current_window)
            raw_predictions.append(raw)
            filtered_predictions.append(filtered)
        else:
            with torch.no_grad():
                inputs = torch.FloatTensor(current_window).unsqueeze(0).to(device)
                pred = model(inputs).cpu().item()
                raw_price = scaler.inverse_transform([[pred]])[0][0]
                raw_predictions.append(raw_price)
    
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    metrics = {
        'raw': calculate_metrics(actual_prices, np.array(raw_predictions))
    }
    
    if use_kalman:
        metrics['filtered'] = calculate_metrics(actual_prices, np.array(filtered_predictions))
    
    return metrics, (actual_prices, raw_predictions, filtered_predictions if use_kalman else None)

def test_with_improved_kalman(model, x_test, y_test, scaler, improved_kf_params):
    """use the improved Kalman filter to test the model"""
    improved_predictor = EnhancedStockPredictor(
        model, scaler, improved_kf_params, use_improved_kf=True
    )
    
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
    raw_predictions = []
    improved_filtered_predictions = []
    
    for i in range(len(x_test)):
        current_window = x_test[i].reshape(-1, 1)
        raw, filtered = improved_predictor.predict_with_filter(
            current_window, look_ahead=2
        )
        raw_predictions.append(raw)
        improved_filtered_predictions.append(filtered)
    
    metrics_improved = {
        'raw': calculate_metrics(actual_prices, np.array(raw_predictions)),
        'filtered': calculate_metrics(actual_prices, np.array(improved_filtered_predictions))
    }
    
    return metrics_improved, (actual_prices, raw_predictions, improved_filtered_predictions)

def compare_kalman_filters(model, x_test, y_test, scaler):
    """compare the standard Kalman filter and the improved Kalman filter"""
    standard_kf_params = {'process_noise': 1e-5, 'measurement_noise': 1e-2}
    metrics_standard, predictions_standard = evaluate_model(
        model, (x_test, y_test), scaler, 
        use_kalman=True, kf_params=standard_kf_params
    )
    
    improved_kf_params = {
        'process_noise': 1e-4,
        'measurement_noise': 5e-2,
        'responsiveness': 0.9,
        'trend_weight': 0.4
    }
    
    metrics_improved, predictions_improved = test_with_improved_kalman(
        model, x_test, y_test, scaler, improved_kf_params
    )
    
    print("\nstandard Kalman filter metrics:")
    for k, v in metrics_standard['filtered'].items():
        print(f"{k}: {v:.4f}")
    
    print("\nimproved Kalman filter metrics:")
    for k, v in metrics_improved['filtered'].items():
        print(f"{k}: {v:.4f}")
    
    plot_comparison(
        predictions_improved[0],  # actual_prices 
        predictions_improved[1],  # raw_predictions
        predictions_standard[2],  # standard_filtered_predictions
        predictions_improved[2]   # improved_filtered_predictions
    )
    
    return metrics_standard, metrics_improved 