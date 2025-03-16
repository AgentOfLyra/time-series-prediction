# LSTM + Kalman Filter for Stock Price Prediction

This project implements a stock price prediction system using LSTM (Long Short-Term Memory) neural networks combined with Kalman filters to improve prediction accuracy and reduce lag issues.

## Project Structure

```
├── data/                   # Directory for stock price data
├── models/                 # Model definitions
│   └── LSTM.py             # LSTM and Kalman filter implementations
├── test/                   # Testing code
│   └── test.py             # Model evaluation functions
├── train/                  # Training code
│   └── train.py            # Model training functions
├── predictions/            # Directory for prediction results
├── Best_Model/             # Directory for saved models
├── main.py                 # Main entry point for training and evaluation
├── Predict_from_model.py   # Script for making predictions using trained models
├── StockData.py            # Utilities for fetching and processing stock data
└── utils.py                # Utility functions for metrics and visualization
```

## Features

- **LSTM Model**: Deep learning model for time series prediction
- **Kalman Filter**: Standard and improved implementations for noise reduction
- **Cross-Validation**: Time series cross-validation for robust model training
- **Visualization**: Comprehensive plotting functions for model evaluation
- **Stock Data Fetching**: Utilities for retrieving historical stock data
- **Future Prediction**: Capability to predict future stock prices

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- yfinance (for fetching stock data)

## Usage

### Install all required libraries
```bash
pip install -r requirements.txt
```

### Fetching Stock Data

```bash
python StockData.py --ticker GOOG --period 1y --interval 1d
```

### Training a Model

```bash
python main.py --data_path ./data/GOOG_1d_20250316.csv --window_size 60 --test_ratio 0.2 --n_splits 5
```

### Making Predictions

```bash
python Predict_from_model.py --model_path ./Best_Model/GOOG_1d_20250316_model.pth --data_path ./data/GOOG_1d_20250316.csv --future_days 30
```

## Model Architecture

The project combines LSTM neural networks with Kalman filters:

1. **LSTM**: Captures temporal patterns in stock price data
2. **Kalman Filter**: Reduces noise and improves prediction stability
3. **Improved Kalman Filter**: Addresses lag issues in standard Kalman filters

## Evaluation Metrics

The model performance is evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

## Visualization

The project includes visualization tools for:
- Cross-validation results
- Prediction comparisons
- Lag analysis
- Future price predictions

## License

This project is for educational and research purposes only. It is not financial advice.

## Acknowledgements

This project was developed as part of research into time series prediction methods combining deep learning and statistical filtering techniques. 