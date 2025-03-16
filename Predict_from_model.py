import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import argparse
from models.LSTM import LSTMStockPredictor, EnhancedStockPredictor, StockDataPreprocessor, ImprovedKalmanFilter

def load_model(filepath):
    """
    加载完整模型/Load the complete model
    """
    model = torch.load(filepath)
    print(f"Model loaded from {filepath}")
    return model

def load_model_state(filepath, device=None):
    """
    加载模型参数和相关信息/Load the model parameters and related information
    
    参数/Parameters:
    filepath: 模型文件路径/Model file path
    device: 设备，如'cuda'或'cpu'/Device, such as 'cuda' or 'cpu'
    
    返回/Returns:
    model: 加载参数后的模型/Model after loading parameters
    model_params: 模型结构参数/Model structure parameters
    additional_info: 其他保存的信息/Other saved information
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载保存的字典/Load the saved dictionary
    checkpoint = torch.load(filepath, map_location=device)
    
    # 提取模型参数/Extract the model parameters
    model_state_dict = checkpoint['model_state_dict']
    model_params = checkpoint['model_params']
    additional_info = checkpoint.get('additional_info', None)
    
    # 创建模型实例/Create the model instance
    model = LSTMStockPredictor(
        input_size=model_params['input_size'],
        hidden_size=model_params['hidden_size'],
        num_layers=model_params['num_layers'],
        output_size=model_params['output_size'],
        text_feature_size=model_params.get('text_feature_size', 0)
    ).to(device)
    
    # 加载参数/Load the parameters
    model.load_state_dict(model_state_dict)
    model.eval()  # 设置为评估模式/Set to evaluation mode
    
    print(f"Model parameters have been loaded from {filepath}")
    return model, model_params, additional_info

def predict_with_loaded_model(model, preprocessor, new_data, use_kalman=True):
    """
    使用加载的模型预测历史数据/Use the loaded model to predict historical data
    
    参数/Parameters:
    model: 加载的模型/Loaded model
    preprocessor: 数据预处理器/Data preprocessor
    new_data: 新数据，DataFrame格式，包含'Close'列/New data, DataFrame format, containing the 'Close' column
    use_kalman: 是否使用卡尔曼滤波/Whether to use the Kalman filter
    
    返回/Returns:
    predictions: 预测结果列表/Prediction results list
    """
    device = next(model.parameters()).device
    
    # 确保数据是正确的格式/Ensure the data is in the correct format
    if isinstance(new_data, pd.DataFrame):
        price_data = new_data[['Close']].values
    else:
        price_data = new_data
    
    # 标准化数据/Standardize the data
    scaled_data = preprocessor.scaler.transform(price_data)
    
    # 创建时间窗口/Create the time window
    x = []
    for i in range(preprocessor.window_size, len(scaled_data)):
        x.append(scaled_data[i-preprocessor.window_size:i, 0])
    
    x = np.array(x)
    
    # 创建预测器/Create the predictor
    if use_kalman:
        improved_kf_params = {
            'process_noise': 1e-4,
            'measurement_noise': 5e-2,
            'responsiveness': 0.9,
            'trend_weight': 0.4
        }
        predictor = EnhancedStockPredictor(
            model, 
            preprocessor.scaler, 
            kf_params=improved_kf_params, 
            use_improved_kf=True
        )
    
    # 进行预测/Perform the prediction
    predictions = []
    for i in range(len(x)):
        current_window = x[i].reshape(-1, 1)
        
        if use_kalman:
            _, filtered = predictor.predict_with_filter(current_window, look_ahead=2)
            predictions.append(filtered)
        else:
            with torch.no_grad():
                inputs = torch.FloatTensor(current_window).unsqueeze(0).to(device)
                pred = model(inputs).cpu().item()
                raw_price = preprocessor.scaler.inverse_transform([[pred]])[0][0]
                predictions.append(raw_price)
    
    return predictions

def predict_future(model, preprocessor, last_window, days=30, use_kalman=True):
    """
    预测未来几天的股票价格/Predict the stock price for the next few days
    
    参数/Parameters:
    model: 训练好的模型/Trained model
    preprocessor: 数据预处理器/Data preprocessor
    last_window: 最后一个时间窗口的数据，用于初始预测/The last time window data, used for initial prediction
    days: 预测的天数/The number of days to predict
    use_kalman: 是否使用卡尔曼滤波/Whether to use the Kalman filter
    
    返回/Returns:
    future_predictions: 未来几天的预测价格/The predicted prices for the next few days
    """
    device = next(model.parameters()).device
    
    # 确保last_window是正确的形状/Make sure last_window is the correct shape
    if len(last_window.shape) == 1:
        last_window = last_window.reshape(-1, 1)
    
    # 创建预测器/Create predictor
    if use_kalman:
        improved_kf_params = {
            'process_noise': 1e-4,
            'measurement_noise': 5e-2,
            'responsiveness': 0.9,
            'trend_weight': 0.4
        }
        predictor = EnhancedStockPredictor(
            model, 
            preprocessor.scaler, 
            kf_params=improved_kf_params, 
            use_improved_kf=True
        )
        # 初始化卡尔曼滤波器状态/Initialize the state of the Kalman filter
        for i in range(min(10, len(last_window))):
            sample = last_window[-10+i:].reshape(-1, 1)
            if len(sample) == preprocessor.window_size:
                predictor.predict_with_filter(sample)
    
    # 复制最后一个窗口用于滚动预测/Copy the last window for rolling prediction
    future_window = last_window.copy()
    future_predictions = []
    
    # 滚动预测未来几天/Rolling prediction for the next few days
    for _ in range(days):
        # 准备输入/Prepare input
        input_window = future_window[-preprocessor.window_size:].reshape(-1, 1)
        
        # 预测下一天/Predict the next day
        if use_kalman:
            _, next_pred = predictor.predict_with_filter(input_window, look_ahead=1)
        else:
            with torch.no_grad():
                inputs = torch.FloatTensor(input_window).unsqueeze(0).to(device)
                pred = model(inputs).cpu().item()
                next_pred = preprocessor.scaler.inverse_transform([[pred]])[0][0]
        
        # 保存预测结果/Save the prediction result
        future_predictions.append(next_pred)
        
        # 更新窗口：移除最早的值，添加新预测的值/Update the window: remove the oldest value, add the new predicted value
        # 首先将预测值转换回标准化空间/First convert the predicted value back to the standardized space
        next_scaled = preprocessor.scaler.transform([[next_pred]])[0][0]
        future_window = np.roll(future_window, -1)
        future_window[-1] = next_scaled
    
    return future_predictions

def plot_future_predictions(future_predictions, future_dates, title="Future Stock Price Prediction"):
    """
    只绘制未来预测，横坐标显示日期/Only plot the future predictions, display the dates on the x-axis
    
    参数/Parameters:
    future_predictions: 未来预测价格/Future prediction prices
    future_dates: 未来日期/Future dates
    title: 图表标题/Chart title
    """
    plt.figure(figsize=(12, 6))
    
    # 绘制未来预测/Plot the future predictions
    plt.plot(future_dates, future_predictions, label='Prediction price', color='red', marker='o')
    
    # 设置x轴格式/Set the format of the x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3))  # 每3天显示一个日期/Display a date every 3 days
    plt.gcf().autofmt_xdate()  # 自动旋转日期标签/Automatically rotate date labels
    
    # 添加网格和图例/Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(title)
    plt.xlabel('Prediction date')
    plt.ylabel('Prediction price')
    
    # 添加数据标签/Add data labels
    for i, (date, price) in enumerate(zip(future_dates, future_predictions)):
        if i % 5 == 0:  # 每5个点添加一个标签，避免过于拥挤/Add a label every 5 points to avoid crowding
            plt.annotate(f'{price:.2f}', 
                        (date, price), 
                        textcoords="offset points",
                        xytext=(0,10), 
                        ha='center')
    
    plt.tight_layout()
    return plt

if __name__ == '__main__':
    # 加载模型/Load model
    parser = argparse.ArgumentParser(description='predict stock price')
    parser.add_argument('--model_path', type=str, default='./Best_Model/GOOG_1d_20250316_model.pth', help='model path')
    parser.add_argument('--data_path', type=str, default='./data/GOOG_1d_20250316.csv', help='data path')
    parser.add_argument('--future_days', type=int, default=30, help='future days')
    args = parser.parse_args()
    
    model_path = args.model_path
    model, model_params, additional_info = load_model_state(model_path)
    
    # 创建预处理器/Create preprocessor
    window_size = model_params.get('window_size', 60)  # 默认值为60/Default value is 60
    preprocessor = StockDataPreprocessor(window_size=window_size)
    
    # 读取历史数据/Read historical data
    historical_data = pd.read_csv(args.data_path)
    
    # 确保数据包含日期列/Ensure the data contains the date column
    if 'Date' not in historical_data.columns:
        # 如果没有日期列，创建一个假的日期序列/If there is no date column, create a fake date sequence
        end_date = datetime.now()
        start_date = end_date - timedelta(days=len(historical_data))
        date_range = pd.date_range(start=start_date, periods=len(historical_data), freq='D')
        historical_data['Date'] = date_range
    
    # 转换日期列为datetime类型/Convert the date column to datetime type
    historical_data['Date'] = pd.to_datetime(historical_data['Date'])
    
    # 获取最后一个窗口的数据用于未来预测/Get the last window data for future prediction
    last_window_data = historical_data['Close'].values[-window_size:]
    last_window_scaled = preprocessor.scaler.fit_transform(historical_data[['Close']].values)[-window_size:]
    
    # 预测未来30天/Predict the next 30 days
    future_days = 30
    future_predictions = predict_future(
        model, 
        preprocessor, 
        last_window_scaled, 
        days=future_days, 
        use_kalman=True
    )
    
    # 创建未来日期/Create the future dates
    last_date = historical_data['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(future_days)]
    
    # 创建输出目录/Create the output directory
    import os
    os.makedirs('predictions', exist_ok=True)
    
    # 只绘制未来预测结果/Only plot the future prediction results
    plt = plot_future_predictions(
        future_predictions,
        future_dates,
        title=f"{args.data_path.split('/')[-1].split('.')[0]} Stock {future_days} days prediction (from {last_date.strftime('%Y-%m-%d')})"
    )
    
    # 保存图表/Save the chart
    plt.savefig(f'predictions/{args.data_path.split("/")[-1].split(".")[0]}_future_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.show()
    
    # 输出预测结果/Output the prediction results
    prediction_df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in future_dates],
        'Predicted_Price': [round(p, 2) for p in future_predictions]
    })
    print("\nFuture 30 days prediction results:")
    print(prediction_df)
    
    # 保存预测结果到CSV/Save the prediction results to CSV
    prediction_df.to_csv(f'predictions/{args.data_path.split("/")[-1].split(".")[0]}_future_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', index=False)
    print(f"Prediction results have been saved to CSV file")




