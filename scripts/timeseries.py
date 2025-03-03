import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

def check_stationarity(timeseries, asset_name):
    result = adfuller(timeseries.dropna())
    print(f'\nTesting stationarity of {asset_name} prices:')
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.4f}')
    if result[1] <= 0.05:
        print("Result: The series is stationary (reject H0)")
    else:
        print("Result: The series is non-stationary (fail to reject H0)")

def time_series_analysis_and_forecasting(asset_name, prices, returns, p=1, d=0, q=1, future_steps=30):
    asset_price = prices[asset_name]
    asset_returns = returns[asset_name]
    
    # Check stationarity
    check_stationarity(asset_price, asset_name)
    check_stationarity(asset_returns, asset_name)
    
    # Time series decomposition
    print(f"\nDecomposing {asset_name} price series:")
    decomposition = seasonal_decompose(asset_price, model='multiplicative', period=252)
    fig = decomposition.plot()
    plt.tight_layout()
    plt.show()
    
    # Train-test split
    train_size = int(len(asset_returns) * 0.9)
    train_data = asset_returns.iloc[:train_size]
    test_data = asset_returns.iloc[train_size:]
    
    # Fit ARIMA model
    print("\nFitting ARIMA model...")
    model = ARIMA(train_data, order=(p, d, q))
    model_fit = model.fit()
    
    # Print model summary
    print(model_fit.summary())
    
    # Forecast and evaluate
    forecast_steps = len(test_data)
    forecast = model_fit.forecast(steps=forecast_steps)
    mae = mean_absolute_error(test_data, forecast)
    rmse = np.sqrt(mean_squared_error(test_data, forecast))
    print(f"\nModel Evaluation on Test Data:")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    
    # Plot actual vs forecast returns
    plt.figure(figsize=(15, 8))
    plt.plot(test_data.index, test_data, label='Actual Returns', color='blue')
    plt.plot(test_data.index, forecast, label='Forecasted Returns', color='red', linestyle='--')
    plt.title(f'{asset_name} - Actual vs Forecasted Returns')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Generate and plot future forecasts
    future_forecast = model_fit.forecast(steps=future_steps)
    future_dates = pd.date_range(start=asset_returns.index[-1], periods=future_steps)
    
    plt.figure(figsize=(15, 8))
    plt.plot(asset_returns.index[-90:], asset_returns.iloc[-90:], label='Historical Returns', color='blue')
    plt.plot(future_dates, future_forecast, label='Future Returns Forecast', color='red', linestyle='--')
    plt.title(f'{asset_name} - Returns Forecast for Next {future_steps} Trading Days')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.axvline(x=asset_returns.index[-1], color='green', linestyle='-', label='Forecast Start')
    plt.legend()
    plt.tight_layout()
    plt.show()
