import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pmdarima import auto_arima
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import numpy as np

# Define the RIC and start date
RIC = 'AAPL'
start_date = '2018-01-02'

# Get time series data using yfinance
df = yf.download(RIC, start=start_date)

# Check if data was retrieved
if df.empty:
    raise ValueError("No data retrieved. Please check the ticker symbol and date range.")
else:
    # Rename 'Adj Close' to 'CLOSE' if necessary
    if 'Adj Close' in df.columns:
        df.rename(columns={'Adj Close': 'CLOSE'}, inplace=True)

    # Check if 'CLOSE' column exists
    if 'CLOSE' not in df.columns:
        raise KeyError("The 'CLOSE' column is not available in the DataFrame. Please check the data source.")

    # Print the last date of the time series data
    print("Last date in time series:", df.index[-1])

# The rest of your code remains the same

def bootstrap_prediction_intervals(model, X, n_bootstrap=1000, alpha=0.05):
    predictions = np.array([model.predict(X) for _ in range(n_bootstrap)])
    lower_bound = np.percentile(predictions, alpha/2*100, axis=0)
    upper_bound = np.percentile(predictions, (1-alpha/2)*100, axis=0)
    return lower_bound, upper_bound

def improved_ARIMA_forecast(df, forecast_end_date):
    try:
        # Prepare the data
        data = df['CLOSE'].resample('D').ffill().dropna()
        print("Data prepared. Shape:", data.shape)

        # Automatically select the best ARIMA model
        model = auto_arima(data, start_p=1, start_q=1, max_p=5, max_q=5, m=1,
                           start_P=0, seasonal=False, d=1, D=1, trace=True,
                           error_action='ignore', suppress_warnings=True, stepwise=True)
        print("ARIMA model selected")

        # Fit the model
        model.fit(data)
        print("ARIMA model fitted")

        # Forecast with ARIMA
        forecast_horizon = (datetime.strptime(forecast_end_date, '%Y-%m-%d') - data.index[-1]).days
        forecast, conf_int = model.predict(n_periods=forecast_horizon, return_conf_int=True)
        print("ARIMA forecast completed")

        # Calculate residuals
        arima_predictions = model.predict_in_sample()
        residuals = data - arima_predictions
        print("Residuals calculated")

        # Prepare features for machine learning
        df_ml = data.to_frame(name='CLOSE')
        df_ml['lag1'] = df_ml['CLOSE'].shift(1)
        df_ml['lag2'] = df_ml['CLOSE'].shift(2)
        df_ml['rolling_mean_7'] = df_ml['CLOSE'].rolling(window=7).mean()
        df_ml['day_of_week'] = df_ml.index.dayofweek
        df_ml['month'] = df_ml.index.month
        df_ml['residual'] = residuals
        df_ml.dropna(inplace=True)
        print("ML features prepared")

        # Cross-validation for ML model
        X = df_ml[['lag1', 'lag2', 'rolling_mean_7', 'day_of_week', 'month']]
        y = df_ml['residual']
        
        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            ml_model = RandomForestRegressor()
            ml_model.fit(X_train, y_train)
            y_pred = ml_model.predict(X_test)
            print(f'Fold MSE: {mean_squared_error(y_test, y_pred)}')
        print("ML model cross-validated and trained")

        # Prepare forecast features and predict
        lag1 = [data.iloc[-1]] + list(forecast[:-1])
        lag2 = [data.iloc[-2]] + list(forecast[:-2])

        # Ensure that lag1, lag2, and forecast are of the same length
        min_length = min(len(lag1), len(lag2), len(forecast), len(conf_int))
        lag1 = lag1[:min_length]
        lag2 = lag2[:min_length]
        forecast = forecast[:min_length]
        conf_int = conf_int[:min_length]

        X_forecast = pd.DataFrame({
            'lag1': lag1,
            'lag2': lag2,
            'rolling_mean_7': pd.Series(lag1).rolling(window=7).mean().fillna(method='bfill').values,
            'day_of_week': pd.date_range(start=data.index[-1] + timedelta(days=1), periods=min_length).dayofweek,
            'month': pd.date_range(start=data.index[-1] + timedelta(days=1), periods=min_length).month
        })
        
        ml_residuals = ml_model.predict(X_forecast)
        print ('.........................................')
        print("ML residuals predicted")

        # Combine forecasts
        final_forecast = forecast + ml_residuals

        # Calculate confidence intervals for the ML model
        ml_lower, ml_upper = bootstrap_prediction_intervals(ml_model, X_forecast)
        ml_lower = ml_lower[:min_length]
        ml_upper = ml_upper[:min_length]

        final_conf_int_lower = conf_int[:, 0] + ml_lower
        final_conf_int_upper = conf_int[:, 1] + ml_upper
        
        date_range = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=min_length)
        forecast_df = pd.DataFrame({'CLOSE': final_forecast}, index=date_range)
        print("Final forecast prepared")

        # Plot the predicted residuals
        plt.figure(figsize=(16,7))
        plt.plot(y_test.index, y_test, label='Actual Residuals')
        plt.plot(y_test.index, ml_model.predict(X_test), label='Predicted Residuals', color='orange')
        plt.legend(loc='best')
        plt.title('Actual vs Predicted Residuals')
        plt.show()
        print("Residuals plot displayed")

        # Plot final forecast with updated confidence intervals
        plt.figure(figsize=(16,7))
        plt.plot(data, label='Actual')
        plt.plot(forecast_df, label='Improved Forecast', color='red')
        plt.fill_between(forecast_df.index, final_conf_int_lower, final_conf_int_upper, color='pink', alpha=0.3)
        plt.legend(loc='best')
        plt.title(f'Improved ARIMA Forecast for {RIC}')
        plt.show()
        print("Final forecast plot displayed")

        return forecast_df

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

# Define the end date for the forecast
forecast_end_date = '2025-08-08'  # Set to a future date

# Print the forecast end date for debugging
print("Forecast end date:", forecast_end_date)

# Call the function to forecast time series with improved ARIMA model
forecast = improved_ARIMA_forecast(df, forecast_end_date)

# Display forecast time series
print(forecast)
