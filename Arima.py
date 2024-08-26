import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pmdarima import auto_arima
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import numpy as np
import streamlit as st

# Title for your app
st.title("ARIMA Machine Learning")

# Text input for ticker and start date
RIC = st.text_input("Enter the ticker:", "AAPL")
start_date = st.text_input("Enter the start date (e.g., 2018-01-02):", "2018-01-02")
forecast_end_date = st.text_input("Enter the forecast end date (e.g., 2025-08-08):", "2025-08-08")

# Get time series data using yfinance
df = yf.download(RIC, start=start_date)

# Check if data was retrieved
if df.empty:
    st.error("No data retrieved. Please check the ticker symbol and date range.")
else:
    if 'Adj Close' in df.columns:
        df.rename(columns={'Adj Close': 'CLOSE'}, inplace=True)

    # Check if 'CLOSE' column exists
    if 'CLOSE' not in df.columns:
        st.error("The 'CLOSE' column is not available in the DataFrame.")
    else:
        st.write("Data retrieved successfully. Last date in time series:", df.index[-1])

        # Proceed with the ARIMA and ML forecast
        def bootstrap_prediction_intervals(model, X, n_bootstrap=1000, alpha=0.05):
            predictions = np.array([model.predict(X) for _ in range(n_bootstrap)])
            lower_bound = np.percentile(predictions, alpha/2*100, axis=0)
            upper_bound = np.percentile(predictions, (1-alpha/2)*100, axis=0)
            return lower_bound, upper_bound

        def improved_ARIMA_forecast(df, forecast_end_date):
            try:
                # Prepare the data
                data = df['CLOSE'].resample('D').ffill().dropna()
                
                # Automatically select the best ARIMA model
                model = auto_arima(data, start_p=1, start_q=1, max_p=5, max_q=5, m=1,
                                   start_P=0, seasonal=False, d=1, D=1, trace=False,
                                   error_action='ignore', suppress_warnings=True, stepwise=True)

                # Fit the model
                model.fit(data)

                # Forecast with ARIMA
                forecast_horizon = (datetime.strptime(forecast_end_date, '%Y-%m-%d') - data.index[-1]).days
                forecast, conf_int = model.predict(n_periods=forecast_horizon, return_conf_int=True)

                # Calculate residuals
                arima_predictions = model.predict_in_sample()
                residuals = data - arima_predictions

                # Prepare features for machine learning
                df_ml = data.to_frame(name='CLOSE')
                df_ml['lag1'] = df_ml['CLOSE'].shift(1)
                df_ml['lag2'] = df_ml['CLOSE'].shift(2)
                df_ml['rolling_mean_7'] = df_ml['CLOSE'].rolling(window=7).mean()
                df_ml['day_of_week'] = df_ml.index.dayofweek
                df_ml['month'] = df_ml.index.month
                df_ml['residual'] = residuals
                df_ml.dropna(inplace=True)

                # Cross-validation for ML model
                X = df_ml[['lag1', 'lag2', 'rolling_mean_7', 'day_of_week', 'month']]
                y = df_ml['residual']

                tscv = TimeSeriesSplit(n_splits=5)
                ml_model = RandomForestRegressor()
                for train_index, test_index in tscv.split(X):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    ml_model.fit(X_train, y_train)
                    y_pred = ml_model.predict(X_test)

                # Prepare forecast features and predict
                lag1 = [data.iloc[-1]] + list(forecast[:-1])
                lag2 = [data.iloc[-2]] + list(forecast[:-2])

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

                # Combine forecasts
                final_forecast = forecast + ml_residuals
                forecast_df = pd.DataFrame({'CLOSE': final_forecast}, index=pd.date_range(start=data.index[-1] + timedelta(days=1), periods=min_length))

                # Plot final forecast with updated confidence intervals
                fig, ax = plt.subplots(figsize=(16, 7))
                ax.plot(data, label='Actual')
                ax.plot(forecast_df, label='Improved Forecast', color='red')
                ax.fill_between(forecast_df.index, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3)
                ax.legend(loc='best')
                ax.set_title(f'Improved ARIMA Forecast for {RIC}')
                st.pyplot(fig)

                return forecast_df

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

        # Call the function to forecast time series with improved ARIMA model
        forecast = improved_ARIMA_forecast(df, forecast_end_date)

        # Display forecast time series
        st.write(forecast)
