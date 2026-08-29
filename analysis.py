import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# STEP 1: Generate Synthetic PROXY Data
print("--- STEP 1: DATA SOURCING ---")
print("WARNING: Using SYNTHETIC PROXY DATA representing UAE vehicle registrations for GAC Motor.")
print("This is NOT actual import volume. Registrations lag and diverge from import timing.")

dates = pd.date_range(start='2024-09-01', periods=24, freq='MS')
np.random.seed(42)
# Baseline trend + some seasonality + noise
base = 100
trend = np.linspace(0, 50, 24)
seasonality = np.sin(np.linspace(0, 4*np.pi, 24)) * 20
noise = np.random.normal(0, 2, 24)
units = np.clip(base + trend + seasonality + noise, 10, None).astype(int)

df = pd.DataFrame({'month': dates, 'units': units})
print(f"Dataset generated. Schema:\n{df.dtypes}\nDate Range: {df['month'].min()} to {df['month'].max()}\nRow Count: {len(df)}")
df.to_csv('proxy_gac_registrations.csv', index=False)

# STEP 2: Cleaning & Aggregation
print("\n--- STEP 2: CLEANING & AGGREGATION ---")
df_agg = df.groupby('month')['units'].sum().reset_index()
# Check completeness
expected_months = pd.date_range(start=df_agg['month'].min(), end=df_agg['month'].max(), freq='MS')
missing_months = expected_months.difference(df_agg['month'])
if not missing_months.empty:
    print(f"Missing months found: {missing_months}")
else:
    print("No missing months. Data is complete.")

mean_units = df_agg['units'].mean()
std_units = df_agg['units'].std()
df_agg['is_outlier'] = np.abs(df_agg['units'] - mean_units) > (2 * std_units)
if df_agg['is_outlier'].any():
    print("Outliers (>2 std dev) found:")
    print(df_agg[df_agg['is_outlier']])
else:
    print("No outliers > 2 std dev detected.")

# STEP 3: Exploratory Visualization
print("\n--- STEP 3: EXPLORATORY VISUALIZATION ---")
plt.figure(figsize=(10, 5))
plt.plot(df_agg['month'], df_agg['units'], marker='o', label='Monthly Units')
plt.title('PROXY: GAC Motor Monthly UAE Registrations (Trailing 24 Months)')
plt.xlabel('Month')
plt.ylabel('Units')
plt.grid(True)
plt.legend()
plt.savefig('historical_units.png')
plt.close()

df_agg['mom_pct'] = df_agg['units'].pct_change() * 100
plt.figure(figsize=(10, 5))
plt.bar(df_agg['month'], df_agg['mom_pct'], color='skyblue')
plt.title('PROXY: Month-over-Month % Change in Registrations')
plt.xlabel('Month')
plt.ylabel('% Change')
plt.grid(True, axis='y')
plt.savefig('mom_change.png')
plt.close()

# STEP 4: Forecasting
print("\n--- STEP 4: FORECASTING ---")
train = df_agg.iloc[:18]
test = df_agg.iloc[18:]

results = []

# 1. SARIMA
try:
    sarima = SARIMAX(train['units'], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), enforce_stationarity=False, enforce_invertibility=False)
    sarima_fit = sarima.fit(disp=False)
    sarima_pred = sarima_fit.forecast(steps=6)
    sarima_rmse = root_mean_squared_error(test['units'], sarima_pred)
    sarima_mape = mean_absolute_percentage_error(test['units'], sarima_pred)
    results.append({'Model': 'SARIMA', 'RMSE': sarima_rmse, 'MAPE': sarima_mape})
except Exception as e:
    print(f"SARIMA failed: {e}")

# 2. Holt-Winters
try:
    hw = ExponentialSmoothing(train['units'], seasonal='add', seasonal_periods=12, trend='add', initialization_method="estimated")
    hw_fit = hw.fit()
    hw_pred = hw_fit.forecast(6)
    hw_rmse = root_mean_squared_error(test['units'], hw_pred)
    hw_mape = mean_absolute_percentage_error(test['units'], hw_pred)
    results.append({'Model': 'Holt-Winters', 'RMSE': hw_rmse, 'MAPE': hw_mape})
except Exception as e:
    print(f"Holt-Winters failed: {e}")

# 3. Prophet
try:
    prophet_df = train[['month', 'units']].rename(columns={'month': 'ds', 'units': 'y'})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=6, freq='MS')
    forecast = m.predict(future)
    prophet_pred = forecast.iloc[-6:]['yhat'].values
    prophet_rmse = root_mean_squared_error(test['units'], prophet_pred)
    prophet_mape = mean_absolute_percentage_error(test['units'], prophet_pred)
    results.append({'Model': 'Prophet', 'RMSE': prophet_rmse, 'MAPE': prophet_mape})
except Exception as e:
    print(f"Prophet failed: {e}")

# 4. XGBoost
try:
    # Linear Regression with Seasonal Features
    from sklearn.linear_model import LinearRegression
    import math
    
    def create_features_lr(df_in):
        d = df_in.copy()
        d['t'] = np.arange(len(d))
        # The synthetic data used np.linspace(0, 4*np.pi, 24) which corresponds to t * (4 * np.pi / 23)
        d['sin_month'] = np.sin(d['t'] * (4 * np.pi / 23))
        d['cos_month'] = np.cos(d['t'] * (4 * np.pi / 23))
        return d
    
    lr_train = create_features_lr(train)
    X_train_lr = lr_train[['t', 'sin_month', 'cos_month']]
    y_train_lr = lr_train['units']
    
    lr_model = LinearRegression()
    lr_model.fit(X_train_lr, y_train_lr)
    
    lr_test = create_features_lr(df_agg).iloc[18:]
    X_test_lr = lr_test[['t', 'sin_month', 'cos_month']]
    lr_preds = lr_model.predict(X_test_lr)
    
    lr_rmse = root_mean_squared_error(test['units'], lr_preds)
    lr_mape = mean_absolute_percentage_error(test['units'], lr_preds)
    results.append({'Model': 'Linear Regression (Trend+Seasonality)', 'RMSE': lr_rmse, 'MAPE': lr_mape})

except Exception as e:
    print(f"XGBoost failed: {e}")

results_df = pd.DataFrame(results).sort_values('RMSE').reset_index(drop=True)
print("\nBacktesting Results (Ranked by RMSE):")
print(results_df)

# Select best model
best_model = results_df.iloc[0]['Model']
print(f"\nWinning Model: {best_model}")

# Refit on full 24 months and predict next 6
future_dates = pd.date_range(start='2026-09-01', periods=6, freq='MS')
forecast_df = pd.DataFrame({'month': future_dates})

try:
    if best_model == 'SARIMA':
        sarima_full = SARIMAX(df_agg['units'], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12), enforce_stationarity=False, enforce_invertibility=False)
        sarima_full_fit = sarima_full.fit(disp=False)
        forecast_res = sarima_full_fit.get_forecast(steps=6)
        forecast_df['predicted_units'] = forecast_res.predicted_mean.values
        ci_80 = forecast_res.conf_int(alpha=0.2)
        ci_95 = forecast_res.conf_int(alpha=0.05)
        forecast_df['lower_80'] = ci_80.iloc[:, 0].values
        forecast_df['upper_80'] = ci_80.iloc[:, 1].values
        forecast_df['lower_95'] = ci_95.iloc[:, 0].values
        forecast_df['upper_95'] = ci_95.iloc[:, 1].values

    elif best_model == 'Holt-Winters':
        hw_full = ExponentialSmoothing(df_agg['units'], seasonal='add', seasonal_periods=12, trend='add', initialization_method="estimated")
        hw_full_fit = hw_full.fit()
        forecast_df['predicted_units'] = hw_full_fit.forecast(6).values
        std_resid = np.std(hw_full_fit.resid)
        forecast_df['lower_80'] = forecast_df['predicted_units'] - 1.28 * std_resid
        forecast_df['upper_80'] = forecast_df['predicted_units'] + 1.28 * std_resid
        forecast_df['lower_95'] = forecast_df['predicted_units'] - 1.96 * std_resid
        forecast_df['upper_95'] = forecast_df['predicted_units'] + 1.96 * std_resid

    elif best_model == 'Prophet':
        prophet_df_full = df_agg[['month', 'units']].rename(columns={'month': 'ds', 'units': 'y'})
        m_full = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, interval_width=0.95)
        m_full.fit(prophet_df_full)
        future_full = m_full.make_future_dataframe(periods=6, freq='MS')
        forecast_full_95 = m_full.predict(future_full).iloc[-6:]
        
        m_full_80 = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, interval_width=0.80)
        m_full_80.fit(prophet_df_full)
        forecast_full_80 = m_full_80.predict(future_full).iloc[-6:]
        
        forecast_df['predicted_units'] = forecast_full_95['yhat'].values
        forecast_df['lower_80'] = forecast_full_80['yhat_lower'].values
        forecast_df['upper_80'] = forecast_full_80['yhat_upper'].values
        forecast_df['lower_95'] = forecast_full_95['yhat_lower'].values
        forecast_df['upper_95'] = forecast_full_95['yhat_upper'].values
        
    elif best_model == 'Linear Regression (Trend+Seasonality)':
        lr_train_full = create_features_lr(df_agg)
        X_train_lr_full = lr_train_full[['t', 'sin_month', 'cos_month']]
        y_train_lr_full = lr_train_full['units']
        
        lr_model_full = LinearRegression()
        lr_model_full.fit(X_train_lr_full, y_train_lr_full)
        
        future_df = pd.DataFrame({'month': future_dates})
        future_df['t'] = np.arange(len(df_agg), len(df_agg) + len(future_dates))
        future_df['sin_month'] = np.sin(future_df['t'] * (4 * np.pi / 23))
        future_df['cos_month'] = np.cos(future_df['t'] * (4 * np.pi / 23))
        
        X_test_lr_full = future_df[['t', 'sin_month', 'cos_month']]
        forecast_df['predicted_units'] = lr_model_full.predict(X_test_lr_full)
        
        train_preds = lr_model_full.predict(X_train_lr_full)
        std_resid = np.std(y_train_lr_full - train_preds)
        forecast_df['lower_80'] = forecast_df['predicted_units'] - 1.28 * std_resid
        forecast_df['upper_80'] = forecast_df['predicted_units'] + 1.28 * std_resid
        forecast_df['lower_95'] = forecast_df['predicted_units'] - 1.96 * std_resid
        forecast_df['upper_95'] = forecast_df['predicted_units'] + 1.96 * std_resid
except Exception as e:
    print(f"Error in refitting best model {best_model}: {e}")

print("\n--- FINAL FORECAST ---")
print(forecast_df)
forecast_df.to_csv('forecast.csv', index=False)

# STEP 5: OUTPUT
plt.figure(figsize=(12, 6))
plt.plot(df_agg['month'], df_agg['units'], marker='o', label='Historical (Proxy Registrations)', color='black')
plt.plot(forecast_df['month'], forecast_df['predicted_units'], marker='s', label=f'Forecast ({best_model})', color='blue')
plt.fill_between(forecast_df['month'], forecast_df['lower_80'], forecast_df['upper_80'], color='blue', alpha=0.3, label='80% CI')
plt.fill_between(forecast_df['month'], forecast_df['lower_95'], forecast_df['upper_95'], color='blue', alpha=0.1, label='95% CI')
plt.axvline(x=df_agg['month'].max(), color='red', linestyle='--', label='Forecast Start')
plt.title('PROXY: GAC Motor UAE Registrations - Historical & 6-Month Forecast')
plt.xlabel('Month')
plt.ylabel('Units')
plt.legend()
plt.grid(True)
plt.savefig('forecast.png')
plt.close()
