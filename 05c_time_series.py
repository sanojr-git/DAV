# EX.NO: 5C - Time Series Analysis on Glucose Levels
# AIM: Perform time series decomposition and moving average on glucose data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

np.random.seed(42)

# Simulate time series glucose data (seasonal pattern)
t = np.arange(200)
glucose = (
    120 +
    20 * np.sin(2 * np.pi * t / 30) +   # seasonal component (30-day cycle)
    np.random.normal(0, 5, 200) +         # noise
    t * 0.05                               # trend
)

df = pd.DataFrame({'Glucose': glucose}, index=pd.date_range('2022-01-01', periods=200, freq='D'))

# Plot original time series
plt.figure(figsize=(12, 4))
plt.plot(df.index, df['Glucose'], color='steelblue', linewidth=1.5, label='Glucose Level')
plt.xlabel('Date'); plt.ylabel('Glucose (mg/dL)')
plt.title('Time Series of Glucose Levels')
plt.legend(); plt.tight_layout()
plt.savefig('glucose_timeseries.png')
plt.show()
print("Saved: glucose_timeseries.png")

# Seasonal Decomposition
decomp = seasonal_decompose(df['Glucose'], model='additive', period=30)
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
df['Glucose'].plot(ax=ax1, title='Original', color='steelblue')
decomp.trend.plot(ax=ax2, title='Trend', color='orange')
decomp.seasonal.plot(ax=ax3, title='Seasonal', color='green')
decomp.resid.plot(ax=ax4, title='Residual', color='red')
plt.tight_layout()
plt.savefig('decomposition.png')
plt.show()
print("Saved: decomposition.png")

# Moving Average Smoothing
df['MA_7']  = df['Glucose'].rolling(window=7).mean()
df['MA_30'] = df['Glucose'].rolling(window=30).mean()

plt.figure(figsize=(12, 4))
plt.plot(df.index, df['Glucose'], alpha=0.4, label='Original', color='steelblue')
plt.plot(df.index, df['MA_7'],  color='orange', linewidth=2, label='7-day MA')
plt.plot(df.index, df['MA_30'], color='red',    linewidth=2, label='30-day MA')
plt.title('Moving Average Smoothing of Glucose Data')
plt.xlabel('Date'); plt.ylabel('Glucose (mg/dL)')
plt.legend(); plt.tight_layout()
plt.savefig('moving_average.png')
plt.show()
print("Saved: moving_average.png")
print("\nTime series analysis complete!")
