import yfinance as yf
import pandas as pd
import numpy as np

# Download data
stock_data = yf.download("AAPL", start="2023-01-01", end="2024-01-01")

# Flatten MultiIndex if needed
if isinstance(stock_data.columns, pd.MultiIndex):
    stock_data.columns = [col[0] for col in stock_data.columns]

# Drop missing values
stock_data = stock_data.dropna()

# Daily returns
stock_data["Daily_return"] = stock_data["Close"].pct_change()

# Moving averages
stock_data["MA20"] = stock_data["Close"].rolling(20).mean()
stock_data["MA50"] = stock_data["Close"].rolling(50).mean()

# RSI function
def compute_RSI(data, window=14):
    delta = data["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain, index=data.index).rolling(window=window).mean()
    avg_loss = pd.Series(loss, index=data.index).rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Apply RSI
stock_data['RSI'] = compute_RSI(stock_data, window=14)

# Volume moving average
stock_data["Volume_MA20"] = stock_data["Volume"].rolling(20).mean()

# Target column (1 if tomorrow's close > today's close, else 0)
stock_data["Target"] = (stock_data["Close"].shift(-1) > stock_data["Close"]).astype(int)

# Drop missing values
stock_data = stock_data.dropna()

# Save to CSV
stock_data.to_csv(r"ml_dataset1.csv", index=False)
print("ML dataset saved successfully")

# Check target distribution
print(stock_data["Target"].value_counts())
