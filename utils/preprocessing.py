import yfinance as yf
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler

def load_data(ticker="AAPL", start="2018-01-01", end="2023-01-01"):
    """
    Download OHLCV data using yfinance.
    """
    df = yf.download(ticker, start=start, end=end)
    # Flatten if multi-index
    df.columns = df.columns.get_level_values(0)
    return df

def add_indicators(df):
    """
    Add RSI, MACD, Aroon, Bollinger Bands to dataframe.
    """
    # RSI
    df['RSI'] = ta.rsi(df['Close'], length=14)

    # MACD
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)

    # Aroon
    aroon = ta.aroon(df['High'], df['Low'], length=14)
    df = pd.concat([df, aroon], axis=1)

    # Bollinger Bands
    bbands = ta.bbands(df['Close'], length=20, std=2)
    df = pd.concat([df, bbands], axis=1)

    return df

def clean_data(df):
    """
    Drop rows with NaN values (from indicator warm-up).
    """
    return df.dropna()

def scale_data(df):
    """
    Normalize features into range [0,1] for LSTM/RL.
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled, index=df.index, columns=df.columns)
    return df_scaled, scaler

def preprocess_pipeline(ticker="AAPL", start="2018-01-01", end="2023-01-01"):
    """
    Full preprocessing pipeline.
    """
    df = load_data(ticker, start, end)
    df = add_indicators(df)
    df = clean_data(df)
    df_scaled, scaler = scale_data(df)
    return df_scaled, scaler
