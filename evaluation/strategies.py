import numpy as np
import pandas as pd


# Random Strategy
def random_strategy(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    np.random.seed(seed)
    signals = np.random.choice([-1, 0, 1], size=len(df))
    return pd.Series(signals, index=df.index, name="signal")


# EMA + RSI Rule Strategy
def rule_strategy(df: pd.DataFrame) -> pd.Series:
    signal = np.zeros(len(df))

    buy_condition = (df["ema_20"] > df["ema_50"]) & (df["rsi_14"] > 55)
    sell_condition = (df["ema_20"] < df["ema_50"]) & (df["rsi_14"] < 45)

    signal[buy_condition] = 1
    signal[sell_condition] = -1

    return pd.Series(signal, index=df.index, name="signal")


# Buy & Hold
def buy_and_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1, index=df.index, name="signal")
