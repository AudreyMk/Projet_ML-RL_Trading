import pandas as pd


def run_backtest(df: pd.DataFrame,
                 signal: pd.Series,
                 transaction_cost: float = 0.0001):

    df = df.copy()
    df["signal"] = signal

    # Position appliquée à t+1 (anti-leakage)
    df["position"] = df["signal"].shift(1).fillna(0)

    # Retour stratégie
    df["strategy_return"] = df["position"] * df["future_return"]

    # Coût transaction si changement de position
    df["trade"] = df["position"].diff().abs()
    df["strategy_return"] -= df["trade"] * transaction_cost

    # Equity curve
    df["equity_curve"] = df["strategy_return"].cumsum()

    return df
