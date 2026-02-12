import numpy as np


def compute_metrics(df):

    returns = df["strategy_return"]

    cumulative_profit = returns.sum()

    # Drawdown
    equity = df["equity_curve"]
    peak = equity.cummax()
    drawdown = equity - peak
    max_drawdown = drawdown.min()

    # Sharpe simplifié
    sharpe = 0
    if returns.std() != 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(96)  
        # 96 bougies M15 par jour

    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = gains / losses if losses != 0 else np.inf

    return {
        "cumulative_profit": cumulative_profit,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "profit_factor": profit_factor
    }
