import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# =============================
# CONFIG
# =============================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DATA_PATH = PROJECT_ROOT / "data" / "features" / "DAT_MT_GBPUSD_M15_2024_features.csv"
ML_SIGNALS_PATH = PROJECT_ROOT / "data" / "rl_outputs" / "signals_ml_2024.csv"
RL_SIGNALS_PATH = PROJECT_ROOT / "data" / "rl_outputs" / "signals_test_2024.csv"

# =============================
# MÉTRIQUES
# =============================

def compute_metrics(df):
    returns = df["strategy_return"]

    cumulative_profit = returns.sum()

    equity = returns.cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    max_drawdown = drawdown.min()

    if returns.std() == 0:
        sharpe = 0
    else:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)

    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())

    if gross_loss == 0:
        profit_factor = np.inf
    else:
        profit_factor = gross_profit / gross_loss

    return {
        "cumulative_profit": float(cumulative_profit),
        "max_drawdown": float(max_drawdown),
        "sharpe": float(sharpe),
        "profit_factor": float(profit_factor)
    }

# =============================
# BACKTEST
# =============================

def backtest(test_df, signals):
    df = test_df.copy()
    df["signal"] = signals
    df["strategy_return"] = df["signal"].shift(1) * df["return"]
    df["strategy_return"] = df["strategy_return"].fillna(0)
    return df

# =============================
# RANDOM STRATEGY
# =============================

def generate_random_signals(df):
    return np.random.choice([-1, 0, 1], size=len(df))

# =============================
# RULE BASED STRATEGY (EXEMPLE SMA)
# =============================

def generate_rule_signals(df):
    df = df.copy()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()

    signal = np.where(df["sma_20"] > df["sma_50"], 1, -1)
    signal[np.isnan(signal)] = 0

    return signal

# =============================
# LOAD DATA
# =============================

def load_data():
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable: {TEST_DATA_PATH}")
    df = pd.read_csv(TEST_DATA_PATH)
    df["return"] = df["close"].pct_change().fillna(0)
    return df

# =============================
# LOAD ML / RL
# =============================

def load_signals(path):
    df = pd.read_csv(path)
    return df["signal"].values


def generate_signals_from_models(test_df):
    signals_dir = PROJECT_ROOT / "data" / "rl_outputs"
    signals_dir.mkdir(parents=True, exist_ok=True)

    # --- ML signals ---
    ml_path = PROJECT_ROOT / "models_registry" / "v1" / "logistic_v1.pkl"
    scaler_path = PROJECT_ROOT / "models_registry" / "v1" / "scaler_v1.pkl"
    meta_path = PROJECT_ROOT / "models_registry" / "v1" / "metadata.json"

    if ml_path.exists() and scaler_path.exists() and meta_path.exists():
        import json
        import joblib

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        feature_list = meta.get("feature_list", [])
        model = joblib.load(ml_path)
        scaler = joblib.load(scaler_path)

        X = test_df.select_dtypes(include=[np.number]).copy()
        for drop in ["future_return", "target_direction"]:
            if drop in X.columns:
                X = X.drop(columns=[drop])
        if feature_list:
            X = X[feature_list]

        X = X.dropna()
        X_scaled = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
        pred = model.predict(X_scaled)
        ml_signals = pd.Series(np.where(pred == 1, 1, -1), index=X_scaled.index, name="signal")
        ml_signals.to_csv(ML_SIGNALS_PATH, index=False)
    else:
        ml_signals = None

    # --- RL signals (Q-table) ---
    rl_path = PROJECT_ROOT / "models_registry" / "rl" / "q_table.pkl"
    if rl_path.exists():
        with open(rl_path, "rb") as f:
            q_obj = pickle.load(f)
        q_table = q_obj.get("q_table")
        bins = q_obj.get("bins")

        def discretize(values, feature_cols, edges):
            out = []
            for v, col in zip(values, feature_cols):
                b = edges[col]
                idx = np.digitize(v, b[1:-1], right=False)
                out.append(int(idx))
            return np.array(out, dtype=int)

        # Use only features that exist in bins
        if bins is not None:
            cols = [col for col in bins.keys() if col in test_df.columns]
        else:
            cols = test_df.select_dtypes(include=[np.number]).columns.tolist()
            for drop in ["future_return", "target_direction"]:
                if drop in cols:
                    cols.remove(drop)

        signals = []
        position = 0
        for i in range(len(test_df)):
            vals = test_df.iloc[i][cols].values
            if bins is None:
                action = 0
            else:
                state_bins = discretize(vals, cols, bins)
                state = tuple(np.append(state_bins, position).astype(int))
                q_vals = q_table.get(state, np.zeros(3)) if q_table is not None else np.zeros(3)
                action = int(np.argmax(q_vals))

            if action == 1:
                position = 1
            elif action == 2:
                position = -1
            else:
                position = 0
            signals.append(position)

        rl_signals = pd.Series(signals, index=test_df.index, name="signal")
        rl_signals.to_csv(RL_SIGNALS_PATH, index=False)
    else:
        rl_signals = None

    return ml_signals, rl_signals

# =============================
# MAIN EVALUATION
# =============================

def run_evaluation():

    print("\n===== ÉVALUATION FINALE 2024 =====\n")

    test_df = load_data()

    # RANDOM
    random_signals = generate_random_signals(test_df)
    random_bt = backtest(test_df, random_signals)
    random_metrics = compute_metrics(random_bt)

    # RULE
    rule_signals = generate_rule_signals(test_df)
    rule_bt = backtest(test_df, rule_signals)
    rule_metrics = compute_metrics(rule_bt)

    # ML / RL signals
    ml_signals, rl_signals = generate_signals_from_models(test_df)
    if ml_signals is None and ML_SIGNALS_PATH.exists():
        ml_signals = load_signals(ML_SIGNALS_PATH)
    ml_bt = backtest(test_df, ml_signals)
    ml_metrics = compute_metrics(ml_bt)

    # RL
    if rl_signals is None and RL_SIGNALS_PATH.exists():
        rl_signals = load_signals(RL_SIGNALS_PATH)
    rl_bt = backtest(test_df, rl_signals)
    rl_metrics = compute_metrics(rl_bt)

    results = {
        "Random": random_metrics,
        "Rule": rule_metrics,
        "ML": ml_metrics,
        "RL": rl_metrics
    }

    results_df = pd.DataFrame(results).T

    print(results_df)

    # Sauvegarde
    results_df.to_csv(PROJECT_ROOT / "data/final_evaluation_2024.csv")

    print("\nRésultats sauvegardés dans final_evaluation_2024.csv")


    # À ajouter dans run_evaluation() après results_df

    # Identifie le meilleur modèle
    best_model = results_df['sharpe'].idxmax()
    print(f"\n🏆 MEILLEUR MODÈLE : {best_model}")
    print(f"   Sharpe: {results_df.loc[best_model, 'sharpe']:.4f}")
    print(f"   Profit: {results_df.loc[best_model, 'cumulative_profit']:.4f}")

    # Validation robustesse
    is_valid = (
        results_df.loc[best_model, 'sharpe'] > 0.5 and
        results_df.loc[best_model, 'max_drawdown'] > -0.10
    )
    print(f"\n{'✓' if is_valid else '✗'} Modèle valide (Sharpe>0.5, DD>-10%): {is_valid}")

    return results_df, best_model, is_valid


if __name__ == "__main__":
    run_evaluation()

