from pathlib import Path
import pandas as pd

from .strategies import random_strategy, rule_strategy, buy_and_hold
from .backtest import run_backtest
from .metrics import compute_metrics


def evaluate_year(year):

    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "features" / f"DAT_MT_GBPUSD_M15_{year}_features.csv"

    df = pd.read_csv(data_path, parse_dates=["timestamp"]).set_index("timestamp")

    # Vérification des colonnes requises pour l'évaluation
    required_cols = ["close", "future_return", "ema_20", "rsi_14"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Fichier de features {data_path} manquant les colonnes: {missing}")

    results = {}

    strategies = {
        "random": random_strategy(df),
        "rule_based": rule_strategy(df),
        "buy_hold": buy_and_hold(df)
    }

    for name, signal in strategies.items():
        bt = run_backtest(df, signal)
        metrics = compute_metrics(bt)
        results[name] = metrics

    return results


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    out_lines = []
    for year in [2022, 2023, 2024]:
        res = evaluate_year(year)
        out_lines.append(f"\n===== {year} =====")
        for strat, metrics in res.items():
            out_lines.append(f"\nStrategy: {strat}")
            for k, v in metrics.items():
                out_lines.append(f"{k}: {round(v, 4)}")

    # Affichage console
    print("\n".join(out_lines))

    # Sauvegarde dans data/baseline.txt
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_file = data_dir / "baseline.txt"
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(out_lines))

    print(f"Saved baseline to: {out_file}")
