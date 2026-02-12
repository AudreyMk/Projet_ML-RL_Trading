"""Train simple ML baselines (T07):

Loads features for 2022/2023/2024, trains LogisticRegression and RandomForest
on 2022, validates on 2023, tests on 2024. Produces financial metrics via
the existing backtest and metrics utilities, saves models and a CSV report.
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import json

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.backtest import run_backtest
from evaluation.metrics import compute_metrics


def load_features(year):
    p = Path(__file__).parent.parent / 'data' / 'features' / f'DAT_MT_GBPUSD_M15_{year}_features.csv'
    df = pd.read_csv(p, parse_dates=['timestamp']).set_index('timestamp')
    return df


def prepare_xy(df):
    # target: 1 if future_return > 0 else 0
    y = (df['future_return'] > 0).astype(int)
    # features: numeric columns except future_return and target
    X = df.select_dtypes(include=[np.number]).copy()
    for drop in ['future_return', 'target_direction']:
        if drop in X.columns:
            X = X.drop(columns=[drop])
    return X, y


def to_signal(pred):
    # map binary prediction to trading signal: 1 -> 1 (long), 0 -> -1 (short)
    return pd.Series(np.where(pred == 1, 1, -1))


def main():
    project_root = PROJECT_ROOT
    years = {'train': 2022, 'val': 2023, 'test': 2024}

    df_train = load_features(years['train'])
    df_val = load_features(years['val'])
    df_test = load_features(years['test'])

    # ----- Data preparation: feature selection and scaler -----
    X_train, y_train = prepare_xy(df_train)
    X_val, y_val = prepare_xy(df_val)
    X_test, y_test = prepare_xy(df_test)

    # Drop warm-up rows if any (rows where any feature is NaN)
    # Keep index alignment for backtest by dropping same rows in df_* as well
    def align_dropna(X, df):
        mask = X.dropna().index
        return X.loc[mask], df.loc[mask]

    X_train, df_train = align_dropna(X_train, df_train)
    X_val, df_val = align_dropna(X_val, df_val)
    X_test, df_test = align_dropna(X_test, df_test)

    # Feature list to keep (numeric columns from X_train)
    feature_list = X_train.columns.tolist()

    # Standardize features using training set only
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=feature_list)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=feature_list)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=feature_list)

    models = {
        'logistic': LogisticRegression(max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    }

    results = []
    models_dir = project_root / 'models_registry' / 'v1'
    models_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        print(f'Training {name}...')
        model.fit(X_train_scaled, y_train.loc[X_train_scaled.index])

        # validation / test signals
        for split, (df_split, X_split) in (('val', (df_val, X_val_scaled)), ('test', (df_test, X_test_scaled))):
            pred = model.predict(X_split)
            sig = to_signal(pred)
            sig.index = df_split.index
            bt = run_backtest(df_split, sig)
            metrics = compute_metrics(bt)
            results.append({
                'model': name,
                'split': split,
                'cumulative_profit': metrics['cumulative_profit'],
                'max_drawdown': metrics['max_drawdown'],
                'sharpe': metrics['sharpe'],
                'profit_factor': metrics['profit_factor']
            })

        # save model
        outp = models_dir / f'{name}_v1.pkl'
        joblib.dump(model, outp)
        print(f'Saved model to {outp}')

    # save scaler and metadata
    meta = {
        'feature_list': feature_list,
        'scaler': 'StandardScaler',
        'model_version': 'v1'
    }
    joblib.dump(scaler, models_dir / 'scaler_v1.pkl')
    with open(models_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f'Saved scaler and metadata to {models_dir}')

    # save results CSV
    out_csv = project_root / 'data' / 'ml_baseline.csv'
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f'Saved ML baseline results to {out_csv}')


if __name__ == '__main__':
    main()
