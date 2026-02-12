"""
Configuration centralisée du projet
"""

from pathlib import Path

# Chemins du projet
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = PROJECT_ROOT / "models_registry"

# Paramètres des données
M1_INTERVAL = "1T"    # 1 minute
M15_INTERVAL = "15T"  # 15 minutes

# Split temporel (OBLIGATOIRE - pas de shuffle)
TRAIN_YEAR = 2022
VAL_YEAR = 2023
TEST_YEAR = 2024

# Trading
TRANSACTION_COST = 0.0001  # 1 pip spread
INITIAL_CAPITAL = 10000.0
SLIPPAGE = 0.00005

# Features (21 au total)
SHORT_TERM_FEATURES = [
    'return_1', 'return_4',
    'ema_20', 'ema_50', 'ema_diff',
    'rsi_14', 'rolling_std_20',
    'range_15m', 'body', 'upper_wick', 'lower_wick'
]

REGIME_FEATURES = [
    'ema_200', 'distance_to_ema200', 'slope_ema50',
    'atr_14', 'rolling_std_100', 'volatility_ratio',
    'adx_14', 'macd', 'macd_signal'
]

ALL_FEATURES = SHORT_TERM_FEATURES + REGIME_FEATURES

# ML & RL
ML_MODELS = ['logistic', 'random_forest', 'xgboost', 'lightgbm']
RL_ACTIONS = ['HOLD', 'BUY', 'SELL']
RL_GAMMA = 0.99

# Random seed
RANDOM_SEED = 42