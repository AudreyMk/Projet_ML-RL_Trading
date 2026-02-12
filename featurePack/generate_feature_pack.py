import numpy as np
import pandas as pd

# =========================================================
# 1️⃣ PRÉPARATION
# =========================================================
def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

# =========================================================
# 2️⃣ FEATURES COURT TERME
# =========================================================
def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    # Log-return pour stabiliser variance et réduire outliers
    for lag in [1, 2, 3, 4, 5, 10]:
        df[f'return_{lag}'] = np.log(df['close'] / df['close'].shift(lag))
    # Winsorisation pour gérer queues lourdes
    for col in [f'return_{lag}' for lag in [1,2,3,4,5,10]]:
        df[col] = df[col].clip(lower=df[col].quantile(0.01), upper=df[col].quantile(0.99))
    return df

def add_ema_features(df: pd.DataFrame) -> pd.DataFrame:
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['ema_diff'] = df['ema_20'] - df['ema_50']
    # Slope EMA50 lissé pour réduire bruit
    df['slope_ema50'] = df['ema_50'].diff().rolling(3).mean()
    df['distance_to_ema200'] = df['close'] - df['ema_200']
    return df

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    return df

def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    # Rolling std sur plusieurs fenêtres
    df['rolling_std_20'] = df['close'].pct_change().rolling(20).std()
    df['rolling_std_60'] = df['close'].pct_change().rolling(60).std()
    df['rolling_std_100'] = df['close'].pct_change().rolling(100).std()
    df['rolling_std_120'] = df['close'].pct_change().rolling(120).std()
    
    # Range et bougie
    df['range_15m'] = df['high'] - df['low']
    df['body'] = df['close'] - df['open']
    df['upper_wick'] = df['high'] - df[['close','open']].max(axis=1)
    df['lower_wick'] = df[['close','open']].min(axis=1) - df['low']
    
    # Volatility ratio
    df['volatility_ratio'] = df['rolling_std_20'] / (df['rolling_std_100'] + 1e-10) if 'rolling_std_100' in df.columns else np.nan
    return df

# =========================================================
# 3️⃣ FEATURES CONTEXTE & RÉGIME
# =========================================================
def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(period).mean()
    # Normalisation ATR par close pour stationnarité
    df['atr_14_ratio'] = df['atr_14'] / df['close']
    return df

def add_directional_strength(df: pd.DataFrame) -> pd.DataFrame:
    # ADX 14
    df['up_move'] = df['high'].diff()
    df['down_move'] = -df['low'].diff()
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0.0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0.0)
    tr = pd.concat([df['high'] - df['low'],
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * (df['plus_dm'].rolling(14).sum() / (atr14 + 1e-10))
    minus_di = 100 * (df['minus_dm'].rolling(14).sum() / (atr14 + 1e-10))
    df['adx_14'] = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)).rolling(14).mean()
    
    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Clean temp columns
    df.drop(columns=['up_move','down_move','plus_dm','minus_dm'], inplace=True)
    return df

# =========================================================
# 4️⃣ FEATURES TEMPORELLES
# =========================================================
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    # Sessions Forex
    df['is_asia'] = df['hour'].between(0, 8).astype(int)
    df['is_london'] = df['hour'].between(7, 15).astype(int)
    df['is_newyork'] = df['hour'].between(13, 21).astype(int)
    # Heure volatile
    df['is_volatile_8h'] = (df['hour'] == 8).astype(int)
    return df

# =========================================================
# 5️⃣ TARGET
# =========================================================
def add_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    df['future_return'] = np.log(df['close'].shift(-horizon) / df['close'])
    df['target_direction'] = np.sign(df['future_return'])
    return df

# =========================================================
# 6️⃣ CLEANING
# =========================================================
def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    return df

# =========================================================
# 7️⃣ BUILD FEATURE PACK V2 COMPLET
# =========================================================
def build_feature_pack_v2(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    df = prepare_dataframe(df)
    df = add_return_features(df)
    df = add_ema_features(df)
    df = add_rsi(df)
    df = add_volatility_features(df)
    df = add_atr(df)
    df = add_directional_strength(df)
    df = add_time_features(df)
    df = add_target(df, horizon=horizon)
    df = clean_features(df)
    
    # Regime de volatilité basé sur rolling_std_20 quantiles
    vol_quantiles = df['rolling_std_20'].quantile([0.33, 0.66])
    df['regime_volatility'] = pd.cut(df['rolling_std_20'], bins=[-np.inf, vol_quantiles[0.33], vol_quantiles[0.66], np.inf],
                                     labels=['low','medium','high'])
    
    return df
