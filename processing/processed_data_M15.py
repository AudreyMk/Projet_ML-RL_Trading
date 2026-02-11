import pandas as pd

## Etape 2: resampling en M15

def resampling(df: pd.DataFrame, freq: str = '15min') -> pd.DataFrame:
    """
    Resample le DataFrame en utilisant la fréquence spécifiée.
    """
    df_resampled = df.resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    })

    df_resampled.to_csv("data/processed_data/DAT_MT_GBPUSD_M15_2022_M15.csv")  # Sauvegarde du DataFrame resamplé
    return df_resampled

file_path = "data/processed_data/DAT_MT_GBPUSD_M1_2022_processed.csv"
df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
df_m15 = resampling(df)
print(df_m15.head())