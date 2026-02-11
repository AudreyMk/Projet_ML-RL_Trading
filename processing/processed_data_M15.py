import pandas as pd

from import_control_data import check_time_irregularities

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



def clean_m15_data(file_path) -> pd.DataFrame:
    """
    Nettoie les données M15 en supprimant les bougies incomplètes, les prix invalides et les incohérences OHLC.
    """
    df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
    df_m15 = resampling(df)

    n_rows_M15_before = len(df_m15)
    n_nan_before = df_m15.isna().sum()

    # supprimer les nan
    df_m15.dropna(inplace=True)

    # vérification prix <=0
    invalid_prices = df_m15[(df_m15 <= 0).any(axis=1)]

    # vérification incohérences OHLC
    invalid_ohlc = len(check_time_irregularities(df_m15))
    periode = [df_m15.index.min(), df_m15.index.max()]

    ## pour compter les bougies M15 avec moins de 15 minutes, on peut compter le nombre de lignes par période de 15 minutes dans le DataFrame original
    m15_count = df.resample('15min').size()

    bougies_m15_count = (m15_count == 15).sum() # nombre de bougies avec exactement 15 minutes
    bougies_m15_inf = (m15_count < 15).sum() # nombre de bougies avec moins de 15 minutes
    bougies_m15_vide = (m15_count == 0).sum() # nombre de bougies vides
    
    return df_m15, {
        'n_rows_M15_before': n_rows_M15_before,
        'n_nan_before': n_nan_before,
        'n_nan_after': df_m15.isna().sum(),
        'n_invalid_prices': len(invalid_prices),
        'n_invalid_ohlc': invalid_ohlc,
        'periode': periode,
        'bougies_m15_count': bougies_m15_count,
        'bougies_m15_inf': bougies_m15_inf,
        'bougies_m15_vide': bougies_m15_vide
    }

file_path = "data/processed_data/DAT_MT_GBPUSD_M1_2022_processed.csv"
df_m15_cleaned, m15_report = clean_m15_data(file_path)
df_m15_cleaned.to_csv("data/processed_data/DAT_MT_GBPUSD_M15_2022_clean.csv")  # Sauvegarde du DataFrame M15 nettoyé
print(m15_report)