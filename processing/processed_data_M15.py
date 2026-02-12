import pandas as pd
import numpy as np
from import_control_data import check_time_irregularities

## Etape 2: resampling en M15


def analyze_gaps(df_m15):
    df = df_m15.copy()

    # 1️⃣ Calcul du gap (entre close t-1 et open t)
    df['gap'] = abs(df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df = df.dropna()

    print("===== ANALYSE DES GAPS =====\n")

    # 2️⃣ Statistiques descriptives
    desc = df['gap'].describe(percentiles=[0.95, 0.99])
    print(desc)
    print("\n")

    # 3️⃣ Seuils statistiques
    mean_gap = df['gap'].mean()
    std_gap = df['gap'].std()

    threshold_sigma = mean_gap + 5 * std_gap
    threshold_percentile = df['gap'].quantile(0.99)

    threshold = max(threshold_sigma, threshold_percentile)

    print(f"Seuil sigma (mean + 5σ) : {threshold_sigma:.6f}")
    print(f"Seuil percentile 99%   : {threshold_percentile:.6f}")
    print(f"Seuil retenu           : {threshold:.6f}\n")

    # 4️⃣ Fréquence des gaps extrêmes
    extreme_mask = df['gap'] > threshold
    extreme_gaps = df[extreme_mask]
    freq = len(extreme_gaps) / len(df)

    print(f"Nombre de gaps extrêmes : {len(extreme_gaps)}")
    print(f"Fréquence               : {freq*100:.4f}%\n")

    # 5️⃣ Décision automatique + application sur df
    if freq < 0.001:
        decision = "SUPPRIMER (probables anomalies rares)"
        df = df[~extreme_mask]  # supprime les lignes extrêmes
    elif freq < 0.01:
        decision = "CLIPPER (événements rares mais réels)"
        # appliquer clipping sur les rendements
        df['return_1'] = np.log(df['close'] / df['close'].shift(1))
        df['return_1'] = np.clip(df['return_1'], -threshold, threshold)
    else:
        decision = "GARDER (régime de marché normal)"
        # rien à faire, on garde tout

    print(f"✅ Décision appliquée : {decision}")

    # Préparer le rapport
    report = {
        "threshold": threshold,
        "frequency": freq,
        "decision": decision,
        "extreme_gaps": extreme_gaps
    }

    return df, report




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

    #df_resampled.to_csv("data/processed_data/DAT_MT_GBPUSD_M15_2022_M15.csv")  # Sauvegarde du DataFrame resamplé
    return df_resampled



def analyse_m15_data(file_path) -> pd.DataFrame:
    """
    Nettoie les données M15 en supprimant les bougies incomplètes, les prix invalides et les incohérences OHLC.
    """
    df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
    df_m15 = resampling(df) 

    n_rows_M15_before = len(df_m15)
    n_nan_before = df_m15.isna().sum()

    # supprimer les nan
    df_m15.dropna(inplace=True)
    n_rows_M15_after = len(df_m15)
    n_nan_after = df_m15.isna().sum()


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

    ### traitement des données M15 et génération du rapport de qualité
    ## suppression des bougies M15 incomplètes
    bougies_completes = m15_count[m15_count == 15].index
    df_m15_clean = df_m15[df_m15.index.isin(bougies_completes)]
    n_rows_M15_clean = len(df_m15_clean)
    

    #print(f"\nAprès filtre bougies complètes (15 min) : {len(df_m15)} bougies")

    # 2. Supprimer les éventuels NaN restants
    df_m15_clean.dropna(inplace=True)
    n_nan_M15_clean = df_m15_clean.isna().sum()

    # analyse des gap annormaux
    df_m15_clean, report = analyze_gaps(df_m15_clean)


    
    return df_m15_clean, {
        'n_rows_M15_before': n_rows_M15_before,
        'n_nan_before': n_nan_before,
        'n_rows_M15_after': n_rows_M15_after,
        'n_nan_after': n_nan_after,
        'n_nan_M15_clean': n_nan_M15_clean,
        'n_invalid_prices': len(invalid_prices),
        'n_invalid_ohlc': invalid_ohlc,
        'periode': periode,
        'bougies_m15_count': bougies_m15_count,
        'bougies_m15_inf': bougies_m15_inf, 
        'bougies_m15_vide': bougies_m15_vide, 
        'gap_analysis_report': report 
        }