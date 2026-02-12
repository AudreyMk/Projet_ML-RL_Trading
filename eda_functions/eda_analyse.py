"""
Fonctions d'analyse exploratoire des données
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pathlib import Path


def load_data(filepath):
    """Charge les données M15"""
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    return df


def compute_returns(df):
    """Calcule les rendements"""
    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    return df


def descriptive_stats(df):
    """Statistiques descriptives"""
    stats = {
        'nb_lignes': len(df),
        'periode_debut': df.index.min(),
        'periode_fin': df.index.max(),
        'prix_min': df['close'].min(),
        'prix_max': df['close'].max(),
        'prix_mean': df['close'].mean(),
        'return_mean': df['return'].mean(),
        'return_std': df['return'].std(),
        'return_skew': df['return'].skew(),
        'return_kurtosis': df['return'].kurtosis(),
        'return_min': df['return'].min(),
        'return_max': df['return'].max()
    }
    return stats


def test_stationarity(series, name=''):
    """Test ADF de stationnarité"""
    result = adfuller(series.dropna())
    
    test = {
        'series': name,
        'adf_statistic': result[0],
        'p_value': result[1],
        'stationnaire': result[1] < 0.05,
        'critical_1%': result[4]['1%'],
        'critical_5%': result[4]['5%'],
        'critical_10%': result[4]['10%']
    }
    return test


def hourly_patterns(df):
    """Analyse des patterns horaires"""
    df = df.copy()
    df['hour'] = df.index.hour
    
    hourly = df.groupby('hour')['return'].agg(['mean', 'std', 'count']).reset_index()
    hourly.columns = ['hour', 'mean_return', 'volatility', 'count']
    
    return hourly


def generate_text_report(year, stats, adf_price, adf_return, hourly):
    """Génère un rapport textuel"""
    lines = []
    lines.append("=" * 80)
    lines.append(f"RAPPORT ANALYSE EXPLORATOIRE - {year}")
    lines.append("=" * 80)
    lines.append("")
    
    # Stats de base
    lines.append("STATISTIQUES DE BASE")
    lines.append(f"  Nombre de bougies : {stats['nb_lignes']:,}")
    lines.append(f"  Période           : {stats['periode_debut']} → {stats['periode_fin']}")
    lines.append(f"  Prix min          : {stats['prix_min']:.5f}")
    lines.append(f"  Prix max          : {stats['prix_max']:.5f}")
    lines.append(f"  Prix moyen        : {stats['prix_mean']:.5f}")
    lines.append("")
    
    # Rendements
    lines.append("RENDEMENTS")
    lines.append(f"  Rendement moyen   : {stats['return_mean']:.6f}")
    lines.append(f"  Volatilité        : {stats['return_std']:.6f}")
    lines.append(f"  Skewness          : {stats['return_skew']:.4f}")
    lines.append(f"  Kurtosis          : {stats['return_kurtosis']:.4f}")
    lines.append(f"  Min               : {stats['return_min']:.6f}")
    lines.append(f"  Max               : {stats['return_max']:.6f}")
    lines.append("")
    
    # Stationnarité
    lines.append("TEST DE STATIONNARITÉ (ADF)")
    lines.append(f"  Prix:")
    lines.append(f"    ADF Statistic   : {adf_price['adf_statistic']:.4f}")
    lines.append(f"    p-value         : {adf_price['p_value']:.4f}")
    lines.append(f"    Stationnaire    : {'OUI' if adf_price['stationnaire'] else 'NON'}")
    lines.append("")
    lines.append(f"  Rendements:")
    lines.append(f"    ADF Statistic   : {adf_return['adf_statistic']:.4f}")
    lines.append(f"    p-value         : {adf_return['p_value']:.4f}")
    lines.append(f"    Stationnaire    : {'OUI' if adf_return['stationnaire'] else 'NON'}")
    lines.append("")
    
    # Patterns horaires
    lines.append("PATTERNS HORAIRES")
    max_vol_hour = hourly.loc[hourly['volatility'].idxmax()]
    min_vol_hour = hourly.loc[hourly['volatility'].idxmin()]
    best_return_hour = hourly.loc[hourly['mean_return'].idxmax()]
    
    lines.append(f"  Heure la plus volatile : {int(max_vol_hour['hour'])}h (std={max_vol_hour['volatility']:.6f})")
    lines.append(f"  Heure la moins volatile: {int(min_vol_hour['hour'])}h (std={min_vol_hour['volatility']:.6f})")
    lines.append(f"  Meilleur rendement     : {int(best_return_hour['hour'])}h (mean={best_return_hour['mean_return']:.6f})")
    lines.append("")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


def plot_price_evolution(df, year, output_dir):
    """Graphique évolution du prix"""
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['close'], linewidth=0.8, color='steelblue')
    plt.title(f'GBP/USD M15 - {year}', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Prix')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = Path(output_dir) / f"price_evolution_{year}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


def plot_return_distribution(df, year, output_dir):
    """Distribution des rendements"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogramme
    axes[0].hist(df['return'].dropna(), bins=100, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_title(f'{year} - Distribution des Rendements', fontweight='bold')
    axes[0].set_xlabel('Rendement')
    axes[0].set_ylabel('Fréquence')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=1)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(df['return'].dropna(), dist="norm", plot=axes[1])
    axes[1].set_title(f'{year} - Q-Q Plot', fontweight='bold')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / f"return_distribution_{year}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


def plot_volatility(df, year, output_dir):
    """Volatilité rolling"""
    df = df.copy()
    df['volatility_20'] = df['return'].rolling(20).std()
    
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['volatility_20'], linewidth=0.8, color='coral')
    plt.title(f'{year} - Volatilité Rolling (20 périodes)', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Volatilité')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = Path(output_dir) / f"volatility_{year}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


def plot_acf_pacf(df, year, output_dir):
    """ACF et PACF"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    plot_acf(df['return'].dropna(), lags=50, ax=axes[0], color='steelblue')
    axes[0].set_title(f'{year} - Autocorrélation (ACF)', fontweight='bold')
    
    plot_pacf(df['return'].dropna(), lags=50, ax=axes[1], color='coral')
    axes[1].set_title(f'{year} - Autocorrélation Partielle (PACF)', fontweight='bold')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / f"acf_pacf_{year}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


def plot_hourly_patterns(hourly, year, output_dir):
    """Patterns horaires"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Rendement moyen
    axes[0].bar(hourly['hour'], hourly['mean_return'], color='steelblue', alpha=0.7)
    axes[0].set_title(f'{year} - Rendement Moyen par Heure', fontweight='bold')
    axes[0].set_xlabel('Heure')
    axes[0].set_ylabel('Rendement Moyen')
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Volatilité
    axes[1].bar(hourly['hour'], hourly['volatility'], color='coral', alpha=0.7)
    axes[1].set_title(f'{year} - Volatilité par Heure', fontweight='bold')
    axes[1].set_xlabel('Heure')
    axes[1].set_ylabel('Volatilité')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / f"hourly_patterns_{year}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(output_path)


def generate_eda_report(df, year, output_dir):
    """
    Genere un rapport complet (texte + figure unique avec subplots).
    Retourne le texte du rapport et le chemin de la figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = compute_returns(df)
    stats = descriptive_stats(df)
    adf_price = test_stationarity(df['close'], 'Prix')
    adf_return = test_stationarity(df['return'], 'Rendements')
    hourly = hourly_patterns(df)

    report_text = generate_text_report(year, stats, adf_price, adf_return, hourly)
    report_path = output_dir / f"eda_report_{year}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    fig, axes = plt.subplots(4, 2, figsize=(16, 18))

    # Prix
    axes[0, 0].plot(df.index, df['close'], linewidth=0.8, color='steelblue')
    axes[0, 0].set_title(f'GBP/USD M15 - {year}', fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Prix')
    axes[0, 0].grid(True, alpha=0.3)

    # Volatilite rolling
    df_vol = df['return'].rolling(20).std()
    axes[0, 1].plot(df.index, df_vol, linewidth=0.8, color='coral')
    axes[0, 1].set_title(f'{year} - Volatilite Rolling (20 periodes)', fontweight='bold')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Volatilite')
    axes[0, 1].grid(True, alpha=0.3)

    # Distribution rendements
    axes[1, 0].hist(df['return'].dropna(), bins=100, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1, 0].set_title(f'{year} - Distribution des Rendements', fontweight='bold')
    axes[1, 0].set_xlabel('Rendement')
    axes[1, 0].set_ylabel('Frequence')
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=1)

    # Q-Q plot
    from scipy import stats as scipy_stats
    scipy_stats.probplot(df['return'].dropna(), dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title(f'{year} - Q-Q Plot', fontweight='bold')

    # ACF / PACF
    plot_acf(df['return'].dropna(), lags=50, ax=axes[2, 0], color='steelblue')
    axes[2, 0].set_title(f'{year} - Autocorrelation (ACF)', fontweight='bold')
    plot_pacf(df['return'].dropna(), lags=50, ax=axes[2, 1], color='coral')
    axes[2, 1].set_title(f'{year} - Autocorrelation Partielle (PACF)', fontweight='bold')

    # Patterns horaires
    axes[3, 0].bar(hourly['hour'], hourly['mean_return'], color='steelblue', alpha=0.7)
    axes[3, 0].set_title(f'{year} - Rendement Moyen par Heure', fontweight='bold')
    axes[3, 0].set_xlabel('Heure')
    axes[3, 0].set_ylabel('Rendement Moyen')
    axes[3, 0].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[3, 0].grid(True, alpha=0.3, axis='y')

    axes[3, 1].bar(hourly['hour'], hourly['volatility'], color='coral', alpha=0.7)
    axes[3, 1].set_title(f'{year} - Volatilite par Heure', fontweight='bold')
    axes[3, 1].set_xlabel('Heure')
    axes[3, 1].set_ylabel('Volatilite')
    axes[3, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = output_dir / f"eda_overview_{year}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    return report_text, str(fig_path)