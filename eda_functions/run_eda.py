"""
Script d'exécution de l'analyse exploratoire pour les 3 années
"""

import os
from pathlib import Path
from eda_analyse import (
    load_data,
    compute_returns,
    descriptive_stats,
    test_stationarity,
    hourly_patterns,
    generate_text_report,
    plot_price_evolution,
    plot_return_distribution,
    plot_volatility,
    plot_acf_pacf,
    plot_hourly_patterns
)


def main():
    """Exécute l'EDA pour les 3 années"""
    
    # Chemins
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed"
    output_dir = project_root / "data" / "processed" / "eda_reports"
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 80)
    print("ANALYSE EXPLORATOIRE DES DONNÉES (EDA)")
    print("=" * 80 + "\n")
    
    # Boucle sur les 3 années
    for year in [2022, 2023, 2024]:
        print(f"\n{'='*80}")
        print(f"TRAITEMENT ANNÉE {year}")
        print(f"{'='*80}")
        
        # Charger
        input_file = data_dir / f"DAT_MT_GBPUSD_M15_{year}_clean.csv"
        print(f" Chargement : {input_file}")
        df = load_data(input_file)
        
        # Calculer rendements
        df = compute_returns(df)
        
        # Statistiques
        print(f" Calcul des statistiques...")
        stats = descriptive_stats(df)
        
        # Tests de stationnarité
        print(f"🔬 Tests ADF...")
        adf_price = test_stationarity(df['close'], 'Prix')
        adf_return = test_stationarity(df['return'], 'Rendements')
        
        # Patterns horaires
        print(f" Analyse patterns horaires...")
        hourly = hourly_patterns(df)
        
        # Rapport textuel
        print(f" Génération rapport textuel...")
        report = generate_text_report(year, stats, adf_price, adf_return, hourly)
        
        # Sauvegarder rapport
        report_path = output_dir / f"eda_report_{year}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f" Rapport sauvegardé : {report_path}")
        
        # Graphiques
        print(f" Génération des graphiques...")
        
        plot_price_evolution(df, year, output_dir)
        print(f" Évolution du prix")
        
        plot_return_distribution(df, year, output_dir)
        print(f" Distribution des rendements")
        
        plot_volatility(df, year, output_dir)
        print(f" Volatilité")
        
        plot_acf_pacf(df, year, output_dir)
        print(f" ACF/PACF")
        
        plot_hourly_patterns(hourly, year, output_dir)
        print(f" Patterns horaires")
        
        print(f"\n Année {year} terminée")
    
    print(f"\n{'='*80}")
    print(" ANALYSE EXPLORATOIRE COMPLÈTE POUR LES 3 ANNÉES")
    print(f"{'='*80}")
    print(f"\n Résultats dans : {output_dir}")
    print(f"  → 3 rapports texte (.txt)")
    print(f"  → 15 graphiques (.png)")


if __name__ == "__main__":
    main()