from datetime import datetime
import os
from processed_data_M15 import analyse_m15_data


def format_quality_report(report, source_path):
    """Retourne un texte formate pour le rapport de qualite."""
    lines = []
    lines.append("RAPPORT DE QUALITE - DONNEES M15")
    lines.append("Source: {}".format(source_path))
    lines.append("Date de generation: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    lines.append("")
    lines.append("Resume:")
    lines.append("- Lignes M15 avant nettoyage: {}".format(report.get("n_rows_M15_before")))
    lines.append("- Valeurs manquantes avant: {}".format(report.get("n_nan_before")))
    lines.append("- Lignes M15 apres nettoyage: {}".format(report.get("n_rows_M15_after")))
    lines.append("- Valeurs manquantes apres: {}".format(report.get("n_nan_after")))
    lines.append("- Prix invalides: {}".format(report.get("n_invalid_prices")))
    lines.append("- OHLC incoherents: {}".format(report.get("n_invalid_ohlc")))
    lines.append("- Periode: {} -> {}".format(
        report.get("periode", [None, None])[0],
        report.get("periode", [None, None])[1]
    ))
    lines.append("")
    lines.append("Couverture M15:")
    lines.append("- Bougies completes (15 minutes): {}".format(report.get("bougies_m15_count")))
    lines.append("- Bougies incompletes (< 15 minutes): {}".format(report.get("bougies_m15_inf")))
    lines.append("- Bougies vides: {}".format(report.get("bougies_m15_vide")))
    lines.append("")
    lines.append("Analyse des gaps:")
    gap_report = report.get("gap_analysis_report", {})
    lines.append("- Seuil retenu: {}".format(gap_report.get("threshold")))
    lines.append("- Frequence: {}".format(gap_report.get("frequency")))
    lines.append("- Decision: {}".format(gap_report.get("decision")))
    extreme_gaps = gap_report.get("extreme_gaps")
    if extreme_gaps is not None:
        lines.append("- Gaps extremes: {}".format(len(extreme_gaps)))

    return "\n".join(lines)


def generate_quality_report(file_path, output_csv_path, output_report_path):
    df_m15, report = analyse_m15_data(file_path)
    df_m15.reset_index().to_csv(output_csv_path, index=False)
    report_text = format_quality_report(report, file_path)
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    return report_text


if __name__ == "__main__":
    # Boucle pour traiter les 3 années
    for year in [2022, 2023, 2024]:
        print(f"\n{'='*60}")
        print(f"GÉNÉRATION RAPPORT QUALITÉ M15 - {year}")
        print(f"{'='*60}")
        
        # Chemins des fichiers
        file_path = os.path.join("data", "processed", f"DAT_MT_GBPUSD_M1_{year}.csv")
        output_csv_path = os.path.join("data", "processed", f"DAT_MT_GBPUSD_M15_{year}_clean.csv")
        output_report_path = os.path.join("data", "processed", f"quality_report_{year}.txt")
        
        # Génération
        report_text = generate_quality_report(file_path, output_csv_path, output_report_path)
        
        print(f" CSV M15 sauvegardé : {output_csv_path}")
        print(f" Rapport sauvegardé : {output_report_path}")
        print(f"\n{report_text}")
    
    print(f"\n{'='*60}")
    print(" TOUS LES RAPPORTS GÉNÉRÉS")
    print(f"{'='*60}")