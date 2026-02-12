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
    file_path_list= ["DAT_MT_GBPUSD_M1_2022_processed", "DAT_MT_GBPUSD_M1_2023_processed", "DAT_MT_GBPUSD_M1_2024_processed"]
    for pat in file_path_list:
        print()
        file_path = "data/processed_data/" + pat + ".csv"
        output_base = pat.replace("_M1_", "_M15_")
        output_csv_path = "data/processed_data/" + output_base + ".csv"
        output_report_path = "data/processed_data/" + output_base + "_quality_report.txt"

        report_text = generate_quality_report(
            file_path,
            output_csv_path,
            output_report_path
        )
        print(report_text)