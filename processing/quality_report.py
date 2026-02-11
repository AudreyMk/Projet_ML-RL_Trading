from datetime import datetime

from processed_data_M15 import clean_m15_data


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

    return "\n".join(lines)


def generate_quality_report(
    file_path,
    output_csv_path,
    output_report_path
):
    df_m15_cleaned, report = clean_m15_data(file_path)
    df_m15_cleaned.to_csv(output_csv_path)
    report_text = format_quality_report(report, file_path)
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    return report_text


if __name__ == "__main__":
    file_path = "data/processed_data/DAT_MT_GBPUSD_M1_2022_processed.csv"
    output_csv_path = "data/processed_data/DAT_MT_GBPUSD_M15_2022_clean.csv"
    output_report_path = "data/processed_data/DAT_MT_GBPUSD_M15_2022_quality_report.txt"

    report_text = generate_quality_report(
        file_path,
        output_csv_path,
        output_report_path
    )
    print(report_text)
