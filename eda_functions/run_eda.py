"""Script d'execution EDA pour les 3 annees"""

from pathlib import Path
from eda_analyse import (
    load_data,
    generate_eda_report
)
import os


def main():
    """Exécute l'EDA pour les 3 annees"""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed_data"
    output_dir = project_root / "data" / "processed" / "eda_reports"
    output_dir.mkdir(exist_ok=True)

    for year in [2022, 2023, 2024]:
        input_file = data_dir / f"DAT_MT_GBPUSD_M15_{year}_processed_clean.csv"
        df = load_data(input_file)
        generate_eda_report(df, year, output_dir)


if __name__ == "__main__":
    main()