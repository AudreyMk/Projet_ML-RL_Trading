from pathlib import Path
import pandas as pd

from generate_feature_pack import build_feature_pack_v2


def run_for_years(years):
    project_root = Path(__file__).parent.parent
    input_dir = project_root / "data" / "processed_data"
    output_dir = project_root / "data" / "features"
    output_dir.mkdir(parents=True, exist_ok=True)

    for year in years:
        input_file = input_dir / f"DAT_MT_GBPUSD_M15_{year}_processed.csv"
        df = pd.read_csv(input_file, parse_dates=['timestamp']).set_index('timestamp')
        feature_df = build_feature_pack_v2(df)
        output_file = output_dir / f"DAT_MT_GBPUSD_M15_{year}_features.csv"
        feature_df.to_csv(output_file)


if __name__ == "__main__":
    run_for_years([2022, 2023, 2024])
