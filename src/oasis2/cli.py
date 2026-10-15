"""Command-line entry point: `python -m oasis2 --demo-csv ... --regional-volumes-csv ...`."""
import argparse
from pathlib import Path

import pandas as pd

from oasis2.config import get_data_dir, get_outputs_dir
from oasis2.data.ingest import find_demographic_csv, load_demographic_csv
from oasis2.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OASIS-2 neurodegeneration pipeline.")
    parser.add_argument("--demo-csv", type=Path, default=None, help="Path to demographics CSV or Excel file.")
    parser.add_argument("--regional-volumes-csv", type=Path, default=None, help="Path to regional volumes CSV.")
    parser.add_argument("--outputs-dir", type=Path, default=None, help="Output directory.")
    parser.add_argument("--dl-epochs", type=int, default=50, help="Number of training epochs.")
    args = parser.parse_args()

    demo_path = args.demo_csv
    if demo_path is None:
        demo_path = find_demographic_csv(get_data_dir("raw"))
    if demo_path is None or not Path(demo_path).exists():
        raise FileNotFoundError("Demographics file not found. Provide --demo-csv.")

    demo_df = load_demographic_csv(demo_path)

    vol_path = args.regional_volumes_csv
    if vol_path is None:
        vol_path = get_outputs_dir() / "regional_volumes.csv"
    if not Path(vol_path).exists():
        raise FileNotFoundError(f"Regional volumes file not found at {vol_path}. Provide --regional-volumes-csv.")

    regional_volumes = pd.read_csv(vol_path)
    run_pipeline(demo_df, regional_volumes, outputs_dir=args.outputs_dir, dl_epochs=args.dl_epochs)


if __name__ == "__main__":
    main()
