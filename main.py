"""Entrypoint for the Abyssal Echo prototype."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from abyssal_echo.data_loader import get_default_paths
from abyssal_echo.main import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Abyssal Echo acoustic reconstruction prototype")
    parser.add_argument("--data-dir", default="data", help="Directory containing the CSV inputs")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated outputs")
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch the Streamlit dashboard after the batch pipeline completes",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = get_default_paths(Path(args.data_dir))
    outputs = run_pipeline(paths=paths, output_dir=args.output_dir)

    latest_speed = outputs.speed_summary.iloc[-1]["Submarine_Speed_knots"]
    latest_depth = outputs.trajectory.iloc[-1]["Estimated_Depth_m"]
    latest_projection = outputs.predicted_path.iloc[-1]
    threat = outputs.threat_summary.iloc[0]

    print("Abyssal Echo pipeline complete.")
    print(f"Cleaned ping rows: {len(outputs.cleaned_pings):,}")
    print(f"Estimated sensor drifts saved to: {Path(args.output_dir) / 'sensor_clock_drift.csv'}")
    print(f"Latest submarine speed: {latest_speed:.2f} knots")
    print(f"Latest reconstructed depth: {latest_depth:.1f} m")
    print(f"Trajectory file: {Path(args.output_dir) / 'reconstructed_trajectory.csv'}")
    print(
        "Projected terminal waypoint: "
        f"({latest_projection['Predicted_X']:.1f}, {latest_projection['Predicted_Y']:.1f}, {latest_projection['Predicted_Z']:.1f})"
    )
    print(f"Current threat level: {threat['Current_Threat_Level']} ({threat['Peak_Threat_Score']:.2f} peak)")
    print(f"Predicted path file: {Path(args.output_dir) / 'predicted_future_path.csv'}")

    if args.dashboard:
        dashboard_path = Path(__file__).resolve().parent / "abyssal_echo" / "dashboard.py"
        command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_path),
            "--",
            args.output_dir,
        ]
        print("Launching dashboard...")
        return subprocess.call(command)

    print("To explore the dashboard, run:")
    print(f"streamlit run {Path('abyssal_echo/dashboard.py')} -- {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
