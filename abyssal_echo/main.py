"""Pipeline orchestration for Abyssal Echo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from abyssal_echo.clock_sync import apply_clock_correction, compute_sensor_drift
from abyssal_echo.data_loader import DatasetPaths, load_datasets, maybe_generate_synthetic_data
from abyssal_echo.doppler_velocity import attach_engine_frequency, solve_submarine_velocity, summarize_speed
from abyssal_echo.echo_filter import flag_primary_signals, keep_primary_signals
from abyssal_echo.future_prediction import (
    compute_interception_window,
    kalman_filter_trajectory,
    project_future_path,
)
from abyssal_echo.sound_speed import enrich_with_sound_speed
from abyssal_echo.tactical_intelligence import (
    attach_prediction_uncertainty,
    detect_anomalies,
    score_threat,
)
from abyssal_echo.triangulation import reconstruct_trajectory


@dataclass(frozen=True)
class PipelineOutputs:
    cleaned_pings: pd.DataFrame
    drift_offsets: pd.DataFrame
    trajectory: pd.DataFrame
    speed_summary: pd.DataFrame
    kalman_track: pd.DataFrame
    predicted_path: pd.DataFrame
    interception_windows: pd.DataFrame
    interception_summary: pd.DataFrame
    anomaly_events: pd.DataFrame
    threat_timeseries: pd.DataFrame
    threat_summary: pd.DataFrame


def run_pipeline(paths: DatasetPaths, output_dir: Path | str = "outputs") -> PipelineOutputs:
    """Execute the full acoustic reconstruction pipeline."""
    maybe_generate_synthetic_data(paths)
    acoustic_pings, engine_logs, ocean_currents, tactical_assets, bathymetry = load_datasets(paths)

    enriched = enrich_with_sound_speed(acoustic_pings)
    flagged = flag_primary_signals(enriched)
    primary = keep_primary_signals(flagged)

    drift = compute_sensor_drift(primary)
    corrected = apply_clock_correction(flagged, drift)
    corrected_primary = corrected.loc[corrected["Is_Primary"]].copy()

    doppler_ready = attach_engine_frequency(corrected_primary, engine_logs)
    doppler = solve_submarine_velocity(doppler_ready)
    trajectory = reconstruct_trajectory(doppler.loc[doppler["Source_Label"].eq("nautilus")].copy())
    speed_summary = summarize_speed(doppler.loc[doppler["Source_Label"].eq("nautilus")].copy())
    kalman_track = kalman_filter_trajectory(trajectory)
    predicted_path = project_future_path(kalman_track, ocean_currents)
    predicted_path = attach_prediction_uncertainty(predicted_path, kalman_track)
    interception_windows, interception_summary = compute_interception_window(
        predicted_path,
        tactical_assets,
        reference_time_ms=float(trajectory["Corrected_Timestamp_ms"].max()),
    )
    anomaly_events = detect_anomalies(kalman_track)
    threat_timeseries, threat_summary = score_threat(
        predicted_path,
        tactical_assets,
        anomaly_events,
        interception_summary,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(out_dir / "cleaned_acoustic_pings.csv", index=False)
    drift.to_csv(out_dir / "sensor_clock_drift.csv", index=False)
    doppler.to_csv(out_dir / "doppler_enriched_pings.csv", index=False)
    speed_summary.to_csv(out_dir / "doppler_speed_summary.csv", index=False)
    trajectory.to_csv(out_dir / "reconstructed_trajectory.csv", index=False)
    kalman_track.to_csv(out_dir / "kalman_smoothed_trajectory.csv", index=False)
    predicted_path.to_csv(out_dir / "predicted_future_path.csv", index=False)
    interception_windows.to_csv(out_dir / "interception_windows.csv", index=False)
    interception_summary.to_csv(out_dir / "interception_summary.csv", index=False)
    tactical_assets.to_csv(out_dir / "tactical_assets_snapshot.csv", index=False)
    bathymetry.to_csv(out_dir / "bathymetry_snapshot.csv", index=False)
    anomaly_events.to_csv(out_dir / "anomaly_events.csv", index=False)
    threat_timeseries.to_csv(out_dir / "threat_timeseries.csv", index=False)
    threat_summary.to_csv(out_dir / "threat_summary.csv", index=False)

    return PipelineOutputs(
        cleaned_pings=corrected,
        drift_offsets=drift,
        trajectory=trajectory,
        speed_summary=speed_summary,
        kalman_track=kalman_track,
        predicted_path=predicted_path,
        interception_windows=interception_windows,
        interception_summary=interception_summary,
        anomaly_events=anomaly_events,
        threat_timeseries=threat_timeseries,
        threat_summary=threat_summary,
    )
