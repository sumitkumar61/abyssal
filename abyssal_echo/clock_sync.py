"""Clock drift estimation and timestamp correction."""

from __future__ import annotations

import numpy as np
import pandas as pd


SYNC_XYZ = np.array([0.0, 0.0, 0.0], dtype=float)


def compute_sensor_drift(pings: pd.DataFrame) -> pd.DataFrame:
    """Estimate a fixed offset per sensor from buoy-origin packet timestamps."""
    sync_rows = pings.loc[pings["Source_Label"].eq("sync_buoy")].copy()
    if sync_rows.empty:
        raise ValueError("Clock synchronization requires sync buoy packets.")

    sensor_xyz = sync_rows[["X", "Y", "Z"]].to_numpy(dtype=float)
    distance_m = np.linalg.norm(sensor_xyz - SYNC_XYZ, axis=1)
    expected_travel_ms = (distance_m / sync_rows["Sound_Speed_mps"].to_numpy(dtype=float)) * 1000.0
    sync_rows["Observed_Drift_ms"] = (
        sync_rows["Timestamp_ms"] - sync_rows["Emission_Timestamp_ms"] - expected_travel_ms
    )

    drift = (
        sync_rows.groupby("Sensor_ID", as_index=False)["Observed_Drift_ms"]
        .median()
        .rename(columns={"Observed_Drift_ms": "Clock_Drift_ms"})
    )
    return drift


def apply_clock_correction(pings: pd.DataFrame, drift: pd.DataFrame) -> pd.DataFrame:
    """Apply per-sensor median drift correction."""
    corrected = pings.merge(drift, on="Sensor_ID", how="left")
    corrected["Clock_Drift_ms"] = corrected["Clock_Drift_ms"].fillna(0.0)
    corrected["Corrected_Timestamp_ms"] = (
        corrected["Timestamp_ms"] - corrected["Clock_Drift_ms"]
    )
    return corrected

