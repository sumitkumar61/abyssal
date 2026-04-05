"""Trajectory reconstruction using multilateration."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def estimate_distances(pings: pd.DataFrame) -> pd.DataFrame:
    """Estimate source-to-sensor distance from corrected travel time."""
    estimated = pings.copy()
    travel_ms = estimated["Corrected_Timestamp_ms"] - estimated["Emission_Timestamp_ms"]
    estimated["Travel_Time_ms"] = travel_ms.clip(lower=0.0)
    estimated["Estimated_Distance_m"] = (
        estimated["Travel_Time_ms"] / 1000.0 * estimated["Sound_Speed_mps"]
    )
    return estimated


def _solve_position(sensor_xyz: np.ndarray, ranges_m: np.ndarray, initial_guess: np.ndarray) -> np.ndarray:
    def residuals(candidate: np.ndarray) -> np.ndarray:
        return np.linalg.norm(sensor_xyz - candidate, axis=1) - ranges_m

    result = least_squares(residuals, x0=initial_guess, method="trf")
    return result.x


def reconstruct_trajectory(pings: pd.DataFrame) -> pd.DataFrame:
    """Compute raw and corrected trajectory estimates for each grouped ping event."""
    ranged = estimate_distances(pings)
    trajectory_rows = []
    initial_guess = np.zeros(3, dtype=float)

    for ping_group, group in ranged.groupby("Ping_Group"):
        sensor_xyz = group[["X", "Y", "Z"]].to_numpy(dtype=float)
        corrected_ranges = group["Estimated_Distance_m"].to_numpy(dtype=float)
        raw_ranges = (
            (group["Timestamp_ms"] - group["Emission_Timestamp_ms"]).clip(lower=0.0) / 1000.0
            * group["Sound_Speed_mps"]
        ).to_numpy(dtype=float)

        if len(group) < 4:
            continue

        corrected_position = _solve_position(sensor_xyz, corrected_ranges, initial_guess)
        raw_position = _solve_position(sensor_xyz, raw_ranges, initial_guess)
        initial_guess = corrected_position

        trajectory_rows.append(
            {
                "Ping_Group": ping_group,
                "Packet_ID": group["Packet_ID"].iloc[0],
                "Emission_Timestamp_ms": group["Emission_Timestamp_ms"].median(),
                "Corrected_Timestamp_ms": group["Corrected_Timestamp_ms"].median(),
                "Raw_X": raw_position[0],
                "Raw_Y": raw_position[1],
                "Raw_Z": raw_position[2],
                "Corrected_X": corrected_position[0],
                "Corrected_Y": corrected_position[1],
                "Corrected_Z": corrected_position[2],
                "Estimated_Depth_m": -corrected_position[2],
                "Mean_Sound_Speed_mps": group["Sound_Speed_mps"].mean(),
            }
        )

    return pd.DataFrame(trajectory_rows).sort_values("Corrected_Timestamp_ms").reset_index(drop=True)

