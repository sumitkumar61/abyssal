"""Doppler-based velocity estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd


MS_TO_KNOTS = 1.9438444924406


def enrich_with_engine_frequency(engine_logs: pd.DataFrame) -> pd.DataFrame:
    """Compute blade-pass frequency from engine RPM and blade count."""
    logs = engine_logs.copy()
    logs["Blade_Frequency_Hz"] = (logs["RPM"] / 60.0) * logs["Blade_Count"]
    return logs


def attach_engine_frequency(pings: pd.DataFrame, engine_logs: pd.DataFrame) -> pd.DataFrame:
    """Align ping rows with the closest engine log in time."""
    logs = enrich_with_engine_frequency(engine_logs).sort_values("Timestamp_ms")
    aligned = pd.merge_asof(
        pings.sort_values("Corrected_Timestamp_ms"),
        logs[["Timestamp_ms", "Blade_Frequency_Hz", "RPM", "Blade_Count"]],
        left_on="Corrected_Timestamp_ms",
        right_on="Timestamp_ms",
        direction="nearest",
    )
    return aligned.rename(columns={"Timestamp_ms_x": "Timestamp_ms", "Timestamp_ms_y": "Engine_Timestamp_ms"})


def solve_submarine_velocity(pings: pd.DataFrame) -> pd.DataFrame:
    """Solve v_source from the simplified Doppler relation and convert to knots."""
    solved = pings.copy()
    ratio = solved["Received_Frequency_Hz"] / solved["Blade_Frequency_Hz"]
    numerator = solved["Sound_Speed_mps"] * (1.0 - ratio)
    denominator = ratio.replace(0, np.nan)
    solved["Submarine_Velocity_mps"] = numerator / denominator
    solved["Submarine_Speed_knots"] = solved["Submarine_Velocity_mps"].abs() * MS_TO_KNOTS
    return solved


def summarize_speed(pings: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Doppler speeds over time."""
    speed_series = (
        pings.groupby("Ping_Group", as_index=False)
        .agg(
            Corrected_Timestamp_ms=("Corrected_Timestamp_ms", "median"),
            Submarine_Velocity_mps=("Submarine_Velocity_mps", "median"),
            Submarine_Speed_knots=("Submarine_Speed_knots", "median"),
        )
        .sort_values("Corrected_Timestamp_ms")
        .reset_index(drop=True)
    )
    return speed_series

