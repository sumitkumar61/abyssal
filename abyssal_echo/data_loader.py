"""Data loading and synthetic data generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from abyssal_echo.sound_speed import compute_sound_speed


DEFAULT_DATA_DIR = Path("data")


@dataclass(frozen=True)
class DatasetPaths:
    acoustic_pings: Path
    engine_logs: Path
    ocean_currents: Path
    tactical_assets: Path
    bathymetry: Path


def get_default_paths(base_dir: Path | None = None) -> DatasetPaths:
    root = (base_dir or DEFAULT_DATA_DIR).resolve()
    return DatasetPaths(
        acoustic_pings=root / "acoustic_pings.csv",
        engine_logs=root / "engine_logs.csv",
        ocean_currents=root / "ocean_currents.csv",
        tactical_assets=root / "tactical_assets.csv",
        bathymetry=root / "bathymetry.csv",
    )


def load_datasets(
    paths: DatasetPaths,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the acoustic ping and engine log datasets."""
    return (
        pd.read_csv(paths.acoustic_pings),
        pd.read_csv(paths.engine_logs),
        pd.read_csv(paths.ocean_currents),
        pd.read_csv(paths.tactical_assets),
        pd.read_csv(paths.bathymetry),
    )


def maybe_generate_synthetic_data(paths: DatasetPaths, seed: int = 7) -> None:
    """Create a coherent demo dataset when CSV files are missing."""
    if (
        paths.acoustic_pings.exists()
        and paths.engine_logs.exists()
        and paths.ocean_currents.exists()
        and paths.tactical_assets.exists()
        and paths.bathymetry.exists()
    ):
        return

    paths.acoustic_pings.parent.mkdir(parents=True, exist_ok=True)
    acoustic, engine, currents, assets, bathymetry = generate_synthetic_data(seed=seed)
    if not paths.acoustic_pings.exists():
        acoustic.to_csv(paths.acoustic_pings, index=False)
    if not paths.engine_logs.exists():
        engine.to_csv(paths.engine_logs, index=False)
    if not paths.ocean_currents.exists():
        currents.to_csv(paths.ocean_currents, index=False)
    if not paths.tactical_assets.exists():
        assets.to_csv(paths.tactical_assets, index=False)
    if not paths.bathymetry.exists():
        bathymetry.to_csv(paths.bathymetry, index=False)


def generate_synthetic_data(
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic submarine pass with reflections and sensor drift."""
    rng = np.random.default_rng(seed)

    sensors = pd.DataFrame(
        [
            ("H1", -4500.0, -2500.0, -1800.0, 12.0),
            ("H2", 4200.0, -2300.0, -1750.0, -18.0),
            ("H3", -3600.0, 3100.0, -1900.0, 9.0),
            ("H4", 4300.0, 3300.0, -1850.0, -24.0),
            ("H5", 0.0, 0.0, -2100.0, 16.0),
        ],
        columns=["Sensor_ID", "X", "Y", "Z", "Sensor_Drift_ms"],
    )

    n_steps = 28
    emission_times = np.arange(n_steps) * 12_000.0
    x = np.linspace(-6200.0, 6400.0, n_steps)
    y = 1200.0 * np.sin(np.linspace(0.0, 2.8 * np.pi, n_steps))
    z = -1450.0 - 300.0 * np.sin(np.linspace(0.0, 1.7 * np.pi, n_steps))
    rpm = 130 + 18 * np.sin(np.linspace(0, 3 * np.pi, n_steps))
    blade_count = np.full(n_steps, 7)

    trajectory = pd.DataFrame(
        {
            "Ping_Group": np.arange(n_steps),
            "Emission_Timestamp_ms": emission_times,
            "True_X": x,
            "True_Y": y,
            "True_Z": z,
            "RPM": rpm,
            "Blade_Count": blade_count,
        }
    )
    trajectory["Temperature_C"] = 3.5 + 0.0022 * (-trajectory["True_Z"]) + rng.normal(0, 0.2, n_steps)
    trajectory["Salinity_PSU"] = 35.0 + 0.15 * np.sin(np.linspace(0, 1.5 * np.pi, n_steps)) + rng.normal(0, 0.03, n_steps)
    trajectory["Sound_Speed_mps"] = compute_sound_speed(
        trajectory["Temperature_C"], trajectory["Salinity_PSU"]
    )
    trajectory["Blade_Frequency_Hz"] = (trajectory["RPM"] / 60.0) * trajectory["Blade_Count"]
    trajectory["True_Speed_mps"] = np.gradient(trajectory["True_X"], 12.0)

    engine_logs = trajectory[["Emission_Timestamp_ms", "RPM", "Blade_Count"]].rename(
        columns={"Emission_Timestamp_ms": "Timestamp_ms"}
    )

    packet_rows: list[dict[str, float | str | int | bool]] = []
    sync_packet_base = 100_000
    sub_packet_base = 200_000

    for sensor in sensors.itertuples(index=False):
        distance_to_buoy = np.linalg.norm(np.array([sensor.X, sensor.Y, sensor.Z], dtype=float))
        buoy_temp = 4.0 + 0.0007 * abs(sensor.Z)
        buoy_salinity = 35.0
        buoy_speed = float(compute_sound_speed(pd.Series([buoy_temp]), pd.Series([buoy_salinity])).iloc[0])
        buoy_travel_ms = distance_to_buoy / buoy_speed * 1000.0

        for sync_idx, sync_emission in enumerate([15_000.0, 150_000.0, 285_000.0]):
            observed = sync_emission + buoy_travel_ms + sensor.Sensor_Drift_ms + rng.normal(0, 0.6)
            packet_rows.append(
                {
                    "Packet_ID": sync_packet_base + sync_idx,
                    "Ping_Group": -(sync_idx + 1),
                    "Timestamp_ms": observed,
                    "Emission_Timestamp_ms": sync_emission,
                    "Sensor_ID": sensor.Sensor_ID,
                    "Received_Frequency_Hz": 0.0,
                    "Intensity_dB": 118.0 + rng.normal(0, 1.0),
                    "Temperature_C": buoy_temp,
                    "Salinity_PSU": buoy_salinity,
                    "Depth_m": abs(sensor.Z),
                    "X": sensor.X,
                    "Y": sensor.Y,
                    "Z": sensor.Z,
                    "Source_Label": "sync_buoy",
                }
            )

    for row in trajectory.itertuples(index=False):
        sub_xyz = np.array([row.True_X, row.True_Y, row.True_Z], dtype=float)
        for sensor in sensors.itertuples(index=False):
            sensor_xyz = np.array([sensor.X, sensor.Y, sensor.Z], dtype=float)
            distance_m = np.linalg.norm(sub_xyz - sensor_xyz)
            travel_ms = distance_m / row.Sound_Speed_mps * 1000.0
            base_arrival = row.Emission_Timestamp_ms + travel_ms + sensor.Sensor_Drift_ms
            received_frequency = row.Blade_Frequency_Hz / (
                1.0 + (row.True_Speed_mps / row.Sound_Speed_mps)
            )
            intensity = 142.0 - 0.0025 * distance_m + rng.normal(0, 1.5)

            primary_packet = {
                "Packet_ID": sub_packet_base + int(row.Ping_Group * 10 + int(sensor.Sensor_ID[-1])),
                "Ping_Group": int(row.Ping_Group),
                "Timestamp_ms": base_arrival + rng.normal(0, 1.2),
                "Emission_Timestamp_ms": row.Emission_Timestamp_ms,
                "Sensor_ID": sensor.Sensor_ID,
                "Received_Frequency_Hz": received_frequency + rng.normal(0, 0.08),
                "Intensity_dB": intensity,
                "Temperature_C": row.Temperature_C + rng.normal(0, 0.08),
                "Salinity_PSU": row.Salinity_PSU + rng.normal(0, 0.02),
                "Depth_m": abs(sensor.Z),
                "X": sensor.X,
                "Y": sensor.Y,
                "Z": sensor.Z,
                "Source_Label": "nautilus",
            }
            packet_rows.append(primary_packet)

            for reflection_idx, delay_ms in enumerate([32.0 + rng.uniform(0, 10), 58.0 + rng.uniform(0, 15)], start=1):
                packet_rows.append(
                    {
                        **primary_packet,
                        "Packet_ID": primary_packet["Packet_ID"],
                        "Timestamp_ms": base_arrival + delay_ms + rng.normal(0, 1.8),
                        "Received_Frequency_Hz": primary_packet["Received_Frequency_Hz"] + rng.normal(0, 0.05),
                        "Intensity_dB": intensity - (8.0 * reflection_idx) + rng.normal(0, 1.0),
                    }
                )

    current_depths = np.array([1200.0, 1450.0, 1700.0, 1950.0, 2200.0])
    current_rows: list[dict[str, float]] = []
    current_times = np.arange(0.0, emission_times.max() + 72_000.0, 12_000.0)
    for timestamp_ms in current_times:
        phase = timestamp_ms / 60_000.0
        for depth_m in current_depths:
            current_rows.append(
                {
                    "Timestamp_ms": timestamp_ms,
                    "Depth_m": depth_m,
                    "Current_U_mps": 0.18 * np.sin(phase / 3.0 + depth_m / 900.0) + rng.normal(0, 0.01),
                    "Current_V_mps": 0.12 * np.cos(phase / 4.0 + depth_m / 1100.0) + rng.normal(0, 0.01),
                    "Current_W_mps": -0.01 * np.sin(phase / 5.0 + depth_m / 1400.0) + rng.normal(0, 0.002),
                }
            )

    tactical_assets = pd.DataFrame(
        [
            {
                "Asset_ID": "P8_POSEIDON",
                "Asset_Type": "Airborne ASW",
                "Base_X_m": 9000.0,
                "Base_Y_m": -2500.0,
                "Base_Z_m": 0.0,
                "Max_Speed_knots": 360.0,
                "Launch_Delay_s": 30.0,
                "Detection_Radius_m": 2500.0,
            },
            {
                "Asset_ID": "SSK_ARGUS",
                "Asset_Type": "Interceptor Sub",
                "Base_X_m": -8200.0,
                "Base_Y_m": -5400.0,
                "Base_Z_m": -1200.0,
                "Max_Speed_knots": 28.0,
                "Launch_Delay_s": 120.0,
                "Detection_Radius_m": 900.0,
            },
            {
                "Asset_ID": "USV_TRIDENT",
                "Asset_Type": "Surface Drone",
                "Base_X_m": 7600.0,
                "Base_Y_m": -1600.0,
                "Base_Z_m": 0.0,
                "Max_Speed_knots": 42.0,
                "Launch_Delay_s": 90.0,
                "Detection_Radius_m": 1200.0,
            },
        ]
    )

    x_grid = np.linspace(-12000.0, 16000.0, 40)
    y_grid = np.linspace(-9000.0, 9000.0, 34)
    xx, yy = np.meshgrid(x_grid, y_grid)
    seabed = (
        2800.0
        + 0.00002 * (xx**2)
        + 0.00003 * (yy**2)
        + 350.0 * np.sin(xx / 3800.0)
        - 220.0 * np.cos(yy / 2700.0)
    )
    bathymetry = pd.DataFrame(
        {
            "X_m": xx.ravel(),
            "Y_m": yy.ravel(),
            "Seabed_Depth_m": seabed.ravel(),
        }
    )

    acoustic = pd.DataFrame(packet_rows).sort_values("Timestamp_ms").reset_index(drop=True)
    ocean_currents = pd.DataFrame(current_rows).sort_values(["Timestamp_ms", "Depth_m"]).reset_index(drop=True)
    return acoustic, engine_logs, ocean_currents, tactical_assets, bathymetry
