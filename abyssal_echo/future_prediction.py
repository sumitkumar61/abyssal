"""Future track projection, current drift, and interception analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator


MS_TO_KNOTS = 1.9438444924406
KNOTS_TO_MPS = 0.514444


def estimate_velocity_vectors(trajectory: pd.DataFrame) -> pd.DataFrame:
    """Estimate velocity vectors from reconstructed positions."""
    track = trajectory.copy().sort_values("Corrected_Timestamp_ms").reset_index(drop=True)
    time_s = track["Corrected_Timestamp_ms"].to_numpy(dtype=float) / 1000.0
    for axis in ["X", "Y", "Z"]:
        positions = track[f"Corrected_{axis}"].to_numpy(dtype=float)
        track[f"Velocity_{axis}_mps"] = np.gradient(positions, time_s, edge_order=1)

    track["Velocity_Magnitude_mps"] = np.linalg.norm(
        track[["Velocity_X_mps", "Velocity_Y_mps", "Velocity_Z_mps"]].to_numpy(dtype=float), axis=1
    )
    track["Velocity_Magnitude_knots"] = track["Velocity_Magnitude_mps"] * MS_TO_KNOTS
    return track


def kalman_filter_trajectory(trajectory: pd.DataFrame, process_noise: float = 4.0, measurement_noise: float = 45.0) -> pd.DataFrame:
    """Smooth the trajectory with a constant-velocity Kalman filter."""
    track = estimate_velocity_vectors(trajectory)
    measurements = track[["Corrected_X", "Corrected_Y", "Corrected_Z"]].to_numpy(dtype=float)
    times = track["Corrected_Timestamp_ms"].to_numpy(dtype=float) / 1000.0

    state = np.zeros(6, dtype=float)
    state[:3] = measurements[0]
    if len(track) > 1:
        initial_velocity = track.loc[0, ["Velocity_X_mps", "Velocity_Y_mps", "Velocity_Z_mps"]].to_numpy(dtype=float)
        state[3:] = initial_velocity

    covariance = np.eye(6) * 250.0
    observation = np.hstack([np.eye(3), np.zeros((3, 3))])
    measurement_cov = np.eye(3) * measurement_noise
    filtered_states: list[np.ndarray] = []

    for idx, measurement in enumerate(measurements):
        dt = 1.0 if idx == 0 else max(times[idx] - times[idx - 1], 1.0)
        transition = np.eye(6)
        transition[0, 3] = dt
        transition[1, 4] = dt
        transition[2, 5] = dt

        q = process_noise
        process_cov = np.array(
            [
                [dt**4 / 4, 0, 0, dt**3 / 2, 0, 0],
                [0, dt**4 / 4, 0, 0, dt**3 / 2, 0],
                [0, 0, dt**4 / 4, 0, 0, dt**3 / 2],
                [dt**3 / 2, 0, 0, dt**2, 0, 0],
                [0, dt**3 / 2, 0, 0, dt**2, 0],
                [0, 0, dt**3 / 2, 0, 0, dt**2],
            ],
            dtype=float,
        ) * q

        state = transition @ state
        covariance = transition @ covariance @ transition.T + process_cov

        innovation = measurement - observation @ state
        innovation_cov = observation @ covariance @ observation.T + measurement_cov
        kalman_gain = covariance @ observation.T @ np.linalg.inv(innovation_cov)
        state = state + kalman_gain @ innovation
        covariance = (np.eye(6) - kalman_gain @ observation) @ covariance
        filtered_states.append(state.copy())

    filtered = track.copy()
    filtered_state_array = np.vstack(filtered_states)
    filtered["Kalman_X"] = filtered_state_array[:, 0]
    filtered["Kalman_Y"] = filtered_state_array[:, 1]
    filtered["Kalman_Z"] = filtered_state_array[:, 2]
    filtered["Kalman_Vx_mps"] = filtered_state_array[:, 3]
    filtered["Kalman_Vy_mps"] = filtered_state_array[:, 4]
    filtered["Kalman_Vz_mps"] = filtered_state_array[:, 5]
    filtered["Kalman_Speed_knots"] = (
        np.linalg.norm(filtered_state_array[:, 3:], axis=1) * MS_TO_KNOTS
    )
    return filtered


def _build_current_interpolator(ocean_currents: pd.DataFrame, column: str) -> LinearNDInterpolator:
    points = ocean_currents[["Timestamp_ms", "Depth_m"]].to_numpy(dtype=float)
    values = ocean_currents[column].to_numpy(dtype=float)
    return LinearNDInterpolator(points, values, fill_value=float(np.nanmean(values)))


def project_future_path(
    kalman_track: pd.DataFrame,
    ocean_currents: pd.DataFrame,
    horizon_steps: int = 20,
    step_seconds: float | None = None,
) -> pd.DataFrame:
    """Project the future path using Kalman state propagation plus ocean drift."""
    track = kalman_track.sort_values("Corrected_Timestamp_ms").reset_index(drop=True)
    if step_seconds is None:
        dt_s = float(np.median(np.diff(track["Corrected_Timestamp_ms"])) / 1000.0)
        step_seconds = dt_s if np.isfinite(dt_s) and dt_s > 0 else 12.0

    u_interp = _build_current_interpolator(ocean_currents, "Current_U_mps")
    v_interp = _build_current_interpolator(ocean_currents, "Current_V_mps")
    w_interp = _build_current_interpolator(ocean_currents, "Current_W_mps")

    last = track.iloc[-1]
    timestamp_ms = float(last["Corrected_Timestamp_ms"])
    state_position = np.array([last["Kalman_X"], last["Kalman_Y"], last["Kalman_Z"]], dtype=float)
    state_velocity = np.array(
        [last["Kalman_Vx_mps"], last["Kalman_Vy_mps"], last["Kalman_Vz_mps"]],
        dtype=float,
    )

    predictions: list[dict[str, float | int]] = []
    for step in range(1, horizon_steps + 1):
        timestamp_ms += step_seconds * 1000.0
        depth_m = abs(state_position[2])
        current = np.array(
            [
                float(u_interp(timestamp_ms, depth_m)),
                float(v_interp(timestamp_ms, depth_m)),
                float(w_interp(timestamp_ms, depth_m)),
            ],
            dtype=float,
        )

        state_position = state_position + (state_velocity + current) * step_seconds
        combined_velocity = state_velocity + current
        predictions.append(
            {
                "Projection_Step": step,
                "Predicted_Timestamp_ms": timestamp_ms,
                "Predicted_X": state_position[0],
                "Predicted_Y": state_position[1],
                "Predicted_Z": state_position[2],
                "Predicted_Depth_m": abs(state_position[2]),
                "Base_Vx_mps": state_velocity[0],
                "Base_Vy_mps": state_velocity[1],
                "Base_Vz_mps": state_velocity[2],
                "Current_U_mps": current[0],
                "Current_V_mps": current[1],
                "Current_W_mps": current[2],
                "Composite_Speed_knots": np.linalg.norm(combined_velocity) * MS_TO_KNOTS,
            }
        )

    return pd.DataFrame(predictions)


def compute_interception_window(
    predicted_path: pd.DataFrame,
    tactical_assets: pd.DataFrame,
    reference_time_ms: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute feasible interception opportunities for each tactical asset."""
    windows: list[dict[str, float | str | bool]] = []

    for asset in tactical_assets.itertuples(index=False):
        asset_xyz = np.array([asset.Base_X_m, asset.Base_Y_m, asset.Base_Z_m], dtype=float)
        asset_speed_mps = float(asset.Max_Speed_knots) * KNOTS_TO_MPS

        for prediction in predicted_path.itertuples(index=False):
            target_xyz = np.array([prediction.Predicted_X, prediction.Predicted_Y, prediction.Predicted_Z], dtype=float)
            distance_m = np.linalg.norm(target_xyz - asset_xyz)
            time_until_target_s = max((prediction.Predicted_Timestamp_ms - reference_time_ms) / 1000.0, 0.0)
            response_time_s = float(asset.Launch_Delay_s) + max(
                (distance_m - float(asset.Detection_Radius_m)) / max(asset_speed_mps, 0.1),
                0.0,
            )
            feasible = response_time_s <= time_until_target_s
            windows.append(
                {
                    "Asset_ID": asset.Asset_ID,
                    "Asset_Type": asset.Asset_Type,
                    "Predicted_Timestamp_ms": prediction.Predicted_Timestamp_ms,
                    "Intercept_X": prediction.Predicted_X,
                    "Intercept_Y": prediction.Predicted_Y,
                    "Intercept_Z": prediction.Predicted_Z,
                    "Time_To_Target_s": time_until_target_s,
                    "Response_Time_s": response_time_s,
                    "Distance_To_Target_m": distance_m,
                    "Feasible_Intercept": feasible,
                    "Confidence_Score": max(0.05, min(0.99, 1.0 - (response_time_s / max(time_until_target_s, 1.0)))),
                }
            )

    window_df = pd.DataFrame(windows).sort_values(["Feasible_Intercept", "Predicted_Timestamp_ms"], ascending=[False, True])
    best = (
        window_df.loc[window_df["Feasible_Intercept"]]
        .groupby(["Asset_ID", "Asset_Type"], as_index=False)
        .first()
    )
    return window_df.reset_index(drop=True), best.reset_index(drop=True)
