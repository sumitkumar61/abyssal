"""Threat scoring, anomaly detection, and prediction uncertainty helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def attach_prediction_uncertainty(predicted_path: pd.DataFrame, kalman_track: pd.DataFrame) -> pd.DataFrame:
    """Attach a widening uncertainty envelope to projected positions."""
    predicted = predicted_path.copy().sort_values("Predicted_Timestamp_ms").reset_index(drop=True)
    kalman = kalman_track.sort_values("Corrected_Timestamp_ms").reset_index(drop=True)

    if len(kalman) > 2:
        residual = np.sqrt(
            (kalman["Corrected_X"] - kalman["Kalman_X"]) ** 2
            + (kalman["Corrected_Y"] - kalman["Kalman_Y"]) ** 2
            + (kalman["Corrected_Z"] - kalman["Kalman_Z"]) ** 2
        )
        base_uncertainty = float(np.nanmedian(residual)) + 60.0
    else:
        base_uncertainty = 120.0

    predicted["Prediction_Horizon_s"] = (
        predicted["Predicted_Timestamp_ms"] - float(kalman["Corrected_Timestamp_ms"].max())
    ) / 1000.0
    predicted["Uncertainty_Radius_m"] = base_uncertainty + 8.0 * np.sqrt(
        predicted["Prediction_Horizon_s"].clip(lower=1.0)
    )
    predicted["Confidence_Score"] = (
        1.0 / (1.0 + predicted["Uncertainty_Radius_m"] / 500.0)
    ).clip(lower=0.12, upper=0.95)
    return predicted


def detect_anomalies(kalman_track: pd.DataFrame) -> pd.DataFrame:
    """Detect aggressive maneuvers, silent running, and abrupt depth shifts."""
    track = kalman_track.copy().sort_values("Corrected_Timestamp_ms").reset_index(drop=True)
    if track.empty:
        return pd.DataFrame(
            columns=[
                "Event_Type",
                "Timestamp_ms",
                "Severity",
                "Description",
            ]
        )

    time_s = track["Corrected_Timestamp_ms"].to_numpy(dtype=float) / 1000.0
    vx = track["Kalman_Vx_mps"].to_numpy(dtype=float)
    vy = track["Kalman_Vy_mps"].to_numpy(dtype=float)
    vz = track["Kalman_Vz_mps"].to_numpy(dtype=float)
    speed = np.linalg.norm(track[["Kalman_Vx_mps", "Kalman_Vy_mps", "Kalman_Vz_mps"]].to_numpy(dtype=float), axis=1)
    heading = np.unwrap(np.arctan2(vy, vx))
    heading_rate = np.gradient(heading, time_s, edge_order=1)
    acceleration = np.gradient(speed, time_s, edge_order=1)
    depth_rate = np.gradient(track["Kalman_Z"].to_numpy(dtype=float), time_s, edge_order=1)

    events: list[dict[str, float | str]] = []
    for idx, row in track.iterrows():
        if abs(heading_rate[idx]) > 0.045:
            events.append(
                {
                    "Event_Type": "Hard Turn",
                    "Timestamp_ms": row["Corrected_Timestamp_ms"],
                    "Severity": min(1.0, abs(heading_rate[idx]) / 0.08),
                    "Description": "Rapid heading change suggests evasive maneuvering.",
                }
            )
        if abs(depth_rate[idx]) > 6.5:
            events.append(
                {
                    "Event_Type": "Depth Excursion",
                    "Timestamp_ms": row["Corrected_Timestamp_ms"],
                    "Severity": min(1.0, abs(depth_rate[idx]) / 10.0),
                    "Description": "Abrupt dive or climb detected in smoothed depth profile.",
                }
            )
        if idx > 0 and acceleration[idx] < -1.8:
            events.append(
                {
                    "Event_Type": "Silent Running",
                    "Timestamp_ms": row["Corrected_Timestamp_ms"],
                    "Severity": min(1.0, abs(acceleration[idx]) / 3.0),
                    "Description": "Sudden deceleration may indicate noise-reduction behavior.",
                }
            )

    if not events:
        events.append(
            {
                "Event_Type": "Nominal Transit",
                "Timestamp_ms": track["Corrected_Timestamp_ms"].iloc[-1],
                "Severity": 0.1,
                "Description": "No major maneuver anomalies detected in the current window.",
            }
        )
    return pd.DataFrame(events).sort_values(["Severity", "Timestamp_ms"], ascending=[False, True]).reset_index(drop=True)


def score_threat(
    predicted_path: pd.DataFrame,
    tactical_assets: pd.DataFrame,
    anomalies: pd.DataFrame,
    interception_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score tactical threat over the prediction horizon."""
    predictions = predicted_path.copy().sort_values("Predicted_Timestamp_ms").reset_index(drop=True)
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame()

    asset_positions = tactical_assets[["Base_X_m", "Base_Y_m", "Base_Z_m"]].to_numpy(dtype=float)
    threat_rows: list[dict[str, float | str]] = []
    anomaly_boost = float(anomalies["Severity"].head(3).sum()) if not anomalies.empty else 0.0
    intercept_bonus = 0.15 if not interception_summary.empty else 0.0

    for row in predictions.itertuples(index=False):
        predicted_xyz = np.array([row.Predicted_X, row.Predicted_Y, row.Predicted_Z], dtype=float)
        min_distance = float(np.min(np.linalg.norm(asset_positions - predicted_xyz, axis=1)))
        proximity_score = np.clip(1.0 - min_distance / 12000.0, 0.0, 1.0)
        speed_score = np.clip(row.Composite_Speed_knots / 120.0, 0.0, 1.0)
        uncertainty_penalty = np.clip(row.Uncertainty_Radius_m / 1500.0, 0.0, 0.4)
        raw_score = 0.45 * proximity_score + 0.25 * speed_score + 0.15 * anomaly_boost + intercept_bonus - uncertainty_penalty
        threat_score = float(np.clip(raw_score, 0.0, 1.0))
        if threat_score >= 0.72:
            level = "High"
        elif threat_score >= 0.42:
            level = "Medium"
        else:
            level = "Low"

        threat_rows.append(
            {
                "Predicted_Timestamp_ms": row.Predicted_Timestamp_ms,
                "Threat_Score": threat_score,
                "Threat_Level": level,
                "Nearest_Asset_Distance_m": min_distance,
                "Composite_Speed_knots": row.Composite_Speed_knots,
                "Uncertainty_Radius_m": row.Uncertainty_Radius_m,
            }
        )

    time_series = pd.DataFrame(threat_rows)
    summary = pd.DataFrame(
        [
            {
                "Current_Threat_Level": time_series["Threat_Level"].mode().iat[0],
                "Peak_Threat_Score": float(time_series["Threat_Score"].max()),
                "Mean_Threat_Score": float(time_series["Threat_Score"].mean()),
                "High_Threat_Windows": int((time_series["Threat_Level"] == "High").sum()),
                "Recommended_Action": (
                    "Vector airborne asset toward earliest feasible intercept and maintain active tracking."
                    if not interception_summary.empty
                    else "Expand prediction horizon and reposition tactical assets for renewed intercept."
                ),
            }
        ]
    )
    return time_series, summary
