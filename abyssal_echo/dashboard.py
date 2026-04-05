"""Streamlit dashboard for Abyssal Echo."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DEFAULT_OUTPUT_DIR = Path("outputs")


@st.cache_data
def load_dashboard_data(
    output_dir: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    root = Path(output_dir)
    cleaned = pd.read_csv(root / "cleaned_acoustic_pings.csv")
    trajectory = pd.read_csv(root / "reconstructed_trajectory.csv")
    speed = pd.read_csv(root / "doppler_speed_summary.csv")
    kalman = pd.read_csv(root / "kalman_smoothed_trajectory.csv")
    predicted = pd.read_csv(root / "predicted_future_path.csv")
    interception = pd.read_csv(root / "interception_summary.csv")
    assets = pd.read_csv(root / "tactical_assets_snapshot.csv")
    bathymetry = pd.read_csv(root / "bathymetry_snapshot.csv")
    anomalies = pd.read_csv(root / "anomaly_events.csv")
    threat = pd.read_csv(root / "threat_timeseries.csv")
    threat_summary = pd.read_csv(root / "threat_summary.csv")
    return cleaned, trajectory, speed, kalman, predicted, interception, assets, bathymetry, anomalies, threat, threat_summary


def render_dashboard(output_dir: str = "outputs") -> None:
    """Render the analytical dashboard."""
    st.set_page_config(page_title="Abyssal Echo", layout="wide")
    st.title("Abyssal Echo – Acoustic Reconnaissance")
    st.caption("Stealth submarine trajectory reconstruction from distorted hydrophone pings.")

    (
        cleaned,
        trajectory,
        speed,
        kalman,
        predicted,
        interception,
        assets,
        bathymetry,
        anomalies,
        threat,
        threat_summary,
    ) = load_dashboard_data(output_dir)
    latest_track = trajectory.iloc[-1]
    latest_speed = speed.iloc[-1]
    latest_prediction = predicted.iloc[-1]
    latest_threat = threat_summary.iloc[0]

    replay_index = st.slider(
        "Replay Timeline",
        min_value=1,
        max_value=len(kalman),
        value=len(kalman),
        help="Scrub the mission timeline to replay reconstruction and prediction buildup.",
    )
    replay_track = trajectory.iloc[:replay_index].copy()
    replay_kalman = kalman.iloc[:replay_index].copy()
    replay_end_time = replay_kalman["Corrected_Timestamp_ms"].iloc[-1]
    replay_predicted = predicted.loc[predicted["Predicted_Timestamp_ms"] >= replay_end_time].copy()
    if replay_predicted.empty:
        replay_predicted = predicted.tail(1).copy()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Depth (m)", f"{latest_track['Estimated_Depth_m']:.1f}")
    col2.metric("Sound Speed (m/s)", f"{latest_track['Mean_Sound_Speed_mps']:.1f}")
    col3.metric("Speed (knots)", f"{latest_speed['Submarine_Speed_knots']:.2f}")
    col4.metric("Projected Depth (m)", f"{latest_prediction['Predicted_Depth_m']:.1f}")

    col5, col6, col7 = st.columns(3)
    col5.metric("Threat Level", latest_threat["Current_Threat_Level"])
    col6.metric("Peak Threat", f"{latest_threat['Peak_Threat_Score']:.2f}")
    col7.metric("Prediction Confidence", f"{latest_prediction['Confidence_Score']:.2f}")

    left, right = st.columns([1.7, 1.0])

    with left:
        st.subheader("Trajectory Reconstruction")
        fig3d = go.Figure()
        fig3d.add_trace(
            go.Scatter3d(
                x=trajectory["Raw_X"],
                y=trajectory["Raw_Y"],
                z=trajectory["Raw_Z"],
                mode="lines+markers",
                name="Raw Path",
                line={"color": "#7f8c8d", "width": 5},
                marker={"size": 3},
            )
        )
        fig3d.add_trace(
            go.Scatter3d(
                x=trajectory["Corrected_X"],
                y=trajectory["Corrected_Y"],
                z=trajectory["Corrected_Z"],
                mode="lines+markers",
                name="Corrected Path",
                line={"color": "#00b894", "width": 7},
                marker={"size": 4},
            )
        )
        fig3d.add_trace(
            go.Scatter3d(
                x=replay_kalman["Kalman_X"],
                y=replay_kalman["Kalman_Y"],
                z=replay_kalman["Kalman_Z"],
                mode="lines",
                name="Kalman Track",
                line={"color": "#f39c12", "width": 6},
            )
        )
        fig3d.add_trace(
            go.Scatter3d(
                x=replay_predicted["Predicted_X"],
                y=replay_predicted["Predicted_Y"],
                z=replay_predicted["Predicted_Z"],
                mode="lines+markers",
                name="Predicted Projection",
                line={"color": "#e84393", "width": 8, "dash": "dash"},
                marker={"size": 4},
            )
        )
        fig3d.update_layout(
            height=560,
            scene={"xaxis_title": "X (m)", "yaxis_title": "Y (m)", "zaxis_title": "Z (m)"},
            legend={"orientation": "h"},
        )
        st.plotly_chart(fig3d, use_container_width=True)

        fig2d = px.line(
            replay_track,
            x="Corrected_X",
            y="Corrected_Y",
            markers=True,
            title="Top-Down Track",
        )
        fig2d.add_scatter(
            x=replay_track["Raw_X"],
            y=replay_track["Raw_Y"],
            mode="lines",
            name="Raw Path",
            line={"dash": "dash", "color": "#636e72"},
        )
        fig2d.add_scatter(
            x=replay_predicted["Predicted_X"],
            y=replay_predicted["Predicted_Y"],
            mode="lines+markers",
            name="Predicted Projection",
            line={"dash": "dot", "color": "#d63031"},
        )
        for row in replay_predicted.itertuples(index=False):
            fig2d.add_shape(
                type="circle",
                xref="x",
                yref="y",
                x0=row.Predicted_X - row.Uncertainty_Radius_m,
                y0=row.Predicted_Y - row.Uncertainty_Radius_m,
                x1=row.Predicted_X + row.Uncertainty_Radius_m,
                y1=row.Predicted_Y + row.Uncertainty_Radius_m,
                line={"color": "rgba(232, 67, 147, 0.14)"},
                fillcolor="rgba(232, 67, 147, 0.08)",
            )
        st.plotly_chart(fig2d, use_container_width=True)

    with right:
        st.subheader("Doppler Speedometer")
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=float(latest_speed["Submarine_Speed_knots"]),
                number={"suffix": " kn"},
                title={"text": "Estimated Speed"},
                gauge={
                    "axis": {"range": [0, max(20.0, speed["Submarine_Speed_knots"].max() * 1.2)]},
                    "bar": {"color": "#0984e3"},
                    "steps": [
                        {"range": [0, 5], "color": "#dfe6e9"},
                        {"range": [5, 12], "color": "#81ecec"},
                        {"range": [12, 20], "color": "#74b9ff"},
                    ],
                },
            )
        )
        gauge.update_layout(height=320, margin={"t": 60, "b": 10})
        st.plotly_chart(gauge, use_container_width=True)

        st.subheader("Environmental HUD")
        hud = trajectory[["Corrected_Timestamp_ms", "Estimated_Depth_m", "Mean_Sound_Speed_mps"]].copy()
        hud["Time_s"] = hud["Corrected_Timestamp_ms"] / 1000.0
        hud_fig = go.Figure()
        hud_fig.add_trace(
            go.Scatter(
                x=hud["Time_s"],
                y=hud["Estimated_Depth_m"],
                mode="lines",
                name="Depth (m)",
                line={"color": "#2d3436"},
                yaxis="y1",
            )
        )
        hud_fig.add_trace(
            go.Scatter(
                x=hud["Time_s"],
                y=hud["Mean_Sound_Speed_mps"],
                mode="lines",
                name="Sound Speed (m/s)",
                line={"color": "#e17055"},
                yaxis="y2",
            )
        )
        hud_fig.update_layout(
            height=260,
            xaxis={"title": "Time (s)"},
            yaxis={"title": "Depth (m)"},
            yaxis2={"title": "Sound Speed (m/s)", "overlaying": "y", "side": "right"},
            legend={"orientation": "h"},
        )
        st.plotly_chart(hud_fig, use_container_width=True)

        st.subheader("Interception Window")
        if interception.empty:
            st.info("No feasible interception opportunity in the current prediction horizon.")
        else:
            best = interception.iloc[0]
            st.metric("Best Asset", best["Asset_ID"])
            st.metric("Time To Intercept (s)", f"{best['Time_To_Target_s']:.0f}")
            st.metric("Confidence", f"{best['Confidence_Score']:.2f}")
            st.dataframe(
                interception[
                    [
                        "Asset_ID",
                        "Asset_Type",
                        "Predicted_Timestamp_ms",
                        "Time_To_Target_s",
                        "Response_Time_s",
                        "Confidence_Score",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Threat Console")
        st.write(latest_threat["Recommended_Action"])
        threat_fig = px.line(
            threat,
            x="Predicted_Timestamp_ms",
            y="Threat_Score",
            color="Threat_Level",
            color_discrete_map={"Low": "#00b894", "Medium": "#fdcb6e", "High": "#d63031"},
            title="Projected Threat Over Time",
        )
        threat_fig.update_layout(height=250, xaxis_title="Prediction Timestamp (ms)", yaxis_title="Threat Score")
        st.plotly_chart(threat_fig, use_container_width=True)

    st.subheader("Signal Waterfall")
    waterfall = cleaned.copy()
    waterfall["Time_s"] = waterfall["Corrected_Timestamp_ms"] / 1000.0
    waterfall["Signal_Type"] = waterfall["Is_Primary"].map({True: "Primary", False: "Echo"})
    waterfall_fig = px.scatter(
        waterfall.sort_values("Corrected_Timestamp_ms"),
        x="Time_s",
        y="Received_Frequency_Hz",
        color="Signal_Type",
        color_discrete_map={"Primary": "#2ecc71", "Echo": "#e74c3c"},
        size="Intensity_dB",
        hover_data=["Packet_ID", "Sensor_ID", "Intensity_dB"],
        title="Incoming Acoustic Events",
    )
    waterfall_fig.update_layout(height=420, xaxis_title="Corrected Time (s)", yaxis_title="Frequency (Hz)")
    st.plotly_chart(waterfall_fig, use_container_width=True)

    st.subheader("Anomaly Feed")
    st.dataframe(
        anomalies[["Event_Type", "Timestamp_ms", "Severity", "Description"]].head(8),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Tactical Ocean Map")
    tactical_map = go.Figure()
    bathy_grid = bathymetry.pivot(index="Y_m", columns="X_m", values="Seabed_Depth_m")
    tactical_map.add_trace(
        go.Contour(
            x=bathy_grid.columns.to_numpy(dtype=float),
            y=bathy_grid.index.to_numpy(dtype=float),
            z=bathy_grid.to_numpy(dtype=float),
            colorscale="Blues",
            opacity=0.45,
            contours={"showlabels": False},
            showscale=False,
            name="Bathymetry",
        )
    )
    tactical_map.add_trace(
        go.Scatter(
            x=replay_kalman["Kalman_X"],
            y=replay_kalman["Kalman_Y"],
            mode="lines",
            name="Kalman Track",
            line={"color": "#fdcb6e", "width": 4},
        )
    )
    tactical_map.add_trace(
        go.Scatter(
            x=replay_predicted["Predicted_X"],
            y=replay_predicted["Predicted_Y"],
            mode="lines+markers",
            name="Projected Path",
            line={"color": "#e84393", "width": 5, "dash": "dash"},
            marker={"size": 7, "symbol": "diamond"},
        )
    )
    tactical_map.add_trace(
        go.Scatter(
            x=assets["Base_X_m"],
            y=assets["Base_Y_m"],
            mode="markers+text",
            name="Tactical Assets",
            text=assets["Asset_ID"],
            textposition="bottom center",
            marker={"size": 14, "color": "#0984e3", "symbol": "triangle-up"},
        )
    )
    if not interception.empty:
        tactical_map.add_trace(
            go.Scatter(
                x=interception["Intercept_X"],
                y=interception["Intercept_Y"],
                mode="markers+text",
                name="Intercept Window",
                text=interception["Asset_ID"],
                textposition="top center",
                marker={"size": 12, "color": "#00cec9", "symbol": "x"},
            )
        )
    tactical_map.add_trace(
        go.Scatter(
            x=[replay_predicted["Predicted_X"].iloc[-1]],
            y=[replay_predicted["Predicted_Y"].iloc[-1]],
            mode="markers",
            name="Projected Contact Zone",
            marker={
                "size": max(20.0, replay_predicted["Uncertainty_Radius_m"].iloc[-1] / 50.0),
                "color": "rgba(214, 48, 49, 0.20)",
                "line": {"color": "#d63031", "width": 2},
                "symbol": "circle",
            },
        )
    )
    tactical_map.update_layout(
        height=500,
        xaxis_title="Atlantic Trench X (m)",
        yaxis_title="Atlantic Trench Y (m)",
        legend={"orientation": "h"},
        paper_bgcolor="#f5f6fa",
        plot_bgcolor="#ecf0f1",
    )
    st.plotly_chart(tactical_map, use_container_width=True)


if __name__ == "__main__":
    selected_output_dir = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_OUTPUT_DIR)
    render_dashboard(selected_output_dir)
