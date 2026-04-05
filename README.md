# Abyssal Echo - Acoustic Reconnaissance

## Overview

**Abyssal Echo** is a scientific signal-processing and tactical analytics prototype for reconstructing and predicting the movement of a stealth submarine from distorted hydrophone acoustic data.

The project starts with noisy underwater pings and tries to answer three progressively harder questions:

1. **Where was the submarine?**
2. **Where is it likely going next?**
3. **What should an interceptor do about it?**

To answer those, the system combines acoustic preprocessing, environmental correction, time synchronization, Doppler analysis, multilateration, Kalman filtering, ocean-current drift modeling, anomaly detection, and tactical scoring.

The implementation is written in Python using:

- `NumPy` for vectorized numerical computation
- `Pandas` for tabular processing
- `SciPy` for least-squares optimization and interpolation
- `Plotly` and `Streamlit` for interactive scientific visualization

---

## Problem Statement

In deep-ocean reconnaissance, hydrophone networks receive distorted acoustic signatures from moving vessels. Those observations are not clean:

- pings contain **echoes and reflections**
- hydrophones can have **clock drift**
- sound propagation changes with **water properties**
- received frequencies are shifted by **relative motion**
- the submarine continues moving after the last measured point

The goal of Abyssal Echo is to transform those imperfect signals into an interpretable operational picture containing:

- reconstructed historical trajectory
- estimated submarine speed
- projected future path
- confidence envelope around predictions
- anomaly alerts
- interception opportunities

---

## Core Features

- Simplified Mackenzie sound-speed calculation
- Echo removal by strongest-return selection
- Clock drift correction using a stationary sync buoy
- Doppler-based velocity estimation
- Trajectory reconstruction via multilateration
- Kalman-filter trajectory smoothing
- Future path projection using velocity extrapolation
- Ocean-current drift compensation
- Prediction uncertainty envelope
- Tactical interception window analysis
- Threat scoring and action recommendation
- Anomaly detection for unusual maneuvers
- Replayable dashboard for storytelling and demo flow
- Synthetic data generation when datasets are missing

---

## How The System Works

The pipeline is intentionally layered. Each stage prepares cleaner, more physically meaningful inputs for the next stage.

### Stage 1: Acoustic Conditioning

Raw hydrophone observations are enriched and cleaned before any reconstruction is attempted.

This stage:

- computes local sound speed from temperature and salinity
- marks primary arrivals vs reflected echoes
- estimates sensor clock drift using sync buoy packets
- corrects timestamps per sensor

Without this stage, later distance and position estimates would be biased by environmental and sensor errors.

### Stage 2: Motion Estimation

After the data is cleaned, the system estimates how the submarine is moving.

This stage:

- aligns each ping with the nearest engine log
- derives blade-pass source frequency from RPM and blade count
- uses Doppler shift to estimate submarine velocity
- reconstructs 3D position from multiple hydrophones

This produces a historical track of the submarine and an estimated speed profile.

### Stage 3: Future Prediction

Historical position alone is not enough for tactical value, so the system projects the submarine forward.

This stage:

- derives velocity vectors from reconstructed positions
- smooths the trajectory using a constant-velocity Kalman filter
- propagates future states in time
- injects synthetic ocean-current drift into the motion model
- expands forecast uncertainty as prediction horizon increases

This produces a projected route rather than a single static endpoint.

### Stage 4: Tactical Intelligence

The final stage turns raw prediction into operational guidance.

This stage:

- checks whether tactical assets can reach the predicted contact zone
- computes feasible interception windows
- detects anomalies such as depth excursions and evasive turns
- scores projected threat level over time
- recommends a tactical action for the operator

This is the part that makes the system feel like a decision-support platform rather than just a signal-processing demo.

---

## Algorithms Used

## 1. Sound Speed Estimation

Implemented in [sound_speed.py](/Users/sumit/Documents/Hackathon_iste/abyssal_echo/sound_speed.py)

Sound propagation underwater depends strongly on environmental conditions. A fixed sound speed would introduce range error, so the system uses a simplified Mackenzie-style approximation:

```text
v = 1449 + 4.6T - 0.055T² + 0.00029T³ + (1.34 - 0.01T)(S - 35)
```

Where:

- `T` = temperature in Celsius
- `S` = salinity in PSU
- `v` = sound speed in meters per second

Why this matters:

- travel time is converted into distance using sound speed
- even small sound-speed errors can distort multilateration
- environmental variation is especially relevant in deep-ocean scenarios

How it is achieved:

- each ping row is vectorized through the formula using Pandas and NumPy
- the result is stored as `Sound_Speed_mps`
- all downstream range calculations use this corrected value

---

## 2. Echo and Reflection Filtering

Implemented in [echo_filter.py](/Users/sumit/Documents/Hackathon_iste/abyssal_echo/echo_filter.py)

Each `Packet_ID` is assumed to appear three times:

- one direct arrival
- two reflected arrivals

The system keeps only the highest-intensity row per packet as the direct signal.

Why this matters:

- reflections arrive later than the direct path
- if echoes are kept, travel time becomes artificially large
- that leads to false distance estimates and incorrect positions

How it is achieved:

- pings are grouped by `Packet_ID`
- `Intensity_dB` is ranked within each group
- the row with rank `1` is marked as `Is_Primary = True`
- the other rows remain available for the waterfall view as discarded echoes

This is a simple but effective physically motivated heuristic for a prototype setting.

---

## 3. Clock Drift Synchronization

Implemented in [clock_sync.py](/Users/sumit/Documents/Hackathon_iste/abyssal_echo/clock_sync.py)

Hydrophone clocks are not assumed to be perfectly synchronized. A stationary sync buoy exists at `(0, 0, 0)`, and its emission times are known.

For each sync-buoy packet:

```text
drift = observed_timestamp - emission_timestamp - expected_travel_time
```

Where:

- `expected_travel_time = distance(sensor, buoy) / sound_speed`

Why this matters:

- even a small timing offset translates into large range error underwater
- multilateration is very sensitive to timestamp bias

How it is achieved:

- identify rows where `Source_Label == "sync_buoy"`
- compute sensor-to-buoy distance from hydrophone coordinates
- compute expected travel time using the local sound speed
- estimate drift per observation
- take the median drift per `Sensor_ID`
- subtract it from all timestamps of that sensor

This produces a corrected time column:

- `Corrected_Timestamp_ms`

---

## 4. Doppler Velocity Estimation

Implemented in [doppler_velocity.py](/Users/sumit/Documents/Hackathon_iste/abyssal_echo/doppler_velocity.py)

The engine logs provide:

- `RPM`
- `Blade_Count`

From that, the source blade-pass frequency is estimated as:

```text
f_source = (RPM / 60) * Blade_Count
```

The simplified Doppler relation used is:

```text
f_received = f_source * v / (v + v_source)
```

Rearranging for submarine velocity:

```text
v_source = v * (1 - r) / r
```

Where:

- `r = f_received / f_source`
- `v` = sound speed

Why this matters:

- it gives a physically grounded estimate of vessel motion from acoustics
- speed estimation is later used in forecasting and tactical inference

How it is achieved:

- engine logs are aligned to pings with `merge_asof`
- blade frequency is computed from engine telemetry
- the Doppler equation is solved row-wise
- results are converted into:
  - `Submarine_Velocity_mps`
  - `Submarine_Speed_knots`

---

## 5. Trajectory Reconstruction by Multilateration

Implemented in [triangulation.py](/Users/sumit/Documents/Hackathon_iste/abyssal_echo/triangulation.py)

After timestamp correction, travel time can be converted into distance:

```text
distance = sound_speed * travel_time
```

For each ping group, each hydrophone gives one sphere of possible source locations. The submarine position is estimated as the point that best satisfies all sphere constraints.

Mathematically, the solver minimizes:

```text
||sensor_position_i - candidate_position|| - estimated_range_i
```

across all sensors.

Why this matters:

- real hydrophone geometry is rarely perfect
- noise and measurement distortion mean spheres do not intersect exactly
- a least-squares estimate is more robust than a closed-form exact solution

How it is achieved:

- group rows by `Ping_Group`
- compute sensor coordinates and estimated ranges
- solve for the best 3D point using `scipy.optimize.least_squares`
- generate both:
  - raw position estimate
  - corrected position estimate

This allows the dashboard to visually compare improvement before and after synchronization.

---

## 6. Velocity Vector Extrapolation

Implemented in [future_prediction.py](/Users/sumit/Documents/Hackathon_iste/abyssal_echo/future_prediction.py)

Once positions are reconstructed, the system estimates velocity vectors from the smoothed position series using numerical gradients:

```text
vx = dX/dt
vy = dY/dt
vz = dZ/dt
```

Why this matters:

- a future path requires both position and direction
- scalar speed alone is not enough for trajectory continuation

How it is achieved:

- sort reconstructed trajectory by corrected time
- compute per-axis gradients with respect to time
- derive magnitude in both meters per second and knots

This forms the base motion model for forecasting.

---

## 7. Kalman Filter Smoothing

Implemented in [future_prediction.py](/Users/sumit/Documents/Hackathon_iste/abyssal_echo/future_prediction.py)

The reconstructed path still contains noise, so the system applies a constant-velocity Kalman filter.

State vector:

```text
[x, y, z, vx, vy, vz]
```

Observation vector:

```text
[x, y, z]
```

Why this matters:

- raw multilateration outputs jitter due to noise and imperfect ranges
- forecasting from noisy positions causes unstable future paths
- the Kalman filter gives a smoother latent state estimate

How it is achieved:

- initialize state from the earliest measured position
- estimate transition matrix using the time delta between samples
- model process noise and measurement noise
- iteratively predict and update state
- save filtered positions and filtered velocities

Produced columns include:

- `Kalman_X`
- `Kalman_Y`
- `Kalman_Z`
- `Kalman_Vx_mps`
- `Kalman_Vy_mps`
- `Kalman_Vz_mps`

---

## 8. Ocean Current Drift Compensation

Implemented in [future_prediction.py](/Users/sumit/Documents/Hackathon_iste/abyssal_echo/future_prediction.py)

Submarine motion is not assumed to occur in still water. A synthetic `ocean_currents.csv` field provides current components over:

- time
- depth

Current components:

- `Current_U_mps`
- `Current_V_mps`
- `Current_W_mps`

Why this matters:

- long-horizon underwater prediction should account for environmental drift
- this makes the forecast more realistic and less purely kinematic

How it is achieved:

- build interpolators over `(Timestamp_ms, Depth_m)`
- evaluate the current vector at each future step
- add the current vector to the Kalman velocity
- propagate future position forward

This creates a current-adjusted projected path rather than a naive straight extrapolation.

---

## 9. Prediction Uncertainty Envelope

Implemented in [tactical_intelligence.py](/Users/sumit/Documents/Hackathon_iste/abyssal_echo/tactical_intelligence.py)

Predictions become less reliable farther into the future. To represent that, the system expands an uncertainty radius with time.

The base uncertainty is derived from Kalman residuals, then increased as a function of prediction horizon.

Why this matters:

- judges and operators trust systems more when uncertainty is explicit
- a single future line can look overconfident
- uncertainty rings turn the output into a believable tactical estimate

How it is achieved:

- compare corrected positions to Kalman positions
- compute a baseline residual scale
- increase radius with square-root growth over time
- derive a confidence score inversely from uncertainty

Output fields include:

- `Uncertainty_Radius_m`
- `Confidence_Score`

---

## 10. Interception Window Analysis

Implemented in [future_prediction.py](/Users/sumit/Documents/Hackathon_iste/abyssal_echo/future_prediction.py)

The project includes synthetic tactical response assets, each with:

- starting location
- maximum speed
- launch delay
- detection radius

For each projected submarine position, the system checks whether an asset can reach the intercept zone before the submarine gets there.

Why this matters:

- it transforms the project from passive monitoring into decision support
- it answers “can we act?” rather than only “what do we know?”

How it is achieved:

- compute asset-to-target distance
- convert asset speed from knots to meters per second
- account for launch delay and detection radius
- compare required response time with time remaining to target
- mark feasible interception opportunities

This produces:

- `interception_windows.csv`
- `interception_summary.csv`

---

## 11. Anomaly Detection

Implemented in [tactical_intelligence.py](/Users/sumit/Documents/Hackathon_iste/abyssal_echo/tactical_intelligence.py)

The system detects unusual maneuver patterns from the Kalman-smoothed trajectory.

Currently detected event types:

- **Hard Turn**
- **Depth Excursion**
- **Silent Running**

Why this matters:

- unusual maneuvers often indicate tactical intent
- anomaly detection helps narrate the mission picture to judges
- it makes the dashboard feel more intelligent

How it is achieved:

- compute heading from horizontal velocity
- compute heading-rate change
- compute acceleration from speed gradient
- compute depth-rate change
- apply thresholds to generate labeled events with severity

This produces:

- `anomaly_events.csv`

---

## 12. Threat Scoring

Implemented in [tactical_intelligence.py](/Users/sumit/Documents/Hackathon_iste/abyssal_echo/tactical_intelligence.py)

Threat scoring combines several pieces of information:

- distance to tactical assets
- projected speed
- anomaly severity
- intercept feasibility
- uncertainty penalty

Why this matters:

- a dashboard is more useful when it summarizes urgency clearly
- a single risk score helps judges immediately understand operational significance

How it is achieved:

- normalize proximity, speed, and anomaly influence
- reward feasible intercept conditions
- penalize large uncertainty
- combine into a bounded score between `0` and `1`
- convert score into:
  - `Low`
  - `Medium`
  - `High`

This produces:

- `threat_timeseries.csv`
- `threat_summary.csv`

The summary also includes a recommended tactical action.

---

## 13. Tactical Ocean Map

Implemented in [dashboard.py](/Users/sumit/Documents/Hackathon_iste/abyssal_echo/dashboard.py)

To make the visualization feel less abstract, the dashboard includes a synthetic trench bathymetry layer.

Why this matters:

- seabed context strengthens the realism of the scenario
- it makes the tactical view more visually memorable
- it provides environmental storytelling for the judges

How it is achieved:

- generate a synthetic seabed grid in `bathymetry.csv`
- draw contour layers with Plotly
- overlay:
  - Kalman-smoothed track
  - projected path
  - tactical assets
  - feasible intercept zones
  - projected contact uncertainty marker

---

## Why These Algorithms Were Chosen

This project is a prototype, so the design balances **physical plausibility**, **clarity**, and **demo value**.

Why these choices work well:

- **Mackenzie-style sound speed** is lightweight and physically meaningful
- **strongest-return echo filtering** is simple and appropriate for the problem statement
- **median clock drift estimation** is robust against noisy sync-buoy observations
- **Doppler velocity estimation** directly connects acoustics to vessel motion
- **least-squares multilateration** is a standard and reliable reconstruction method
- **Kalman filtering** is a natural upgrade from noisy position estimates to stable state estimation
- **current-drift propagation** makes future prediction more realistic
- **threat scoring and anomalies** turn technical output into mission intelligence

In short, the project is not just mathematically correct enough for a prototype, but also shaped to communicate well during a hackathon demo.

---

## Synthetic Data Strategy

Implemented primarily in [data_loader.py](/Users/sumit/Documents/Hackathon_iste/abyssal_echo/data_loader.py)

If real datasets are missing, the system generates synthetic but internally consistent data for:

- acoustic pings
- engine logs
- ocean currents
- tactical assets
- bathymetry

The synthetic generator simulates:

- multiple hydrophones with fixed drift offsets
- a moving submarine track through the trench
- sync-buoy emissions
- direct arrivals and reflections
- environmental variation in temperature and salinity
- blade-pass frequencies from RPM
- current drift over depth and time
- interceptor starting positions and capabilities

This is important because it makes the project:

- immediately runnable
- easy to demo
- self-contained for evaluation

---

## Project Structure

```text
Hackathon_iste/
├── abyssal_echo/
│   ├── __init__.py
│   ├── clock_sync.py               # Sync buoy drift estimation and correction
│   ├── dashboard.py                # Streamlit + Plotly tactical dashboard
│   ├── data_loader.py              # CSV loading and synthetic data generation
│   ├── doppler_velocity.py         # Blade frequency alignment and Doppler speed solving
│   ├── echo_filter.py              # Primary arrival detection and echo labeling
│   ├── future_prediction.py        # Velocity vectors, Kalman filter, projection, interception
│   ├── main.py                     # End-to-end pipeline orchestration
│   ├── sound_speed.py              # Sound speed enrichment
│   ├── tactical_intelligence.py    # Uncertainty, anomalies, threat scoring
│   └── triangulation.py            # Range estimation and multilateration
├── data/                           # Input datasets or generated demo inputs
├── outputs/                        # Generated pipeline outputs
├── main.py                         # CLI entrypoint
├── README.md
└── requirements.txt
```

---

## Input Data

### `acoustic_pings.csv`

Core columns:

- `Packet_ID`
- `Timestamp_ms`
- `Sensor_ID`
- `Received_Frequency_Hz`
- `Intensity_dB`
- `Temperature_C`
- `Salinity_PSU`
- `Depth_m`
- `X`
- `Y`
- `Z`

Additional prototype columns used internally:

- `Emission_Timestamp_ms`
- `Ping_Group`
- `Source_Label`

### `engine_logs.csv`

- `Timestamp_ms`
- `RPM`
- `Blade_Count`

### `ocean_currents.csv`

- `Timestamp_ms`
- `Depth_m`
- `Current_U_mps`
- `Current_V_mps`
- `Current_W_mps`

### `tactical_assets.csv`

- `Asset_ID`
- `Asset_Type`
- `Base_X_m`
- `Base_Y_m`
- `Base_Z_m`
- `Max_Speed_knots`
- `Launch_Delay_s`
- `Detection_Radius_m`

### `bathymetry.csv`

- `X_m`
- `Y_m`
- `Seabed_Depth_m`

---

## Output Files

After a successful run, `outputs/` contains:

- `cleaned_acoustic_pings.csv`
- `sensor_clock_drift.csv`
- `doppler_enriched_pings.csv`
- `doppler_speed_summary.csv`
- `reconstructed_trajectory.csv`
- `kalman_smoothed_trajectory.csv`
- `predicted_future_path.csv`
- `interception_windows.csv`
- `interception_summary.csv`
- `tactical_assets_snapshot.csv`
- `bathymetry_snapshot.csv`
- `anomaly_events.csv`
- `threat_timeseries.csv`
- `threat_summary.csv`

---

## Dashboard Views

The Streamlit dashboard presents the analysis as an interactive mission console.

It includes:

- 3D comparison of raw, corrected, smoothed, and predicted trajectories
- top-down tactical projection map
- replay timeline slider
- sound-speed and depth HUD
- Doppler speedometer
- signal waterfall showing primary vs echo arrivals
- interception window panel
- anomaly feed
- threat console with recommended action
- bathymetry-backed tactical ocean map
- uncertainty-aware contact zone visualization

---

## Running The Project

## Setup

macOS/Homebrew Python may block global installation with the `externally-managed-environment` message. Use a virtual environment:

```bash
cd /Users/sumit/Documents/Hackathon_iste
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Run The Pipeline

```bash
python3 main.py
```

This will:

- load real datasets if present
- otherwise generate synthetic demo datasets
- run the full reconstruction and prediction pipeline
- save all outputs into `outputs/`

## Launch The Dashboard

```bash
python3 main.py --dashboard
```

Or after the outputs already exist:

```bash
streamlit run /Users/sumit/Documents/Hackathon_iste/abyssal_echo/dashboard.py -- outputs
```

Notes:

- Streamlit may ask for an email on first launch
- this is optional
- press `Enter` to skip

---

## Main Processing Flow

1. Load or generate datasets
2. Compute local sound speed
3. Label and filter reflections
4. Estimate hydrophone clock drift
5. Correct timestamps
6. Align engine logs
7. Compute Doppler speed
8. Estimate distances from travel time
9. Reconstruct submarine position by multilateration
10. Smooth state with a Kalman filter
11. Project future path with current drift
12. Expand uncertainty envelope
13. Evaluate tactical interception opportunities
14. Detect anomalies
15. Score threat and recommend an action
16. Visualize everything in the dashboard

---

## Verification

Verified locally in the project virtual environment with:

```bash
python3 -m compileall main.py abyssal_echo
./.venv/bin/python main.py
```

Example current output summary:

- pipeline completes successfully
- predicted future path is generated
- threat summary is populated
- interception summary contains feasible tactical options

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'numpy'`

You are likely not inside the project virtual environment:

```bash
source /Users/sumit/Documents/Hackathon_iste/.venv/bin/activate
python3 -m pip install -r requirements.txt
python3 main.py
```

### `externally-managed-environment`

Do not install packages globally. Use the virtual environment shown above.

### Streamlit asks for email

This is a Streamlit onboarding prompt, not part of the project. Press `Enter` to skip it.

---

## Entry Points

- Batch entrypoint: [/Users/sumit/Documents/Hackathon_iste/main.py](/Users/sumit/Documents/Hackathon_iste/main.py)
- Pipeline orchestrator: [/Users/sumit/Documents/Hackathon_iste/abyssal_echo/main.py](/Users/sumit/Documents/Hackathon_iste/abyssal_echo/main.py)
- Dashboard: [/Users/sumit/Documents/Hackathon_iste/abyssal_echo/dashboard.py](/Users/sumit/Documents/Hackathon_iste/abyssal_echo/dashboard.py)

---

## Final Note

What makes Abyssal Echo strong as a hackathon project is that it does not stop at cleaning data or drawing a path. It combines:

- scientific reasoning
- signal-processing logic
- environmental modeling
- probabilistic filtering
- tactical decision support
- strong visual storytelling

That combination makes it both technically interesting and demo-friendly.
