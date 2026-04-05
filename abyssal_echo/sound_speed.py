"""Sound speed calculations."""

from __future__ import annotations

import pandas as pd


def compute_sound_speed(temperature_c: pd.Series, salinity_psu: pd.Series) -> pd.Series:
    """Compute seawater sound speed with a simplified Mackenzie relation."""
    t = temperature_c.astype(float)
    s = salinity_psu.astype(float)
    return (
        1449
        + 4.6 * t
        - 0.055 * (t**2)
        + 0.00029 * (t**3)
        + (1.34 - 0.01 * t) * (s - 35)
    )


def enrich_with_sound_speed(pings: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the ping frame with a Sound_Speed_mps column."""
    enriched = pings.copy()
    enriched["Sound_Speed_mps"] = compute_sound_speed(
        enriched["Temperature_C"], enriched["Salinity_PSU"]
    )
    return enriched

