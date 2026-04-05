"""Echo and reflection filtering helpers."""

from __future__ import annotations

import pandas as pd


def flag_primary_signals(pings: pd.DataFrame) -> pd.DataFrame:
    """Mark the highest-intensity row per packet as the primary signal."""
    ranked = pings.copy()
    ranked["Echo_Rank"] = ranked.groupby("Packet_ID")["Intensity_dB"].rank(
        method="first", ascending=False
    )
    ranked["Is_Primary"] = ranked["Echo_Rank"] == 1
    return ranked


def keep_primary_signals(pings: pd.DataFrame) -> pd.DataFrame:
    """Filter the ping frame to keep only direct arrivals."""
    flagged = flag_primary_signals(pings)
    return (
        flagged.loc[flagged["Is_Primary"]]
        .drop(columns=["Echo_Rank"])
        .sort_values("Timestamp_ms")
        .reset_index(drop=True)
    )

