import streamlit as st
import pandas as pd
from datetime import datetime

from data_utils import load_player_stats

# --------------------------------------------------------
# Page config
# --------------------------------------------------------
st.set_page_config(
    page_title="Mojo Analytics — Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------
# Header
# --------------------------------------------------------
st.title("Mojo Analytics — Demo")
st.subheader("Fantasy value explained in plain English.")
st.info("This is a cloud-hosted demo. No betting advice, no guarantees.")

# --------------------------------------------------------
# Load REAL weekly stats (WORKING PIPELINE)
# --------------------------------------------------------
@st.cache_data
def get_player_stats():
    df = load_player_stats([2023, 2024, 2025])
    df = df.dropna(subset=["player_display_name", "week", "season"])
    return df

stats = get_player_stats()

# --------------------------------------------------------
# Sidebar: Season + Week
# --------------------------------------------------------
with st.sidebar:
    st.markdown("### 📅 Season / Week")

    seasons = sorted(stats["season"].unique().tolist())
    season = st.selectbox("Season", seasons, index=len(seasons) - 1)

    season_df = stats[stats["season"] == season]
    max_week = int(season_df["week"].max())

    week = st.slider(
        "Evaluate trends up to week",
        min_value=3,
        max_value=max_week,
        value=max_week,
        step=1,
    )

# --------------------------------------------------------
# Compute DraftKings fantasy points (1 decimal)
# --------------------------------------------------------
def dk_points(row):
    pts = 0.0
    pts += row.get("passing_yards", 0) * 0.04
    pts += row.get("passing_tds", 0) * 4
    pts -= row.get("interceptions", 0)
    pts += row.get("rushing_yards", 0) * 0.1
    pts += row.get("rushing_tds", 0) * 6
    pts += row.get("receiving_yards", 0) * 0.1
    pts += row.get("receiving_tds", 0) * 6
    pts += row.get("receptions", 0)
    pts -= row.get("fumbles_lost", 0)
    return round(pts, 1)

stats["dk_points"] = stats.apply(dk_points, axis=1)

# --------------------------------------------------------
# Last-3 games (chronological)
# --------------------------------------------------------
recent = (
    stats[
        (stats["season"] == season) &
        (stats["week"] <= week)
    ]
    .sort_values(["player_display_name", "week"])
    .groupby("player_display_name")
    .tail(3)
)

# Label weeks (W18, W19, etc.)
recent["WeekLabel"] = recent["week"].apply(lambda w: f"W{int(w)}")

# --------------------------------------------------------
# Aggregate per player
# --------------------------------------------------------
agg = (
    recent
    .groupby("player_display_name")
    .agg(
        Position=("position", "first"),
        Team=("recent_team", "first"),
        Weeks=("WeekLabel", list),
        Pts=("dk_points", list),
        Avg_Last3=("dk_points", "mean"),
    )
    .reset_index()
)

agg["Avg_Last3"] = agg["Avg_Last3"].round(1)

# --------------------------------------------------------
# Trend arrow
# --------------------------------------------------------
def trend_arrow(vals):
    if len(vals) < 2:
        return "➖"
    if vals[-1] > vals[0]:
        return "🔺"
    if vals[-1] < vals[0]:
        return "🔻"
    return "➖"

agg["Trend"] = agg["Pts"].apply(trend_arrow)

# --------------------------------------------------------
# Sparkline (OLD → NEW)
# --------------------------------------------------------
def sparkline(vals):
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return blocks[3] * len(vals)
    return "".join(
        blocks[int((v - lo) / (hi - lo) * (len(blocks) - 1))]
        for v in vals
    )

agg["Last3_Spark"] = agg["Pts"].apply(sparkline)

# --------------------------------------------------------
# Player Pool
# --------------------------------------------------------
st.markdown("## 🧾 Player Pool")

c1, c2 = st.columns([1, 2])
with c1:
    pos_filter = st.selectbox(
        "Position",
        ["ALL"] + sorted(agg["Position"].dropna().unique().tolist())
    )
with c2:
    search = st.text_input("Search player")

pool = agg.copy()
if pos_filter != "ALL":
    pool = pool[pool["Position"] == pos_filter]
if search:
    pool = pool[pool["player_display_name"].str.lower().str.contains(search.lower())]

pool_view = pool[
    ["player_display_name", "Position", "Team", "Last3_Spark", "Avg_Last3", "Trend"]
].rename(columns={
    "player_display_name": "Player",
    "Avg_Last3": "Avg DK (Last 3)"
})

st.dataframe(pool_view, use_container_width=True, hide_index=True)

# --------------------------------------------------------
# Player Details
# --------------------------------------------------------
st.markdown("## 🔎 Player Details")

if pool.empty:
    st.stop()

player = st.selectbox("Select player", pool["player_display_name"].tolist())
row = pool[pool["player_display_name"] == player].iloc[0]

st.markdown(f"### {player} ({row['Position']} – {row['Team']})")

m1, m2, m3 = st.columns(3)
m1.metric("Avg DK (Last 3)", f"{row['Avg_Last3']:.1f}")
m2.metric("Trend", row["Trend"])
m3.metric("Games Used", len(row["Pts"]))

detail_df = pd.DataFrame({
    "Week": row["Weeks"],
    "DK Points": [round(v, 1) for v in row["Pts"]],
})

st.dataframe(detail_df, use_container_width=True, hide_index=True)
st.line_chart(detail_df.set_index("Week"))

st.caption(
    "DraftKings scoring only. "
    "Last-3 games shown oldest → newest (trend-correct). "
    "Data sourced via nflreadpy (same pipeline as betting analyzer)."
)
