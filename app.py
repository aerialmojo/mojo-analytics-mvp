# app.py — Mojo Analytics (NFL) — Streamlit
# Requirements (requirements.txt):
# streamlit>=1.37,<2
# pandas>=2.1,<3
# numpy>=1.26,<3
# nfl-data-py

import re
import numpy as np
import pandas as pd
import streamlit as st

# If you have data_utils.py in your repo (recommended):
from data_utils import load_player_stats


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Mojo Analytics — Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Header
# -----------------------------
st.title("Mojo Analytics — Demo")
st.subheader("Fantasy value explained in plain English.")
st.info("This is a cloud-hosted demo. No betting advice, no guarantees.")


# -----------------------------
# Helpers
# -----------------------------
def normalize_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s).strip()
    return s


def sparkline(vals):
    blocks = "▁▂▃▄▅▆▇█"
    vals = [0 if v is None else float(v) for v in vals]
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        return blocks[3] * len(vals)
    out = ""
    for v in vals:
        idx = int((v - vmin) / (vmax - vmin) * (len(blocks) - 1))
        out += blocks[idx]
    return out


def trend_arrow(v1, v2, v3, flat_eps: float = 0.75) -> str:
    """Compare L3 to L1; small delta treated as flat."""
    try:
        a, c = float(v1), float(v3)
    except Exception:
        return "→"
    delta = c - a
    if abs(delta) <= flat_eps:
        return "→"
    return "↑" if delta > 0 else "↓"


def ensure_column(df: pd.DataFrame, target: str, fallbacks: list[str], default_val=""):
    """
    Ensure df[target] exists. If not, copy from first existing fallback,
    otherwise set to default_val.
    """
    if target in df.columns:
        return df
    for fb in fallbacks:
        if fb in df.columns:
            df[target] = df[fb]
            return df
    df[target] = default_val
    return df


def compute_dk_points(df: pd.DataFrame) -> pd.Series:
    """
    DK scoring (practical, 1-decimal):
      Passing: 0.04/yd, 4/TD, -1 INT, +3 at 300+
      Rushing: 0.1/yd, 6/TD, +3 at 100+
      Receiving: 1/rec, 0.1/yd, 6/TD, +3 at 100+
      Fumbles lost: -1 each
      2PT: +2 each (pass/rush/rec)
    """
    needed = [
        "passing_yards", "passing_tds", "interceptions",
        "rushing_yards", "rushing_tds",
        "receptions", "receiving_yards", "receiving_tds",
        "fumbles_lost",
        "passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions",
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = 0

    pass_yds = pd.to_numeric(df["passing_yards"], errors="coerce").fillna(0)
    pass_td  = pd.to_numeric(df["passing_tds"], errors="coerce").fillna(0)
    ints     = pd.to_numeric(df["interceptions"], errors="coerce").fillna(0)

    rush_yds = pd.to_numeric(df["rushing_yards"], errors="coerce").fillna(0)
    rush_td  = pd.to_numeric(df["rushing_tds"], errors="coerce").fillna(0)

    rec      = pd.to_numeric(df["receptions"], errors="coerce").fillna(0)
    rec_yds  = pd.to_numeric(df["receiving_yards"], errors="coerce").fillna(0)
    rec_td   = pd.to_numeric(df["receiving_tds"], errors="coerce").fillna(0)

    fum_lost = pd.to_numeric(df["fumbles_lost"], errors="coerce").fillna(0)

    p2  = pd.to_numeric(df["passing_2pt_conversions"], errors="coerce").fillna(0)
    r2  = pd.to_numeric(df["rushing_2pt_conversions"], errors="coerce").fillna(0)
    rc2 = pd.to_numeric(df["receiving_2pt_conversions"], errors="coerce").fillna(0)

    pts = (
        pass_yds * 0.04 +
        pass_td  * 4.0 +
        ints     * -1.0 +
        rush_yds * 0.10 +
        rush_td  * 6.0 +
        rec      * 1.0 +
        rec_yds  * 0.10 +
        rec_td   * 6.0 +
        fum_lost * -1.0 +
        (p2 + r2 + rc2) * 2.0
    )

    pts = pts + (pass_yds >= 300).astype(int) * 3.0
    pts = pts + (rush_yds >= 100).astype(int) * 3.0
    pts = pts + (rec_yds >= 100).astype(int) * 3.0

    return pts.round(1)


def deterministic_salary(player_key: str, pos: str, seed_tag: str) -> int:
    ranges = {
        "QB": (5200, 8800),
        "RB": (4000, 9200),
        "WR": (3000, 9000),
        "TE": (2500, 7800),
        "DST": (2000, 4500),
    }
    lo, hi = ranges.get(pos, (3000, 8000))
    h = abs(hash(f"{seed_tag}:{pos}:{player_key}")) % (10**9)
    val = lo + (h % (hi - lo + 1))
    return int(round(val / 100) * 100)


# -----------------------------
# Load weekly stats (real)
# -----------------------------
@st.cache_data(ttl=6 * 60 * 60)
def get_weekly(seasons):
    df = load_player_stats(seasons)

    # Normalize key columns across datasets
    df = ensure_column(df, "player_display_name", ["player_name", "name", "player"], default_val=None)
    df = ensure_column(df, "season", ["year"], default_val=None)
    df = ensure_column(df, "week", ["week_num", "game_week"], default_val=None)

    # IMPORTANT: your crash is here — make these exist
    df = ensure_column(df, "position", ["pos", "position_group"], default_val="")
    df = ensure_column(df, "recent_team", ["team", "posteam", "team_abbr", "club"], default_val="")

    # Clean types
    df = df.dropna(subset=["player_display_name", "season", "week"])
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df["position"] = df["position"].astype(str).str.upper().str.strip()
    df["recent_team"] = df["recent_team"].astype(str).str.upper().str.strip()

    return df


# -----------------------------
# Sidebar: Season
# -----------------------------
with st.sidebar:
    st.markdown("### 📅 Season")
    season = st.selectbox("Choose season", [2025, 2024, 2023], index=0)
    st.caption("Stats pulled automatically via nfl-data-py (free).")

try:
    weekly_all = get_weekly([2023, 2024, 2025])
except Exception as e:
    st.error("Could not load weekly player stats.")
    st.code(str(e))
    st.stop()

weekly = weekly_all[weekly_all["season"] == int(season)].copy()
if weekly.empty:
    st.warning(f"No weekly data found for season {season}.")
    st.stop()

# Compute DK points
weekly["dk_points"] = compute_dk_points(weekly)

latest_week = int(pd.to_numeric(weekly["week"], errors="coerce").max())

# -----------------------------
# Sidebar: Week slider
# -----------------------------
with st.sidebar:
    st.markdown("### 🗓️ Week")
    mode = st.radio("Last-3 mode", ["Latest available", "Choose a week"], index=0, horizontal=False)
    if mode == "Latest available":
        up_to_week = latest_week + 1
        st.caption(f"Using last-3 games prior to Week {up_to_week} (latest in data: Week {latest_week}).")
    else:
        up_to_week = st.slider(
            "Compute last-3 games before this week",
            min_value=1,
            max_value=latest_week + 1,
            value=latest_week + 1,
            step=1
        )

weekly_cut = weekly[weekly["week"] < int(up_to_week)].copy()
if weekly_cut.empty:
    st.warning("No games found before that week. Choose a later week.")
    st.stop()

# -----------------------------
# Build last-3 + season average metrics
# -----------------------------
weekly_cut = weekly_cut.sort_values(["player_display_name", "week"], ascending=[True, False])

last3 = weekly_cut.groupby("player_display_name").head(3).copy()
last3["rank"] = last3.groupby("player_display_name").cumcount() + 1

def pivot_last3(col, pref):
    if col not in last3.columns:
        return None
    p = last3.pivot_table(index="player_display_name", columns="rank", values=col, aggfunc="first")
    return p.rename(columns={1: f"{pref}_L1", 2: f"{pref}_L2", 3: f"{pref}_L3"}).reset_index()

metrics = pivot_last3("dk_points", "Pts")
if metrics is None:
    metrics = pd.DataFrame({"player_display_name": weekly_cut["player_display_name"].unique()})

for col, pref in [
    ("passing_yards", "PassYds"),
    ("passing_tds", "PassTD"),
    ("rushing_yards", "RushYds"),
    ("receiving_yards", "RecYds"),
    ("receptions", "Rec"),
]:
    p = pivot_last3(col, pref)
    if p is not None:
        metrics = metrics.merge(p, on="player_display_name", how="left")

season_avg = (
    weekly_cut.groupby("player_display_name")["dk_points"]
    .mean()
    .reset_index()
    .rename(columns={"dk_points": "PPG_Season"})
)
metrics = metrics.merge(season_avg, on="player_display_name", how="left")

# Identity from most recent game in cut (SAFE because we forced cols above)
identity = (
    weekly_cut
    .sort_values(["player_display_name", "week"], ascending=[True, False])
    .drop_duplicates("player_display_name")[["player_display_name", "position", "recent_team"]]
    .rename(columns={"position": "Position", "recent_team": "Team"})
)
metrics = metrics.merge(identity, on="player_display_name", how="left")

# Clean + fill numeric
for c in metrics.columns:
    if c not in ["player_display_name", "Position", "Team"]:
        metrics[c] = pd.to_numeric(metrics[c], errors="coerce").fillna(0)

metrics["Player"] = metrics["player_display_name"].astype(str)
metrics["Position"] = metrics["Position"].astype(str).str.upper().str.strip()
metrics["Team"] = metrics["Team"].fillna("").astype(str).str.upper().str.strip()

# Keep core DK roster positions
metrics = metrics[metrics["Position"].isin(["QB", "RB", "WR", "TE"])].copy()

# Salaries: deterministic placeholders (until DK salary integration)
seed_tag = f"{season}-W{up_to_week}"
metrics["player_key"] = metrics["Player"].apply(normalize_name)
metrics["Salary"] = metrics.apply(lambda r: deterministic_salary(r["player_key"], r["Position"], seed_tag), axis=1)

# Derived metrics (1 decimal)
metrics["Avg_Last3"] = metrics[["Pts_L1", "Pts_L2", "Pts_L3"]].mean(axis=1).round(1)
metrics["Value_per_1k"] = (metrics["Avg_Last3"] / (metrics["Salary"] / 1000)).replace([np.inf, -np.inf], 0).fillna(0).round(1)
metrics["Last3_Spark"] = metrics.apply(lambda r: sparkline([r["Pts_L1"], r["Pts_L2"], r["Pts_L3"]]), axis=1)
metrics["Trend"] = metrics.apply(lambda r: trend_arrow(r["Pts_L1"], r["Pts_L2"], r["Pts_L3"]), axis=1)

df = metrics[
    ["Player", "Position", "Team", "Salary", "PPG_Season",
     "Pts_L1", "Pts_L2", "Pts_L3",
     "PassYds_L1", "PassYds_L2", "PassYds_L3",
     "PassTD_L1", "PassTD_L2", "PassTD_L3",
     "RushYds_L1", "RushYds_L2", "RushYds_L3",
     "RecYds_L1", "RecYds_L2", "RecYds_L3",
     "Rec_L1", "Rec_L2", "Rec_L3",
     "Avg_Last3", "Value_per_1k", "Last3_Spark", "Trend"]
].copy()

# Add simple DST placeholders
dst_pool = pd.DataFrame({
    "Player": ["49ers DST", "Cowboys DST", "Eagles DST", "Chiefs DST"],
    "Position": ["DST", "DST", "DST", "DST"],
    "Team": ["SF", "DAL", "PHI", "KC"],
    "Salary": [3200, 3500, 3300, 3400],
    "PPG_Season": [0.0, 0.0, 0.0, 0.0],
    "Pts_L1": [0.0, 0.0, 0.0, 0.0],
    "Pts_L2": [0.0, 0.0, 0.0, 0.0],
    "Pts_L3": [0.0, 0.0, 0.0, 0.0],
    "PassYds_L1": [0.0]*4, "PassYds_L2": [0.0]*4, "PassYds_L3": [0.0]*4,
    "PassTD_L1": [0.0]*4, "PassTD_L2": [0.0]*4, "PassTD_L3": [0.0]*4,
    "RushYds_L1": [0.0]*4, "RushYds_L2": [0.0]*4, "RushYds_L3": [0.0]*4,
    "RecYds_L1": [0.0]*4, "RecYds_L2": [0.0]*4, "RecYds_L3": [0.0]*4,
    "Rec_L1": [0.0]*4, "Rec_L2": [0.0]*4, "Rec_L3": [0.0]*4,
    "Avg_Last3": [0.0]*4,
    "Value_per_1k": [0.0]*4,
    "Last3_Spark": ["▁▁▁"]*4,
    "Trend": ["→"]*4,
})

df_all = pd.concat([df, dst_pool[df.columns]], ignore_index=True)

st.caption(
    f"Season {season}: data through Week {latest_week}. "
    f"Last-3 shows last 3 games played before Week {up_to_week}."
)

# -----------------------------
# Roster config (DraftKings-style)
# -----------------------------
salary_cap = 50000
slots = [
    ("QB", ["QB"]),
    ("RB1", ["RB"]),
    ("RB2", ["RB"]),
    ("WR1", ["WR"]),
    ("WR2", ["WR"]),
    ("WR3", ["WR"]),
    ("TE", ["TE"]),
    ("FLEX", ["RB", "WR", "TE"]),
    ("DST", ["DST"]),
]
slot_names = [s[0] for s in slots]

for slot_name, _ in slots:
    st.session_state.setdefault(f"slot_{slot_name}", "—")

def get_selected_players():
    out = []
    for slot_name in slot_names:
        v = st.session_state.get(f"slot_{slot_name}", "—")
        if v and v != "—":
            out.append(v)
    return out

# -----------------------------
# Sidebar: Roster Builder
# -----------------------------
with st.sidebar:
    st.markdown("## 🧩 Roster Builder")
    st.caption("DraftKings-style demo roster")

    selected_now = get_selected_players()

    for slot_name, allowed_positions in slots:
        pool_slot = df_all[df_all["Position"].isin(allowed_positions)].copy()

        current_val = st.session_state.get(f"slot_{slot_name}", "—")
        exclude = set(selected_now)
        if current_val != "—":
            exclude.discard(current_val)
        pool_slot = pool_slot[~pool_slot["Player"].isin(exclude)]

        pool_slot["Label"] = pool_slot.apply(
            lambda r: f'{r["Player"]} ({r["Team"]}) — ${int(r["Salary"]):,} — {r["Trend"]} V {r["Value_per_1k"]:.1f}',
            axis=1
        )

        label_to_player = dict(zip(pool_slot["Label"], pool_slot["Player"]))
        player_to_label = {v: k for k, v in label_to_player.items()}

        options = ["—"] + pool_slot["Label"].tolist()

        default_label = "—"
        if current_val != "—" and current_val in player_to_label:
            default_label = player_to_label[current_val]

        picked_label = st.selectbox(
            slot_name,
            options,
            index=options.index(default_label) if default_label in options else 0,
            key=f"ui_{slot_name}",
        )

        st.session_state[f"slot_{slot_name}"] = label_to_player.get(picked_label, "—")

    chosen = get_selected_players()
    total_salary = int(df_all[df_all["Player"].isin(chosen)]["Salary"].sum()) if chosen else 0

    st.markdown("---")
    st.metric("Salary Used", f"${total_salary:,}")
    st.metric("Remaining", f"${salary_cap - total_salary:,}")

    if total_salary > salary_cap:
        st.error("Over the $50,000 salary cap. Try swapping to lower-cost players.")

    if st.button("Reset Lineup"):
        for sname in slot_names:
            st.session_state[f"slot_{sname}"] = "—"
            st.session_state[f"ui_{sname}"] = "—"
        st.rerun()

# -----------------------------
# Main: Selected lineup summary
# -----------------------------
chosen_players = get_selected_players()

st.markdown("## ✅ Selected Lineup")
if chosen_players:
    lineup_df = df_all[df_all["Player"].isin(chosen_players)][
        ["Player", "Position", "Team", "Salary", "PPG_Season", "Trend", "Last3_Spark", "Avg_Last3", "Value_per_1k"]
    ].copy()

    lineup_df = lineup_df.rename(columns={
        "PPG_Season": "SeasonAvg(DK)",
        "Avg_Last3": "Last3Avg(DK)",
        "Value_per_1k": "Value/$1k",
    })

    st.dataframe(lineup_df, use_container_width=True, hide_index=True)
else:
    st.caption("Use the sidebar to build a lineup, then browse the pool for details.")

st.markdown("---")

# -----------------------------
# Main: Player Pool
# -----------------------------
st.markdown("## 🧾 Player Pool")

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    pos_filter = st.selectbox("Position", ["ALL", "QB", "RB", "WR", "TE", "DST"], index=0)
with c2:
    team_filter = st.selectbox("Team", ["ALL"] + sorted(df_all["Team"].dropna().unique().tolist()))
with c3:
    search = st.text_input("Search player name", value="").strip().lower()

pool_df = df_all.copy()
if pos_filter != "ALL":
    pool_df = pool_df[pool_df["Position"] == pos_filter]
if team_filter != "ALL":
    pool_df = pool_df[pool_df["Team"] == team_filter]
if search:
    pool_df = pool_df[pool_df["Player"].str.lower().str.contains(search)]

display_cols = ["Player", "Position", "Team", "Salary", "PPG_Season", "Trend", "Last3_Spark", "Avg_Last3", "Value_per_1k"]
pool_view = pool_df[display_cols].sort_values(["Position", "Value_per_1k"], ascending=[True, False]).copy()

pool_view = pool_view.rename(columns={
    "PPG_Season": "SeasonAvg(DK)",
    "Avg_Last3": "Last3Avg(DK)",
    "Value_per_1k": "Value/$1k",
})

st.data_editor(pool_view, use_container_width=True, hide_index=True, disabled=True)

# -----------------------------
# Main: Player Details
# -----------------------------
st.markdown("## 🔎 Player Details")

if pool_df.empty:
    st.warning("No players found for this filter/search.")
    st.stop()

detail_player = st.selectbox(
    "Select a player from this pool",
    pool_df["Player"].tolist(),
    key="detail_player",
)

row = df_all[df_all["Player"] == detail_player].iloc[0]
pos = row["Position"]

st.markdown(f"#### {row['Player']} ({pos}) — {row['Team']}")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Salary", f"${int(row['Salary']):,}")
m2.metric("Season Avg (DK)", f"{float(row['PPG_Season']):.1f}")
m3.metric("Avg Last 3 (DK)", f"{float(row['Avg_Last3']):.1f}")
m4.metric("Value / $1k", f"{float(row['Value_per_1k']):.1f}")
m5.metric("Trend", f"{row['Trend']}")

st.markdown("**Last 3 Fantasy Points (DK calc)**")
pts_df = pd.DataFrame({
    "Game": ["L1", "L2", "L3"],
    "DK_Points": [float(row["Pts_L1"]), float(row["Pts_L2"]), float(row["Pts_L3"])],
})
st.dataframe(pts_df, use_container_width=True, hide_index=True)
st.line_chart(pts_df.set_index("Game")["DK_Points"])

st.markdown("**Last 3 Stat Breakdown**")
if pos == "QB":
    detail_df = pd.DataFrame({
        "Game": ["L1", "L2", "L3"],
        "PassYds": [float(row["PassYds_L1"]), float(row["PassYds_L2"]), float(row["PassYds_L3"])],
        "PassTD":  [float(row["PassTD_L1"]),  float(row["PassTD_L2"]),  float(row["PassTD_L3"])],
        "RushYds": [float(row["RushYds_L1"]), float(row["RushYds_L2"]), float(row["RushYds_L3"])],
        "DK_Points": [float(row["Pts_L1"]), float(row["Pts_L2"]), float(row["Pts_L3"])],
    })
elif pos in ["RB", "WR", "TE"]:
    detail_df = pd.DataFrame({
        "Game": ["L1", "L2", "L3"],
        "RushYds": [float(row["RushYds_L1"]), float(row["RushYds_L2"]), float(row["RushYds_L3"])],
        "Receptions": [float(row["Rec_L1"]), float(row["Rec_L2"]), float(row["Rec_L3"])],
        "RecYds": [float(row["RecYds_L1"]), float(row["RecYds_L2"]), float(row["RecYds_L3"])],
        "DK_Points": [float(row["Pts_L1"]), float(row["Pts_L2"]), float(row["Pts_L3"])],
    })
else:
    detail_df = pd.DataFrame({
        "Game": ["L1", "L2", "L3"],
        "DK_Points": [float(row["Pts_L1"]), float(row["Pts_L2"]), float(row["Pts_L3"])],
    })

st.dataframe(detail_df, use_container_width=True, hide_index=True)

st.caption(
    "DK points are computed in-app (1 decimal). "
    "Salaries are demo placeholders until you connect real DraftKings salaries."
)
