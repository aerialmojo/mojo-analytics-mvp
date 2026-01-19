import re
import requests
import streamlit as st
import pandas as pd

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
    # Remove punctuation/periods/apostrophes, collapse whitespace
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    # common suffixes
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s).strip()
    return s

def safe_col(df: pd.DataFrame, col: str, default=0):
    return df[col] if col in df.columns else default

def dk_fantasy_points(row: pd.Series) -> float:
    """
    DraftKings NFL classic scoring (core pieces).
    Notes:
    - Passing: 0.04/yd, 4/TD, -1/INT, +3 bonus for 300+ pass yds
    - Rushing: 0.1/yd, 6/TD, +3 bonus for 100+ rush yds
    - Receiving: 1/reception, 0.1/yd, 6/TD, +3 bonus for 100+ rec yds
    - Fumbles lost: -1 (if available)
    """
    pass_yds = float(row.get("passing_yards", 0) or 0)
    pass_td = float(row.get("passing_tds", 0) or 0)
    ints = float(row.get("interceptions", 0) or 0)

    rush_yds = float(row.get("rushing_yards", 0) or 0)
    rush_td = float(row.get("rushing_tds", 0) or 0)

    rec = float(row.get("receptions", 0) or 0)
    rec_yds = float(row.get("receiving_yards", 0) or 0)
    rec_td = float(row.get("receiving_tds", 0) or 0)

    fum_lost = float(row.get("fumbles_lost", 0) or 0)

    pts = 0.0
    pts += pass_yds * 0.04
    pts += pass_td * 4.0
    pts += ints * -1.0
    if pass_yds >= 300:
        pts += 3.0

    pts += rush_yds * 0.10
    pts += rush_td * 6.0
    if rush_yds >= 100:
        pts += 3.0

    pts += rec * 1.0
    pts += rec_yds * 0.10
    pts += rec_td * 6.0
    if rec_yds >= 100:
        pts += 3.0

    pts += fum_lost * -1.0
    return round(pts, 2)

def sparkline(vals):
    blocks = "▁▂▃▄▅▆▇█"
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        return blocks[3] * len(vals)
    out = ""
    for v in vals:
        idx = int((v - vmin) / (vmax - vmin) * (len(blocks) - 1))
        out += blocks[idx]
    return out

# -----------------------------
# Load DraftKings weekly pool (CSV upload)
# -----------------------------
@st.cache_data(ttl=3600)
def load_dk_csv(uploaded_file) -> pd.DataFrame:
    raw = pd.read_csv(uploaded_file)

    # Normalize likely DK columns
    cols = {c.lower(): c for c in raw.columns}

    def pick(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    name_col = pick("Name", "Player", "Nickname")
    pos_col  = pick("Position", "Roster Position", "RosterPosition")
    sal_col  = pick("Salary")
    team_col = pick("TeamAbbrev", "Team", "Team Abbrev", "team")

    if not (name_col and pos_col and sal_col):
        raise ValueError(f"DK CSV missing required columns. Found: {list(raw.columns)}")

    df = pd.DataFrame({
        "Player": raw[name_col].astype(str),
        "Position": raw[pos_col].astype(str).str.upper().str.strip(),
        "Salary": pd.to_numeric(raw[sal_col], errors="coerce").fillna(0).astype(int),
        "Team": raw[team_col].astype(str) if team_col else ""
    })

    # Keep classic NFL positions
    df = df[df["Position"].isin(["QB", "RB", "WR", "TE", "DST"])].copy()
    df["player_key"] = df["Player"].apply(normalize_name)

    # DK DST names may be like "49ers" or "San Francisco"; keep as-is for now
    return df.reset_index(drop=True)

# -----------------------------
# Load free weekly player stats (nflverse)
# -----------------------------
@st.cache_data(ttl=3600)
def load_nflverse_weekly_player_stats(season: int) -> pd.DataFrame:
    # Public release CSVs (free). Example known pattern in nflverse-data releases.
    # This is the common filename used by the community.
    url = f"https://github.com/nflverse/nflverse-data/releases/download/player_stats/stats_player_week_{season}.csv"
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    stats = pd.read_csv(pd.io.common.BytesIO(r.content))

    # Try to normalize expected columns across versions
    # Common nflverse columns include:
    # player_name, position, week, passing_yards, passing_tds, interceptions,
    # rushing_yards, rushing_tds, receiving_yards, receiving_tds, receptions, fumbles_lost, etc.
    if "player_name" not in stats.columns:
        # Sometimes it might be "player_display_name" or similar
        for alt in ["player_display_name", "name", "player"]:
            if alt in stats.columns:
                stats = stats.rename(columns={alt: "player_name"})
                break

    if "week" not in stats.columns:
        for alt in ["week_num", "game_week"]:
            if alt in stats.columns:
                stats = stats.rename(columns={alt: "week"})
                break

    if "position" not in stats.columns:
        for alt in ["pos"]:
            if alt in stats.columns:
                stats = stats.rename(columns={alt: "position"})
                break

    # Keep only skill positions + QB for now; DST handled separately
    stats["player_key"] = stats["player_name"].apply(normalize_name)
    stats["position"] = stats["position"].astype(str).str.upper().str.strip()
    stats["week"] = pd.to_numeric(stats["week"], errors="coerce")

    # Compute DK fantasy points from available stat columns
    stats["dk_points"] = stats.apply(dk_fantasy_points, axis=1)

    return stats

def build_last3_and_season_metrics(stats: pd.DataFrame, up_to_week: int) -> pd.DataFrame:
    """
    Returns one row per player with:
    - last 3 fantasy points (Pts_L1/L2/L3)
    - last 3 passing/rushing/receiving stats
    - season avg DK points up to (up_to_week-1)
    """
    s = stats.copy()
    s = s[s["week"].notna()].copy()
    s = s[s["week"] < up_to_week].copy()

    # If user picks week 1, there is no "previous" data; handle gracefully
    if s.empty:
        return pd.DataFrame(columns=["player_key"])

    # Sort so we can take last 3 games
    s = s.sort_values(["player_key", "week"], ascending=[True, False])

    # Take last 3 rows per player
    last3 = s.groupby("player_key").head(3).copy()

    # Rank within last3 so we can pivot into L1/L2/L3
    last3["rank"] = last3.groupby("player_key").cumcount() + 1  # 1..3

    # Pivot DK points
    pts_pivot = last3.pivot_table(index="player_key", columns="rank", values="dk_points", aggfunc="first")
    pts_pivot = pts_pivot.rename(columns={1: "Pts_L1", 2: "Pts_L2", 3: "Pts_L3"}).reset_index()

    # Pivot some core stats we care about
    def pivot_stat(col, out_prefix):
        if col not in last3.columns:
            return None
        p = last3.pivot_table(index="player_key", columns="rank", values=col, aggfunc="first")
        p = p.rename(columns={1: f"{out_prefix}_L1", 2: f"{out_prefix}_L2", 3: f"{out_prefix}_L3"}).reset_index()
        return p

    pivots = [pts_pivot]

    for col, pref in [
        ("passing_yards", "PassYds"),
        ("passing_tds", "PassTD"),
        ("rushing_yards", "RushYds"),
        ("receiving_yards", "RecYds"),
        ("receptions", "Rec"),
    ]:
        p = pivot_stat(col, pref)
        if p is not None:
            pivots.append(p)

    # Merge all pivots
    out = pivots[0]
    for p in pivots[1:]:
        out = out.merge(p, on="player_key", how="left")

    # Season average DK points up to week-1
    season_avg = s.groupby("player_key")["dk_points"].mean().reset_index().rename(columns={"dk_points": "PPG_Season"})
    out = out.merge(season_avg, on="player_key", how="left")

    # Fill missing last3 with NaN -> 0 (players with <3 games)
    for c in out.columns:
        if c != "player_key":
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out

# -----------------------------
# UI: choose week/season + upload DK CSV
# -----------------------------
st.markdown("## 📥 Data Source")

cA, cB, cC = st.columns([1, 1, 2])

with cA:
    season = st.number_input("Season", min_value=2016, max_value=2030, value=2025, step=1)

with cB:
    week = st.number_input("Current Week", min_value=1, max_value=22, value=1, step=1)
    st.caption("Last-3 + season stats are computed using games **before** this week.")

with cC:
    uploaded = st.file_uploader("Upload DraftKings NFL salary CSV", type=["csv"])

if not uploaded:
    st.info("Upload a DraftKings salary CSV to populate the real weekly pool.")
    st.stop()

dk_pool = load_dk_csv(uploaded)

# Load free weekly stats and build features
stats = load_nflverse_weekly_player_stats(int(season))
metrics = build_last3_and_season_metrics(stats, up_to_week=int(week))

# Merge metrics into DK pool
df = dk_pool.merge(metrics, on="player_key", how="left")

# Default missing stats to 0 (and PPG to 0 if no games yet)
for col in [
    "Pts_L1", "Pts_L2", "Pts_L3",
    "PassYds_L1", "PassYds_L2", "PassYds_L3",
    "PassTD_L1", "PassTD_L2", "PassTD_L3",
    "RushYds_L1", "RushYds_L2", "RushYds_L3",
    "RecYds_L1", "RecYds_L2", "RecYds_L3",
    "Rec_L1", "Rec_L2", "Rec_L3",
    "PPG_Season",
]:
    if col not in df.columns:
        df[col] = 0
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# Derived metrics for your UI
df["Avg_Last3"] = df[["Pts_L1", "Pts_L2", "Pts_L3"]].mean(axis=1).round(1)
df["Value_Last3_per_$1k"] = (df["Avg_Last3"] / (df["Salary"] / 1000)).replace([pd.NA, pd.NaT], 0).fillna(0).round(2)
df["Value_Season_per_$1k"] = (df["PPG_Season"] / (df["Salary"] / 1000)).replace([pd.NA, pd.NaT], 0).fillna(0).round(2)
df["Last3_Spark"] = df.apply(lambda r: sparkline([r["Pts_L1"], r["Pts_L2"], r["Pts_L3"]]), axis=1)

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

# Ensure session_state keys exist
for slot_name, _ in slots:
    st.session_state.setdefault(f"slot_{slot_name}", "—")

def get_selected_players():
    selected = []
    for slot_name in slot_names:
        val = st.session_state.get(f"slot_{slot_name}", "—")
        if val and val != "—":
            selected.append(val)
    return selected

# -----------------------------
# Sidebar: Roster Builder
# -----------------------------
with st.sidebar:
    st.markdown("## 🧩 Roster Builder")
    st.caption("DraftKings-style demo roster")

    selected_now = get_selected_players()

    for slot_name, allowed_positions in slots:
        pool = df[df["Position"].isin(allowed_positions)].copy()

        current_val = st.session_state.get(f"slot_{slot_name}", "—")
        exclude = set(selected_now)
        if current_val != "—":
            exclude.discard(current_val)

        pool = pool[~pool["Player"].isin(exclude)]

        pool["Label"] = pool.apply(
            lambda r: f'{r["Player"]} — ${int(r["Salary"]):,} — V3 {r["Value_Last3_per_$1k"]}',
            axis=1
        )

        label_to_player = dict(zip(pool["Label"], pool["Player"]))
        player_to_label = {v: k for k, v in label_to_player.items()}

        options = ["—"] + pool["Label"].tolist()

        default_label = "—"
        if current_val != "—" and current_val in player_to_label:
            default_label = player_to_label[current_val]

        picked_label = st.selectbox(
            slot_name,
            options,
            index=options.index(default_label) if default_label in options else 0,
            key=f"ui_{slot_name}"
        )

        st.session_state[f"slot_{slot_name}"] = label_to_player.get(picked_label, "—")

    chosen_players = get_selected_players()
    total_salary = int(df[df["Player"].isin(chosen_players)]["Salary"].sum()) if chosen_players else 0

    st.markdown("---")
    st.metric("Salary Used", f"${total_salary:,}")
    st.metric("Remaining", f"${salary_cap - total_salary:,}")

    if total_salary > salary_cap:
        st.error("Over the $50,000 salary cap. Try swapping to lower-cost players.")

    if st.button("Reset Lineup"):
        for slot_name in slot_names:
            st.session_state[f"slot_{slot_name}"] = "—"
            st.session_state[f"ui_{slot_name}"] = "—"
        st.rerun()

# -----------------------------
# Main: Selected lineup summary
# -----------------------------
chosen_players = get_selected_players()

st.markdown("## ✅ Selected Lineup")
if chosen_players:
    lineup_df = df[df["Player"].isin(chosen_players)][
        ["Player", "Position", "Salary", "PPG_Season", "Last3_Spark", "Avg_Last3", "Value_Last3_per_$1k"]
    ].copy()

    st.dataframe(lineup_df, use_container_width=True, hide_index=True)
else:
    st.caption("Use the sidebar to build a lineup, then browse the weekly pool for details.")

st.markdown("---")

# -----------------------------
# Main: Player Pool
# -----------------------------
st.markdown("## 🧾 Player Pool (This Week)")

pos_filter = st.selectbox("Filter by Position", ["ALL", "QB", "RB", "WR", "TE", "DST"], index=0)
pool_df = df.copy()
if pos_filter != "ALL":
    pool_df = pool_df[pool_df["Position"] == pos_filter]

display_cols = ["Player", "Position", "Team", "Salary", "PPG_Season", "Last3_Spark", "Avg_Last3", "Value_Last3_per_$1k"]
pool_view = pool_df[display_cols].copy()

st.data_editor(pool_view, use_container_width=True, hide_index=True, disabled=True)

# -----------------------------
# Main: Player Details
# -----------------------------
st.markdown("## 🔎 Player Details")

if pool_df.empty:
    st.warning("No players found for this filter.")
    st.stop()

detail_player = st.selectbox(
    "Select a player from this week's pool",
    pool_df["Player"].tolist(),
    key="detail_player"
)

row = df[df["Player"] == detail_player].iloc[0]
pos = row["Position"]

st.markdown(f"#### {row['Player']} ({pos})")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Salary", f"${int(row['Salary']):,}")
m2.metric("Season Avg (DK pts)", f"{row['PPG_Season']:.2f}")
m3.metric("Avg Last 3 (DK pts)", f"{row['Avg_Last3']:.2f}")
m4.metric("Value (Last 3) / $1k", f"{row['Value_Last3_per_$1k']:.2f}")

# Last 3 fantasy points
st.markdown("**Last 3 Fantasy Points (DraftKings scoring)**")
pts_df = pd.DataFrame({
    "Game": ["L1", "L2", "L3"],
    "FantasyPts": [row["Pts_L1"], row["Pts_L2"], row["Pts_L3"]],
})
st.dataframe(pts_df, use_container_width=True, hide_index=True)
st.line_chart(pts_df.set_index("Game")["FantasyPts"])

# Position-specific breakdown
st.markdown("**Last 3 Stat Breakdown**")

if pos == "QB":
    detail_df = pd.DataFrame({
        "Game": ["L1", "L2", "L3"],
        "PassYds": [row["PassYds_L1"], row["PassYds_L2"], row["PassYds_L3"]],
        "PassTD":  [row["PassTD_L1"],  row["PassTD_L2"],  row["PassTD_L3"]],
        "RushYds": [row["RushYds_L1"], row["RushYds_L2"], row["RushYds_L3"]],
        "FantasyPts": [row["Pts_L1"], row["Pts_L2"], row["Pts_L3"]],
    })
elif pos in ["RB", "WR", "TE"]:
    detail_df = pd.DataFrame({
        "Game": ["L1", "L2", "L3"],
        "RushYds": [row["RushYds_L1"], row["RushYds_L2"], row["RushYds_L3"]],
        "Receptions": [row["Rec_L1"], row["Rec_L2"], row["Rec_L3"]],
        "RecYds": [row["RecYds_L1"], row["RecYds_L2"], row["RecYds_L3"]],
        "FantasyPts": [row["Pts_L1"], row["Pts_L2"], row["Pts_L3"]],
    })
else:  # DST – we don’t have free standardized DST boxscore stats in this file
    detail_df = pd.DataFrame({
        "Game": ["L1", "L2", "L3"],
        "FantasyPts": [row["Pts_L1"], row["Pts_L2"], row["Pts_L3"]],
    })

st.dataframe(detail_df, use_container_width=True, hide_index=True)

st.caption(
    "Weekly player stats are pulled from nflverse public data files; DK fantasy points are computed in-app. "
    "Name matching is best-effort; a small manual mapping table can improve edge cases."
)
