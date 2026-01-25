import re
import hashlib
import numpy as np
import pandas as pd
import streamlit as st

# Free stats source
import nfl_data_py as nfl


# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Mojo Analytics — Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Mojo Analytics — Demo")
st.subheader("Fantasy value explained in plain English.")
st.info("This is a cloud-hosted demo. No betting advice, no guarantees.")


# =========================================================
# Helpers
# =========================================================
def normalize_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s).strip()
    return s


def sparkline(vals):
    """Unicode sparkline. Input should already be in display order (oldest -> newest)."""
    blocks = "▁▂▃▄▅▆▇█"
    vals = [float(v) for v in vals]
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        return blocks[3] * len(vals)
    out = ""
    for v in vals:
        idx = int((v - vmin) / (vmax - vmin) * (len(blocks) - 1))
        out += blocks[idx]
    return out


def trend_arrow(oldest: float, newest: float, threshold: float = 0.75) -> str:
    """Simple trend arrow based on last3 delta."""
    delta = float(newest) - float(oldest)
    if delta >= threshold:
        return "⬆️"
    if delta <= -threshold:
        return "⬇️"
    return "➡️"


def dk_points(row: pd.Series) -> float:
    """DraftKings-style scoring (offense). Handles missing columns safely."""
    def g(col):
        v = row.get(col, 0)
        if pd.isna(v):
            return 0.0
        try:
            return float(v)
        except Exception:
            return 0.0

    pass_yds = g("passing_yards")
    pass_td  = g("passing_tds")
    ints     = g("interceptions")
    # some data uses passing_interceptions
    ints_alt = g("passing_interceptions")
    if ints == 0 and ints_alt != 0:
        ints = ints_alt

    rush_yds = g("rushing_yards")
    rush_td  = g("rushing_tds")

    rec      = g("receptions")
    rec_yds  = g("receiving_yards")
    rec_td   = g("receiving_tds")

    fum_lost = g("fumbles_lost")

    # 2pt conversions (varies by dataset)
    tp_total = g("two_point_conversions")
    if tp_total == 0:
        tp_total = (
            g("two_point_conversions_passed")
            + g("two_point_conversions_received")
            + g("two_point_conversions_rushed")
        )

    pts = 0.0

    # Passing
    pts += pass_yds * 0.04
    pts += pass_td * 4.0
    pts += ints * -1.0
    if pass_yds >= 300:
        pts += 3.0

    # Rushing
    pts += rush_yds * 0.10
    pts += rush_td * 6.0
    if rush_yds >= 100:
        pts += 3.0

    # Receiving (PPR)
    pts += rec * 1.0
    pts += rec_yds * 0.10
    pts += rec_td * 6.0
    if rec_yds >= 100:
        pts += 3.0

    # Fumbles lost
    pts += fum_lost * -1.0

    # 2pt conversions
    pts += tp_total * 2.0

    return float(round(pts, 3))


def deterministic_salary(player_key: str, pos: str, seed_tag: str) -> int:
    """Deterministic demo salary so UI works without DK salary CSV."""
    ranges = {
        "QB": (5200, 8800),
        "RB": (4000, 9200),
        "WR": (3000, 9000),
        "TE": (2500, 7800),
        "DST": (2000, 4500),
    }
    lo, hi = ranges.get(pos, (3000, 8000))
    h = hashlib.md5(f"{seed_tag}:{pos}:{player_key}".encode("utf-8")).hexdigest()
    n = int(h[:8], 16)
    val = lo + (n % (hi - lo + 1))
    return int(round(val / 100) * 100)


# =========================================================
# Load data (nfl_data_py weekly)
# =========================================================
@st.cache_data(ttl=3600)
def load_weekly(seasons: list[int]) -> pd.DataFrame:
    df = nfl.import_weekly_data(seasons)
    # Standardize expected columns (best-effort)
    # nfl_data_py typically includes: player_display_name, position, recent_team, opponent_team, season, week, etc.
    return df


# Choose last 3 seasons (including 2025 if present)
# We’ll infer available seasons from what loads successfully.
default_seasons_try = [2025, 2024, 2023, 2022, 2021]

weekly_all = None
loaded_seasons = []

for s in default_seasons_try:
    try:
        tmp = load_weekly([s])
        if tmp is not None and not tmp.empty and "season" in tmp.columns:
            loaded_seasons.append(s)
    except Exception:
        pass

if len(loaded_seasons) == 0:
    st.error("Could not load weekly NFL data right now via nfl_data_py.")
    st.stop()

# keep last 3 seasons, newest-first
loaded_seasons = sorted(loaded_seasons, reverse=True)[:3]

with st.sidebar:
    st.markdown("### 📅 Season")
    season = st.selectbox("Choose season", loaded_seasons, index=0)
    st.caption("Stats pulled automatically via nfl_data_py (free).")

weekly = load_weekly([int(season)]).copy()

# Ensure needed columns exist
for col in ["player_display_name", "position", "recent_team", "opponent_team", "season", "week"]:
    if col not in weekly.columns:
        weekly[col] = np.nan

weekly["week"] = pd.to_numeric(weekly["week"], errors="coerce")
weekly["season"] = pd.to_numeric(weekly["season"], errors="coerce")

weekly = weekly.dropna(subset=["player_display_name", "week"]).copy()
weekly["player_display_name"] = weekly["player_display_name"].astype(str).str.strip()
weekly["position"] = weekly["position"].astype(str).str.upper().str.strip()
weekly["recent_team"] = weekly["recent_team"].astype(str).str.upper().str.strip()

# Compute DK points for each row
weekly["dk_points"] = weekly.apply(dk_points, axis=1)

latest_week = int(pd.to_numeric(weekly["week"], errors="coerce").max())


with st.sidebar:
    st.markdown("### 🗓️ Week")
    mode = st.radio(
        "Last-3 mode",
        ["Latest available", "Choose a week"],
        index=0,
        horizontal=True,
    )
    if mode == "Latest available":
        up_to_week = latest_week + 1
        st.caption(f"Using last-3 games prior to Week {up_to_week} (latest in data: Week {latest_week}).")
    else:
        up_to_week = st.slider(
            "Compute last-3 games prior to this week",
            min_value=1,
            max_value=latest_week + 1,
            value=latest_week + 1,
            step=1,
        )


# =========================================================
# Build player pool + last3 metrics (before up_to_week)
# =========================================================
def build_pool_and_metrics(weekly_df: pd.DataFrame, up_to_week: int) -> pd.DataFrame:
    w = weekly_df.copy()
    w = w[(w["week"].notna()) & (w["week"] < int(up_to_week))].copy()

    if w.empty:
        return pd.DataFrame(columns=["Player", "Position", "Team", "player_key"])

    # Sort to get last games
    w = w.sort_values(["player_display_name", "week"], ascending=[True, False])

    # Take last 3 games per player (rank 1 = most recent)
    last3 = w.groupby("player_display_name").head(3).copy()
    last3["rank"] = last3.groupby("player_display_name").cumcount() + 1

    # Pivot helper
    def pivot(col: str, prefix: str):
        if col not in last3.columns:
            return None
        p = last3.pivot_table(
            index="player_display_name",
            columns="rank",
            values=col,
            aggfunc="first"
        )
        p = p.rename(columns={
            1: f"{prefix}_L1",  # most recent
            2: f"{prefix}_L2",
            3: f"{prefix}_L3",  # oldest (of last3)
        }).reset_index()
        return p

    # Always capture week numbers for last3
    wk = pivot("week", "Week")
    pts = pivot("dk_points", "Pts")

    # Other stat pivots (best-effort)
    pivots = [wk, pts]
    for col, pref in [
        ("passing_yards", "PassYds"),
        ("passing_tds", "PassTD"),
        ("interceptions", "Ints"),
        ("passing_interceptions", "PassInts"),
        ("rushing_yards", "RushYds"),
        ("rushing_tds", "RushTD"),
        ("receptions", "Rec"),
        ("receiving_yards", "RecYds"),
        ("receiving_tds", "RecTD"),
        ("fumbles_lost", "FumLost"),
        ("two_point_conversions", "TwoPt"),
        ("two_point_conversions_passed", "TwoPtPass"),
        ("two_point_conversions_received", "TwoPtRec"),
        ("two_point_conversions_rushed", "TwoPtRush"),
    ]:
        p = pivot(col, pref)
        if p is not None:
            pivots.append(p)

    # Merge pivots
    metrics = pivots[0]
    for p in pivots[1:]:
        metrics = metrics.merge(p, on="player_display_name", how="outer")

    # Season average DK points (before up_to_week)
    season_avg = (
        w.groupby("player_display_name")["dk_points"]
        .mean()
        .reset_index()
        .rename(columns={"dk_points": "PPG_Season_DK"})
    )
    metrics = metrics.merge(season_avg, on="player_display_name", how="left")

    # Player base info: take the most recent row prior to up_to_week
    latest_rows = (
        w.sort_values(["player_display_name", "week"], ascending=[True, False])
         .drop_duplicates("player_display_name")
         [["player_display_name", "position", "recent_team"]]
         .rename(columns={"recent_team": "Team", "position": "Position"})
    )

    out = latest_rows.merge(metrics, on="player_display_name", how="left")
    out = out.rename(columns={"player_display_name": "Player"})
    out["player_key"] = out["Player"].apply(normalize_name)

    # Limit to main DK positions for now
    out = out[out["Position"].isin(["QB", "RB", "WR", "TE"])].copy()

    # Add a small MVP DST list (real DST stats can come later)
    dst = pd.DataFrame({
        "Player": ["49ers DST", "Cowboys DST", "Eagles DST", "Chiefs DST"],
        "Position": ["DST", "DST", "DST", "DST"],
        "Team": ["SF", "DAL", "PHI", "KC"],
    })
    dst["player_key"] = dst["Player"].apply(normalize_name)
    out = pd.concat([out, dst], ignore_index=True)

    return out


df = build_pool_and_metrics(weekly, up_to_week=int(up_to_week))

if df.empty:
    st.warning("No player data found for this season/week selection.")
    st.stop()

# Fill missing numeric columns
need_cols = [
    "Week_L1", "Week_L2", "Week_L3",
    "Pts_L1", "Pts_L2", "Pts_L3",
    "PassYds_L1", "PassYds_L2", "PassYds_L3",
    "PassTD_L1", "PassTD_L2", "PassTD_L3",
    "RushYds_L1", "RushYds_L2", "RushYds_L3",
    "RecYds_L1", "RecYds_L2", "RecYds_L3",
    "Rec_L1", "Rec_L2", "Rec_L3",
    "FumLost_L1", "FumLost_L2", "FumLost_L3",
    "TwoPt_L1", "TwoPt_L2", "TwoPt_L3",
    "PPG_Season_DK",
]
for c in need_cols:
    if c not in df.columns:
        df[c] = 0
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# Combine two-point conversion components if needed (if TwoPt_* columns are 0 but others exist)
for rank in ["L1", "L2", "L3"]:
    base = f"TwoPt_{rank}"
    if base not in df.columns:
        df[base] = 0
    # Only add if component cols exist
    for comp in [f"TwoPtPass_{rank}", f"TwoPtRec_{rank}", f"TwoPtRush_{rank}"]:
        if comp in df.columns:
            df[base] = df[base] + pd.to_numeric(df[comp], errors="coerce").fillna(0)

# Demo salaries (deterministic; replace later with DK salary import)
seed_tag = f"{season}-W{up_to_week}"
df["Salary"] = df.apply(lambda r: deterministic_salary(r["player_key"], r["Position"], seed_tag), axis=1)

# Derived metrics (1 decimal where appropriate)
df["Avg_Last3"] = df[["Pts_L1", "Pts_L2", "Pts_L3"]].mean(axis=1).round(1)
df["Value_per_1k"] = (df["Avg_Last3"] / (df["Salary"] / 1000)).replace([np.inf, -np.inf], 0).fillna(0).round(1)

# Sparkline should be OLDEST -> NEWEST: L3, L2, L1
df["Last3_Spark"] = df.apply(lambda r: sparkline([r["Pts_L3"], r["Pts_L2"], r["Pts_L1"]]), axis=1)

# Trend arrow based on oldest vs newest DK points
df["Trend"] = df.apply(lambda r: trend_arrow(r["Pts_L3"], r["Pts_L1"]), axis=1)

st.caption(
    f"Season {season}: data detected through Week {latest_week}. "
    f"Last-3 shows games played prior to Week {up_to_week}."
)


# =========================================================
# Roster config (DraftKings-style)
# =========================================================
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
    selected = []
    for slot_name in slot_names:
        val = st.session_state.get(f"slot_{slot_name}", "—")
        if val and val != "—":
            selected.append(val)
    return selected


# =========================================================
# Sidebar: Roster Builder
# =========================================================
with st.sidebar:
    st.markdown("## 🧩 Roster Builder")
    st.caption("DraftKings-style demo roster")

    selected_now = get_selected_players()

    for slot_name, allowed_positions in slots:
        pool_slot = df[df["Position"].isin(allowed_positions)].copy()

        current_val = st.session_state.get(f"slot_{slot_name}", "—")
        exclude = set(selected_now)
        if current_val != "—":
            exclude.discard(current_val)

        pool_slot = pool_slot[~pool_slot["Player"].isin(exclude)]

        pool_slot["Label"] = pool_slot.apply(
            lambda r: f'{r["Player"]} {r["Trend"]} — ${int(r["Salary"]):,} — V {r["Value_per_1k"]:.1f}',
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


# =========================================================
# Main: Selected lineup summary
# =========================================================
chosen_players = get_selected_players()

st.markdown("## ✅ Selected Lineup")
if chosen_players:
    lineup_df = df[df["Player"].isin(chosen_players)][
        ["Player", "Position", "Team", "Salary", "Trend", "Last3_Spark", "Avg_Last3", "Value_per_1k"]
    ].copy()

    st.dataframe(
        lineup_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Salary": st.column_config.NumberColumn(format="$%d"),
            "Avg_Last3": st.column_config.NumberColumn(format="%.1f"),
            "Value_per_1k": st.column_config.NumberColumn(format="%.1f"),
        },
    )
else:
    st.caption("Use the sidebar to build a lineup, then browse the pool for details.")

st.markdown("---")


# =========================================================
# Main: Player Pool
# =========================================================
st.markdown("## 🧾 Player Pool")

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    pos_filter = st.selectbox("Filter by Position", ["ALL", "QB", "RB", "WR", "TE", "DST"], index=0)
with c2:
    team_filter = st.selectbox("Team", ["ALL"] + sorted([t for t in df["Team"].dropna().unique().tolist() if t]), index=0)
with c3:
    search = st.text_input("Search player name", value="").strip().lower()

pool_df = df.copy()
if pos_filter != "ALL":
    pool_df = pool_df[pool_df["Position"] == pos_filter]
if team_filter != "ALL":
    pool_df = pool_df[pool_df["Team"] == team_filter]
if search:
    pool_df = pool_df[pool_df["Player"].str.lower().str.contains(search)]

display_cols = ["Player", "Position", "Team", "Salary", "Trend", "Last3_Spark", "Avg_Last3", "Value_per_1k"]
pool_view = (
    pool_df[display_cols]
    .sort_values(["Position", "Value_per_1k"], ascending=[True, False])
    .copy()
)

st.data_editor(
    pool_view,
    use_container_width=True,
    hide_index=True,
    disabled=True,
    column_config={
        "Salary": st.column_config.NumberColumn(format="$%d"),
        "Avg_Last3": st.column_config.NumberColumn(format="%.1f"),
        "Value_per_1k": st.column_config.NumberColumn(format="%.1f"),
    },
)


# =========================================================
# Main: Player Details (chronological last 3 with week labels)
# =========================================================
st.markdown("## 🔎 Player Details")

if pool_df.empty:
    st.warning("No players found for this filter/search.")
    st.stop()

detail_player = st.selectbox(
    "Select a player from this pool",
    pool_df["Player"].tolist(),
    key="detail_player",
)

row = df[df["Player"] == detail_player].iloc[0]
pos = row["Position"]

st.markdown(f"#### {row['Player']} ({pos}) — {row.get('Team','')}")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Salary (demo)", f"${int(row['Salary']):,}")
m2.metric("Season Avg (DK)", f"{float(row['PPG_Season_DK']):.1f}")
m3.metric("Avg Last 3 (DK)", f"{float(row['Avg_Last3']):.1f}")
m4.metric("Value / $1k", f"{float(row['Value_per_1k']):.1f} {row['Trend']}")

# Build chronological labels: oldest->newest = L3, L2, L1
week_labels = [
    f"W{int(row['Week_L3'])}" if row["Week_L3"] else "W?",
    f"W{int(row['Week_L2'])}" if row["Week_L2"] else "W?",
    f"W{int(row['Week_L1'])}" if row["Week_L1"] else "W?",
]

st.markdown("**Last 3 Fantasy Points (DK calc) — chronological**")
pts_df = pd.DataFrame({
    "Week": week_labels,
    "DK_Points": [float(row["Pts_L3"]), float(row["Pts_L2"]), float(row["Pts_L1"])],
}).round(1)

st.dataframe(pts_df, use_container_width=True, hide_index=True)
st.line_chart(pts_df.set_index("Week")["DK_Points"])

st.markdown("**Last 3 Stat Breakdown — chronological**")

if pos == "QB":
    detail_df = pd.DataFrame({
        "Week": week_labels,
        "PassYds": [row["PassYds_L3"], row["PassYds_L2"], row["PassYds_L1"]],
        "PassTD":  [row["PassTD_L3"],  row["PassTD_L2"],  row["PassTD_L1"]],
        "RushYds": [row["RushYds_L3"], row["RushYds_L2"], row["RushYds_L1"]],
        "FumLost": [row["FumLost_L3"], row["FumLost_L2"], row["FumLost_L1"]],
        "TwoPt":   [row["TwoPt_L3"],   row["TwoPt_L2"],   row["TwoPt_L1"]],
        "DK_Pts":  [row["Pts_L3"],     row["Pts_L2"],     row["Pts_L1"]],
    })
elif pos in ["RB", "WR", "TE"]:
    detail_df = pd.DataFrame({
        "Week": week_labels,
        "RushYds":     [row["RushYds_L3"], row["RushYds_L2"], row["RushYds_L1"]],
        "Receptions":  [row["Rec_L3"],     row["Rec_L2"],     row["Rec_L1"]],
        "RecYds":      [row["RecYds_L3"],  row["RecYds_L2"],  row["RecYds_L1"]],
        "FumLost":     [row["FumLost_L3"], row["FumLost_L2"], row["FumLost_L1"]],
        "TwoPt":       [row["TwoPt_L3"],   row["TwoPt_L2"],   row["TwoPt_L1"]],
        "DK_Pts":      [row["Pts_L3"],     row["Pts_L2"],     row["Pts_L1"]],
    })
else:  # DST
    detail_df = pd.DataFrame({
        "Week": week_labels,
        "DK_Pts": [row["Pts_L3"], row["Pts_L2"], row["Pts_L1"]],
    })

# Round numeric to 1 decimal for display consistency
for col in detail_df.columns:
    if col != "Week":
        detail_df[col] = pd.to_numeric(detail_df[col], errors="coerce").fillna(0).round(1)

st.dataframe(detail_df, use_container_width=True, hide_index=True)

st.caption(
    "Stats are pulled automatically from nfl_data_py weekly data. "
    "DK points are calculated in-app. Salaries are deterministic demo values until you plug in real DraftKings salaries."
)
