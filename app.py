import re
import hashlib
import requests
import streamlit as st
import pandas as pd
from datetime import datetime

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
    s = re.sub(r"[^a-z0-9\s]", "", s)   # remove punctuation
    s = re.sub(r"\s+", " ", s)          # collapse spaces
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s).strip()
    return s

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

def dk_fantasy_points(row: pd.Series) -> float:
    """
    DraftKings NFL classic scoring (core pieces).
    Passing: 0.04/yd, 4/TD, -1/INT, +3 bonus for 300+ pass yds
    Rushing: 0.1/yd, 6/TD, +3 bonus for 100+ rush yds
    Receiving: 1/reception, 0.1/yd, 6/TD, +3 bonus for 100+ rec yds
    Fumbles lost: -1 (if available)
    """
    pass_yds = float(row.get("passing_yards", 0) or 0)
    pass_td  = float(row.get("passing_tds", 0) or 0)
    ints     = float(row.get("interceptions", 0) or 0)

    rush_yds = float(row.get("rushing_yards", 0) or 0)
    rush_td  = float(row.get("rushing_tds", 0) or 0)

    rec      = float(row.get("receptions", 0) or 0)
    rec_yds  = float(row.get("receiving_yards", 0) or 0)
    rec_td   = float(row.get("receiving_tds", 0) or 0)

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

def deterministic_salary(player_key: str, pos: str, seed_tag: str) -> int:
    """
    Deterministic demo salary so the UI works without DK salary CSV.
    seed_tag can include season/week so salaries shift slightly week to week.
    """
    ranges = {
        "QB": (5200, 8800),
        "RB": (4000, 9200),
        "WR": (3000, 9000),
        "TE": (2500, 7800),
        "DST": (2000, 4500),
    }
    lo, hi = ranges.get(pos, (3000, 8000))

    h = hashlib.md5(f"{seed_tag}:{pos}:{player_key}".encode("utf-8")).hexdigest()
    n = int(h[:8], 16)  # 32-bit chunk
    val = lo + (n % (hi - lo + 1))

    # round to nearest 100 like DK
    return int(round(val / 100) * 100)

# -----------------------------
# nflverse fetch + processing
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_player_week_stats(season: int) -> pd.DataFrame:
    """
    Pull weekly player stats from nflverse-data releases (CSV over HTTP).
    """
    url = f"https://github.com/nflverse/nflverse-data/releases/download/player_stats/stats_player_week_{season}.csv"
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    stats = pd.read_csv(pd.io.common.BytesIO(r.content))

    # normalize common alt column names
    if "player_name" not in stats.columns:
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

    stats["player_name"] = stats["player_name"].astype(str)
    stats["player_key"] = stats["player_name"].apply(normalize_name)
    stats["position"] = stats["position"].astype(str).str.upper().str.strip()
    stats["week"] = pd.to_numeric(stats["week"], errors="coerce")

    # DK points computed in app
    stats["dk_points"] = stats.apply(dk_fantasy_points, axis=1)

    return stats

@st.cache_data(ttl=3600)
def find_latest_available_season(start_year: int) -> int:
    """
    Try current year, then back a few years until we find an nflverse stats file that exists.
    Keeps UX smooth without user needing to know which season is available.
    """
    for y in range(start_year, start_year - 6, -1):
        url = f"https://github.com/nflverse/nflverse-data/releases/download/player_stats/stats_player_week_{y}.csv"
        try:
            resp = requests.head(url, timeout=15, allow_redirects=True)
            if resp.status_code == 200:
                return y
        except Exception:
            continue
    return start_year - 1  # fallback

def build_player_pool(stats: pd.DataFrame) -> pd.DataFrame:
    """
    Build the weekly pool from players that exist in stats (QB/RB/WR/TE).
    """
    base = stats[["player_key", "player_name", "position"]].dropna().copy()
    base = base[base["position"].isin(["QB", "RB", "WR", "TE"])].copy()

    # one row per player key
    base = base.drop_duplicates(subset=["player_key"], keep="first")

    pool = base.rename(columns={"player_name": "Player", "position": "Position"})[
        ["Player", "Position", "player_key"]
    ].copy()

    # Simple DST list for MVP (DST boxscore integration can be added later)
    dst = pd.DataFrame({
        "Player": ["49ers DST", "Cowboys DST", "Eagles DST", "Chiefs DST"],
        "Position": ["DST", "DST", "DST", "DST"],
    })
    dst["player_key"] = dst["Player"].apply(normalize_name)

    return pd.concat([pool, dst], ignore_index=True)

def build_last3_and_season_metrics(stats: pd.DataFrame, up_to_week: int) -> pd.DataFrame:
    """
    One row per player_key with:
      - PPG_Season (avg DK points) using games before up_to_week
      - last 3 DK points
      - last 3 pass/rush/rec stats (if present)
    """
    s = stats.copy()
    s = s[s["week"].notna()].copy()
    s = s[s["week"] < up_to_week].copy()

    if s.empty:
        return pd.DataFrame(columns=["player_key"])

    s = s.sort_values(["player_key", "week"], ascending=[True, False])
    last3 = s.groupby("player_key").head(3).copy()
    last3["rank"] = last3.groupby("player_key").cumcount() + 1  # 1..3

    def pivot(col, pref):
        if col not in last3.columns:
            return None
        p = last3.pivot_table(index="player_key", columns="rank", values=col, aggfunc="first")
        return p.rename(columns={1: f"{pref}_L1", 2: f"{pref}_L2", 3: f"{pref}_L3"}).reset_index()

    out = pivot("dk_points", "Pts")
    if out is None:
        out = pd.DataFrame({"player_key": s["player_key"].unique()})

    for col, pref in [
        ("passing_yards", "PassYds"),
        ("passing_tds", "PassTD"),
        ("rushing_yards", "RushYds"),
        ("receiving_yards", "RecYds"),
        ("receptions", "Rec"),
    ]:
        p = pivot(col, pref)
        if p is not None:
            out = out.merge(p, on="player_key", how="left")

    season_avg = s.groupby("player_key")["dk_points"].mean().reset_index().rename(columns={"dk_points": "PPG_Season"})
    out = out.merge(season_avg, on="player_key", how="left")

    # numeric cleanup
    for c in out.columns:
        if c != "player_key":
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out

# -----------------------------
# Build data (auto, no upload)
# -----------------------------
current_year = datetime.now().year

# Put settings in sidebar so main UI looks like your old version
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    with st.expander("Data settings (free nflverse stats)", expanded=False):
        auto_season = find_latest_available_season(current_year)
        season = st.number_input("Season", min_value=2016, max_value=2030, value=int(auto_season), step=1)
        week = st.number_input("Week (compute last-3 before this week)", min_value=1, max_value=22, value=2, step=1)
        st.caption("If Week=1, last-3 and season averages will be 0 because no prior games exist.")

try:
    stats = fetch_player_week_stats(int(season))
except Exception as e:
    st.error("Could not load nflverse weekly stats right now.")
    st.code(str(e))
    st.stop()

pool = build_player_pool(stats)
metrics = build_last3_and_season_metrics(stats, up_to_week=int(week))

df = pool.merge(metrics, on="player_key", how="left")

# Fill missing numeric columns with 0
numeric_cols = [
    "Pts_L1", "Pts_L2", "Pts_L3",
    "PassYds_L1", "PassYds_L2", "PassYds_L3",
    "PassTD_L1", "PassTD_L2", "PassTD_L3",
    "RushYds_L1", "RushYds_L2", "RushYds_L3",
    "RecYds_L1", "RecYds_L2", "RecYds_L3",
    "Rec_L1", "Rec_L2", "Rec_L3",
    "PPG_Season",
]
for c in numeric_cols:
    if c not in df.columns:
        df[c] = 0
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# Demo salaries (deterministic)
seed_tag = f"{season}-W{week}"
df["Salary"] = df.apply(lambda r: deterministic_salary(r["player_key"], r["Position"], seed_tag), axis=1)

# Derived metrics
df["Avg_Last3"] = df[["Pts_L1", "Pts_L2", "Pts_L3"]].mean(axis=1).round(2)
df["Value_Last3_per_$1k"] = (df["Avg_Last3"] / (df["Salary"] / 1000)).replace([pd.NA, pd.NaT], 0).fillna(0).round(2)
df["Value_Season_per_$1k"] = (df["PPG_Season"] / (df["Salary"] / 1000)).replace([pd.NA, pd.NaT], 0).fillna(0).round(2)
df["Last3_Spark"] = df.apply(lambda r: sparkline([r["Pts_L1"], r["Pts_L2"], r["Pts_L3"]]), axis=1)

# Team column placeholder (nflverse file may include team columns; keep blank for now)
if "Team" not in df.columns:
    df["Team"] = ""

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
        pool_slot = df[df["Position"].isin(allowed_positions)].copy()

        current_val = st.session_state.get(f"slot_{slot_name}", "—")
        exclude = set(selected_now)
        if current_val != "—":
            exclude.discard(current_val)

        pool_slot = pool_slot[~pool_slot["Player"].isin(exclude)]

        # Rich labels
        pool_slot["Label"] = pool_slot.apply(
            lambda r: f'{r["Player"]} — ${int(r["Salary"]):,} — V3 {r["Value_Last3_per_$1k"]}',
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

c1, c2 = st.columns([1, 2])
with c1:
    pos_filter = st.selectbox("Filter by Position", ["ALL", "QB", "RB", "WR", "TE", "DST"], index=0)
with c2:
    search = st.text_input("Search player name", value="").strip().lower()

pool_df = df.copy()
if pos_filter != "ALL":
    pool_df = pool_df[pool_df["Position"] == pos_filter]
if search:
    pool_df = pool_df[pool_df["Player"].str.lower().str.contains(search)]

display_cols = ["Player", "Position", "Salary", "PPG_Season", "Last3_Spark", "Avg_Last3", "Value_Last3_per_$1k"]
pool_view = pool_df[display_cols].sort_values(["Position", "Value_Last3_per_$1k"], ascending=[True, False]).copy()

st.data_editor(pool_view, use_container_width=True, hide_index=True, disabled=True)

# -----------------------------
# Main: Player Details
# -----------------------------
st.markdown("## 🔎 Player Details")

if pool_df.empty:
    st.warning("No players found for this filter/search.")
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
m1.metric("Salary (demo)", f"${int(row['Salary']):,}")
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
else:  # DST
    detail_df = pd.DataFrame({
        "Game": ["L1", "L2", "L3"],
        "FantasyPts": [row["Pts_L1"], row["Pts_L2"], row["Pts_L3"]],
    })

st.dataframe(detail_df, use_container_width=True, hide_index=True)

st.caption(
    "Stats are pulled automatically from free nflverse weekly player data. "
    "DraftKings points are computed in-app. Salaries are deterministic demo values until you plug in real DK salaries."
)
