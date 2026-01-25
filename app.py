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
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s)
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

# -----------------------------
# nflverse "free API" access
# -----------------------------
def nflverse_weekly_url(season: int) -> str:
    return f"https://github.com/nflverse/nflverse-data/releases/download/player_stats/stats_player_week_{season}.csv"

@st.cache_data(ttl=86400)
def season_exists(season: int) -> bool:
    """
    More reliable than HEAD: do a tiny GET using Range.
    GitHub release assets sometimes behave oddly for HEAD requests.
    """
    url = nflverse_weekly_url(season)
    try:
        r = requests.get(
            url,
            headers={"Range": "bytes=0-1024"},
            timeout=20,
            allow_redirects=True
        )
        return r.status_code in (200, 206) and len(r.content) > 0
    except Exception:
        return False

@st.cache_data(ttl=3600)
def fetch_player_week_stats(season: int) -> pd.DataFrame:
    url = nflverse_weekly_url(season)
    r = requests.get(url, timeout=40)
    r.raise_for_status()
    stats = pd.read_csv(pd.io.common.BytesIO(r.content))

    # Normalize common alt column names
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

    stats["dk_points"] = stats.apply(dk_fantasy_points, axis=1)
    return stats

def build_player_pool(stats: pd.DataFrame) -> pd.DataFrame:
    base = stats[["player_key", "player_name", "position"]].dropna().copy()
    base = base[base["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    base = base.drop_duplicates(subset=["player_key"], keep="first")

    pool = base.rename(columns={"player_name": "Player", "position": "Position"})[
        ["Player", "Position", "player_key"]
    ].copy()

    # MVP DST list (defense-by-week stats can be added later)
    dst = pd.DataFrame({
        "Player": ["49ers DST", "Cowboys DST", "Eagles DST", "Chiefs DST"],
        "Position": ["DST", "DST", "DST", "DST"],
    })
    dst["player_key"] = dst["Player"].apply(normalize_name)

    return pd.concat([pool, dst], ignore_index=True)

def build_last3_and_season_metrics(stats: pd.DataFrame, up_to_week: int) -> tuple[pd.DataFrame, int]:
    """
    Compute last-3 games played + season average for games BEFORE up_to_week.
    Returns (metrics_df, latest_week_in_data)
    """
    s = stats.copy()
    s = s[s["week"].notna()].copy()
    if s.empty:
        return pd.DataFrame(columns=["player_key"]), 0

    latest_week = int(s["week"].max())

    # Only games before the selected week
    s = s[s["week"] < int(up_to_week)].copy()
    if s.empty:
        return pd.DataFrame(columns=["player_key"]), latest_week

    s = s.sort_values(["player_key", "week"], ascending=[True, False])
    last3 = s.groupby("player_key").head(3).copy()
    last3["rank"] = last3.groupby("player_key").cumcount() + 1

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

    season_avg = (
        s.groupby("player_key")["dk_points"]
        .mean()
        .reset_index()
        .rename(columns={"dk_points": "PPG_Season"})
    )
    out = out.merge(season_avg, on="player_key", how="left")

    for c in out.columns:
        if c != "player_key":
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out, latest_week

# -----------------------------
# Sidebar: Season dropdown (last 3 available seasons)
# -----------------------------
current_year = datetime.now().year

# Prefer last 3 seasons by label, but only include those that exist
preferred = [current_year - 1, current_year - 2, current_year - 3]

available = [y for y in preferred if season_exists(y)]

# Fallback search if needed
if len(available) < 3:
    for y in range(current_year, current_year - 10, -1):
        if y not in available and season_exists(y):
            available.append(y)
        if len(available) >= 3:
            break

if len(available) == 0:
    st.error("Could not find any available nflverse seasons right now.")
    st.stop()

available = available[:3]  # show last 3 we found

with st.sidebar:
    st.markdown("### 📅 Season")
    season = st.selectbox("Choose season", available, index=0)
    st.caption("Stats are pulled automatically from nflverse weekly data (free).")

# -----------------------------
# Build dataset for selected season
# -----------------------------
try:
    stats = fetch_player_week_stats(int(season))
except Exception as e:
    st.error("Could not load nflverse weekly stats right now.")
    st.code(str(e))
    st.stop()

pool = build_player_pool(stats)

# -----------------------------
# Sidebar: Week slider (Latest or Choose)
# -----------------------------
latest_week = int(pd.to_numeric(stats["week"], errors="coerce").max())

with st.sidebar:
    st.markdown("### 🗓️ Week")
    mode = st.radio("Last-3 mode", ["Latest available", "Choose a week"], index=0, horizontal=True)

    if mode == "Latest available":
        up_to_week = latest_week + 1
        st.caption(f"Using last-3 games before Week {up_to_week} (latest in data: Week {latest_week}).")
    else:
        up_to_week = st.slider(
            "Compute last-3 games before this week",
            min_value=1,
            max_value=latest_week + 1,
            value=latest_week + 1,
            step=1
        )

metrics, _latest_week_check = build_last3_and_season_metrics(stats, up_to_week=up_to_week)

df = pool.merge(metrics, on="player_key", how="left")

# Fill missing numeric columns
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

# Demo salaries (deterministic, season+week-based so values shift a bit)
seed_tag = f"{season}-W{up_to_week}"
df["Salary"] = df.apply(lambda r: deterministic_salary(r["player_key"], r["Position"], seed_tag), axis=1)

# Derived metrics
df["Avg_Last3"] = df[["Pts_L1", "Pts_L2", "Pts_L3"]].mean(axis=1).round(2)
df["Value_Last3_per_$1k"] = (df["Avg_Last3"] / (df["Salary"] / 1000)).replace([pd.NA, pd.NaT], 0).fillna(0).round(2)
df["Value_Season_per_$1k"] = (df["PPG_Season"] / (df["Salary"] / 1000)).replace([pd.NA, pd.NaT], 0).fillna(0).round(2)
df["Last3_Spark"] = df.apply(lambda r: sparkline([r["Pts_L1"], r["Pts_L2"], r["Pts_L3"]]), axis=1)

if "Team" not in df.columns:
    df["Team"] = ""

st.caption(
    f"Season {season}: data detected through Week {latest_week}. "
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
    st.caption("Use the sidebar to build a lineup, then browse the pool for details.")

st.markdown("---")

# -----------------------------
# Main: Player Pool
# -----------------------------
st.markdown("## 🧾 Player Pool")

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
    "Select a player from this pool",
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

st.markdown("**Last 3 Fantasy Points (DraftKings scoring)**")
pts_df = pd.DataFrame({
    "Game": ["L1", "L2", "L3"],
    "FantasyPts": [row["Pts_L1"], row["Pts_L2"], row["Pts_L3"]],
})
st.dataframe(pts_df, use_container_width=True, hide_index=True)
st.line_chart(pts_df.set_index("Game")["FantasyPts"])

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
else:
    detail_df = pd.DataFrame({
        "Game": ["L1", "L2", "L3"],
        "FantasyPts": [row["Pts_L1"], row["Pts_L2"], row["Pts_L3"]],
    })

st.dataframe(detail_df, use_container_width=True, hide_index=True)

st.caption(
    "Stats are pulled automatically from free nflverse weekly player data. "
    "DraftKings points are computed in-app. Salaries are deterministic demo values until you plug in real DK salaries."
)
