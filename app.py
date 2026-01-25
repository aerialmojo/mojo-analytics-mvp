import re
import hashlib
import streamlit as st
import pandas as pd

# IMPORTANT: this file must exist in your repo (same folder as app.py)
# and your requirements.txt must include nflreadpy (since your data_utils uses it).
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


def safe_num(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        df[col] = 0
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)


def dk_points_offense(row: pd.Series) -> float:
    """
    DraftKings NFL Classic (offense) scoring approximation:
    - Pass: 0.04/yd, 4/TD, -1/INT, +3 bonus 300+ pass yds
    - Rush: 0.1/yd, 6/TD, +3 bonus 100+ rush yds
    - Rec: 1/reception, 0.1/yd, 6/TD, +3 bonus 100+ rec yds
    - -1 fumble lost
    - +2 per 2-pt conversion
    """
    pass_yds = float(row.get("passing_yards", 0) or 0)
    pass_td = float(row.get("passing_tds", 0) or 0)

    ints = row.get("passing_interceptions", None)
    if ints is None:
        ints = row.get("interceptions", 0)
    ints = float(ints or 0)

    rush_yds = float(row.get("rushing_yards", 0) or 0)
    rush_td = float(row.get("rushing_tds", 0) or 0)

    rec = float(row.get("receptions", 0) or 0)
    rec_yds = float(row.get("receiving_yards", 0) or 0)
    rec_td = float(row.get("receiving_tds", 0) or 0)

    fum_lost = float(row.get("fumbles_lost", 0) or 0)

    two_pt = row.get("two_point_conversions", None)
    if two_pt is None:
        two_pt = row.get("two_pt_conversions", None)
    if two_pt is None:
        two_pt = row.get("two_point_conversions_made", 0)
    two_pt = float(two_pt or 0)

    pts = 0.0

    pts += pass_yds * 0.04
    pts += pass_td * 4.0
    pts -= ints * 1.0
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

    pts -= fum_lost * 1.0
    pts += two_pt * 2.0

    return round(pts, 2)


# -----------------------------
# Load REAL weekly stats (same idea as betting analyzer)
# -----------------------------
@st.cache_data(ttl=60 * 60)
def get_player_stats():
    df = load_player_stats([2023, 2024, 2025])
    df = df.dropna(subset=["player_display_name"])
    return df


raw = get_player_stats()

# Normalize/ensure base columns exist
if "team" not in raw.columns:
    raw["team"] = None
if "position" not in raw.columns:
    raw["position"] = None

raw["player_display_name"] = raw["player_display_name"].astype(str).str.strip()
raw["player_key"] = raw["player_display_name"].apply(normalize_name)

safe_num(raw, "week")
safe_num(raw, "season")

# Make sure underlying stats exist as numeric
for c in [
    "passing_yards", "passing_tds", "passing_interceptions", "interceptions",
    "rushing_yards", "rushing_tds",
    "receptions", "receiving_yards", "receiving_tds",
    "fumbles_lost",
    "two_point_conversions",
    "fantasy_points",  # source-provided points (non-DK)
]:
    safe_num(raw, c)

# Compute DK points ourselves
raw["dk_points"] = raw.apply(dk_points_offense, axis=1)

raw["position"] = raw["position"].astype(str).str.upper().str.strip()

# -----------------------------
# Sidebar: Season + Week cutoff + scoring toggle
# -----------------------------
available_seasons = sorted([int(x) for x in raw["season"].dropna().unique().tolist()])
if not available_seasons:
    st.error("No seasons found in the loaded dataset.")
    st.stop()

default_season = 2025 if 2025 in available_seasons else available_seasons[-1]

with st.sidebar:
    st.markdown("### 📅 Season")
    season = st.selectbox("Choose season", available_seasons, index=available_seasons.index(default_season))

season_df = raw[raw["season"] == season].copy()
latest_week = int(season_df["week"].dropna().max()) if season_df["week"].notna().any() else 1

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
            max_value=max(2, latest_week + 1),
            value=latest_week + 1,
            step=1,
        )

with st.sidebar:
    st.markdown("### 🧮 Scoring")
    scoring_mode = st.radio("Use for Value/Trends", ["DraftKings (computed)", "Source (feed)"], index=0)
    st.caption("We show both side-by-side. This selection controls Value/Trend calculations.")

st.caption(
    f"Season {season}: data detected through Week {latest_week}. "
    f"Last-3 shows last 3 games played before Week {up_to_week}."
)

# -----------------------------
# Work set: only games prior to cutoff week
# -----------------------------
work = season_df[season_df["week"] < up_to_week].copy()

# -----------------------------
# Build player pool (offense positions) + add DST placeholders
# -----------------------------
pool = work[["player_key", "player_display_name", "position"]].dropna().copy()
pool = pool[pool["position"].isin(["QB", "RB", "WR", "TE"])].copy()
pool = pool.drop_duplicates(subset=["player_key"], keep="first")
pool = pool.rename(columns={"player_display_name": "Player", "position": "Position"})[["Player", "Position", "player_key"]].copy()

dst = pd.DataFrame({
    "Player": ["49ers DST", "Cowboys DST", "Eagles DST", "Chiefs DST"],
    "Position": ["DST", "DST", "DST", "DST"],
})
dst["player_key"] = dst["Player"].apply(normalize_name)

pool = pd.concat([pool, dst], ignore_index=True)

# -----------------------------
# Compute last 3 games played per player_key
# -----------------------------
work = work.sort_values(["player_key", "week"], ascending=[True, False])
last3 = work.groupby("player_key").head(3).copy()
last3["rank"] = last3.groupby("player_key").cumcount() + 1

def pivot_last3(col: str, pref: str) -> pd.DataFrame:
    if col not in last3.columns:
        return pd.DataFrame({"player_key": last3["player_key"].unique()})
    p = last3.pivot_table(index="player_key", columns="rank", values=col, aggfunc="first")
    p = p.rename(columns={1: f"{pref}_L1", 2: f"{pref}_L2", 3: f"{pref}_L3"}).reset_index()
    return p

# Points pivots (side-by-side)
metrics_src = pivot_last3("fantasy_points", "SrcFP")
metrics_dk  = pivot_last3("dk_points", "DKFP")
metrics = metrics_src.merge(metrics_dk, on="player_key", how="outer")

# Stat pivots (shared for detail)
metrics = metrics.merge(pivot_last3("passing_yards", "PassYds"), on="player_key", how="left")
metrics = metrics.merge(pivot_last3("passing_tds", "PassTD"), on="player_key", how="left")
metrics = metrics.merge(pivot_last3("passing_interceptions", "PassINT"), on="player_key", how="left")
metrics = metrics.merge(pivot_last3("rushing_yards", "RushYds"), on="player_key", how="left")
metrics = metrics.merge(pivot_last3("rushing_tds", "RushTD"), on="player_key", how="left")
metrics = metrics.merge(pivot_last3("receiving_yards", "RecYds"), on="player_key", how="left")
metrics = metrics.merge(pivot_last3("receptions", "Rec"), on="player_key", how="left")
metrics = metrics.merge(pivot_last3("receiving_tds", "RecTD"), on="player_key", how="left")
metrics = metrics.merge(pivot_last3("fumbles_lost", "FumLost"), on="player_key", how="left")
metrics = metrics.merge(pivot_last3("two_point_conversions", "TwoPt"), on="player_key", how="left")

# Season averages (side-by-side)
season_avg_src = (
    work.groupby("player_key")["fantasy_points"].mean().reset_index()
    .rename(columns={"fantasy_points": "PPG_Season_Src"})
)
season_avg_dk = (
    work.groupby("player_key")["dk_points"].mean().reset_index()
    .rename(columns={"dk_points": "PPG_Season_DK"})
)

metrics = metrics.merge(season_avg_src, on="player_key", how="left")
metrics = metrics.merge(season_avg_dk, on="player_key", how="left")

# Merge pool + metrics
df = pool.merge(metrics, on="player_key", how="left")

# Fill missing numeric columns so UI never breaks
numeric_cols = [
    "SrcFP_L1","SrcFP_L2","SrcFP_L3",
    "DKFP_L1","DKFP_L2","DKFP_L3",
    "PPG_Season_Src","PPG_Season_DK",
    "PassYds_L1","PassYds_L2","PassYds_L3",
    "PassTD_L1","PassTD_L2","PassTD_L3",
    "PassINT_L1","PassINT_L2","PassINT_L3",
    "RushYds_L1","RushYds_L2","RushYds_L3",
    "RushTD_L1","RushTD_L2","RushTD_L3",
    "RecYds_L1","RecYds_L2","RecYds_L3",
    "Rec_L1","Rec_L2","Rec_L3",
    "RecTD_L1","RecTD_L2","RecTD_L3",
    "FumLost_L1","FumLost_L2","FumLost_L3",
    "TwoPt_L1","TwoPt_L2","TwoPt_L3",
]
for c in numeric_cols:
    if c not in df.columns:
        df[c] = 0
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# Demo salaries (until DK salaries are integrated)
seed_tag = f"{season}-W{up_to_week}"
df["Salary"] = df.apply(lambda r: deterministic_salary(r["player_key"], r["Position"], seed_tag), axis=1)

# Derived metrics for both scoring systems
df["Avg_Last3_Src"] = df[["SrcFP_L1","SrcFP_L2","SrcFP_L3"]].mean(axis=1).round(2)
df["Avg_Last3_DK"]  = df[["DKFP_L1","DKFP_L2","DKFP_L3"]].mean(axis=1).round(2)

df["Value_Last3_Src_per_$1k"] = (df["Avg_Last3_Src"] / (df["Salary"] / 1000)).fillna(0).round(2)
df["Value_Last3_DK_per_$1k"]  = (df["Avg_Last3_DK"]  / (df["Salary"] / 1000)).fillna(0).round(2)

df["Last3_Spark_Src"] = df.apply(lambda r: sparkline([r["SrcFP_L1"], r["SrcFP_L2"], r["SrcFP_L3"]]), axis=1)
df["Last3_Spark_DK"]  = df.apply(lambda r: sparkline([r["DKFP_L1"],  r["DKFP_L2"],  r["DKFP_L3"]]),  axis=1)

# Which scoring powers "value" in dropdown labels + sorting
use_dk = (scoring_mode == "DraftKings (computed)")
value_col = "Value_Last3_DK_per_$1k" if use_dk else "Value_Last3_Src_per_$1k"

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
            lambda r: f'{r["Player"]} — ${int(r["Salary"]):,} — V3 {r[value_col]}',
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

    # Optional: lightweight data debug
    with st.expander("🔎 Data Debug"):
        st.write("Seasons in data:", sorted(raw["season"].dropna().unique().tolist()))
        st.write(f"{season} max week:", int(season_df["week"].max()) if not season_df.empty else None)
        st.write("Rows (season):", int(len(season_df)))

# -----------------------------
# Main: Selected lineup summary
# -----------------------------
chosen_players = get_selected_players()

st.markdown("## ✅ Selected Lineup")
if chosen_players:
    lineup_df = df[df["Player"].isin(chosen_players)][
        [
            "Player", "Position", "Salary",
            "PPG_Season_Src", "PPG_Season_DK",
            "Last3_Spark_Src", "Avg_Last3_Src", "Value_Last3_Src_per_$1k",
            "Last3_Spark_DK",  "Avg_Last3_DK",  "Value_Last3_DK_per_$1k",
        ]
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

display_cols = [
    "Player", "Position", "Salary",
    "PPG_Season_Src", "PPG_Season_DK",
    "Last3_Spark_Src", "Avg_Last3_Src", "Value_Last3_Src_per_$1k",
    "Last3_Spark_DK",  "Avg_Last3_DK",  "Value_Last3_DK_per_$1k",
]
pool_view = pool_df[display_cols].sort_values(["Position", value_col], ascending=[True, False]).copy()

st.data_editor(pool_view, use_container_width=True, hide_index=True, disabled=True)

# -----------------------------
# Main: Player Details
# -----------------------------
st.markdown("## 🔎 Player Details")

if pool_df.empty:
    st.warning("No players found for this filter/search.")
    st.stop()

detail_player = st.selectbox("Select a player from this pool", pool_df["Player"].tolist(), key="detail_player")

row = df[df["Player"] == detail_player].iloc[0]
pos = row["Position"]

st.markdown(f"#### {row['Player']} ({pos})")

# Metrics: Source vs DK side-by-side
cA, cB = st.columns(2)

with cA:
    st.markdown("### Source Scoring (Feed)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Salary (demo)", f"${int(row['Salary']):,}")
    m2.metric("Season Avg (Src)", f"{row['PPG_Season_Src']:.2f}")
    m3.metric("Avg Last 3 (Src)", f"{row['Avg_Last3_Src']:.2f}")
    m4.metric("Value / $1k (Src)", f"{row['Value_Last3_Src_per_$1k']:.2f}")

with cB:
    st.markdown("### DraftKings Scoring (Computed)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Salary (demo)", f"${int(row['Salary']):,}")
    m2.metric("Season Avg (DK)", f"{row['PPG_Season_DK']:.2f}")
    m3.metric("Avg Last 3 (DK)", f"{row['Avg_Last3_DK']:.2f}")
    m4.metric("Value / $1k (DK)", f"{row['Value_Last3_DK_per_$1k']:.2f}")

st.markdown("---")

# Last 3 points table + charts
st.markdown("### Last 3 Fantasy Points (Side-by-Side)")

pts_df = pd.DataFrame({
    "Game": ["L1", "L2", "L3"],
    "Source_FP": [row["SrcFP_L1"], row["SrcFP_L2"], row["SrcFP_L3"]],
    "DK_FP":     [row["DKFP_L1"],  row["DKFP_L2"],  row["DKFP_L3"]],
})
st.dataframe(pts_df, use_container_width=True, hide_index=True)

ch = pts_df.set_index("Game")[["Source_FP", "DK_FP"]]
st.line_chart(ch)

# Position-specific stat breakdown (includes fumbles lost + 2pt)
st.markdown("### Last 3 Stat Breakdown")

if pos == "QB":
    detail_df = pd.DataFrame({
        "Game": ["L1", "L2", "L3"],
        "PassYds": [row["PassYds_L1"], row["PassYds_L2"], row["PassYds_L3"]],
        "PassTD":  [row["PassTD_L1"],  row["PassTD_L2"],  row["PassTD_L3"]],
        "PassINT": [row["PassINT_L1"], row["PassINT_L2"], row["PassINT_L3"]],
        "RushYds": [row["RushYds_L1"], row["RushYds_L2"], row["RushYds_L3"]],
        "RushTD":  [row["RushTD_L1"],  row["RushTD_L2"],  row["RushTD_L3"]],
        "FumLost": [row["FumLost_L1"], row["FumLost_L2"], row["FumLost_L3"]],
        "TwoPt":   [row["TwoPt_L1"],   row["TwoPt_L2"],   row["TwoPt_L3"]],
        "SrcFP":   [row["SrcFP_L1"],   row["SrcFP_L2"],   row["SrcFP_L3"]],
        "DKFP":    [row["DKFP_L1"],    row["DKFP_L2"],    row["DKFP_L3"]],
    })
elif pos in ["RB", "WR", "TE"]:
    detail_df = pd.DataFrame({
        "Game": ["L1", "L2", "L3"],
        "RushYds": [row["RushYds_L1"], row["RushYds_L2"], row["RushYds_L3"]],
        "RushTD":  [row["RushTD_L1"],  row["RushTD_L2"],  row["RushTD_L3"]],
        "Receptions": [row["Rec_L1"], row["Rec_L2"], row["Rec_L3"]],
        "RecYds":  [row["RecYds_L1"], row["RecYds_L2"], row["RecYds_L3"]],
        "RecTD":   [row["RecTD_L1"],  row["RecTD_L2"],  row["RecTD_L3"]],
        "FumLost": [row["FumLost_L1"], row["FumLost_L2"], row["FumLost_L3"]],
        "TwoPt":   [row["TwoPt_L1"],   row["TwoPt_L2"],   row["TwoPt_L3"]],
        "SrcFP":   [row["SrcFP_L1"],   row["SrcFP_L2"],   row["SrcFP_L3"]],
        "DKFP":    [row["DKFP_L1"],    row["DKFP_L2"],    row["DKFP_L3"]],
    })
else:
    detail_df = pd.DataFrame({
        "Game": ["L1", "L2", "L3"],
        "SrcFP": [row["SrcFP_L1"], row["SrcFP_L2"], row["SrcFP_L3"]],
        "DKFP":  [row["DKFP_L1"],  row["DKFP_L2"],  row["DKFP_L3"]],
    })

st.dataframe(detail_df, use_container_width=True, hide_index=True)

st.caption(
    "Source_FP uses the feed's fantasy_points. DK_FP is computed in-app using DK NFL Classic rules, "
    "including fumbles lost (-1) and 2-pt conversions (+2) when those stats are available in the dataset. "
    "Salaries are deterministic demo values until you integrate real DraftKings salaries."
)
