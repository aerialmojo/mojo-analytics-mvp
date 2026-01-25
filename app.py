import re
import hashlib
import streamlit as st
import pandas as pd

# IMPORTANT: this must exist in your repo (copy from betting analyzer project)
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

# -----------------------------
# Load REAL weekly stats (same as betting analyzer)
# -----------------------------
@st.cache_data(ttl=60 * 60)
def get_player_stats():
    df = load_player_stats([2023, 2024, 2025])  # <- EXACT idea from analyzer
    df = df.dropna(subset=["player_display_name"])
    return df

raw = get_player_stats()

# Normalize expected columns (your analyzer already guards team/position)
if "team" not in raw.columns:
    raw["team"] = None
if "position" not in raw.columns:
    raw["position"] = None

# Normalize naming + weeks
raw["player_display_name"] = raw["player_display_name"].astype(str).str.strip()
raw["player_key"] = raw["player_display_name"].apply(normalize_name)

if "week" in raw.columns:
    raw["week"] = pd.to_numeric(raw["week"], errors="coerce")
if "season" in raw.columns:
    raw["season"] = pd.to_numeric(raw["season"], errors="coerce")

# Ensure common stat columns exist (so UI never breaks)
for c in [
    "passing_yards", "passing_tds", "interceptions",
    "rushing_yards", "rushing_tds",
    "receptions", "receiving_yards", "receiving_tds",
    "fantasy_points",
]:
    safe_num(raw, c)

# -----------------------------
# Optional: Apply roster mapping (same as analyzer)
# -----------------------------
@st.cache_data(ttl=24 * 60 * 60)
def load_roster_if_exists():
    # Keep it optional so the app still runs if the file isn't present
    try:
        roster_df = pd.read_csv("2025_roster.csv")
        roster_df["player_display_name"] = roster_df["player_display_name"].astype(str).str.strip()
        roster_df["team"] = roster_df["team"].astype(str).str.strip()
        roster_df["position"] = roster_df["position"].astype(str).str.strip()
        return roster_df
    except Exception:
        return None

roster_df = load_roster_if_exists()
if roster_df is not None:
    team_map = dict(zip(roster_df["player_display_name"], roster_df["team"]))
    pos_map  = dict(zip(roster_df["player_display_name"], roster_df["position"]))
    raw["team"] = raw["player_display_name"].map(team_map).fillna(raw["team"])
    raw["position"] = raw["player_display_name"].map(pos_map).fillna(raw["position"])

raw["position"] = raw["position"].astype(str).str.upper().str.strip()

# -----------------------------
# Sidebar: Season + Week cutoff
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
            max_value=latest_week + 1,
            value=latest_week + 1,
            step=1,
        )

st.caption(
    f"Season {season}: data detected through Week {latest_week}. "
    f"Last-3 shows last 3 games played before Week {up_to_week}."
)

# -----------------------------
# Build player pool + last-3 metrics (for this season & cutoff)
# -----------------------------
# Only games prior to cutoff week
work = season_df[season_df["week"] < up_to_week].copy()

# Build pool from whoever has stats; fallback positions if missing
pool = work[["player_key", "player_display_name", "position"]].dropna().copy()
pool = pool[pool["position"].isin(["QB", "RB", "WR", "TE"])].copy()
pool = pool.drop_duplicates(subset=["player_key"], keep="first")

pool = pool.rename(columns={"player_display_name": "Player", "position": "Position"})[
    ["Player", "Position", "player_key"]
].copy()

# Add simple DST placeholders (same style as before)
dst = pd.DataFrame({
    "Player": ["49ers DST", "Cowboys DST", "Eagles DST", "Chiefs DST"],
    "Position": ["DST", "DST", "DST", "DST"],
})
dst["player_key"] = dst["Player"].apply(normalize_name)

pool = pd.concat([pool, dst], ignore_index=True)

# Compute last 3 games played per player_key
work = work.sort_values(["player_key", "week"], ascending=[True, False])
last3 = work.groupby("player_key").head(3).copy()
last3["rank"] = last3.groupby("player_key").cumcount() + 1

def pivot_last3(col: str, pref: str) -> pd.DataFrame:
    p = last3.pivot_table(index="player_key", columns="rank", values=col, aggfunc="first")
    p = p.rename(columns={1: f"{pref}_L1", 2: f"{pref}_L2", 3: f"{pref}_L3"}).reset_index()
    return p

# fantasy_points here comes from your loader (same as betting analyzer)
metrics = pivot_last3("fantasy_points", "Pts")

metrics = metrics.merge(pivot_last3("passing_yards", "PassYds"), on="player_key", how="left")
metrics = metrics.merge(pivot_last3("passing_tds", "PassTD"), on="player_key", how="left")
metrics = metrics.merge(pivot_last3("rushing_yards", "RushYds"), on="player_key", how="left")
metrics = metrics.merge(pivot_last3("receiving_yards", "RecYds"), on="player_key", how="left")
metrics = metrics.merge(pivot_last3("receptions", "Rec"), on="player_key", how="left")

season_avg = (
    work.groupby("player_key")["fantasy_points"]
    .mean()
    .reset_index()
    .rename(columns={"fantasy_points": "PPG_Season"})
)

metrics = metrics.merge(season_avg, on="player_key", how="left")

# Merge pool + metrics
df = pool.merge(metrics, on="player_key", how="left")

# Fill missing numeric values so UI doesn't break
for c in [
    "Pts_L1","Pts_L2","Pts_L3",
    "PassYds_L1","PassYds_L2","PassYds_L3",
    "PassTD_L1","PassTD_L2","PassTD_L3",
    "RushYds_L1","RushYds_L2","RushYds_L3",
    "RecYds_L1","RecYds_L2","RecYds_L3",
    "Rec_L1","Rec_L2","Rec_L3",
    "PPG_Season",
]:
    if c not in df.columns:
        df[c] = 0
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# Salaries (demo placeholder until you merge DK)
seed_tag = f"{season}-W{up_to_week}"
df["Salary"] = df.apply(lambda r: deterministic_salary(r["player_key"], r["Position"], seed_tag), axis=1)

# Derived metrics
df["Avg_Last3"] = df[["Pts_L1", "Pts_L2", "Pts_L3"]].mean(axis=1).round(2)
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
# Sidebar: Roster Builder (CURRENT UI)
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
# Main: Player Pool (CURRENT UI)
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
# Main: Player Details (CURRENT UI)
# -----------------------------
st.markdown("## 🔎 Player Details")

if pool_df.empty:
    st.warning("No players found for this filter/search.")
    st.stop()

detail_player = st.selectbox("Select a player from this pool", pool_df["Player"].tolist(), key="detail_player")

row = df[df["Player"] == detail_player].iloc[0]
pos = row["Position"]

st.markdown(f"#### {row['Player']} ({pos})")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Salary (demo)", f"${int(row['Salary']):,}")
m2.metric("Season Avg (FP)", f"{row['PPG_Season']:.2f}")
m3.metric("Avg Last 3 (FP)", f"{row['Avg_Last3']:.2f}")
m4.metric("Value (Last 3) / $1k", f"{row['Value_Last3_per_$1k']:.2f}")

st.markdown("**Last 3 Fantasy Points**")
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
    "Stats are loaded using the same `load_player_stats([2023, 2024, 2025])` pipeline as the betting analyzer. "
    "Salaries are demo placeholders until you merge DraftKings salaries."
)
