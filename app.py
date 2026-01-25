import re
import hashlib
import streamlit as st
import pandas as pd
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
st.subheader("DraftKings fantasy value, simplified.")
st.info("This is a cloud-hosted demo. No betting advice, no guarantees.")

# -----------------------------
# Helpers
# -----------------------------
def normalize_name(s):
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"\s+", " ", s)

def sparkline(vals):
    blocks = "▁▂▃▄▅▆▇█"
    vals = [float(v or 0) for v in vals]
    mn, mx = min(vals), max(vals)
    if mn == mx:
        return blocks[3] * len(vals)
    return "".join(blocks[int((v-mn)/(mx-mn)*(len(blocks)-1))] for v in vals)

def dk_points(row):
    pts = 0
    pts += row.passing_yards * 0.04
    pts += row.passing_tds * 4
    pts -= row.passing_interceptions
    if row.passing_yards >= 300: pts += 3

    pts += row.rushing_yards * 0.1
    pts += row.rushing_tds * 6
    if row.rushing_yards >= 100: pts += 3

    pts += row.receptions
    pts += row.receiving_yards * 0.1
    pts += row.receiving_tds * 6
    if row.receiving_yards >= 100: pts += 3

    pts -= row.fumbles_lost
    pts += row.two_point_conversions * 2
    return round(pts, 1)

def salary(player_key, pos, seed):
    ranges = {
        "QB": (5200, 8800),
        "RB": (4000, 9200),
        "WR": (3000, 9000),
        "TE": (2500, 7800),
        "DST": (2000, 4500),
    }
    lo, hi = ranges.get(pos, (3000, 8000))
    h = int(hashlib.md5(f"{seed}:{player_key}".encode()).hexdigest()[:8], 16)
    return int(round((lo + h % (hi-lo)) / 100) * 100)

# -----------------------------
# Load REAL weekly data
# -----------------------------
@st.cache_data(ttl=3600)
def load_data():
    df = load_player_stats([2023, 2024, 2025])
    df = df.dropna(subset=["player_display_name"])
    return df

raw = load_data()

raw["player_key"] = raw["player_display_name"].apply(normalize_name)
raw["position"] = raw["position"].str.upper()

num_cols = [
    "passing_yards","passing_tds","passing_interceptions",
    "rushing_yards","rushing_tds",
    "receiving_yards","receiving_tds","receptions",
    "fumbles_lost","two_point_conversions",
    "week","season"
]
for c in num_cols:
    raw[c] = pd.to_numeric(raw.get(c, 0), errors="coerce").fillna(0)

raw["dk_points"] = raw.apply(dk_points, axis=1)

# -----------------------------
# Sidebar controls
# -----------------------------
seasons = sorted(raw.season.unique())
season = st.sidebar.selectbox("Season", seasons, index=seasons.index(2025))

season_df = raw[raw.season == season]
latest_week = int(season_df.week.max())

up_to_week = st.sidebar.slider(
    "Analyze games before week",
    1, latest_week + 1, latest_week + 1
)

work = season_df[season_df.week < up_to_week]

# -----------------------------
# Build last-3 metrics
# -----------------------------
work = work.sort_values(["player_key","week"], ascending=[True, False])
last3 = work.groupby("player_key").head(3).copy()
last3["rank"] = last3.groupby("player_key").cumcount()+1

def pivot(col):
    return last3.pivot_table(
        index="player_key", columns="rank", values=col
    ).rename(columns={1:"L1",2:"L2",3:"L3"}).reset_index()

pts = pivot("dk_points")
avg = work.groupby("player_key").dk_points.mean().round(1).reset_index(name="PPG_Season")

df = (
    last3[["player_key","player_display_name","position"]]
    .drop_duplicates("player_key")
    .merge(pts, on="player_key", how="left")
    .merge(avg, on="player_key", how="left")
)

df = df.rename(columns={"player_display_name":"Player","position":"Position"})
df[["L1","L2","L3"]] = df[["L1","L2","L3"]].fillna(0).round(1)

df["Avg_Last3"] = df[["L1","L2","L3"]].mean(axis=1).round(1)
df["Last3_Spark"] = df.apply(lambda r: sparkline([r.L1,r.L2,r.L3]), axis=1)
df["Salary"] = df.apply(lambda r: salary(r.player_key,r.Position,f"{season}-{up_to_week}"), axis=1)
df["Value_per_$1k"] = (df.Avg_Last3 / (df.Salary/1000)).round(1)

# -----------------------------
# UI: Player Pool
# -----------------------------
st.markdown("## 🧾 Player Pool")

pool_view = df[
    ["Player","Position","Salary","PPG_Season","Last3_Spark","Avg_Last3","Value_per_$1k"]
].sort_values("Value_per_$1k", ascending=False)

st.dataframe(pool_view, use_container_width=True, hide_index=True)

# -----------------------------
# Player Details
# -----------------------------
st.markdown("## 🔍 Player Details")

player = st.selectbox("Select player", pool_view.Player.tolist())
row = df[df.Player == player].iloc[0]

st.metric("Season Avg (DK)", row.PPG_Season)
st.metric("Last 3 Avg (DK)", row.Avg_Last3)
st.metric("Value / $1k", row.Value_per_$1k)

pts_df = pd.DataFrame({
    "Game": ["L1","L2","L3"],
    "DK Points": [row.L1,row.L2,row.L3]
})

st.line_chart(pts_df.set_index("Game"))
