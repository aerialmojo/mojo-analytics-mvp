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
# Sample static player data (MVP only)
# NOTE: Added last-3 passing/rushing/receiving stats.
# Use 0s for non-applicable stats (e.g., WR passing yards).
# -----------------------------
players = [
    # QBs
    {"Player": "Jalen Hurts", "Position": "QB", "Salary": 8200, "PPG_Season": 22.8,
     "Pts_L1": 20.1, "Pts_L2": 27.4, "Pts_L3": 24.8,
     "PassYds_L1": 210, "PassYds_L2": 285, "PassYds_L3": 240,
     "PassTD_L1": 1, "PassTD_L2": 2, "PassTD_L3": 2,
     "RushYds_L1": 45, "RushYds_L2": 62, "RushYds_L3": 51,
     "RecYds_L1": 0, "RecYds_L2": 0, "RecYds_L3": 0,
     "Rec_L1": 0, "Rec_L2": 0, "Rec_L3": 0},

    {"Player": "Josh Allen", "Position": "QB", "Salary": 8600, "PPG_Season": 23.5,
     "Pts_L1": 18.7, "Pts_L2": 26.9, "Pts_L3": 23.4,
     "PassYds_L1": 240, "PassYds_L2": 318, "PassYds_L3": 265,
     "PassTD_L1": 1, "PassTD_L2": 3, "PassTD_L3": 2,
     "RushYds_L1": 31, "RushYds_L2": 22, "RushYds_L3": 41,
     "RecYds_L1": 0, "RecYds_L2": 0, "RecYds_L3": 0,
     "Rec_L1": 0, "Rec_L2": 0, "Rec_L3": 0},

    {"Player": "Lamar Jackson", "Position": "QB", "Salary": 8000, "PPG_Season": 22.1,
     "Pts_L1": 21.0, "Pts_L2": 19.6, "Pts_L3": 28.1,
     "PassYds_L1": 205, "PassYds_L2": 190, "PassYds_L3": 275,
     "PassTD_L1": 1, "PassTD_L2": 1, "PassTD_L3": 2,
     "RushYds_L1": 68, "RushYds_L2": 54, "RushYds_L3": 74,
     "RecYds_L1": 0, "RecYds_L2": 0, "RecYds_L3": 0,
     "Rec_L1": 0, "Rec_L2": 0, "Rec_L3": 0},

    # RBs
    {"Player": "Christian McCaffrey", "Position": "RB", "Salary": 9000, "PPG_Season": 24.9,
     "Pts_L1": 29.2, "Pts_L2": 22.4, "Pts_L3": 27.6,
     "PassYds_L1": 0, "PassYds_L2": 0, "PassYds_L3": 0,
     "PassTD_L1": 0, "PassTD_L2": 0, "PassTD_L3": 0,
     "RushYds_L1": 112, "RushYds_L2": 86, "RushYds_L3": 104,
     "RecYds_L1": 38, "RecYds_L2": 22, "RecYds_L3": 44,
     "Rec_L1": 5, "Rec_L2": 3, "Rec_L3": 6},

    {"Player": "Saquon Barkley", "Position": "RB", "Salary": 7600, "PPG_Season": 18.6,
     "Pts_L1": 16.2, "Pts_L2": 21.1, "Pts_L3": 17.9,
     "PassYds_L1": 0, "PassYds_L2": 0, "PassYds_L3": 0,
     "PassTD_L1": 0, "PassTD_L2": 0, "PassTD_L3": 0,
     "RushYds_L1": 74, "RushYds_L2": 98, "RushYds_L3": 81,
     "RecYds_L1": 18, "RecYds_L2": 34, "RecYds_L3": 21,
     "Rec_L1": 3, "Rec_L2": 4, "Rec_L3": 2},

    # WRs
    {"Player": "Tyreek Hill", "Position": "WR", "Salary": 8800, "PPG_Season": 23.7,
     "Pts_L1": 19.4, "Pts_L2": 31.0, "Pts_L3": 22.7,
     "PassYds_L1": 0, "PassYds_L2": 0, "PassYds_L3": 0,
     "PassTD_L1": 0, "PassTD_L2": 0, "PassTD_L3": 0,
     "RushYds_L1": 6, "RushYds_L2": 0, "RushYds_L3": 9,
     "RecYds_L1": 96, "RecYds_L2": 142, "RecYds_L3": 104,
     "Rec_L1": 7, "Rec_L2": 10, "Rec_L3": 8},

    {"Player": "A.J. Brown", "Position": "WR", "Salary": 8100, "PPG_Season": 20.2,
     "Pts_L1": 14.8, "Pts_L2": 24.1, "Pts_L3": 18.9,
     "PassYds_L1": 0, "PassYds_L2": 0, "PassYds_L3": 0,
     "PassTD_L1": 0, "PassTD_L2": 0, "PassTD_L3": 0,
     "RushYds_L1": 0, "RushYds_L2": 0, "RushYds_L3": 0,
     "RecYds_L1": 72, "RecYds_L2": 118, "RecYds_L3": 84,
     "Rec_L1": 5, "Rec_L2": 8, "Rec_L3": 6},

    {"Player": "CeeDee Lamb", "Position": "WR", "Salary": 8300, "PPG_Season": 21.0,
     "Pts_L1": 26.3, "Pts_L2": 17.2, "Pts_L3": 21.4,
     "PassYds_L1": 0, "PassYds_L2": 0, "PassYds_L3": 0,
     "PassTD_L1": 0, "PassTD_L2": 0, "PassTD_L3": 0,
     "RushYds_L1": 0, "RushYds_L2": 4, "RushYds_L3": 0,
     "RecYds_L1": 121, "RecYds_L2": 66, "RecYds_L3": 103,
     "Rec_L1": 9, "Rec_L2": 6, "Rec_L3": 8},

    # TEs
    {"Player": "Travis Kelce", "Position": "TE", "Salary": 6800, "PPG_Season": 15.1,
     "Pts_L1": 10.4, "Pts_L2": 18.9, "Pts_L3": 14.2,
     "PassYds_L1": 0, "PassYds_L2": 0, "PassYds_L3": 0,
     "PassTD_L1": 0, "PassTD_L2": 0, "PassTD_L3": 0,
     "RushYds_L1": 0, "RushYds_L2": 0, "RushYds_L3": 0,
     "RecYds_L1": 54, "RecYds_L2": 88, "RecYds_L3": 61,
     "Rec_L1": 4, "Rec_L2": 7, "Rec_L3": 5},

    {"Player": "Mark Andrews", "Position": "TE", "Salary": 6200, "PPG_Season": 13.9,
     "Pts_L1": 12.1, "Pts_L2": 15.8, "Pts_L3": 9.6,
     "PassYds_L1": 0, "PassYds_L2": 0, "PassYds_L3": 0,
     "PassTD_L1": 0, "PassTD_L2": 0, "PassTD_L3": 0,
     "RushYds_L1": 0, "RushYds_L2": 0, "RushYds_L3": 0,
     "RecYds_L1": 63, "RecYds_L2": 71, "RecYds_L3": 42,
     "Rec_L1": 6, "Rec_L2": 6, "Rec_L3": 4},

    # DST (keep fantasy points only in MVP)
    {"Player": "49ers DST", "Position": "DST", "Salary": 3200, "PPG_Season": 8.7,
     "Pts_L1": 7.0, "Pts_L2": 11.0, "Pts_L3": 10.2,
     "PassYds_L1": 0, "PassYds_L2": 0, "PassYds_L3": 0,
     "PassTD_L1": 0, "PassTD_L2": 0, "PassTD_L3": 0,
     "RushYds_L1": 0, "RushYds_L2": 0, "RushYds_L3": 0,
     "RecYds_L1": 0, "RecYds_L2": 0, "RecYds_L3": 0,
     "Rec_L1": 0, "Rec_L2": 0, "Rec_L3": 0},

    {"Player": "Cowboys DST", "Position": "DST", "Salary": 3500, "PPG_Season": 9.1,
     "Pts_L1": 6.0, "Pts_L2": 13.4, "Pts_L3": 9.8,
     "PassYds_L1": 0, "PassYds_L2": 0, "PassYds_L3": 0,
     "PassTD_L1": 0, "PassTD_L2": 0, "PassTD_L3": 0,
     "RushYds_L1": 0, "RushYds_L2": 0, "RushYds_L3": 0,
     "RecYds_L1": 0, "RecYds_L2": 0, "RecYds_L3": 0,
     "Rec_L1": 0, "Rec_L2": 0, "Rec_L3": 0},
]

df = pd.DataFrame(players)

# -----------------------------
# Derived metrics
# -----------------------------
df["Avg_Last3"] = df[["Pts_L1", "Pts_L2", "Pts_L3"]].mean(axis=1).round(1)
df["Value_Last3_per_$1k"] = (df["Avg_Last3"] / (df["Salary"] / 1000)).round(2)
df["Value_Season_per_$1k"] = (df["PPG_Season"] / (df["Salary"] / 1000)).round(2)

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

display_cols = ["Player", "Position", "Salary", "PPG_Season", "Last3_Spark", "Avg_Last3", "Value_Last3_per_$1k"]
pool_view = pool_df[display_cols].copy()

st.data_editor(pool_view, use_container_width=True, hide_index=True, disabled=True)

# -----------------------------
# Main: Player Details (replaces old Value Overview)
# -----------------------------
st.markdown("## 🔎 Player Details")

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
m2.metric("Season Avg", f"{row['PPG_Season']:.1f}")
m3.metric("Avg Last 3", f"{row['Avg_Last3']:.1f}")
m4.metric("Value (Last 3) / $1k", f"{row['Value_Last3_per_$1k']:.2f}")

# Last 3 fantasy points
st.markdown("**Last 3 Fantasy Points**")
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

st.caption("This is MVP demo data. In the real app, these stats would be pulled for the current slate/week.")
