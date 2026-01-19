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
# -----------------------------
players = [
    {"Player": "Jalen Hurts", "Position": "QB", "Salary": 8200, "PPG_Season": 22.8, "Pts_L1": 20.1, "Pts_L2": 27.4, "Pts_L3": 24.8},
    {"Player": "Josh Allen", "Position": "QB", "Salary": 8600, "PPG_Season": 23.5, "Pts_L1": 18.7, "Pts_L2": 26.9, "Pts_L3": 23.4},
    {"Player": "Lamar Jackson", "Position": "QB", "Salary": 8000, "PPG_Season": 22.1, "Pts_L1": 21.0, "Pts_L2": 19.6, "Pts_L3": 28.1},

    {"Player": "Christian McCaffrey", "Position": "RB", "Salary": 9000, "PPG_Season": 24.9, "Pts_L1": 29.2, "Pts_L2": 22.4, "Pts_L3": 27.6},
    {"Player": "Saquon Barkley", "Position": "RB", "Salary": 7600, "PPG_Season": 18.6, "Pts_L1": 16.2, "Pts_L2": 21.1, "Pts_L3": 17.9},

    {"Player": "Tyreek Hill", "Position": "WR", "Salary": 8800, "PPG_Season": 23.7, "Pts_L1": 19.4, "Pts_L2": 31.0, "Pts_L3": 22.7},
    {"Player": "A.J. Brown", "Position": "WR", "Salary": 8100, "PPG_Season": 20.2, "Pts_L1": 14.8, "Pts_L2": 24.1, "Pts_L3": 18.9},
    {"Player": "CeeDee Lamb", "Position": "WR", "Salary": 8300, "PPG_Season": 21.0, "Pts_L1": 26.3, "Pts_L2": 17.2, "Pts_L3": 21.4},

    {"Player": "Travis Kelce", "Position": "TE", "Salary": 6800, "PPG_Season": 15.1, "Pts_L1": 10.4, "Pts_L2": 18.9, "Pts_L3": 14.2},
    {"Player": "Mark Andrews", "Position": "TE", "Salary": 6200, "PPG_Season": 13.9, "Pts_L1": 12.1, "Pts_L2": 15.8, "Pts_L3": 9.6},

    {"Player": "49ers DST", "Position": "DST", "Salary": 3200, "PPG_Season": 8.7, "Pts_L1": 7.0, "Pts_L2": 11.0, "Pts_L3": 10.2},
    {"Player": "Cowboys DST", "Position": "DST", "Salary": 3500, "PPG_Season": 9.1, "Pts_L1": 6.0, "Pts_L2": 13.4, "Pts_L3": 9.8},
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

# -----------------------------
# Helper: get currently selected players (as names)
# -----------------------------
def get_selected_players():
    selected = []
    for slot_name in slot_names:
        val = st.session_state.get(f"slot_{slot_name}", "—")
        if val and val != "—":
            selected.append(val)
    return selected

# -----------------------------
# Sidebar: Roster Builder (Option A quick pick)
# -----------------------------
with st.sidebar:
    st.markdown("## 🧩 Roster Builder")
    st.caption("DraftKings-style demo roster")

    selected_now = get_selected_players()

    # Dropdowns per slot
    for slot_name, allowed_positions in slots:
        # Build pool for this slot
        pool = df[df["Position"].isin(allowed_positions)].copy()

        # Prevent duplicates: exclude players selected in OTHER slots
        current_val = st.session_state.get(f"slot_{slot_name}", "—")
        exclude = set(selected_now)
        if current_val != "—":
            exclude.discard(current_val)

        pool = pool[~pool["Player"].isin(exclude)]

        # Rich label in dropdown
        pool["Label"] = pool.apply(
            lambda r: f'{r["Player"]} — ${int(r["Salary"]):,} — V3 {r["Value_Last3_per_$1k"]}',
            axis=1
        )

        # Map label -> player
        label_to_player = dict(zip(pool["Label"], pool["Player"]))
        player_to_label = {v: k for k, v in label_to_player.items()}

        options = ["—"] + pool["Label"].tolist()

        # Make default selection show correctly
        default_label = "—"
        if current_val != "—" and current_val in player_to_label:
            default_label = player_to_label[current_val]

        # Use a separate widget key so we can store player name in session_state
        picked_label = st.selectbox(
            slot_name,
            options,
            index=options.index(default_label) if default_label in options else 0,
            key=f"ui_{slot_name}"
        )

        # Persist as player name (not label)
        st.session_state[f"slot_{slot_name}"] = label_to_player.get(picked_label, "—")

    # Salary totals
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

    st.dataframe(
        lineup_df,
        use_container_width=True,
        hide_index=True
    )
else:
    st.caption("Use the sidebar to build a lineup, or assign players from the Player Pool below.")

st.markdown("---")

# -----------------------------
# Main: Player Pool (Option B browse + assign)
# -----------------------------
st.markdown("## 🧾 Player Pool (This Week)")

pos_filter = st.selectbox("Filter by Position", ["ALL", "QB", "RB", "WR", "TE", "DST"], index=0)
pool_df = df.copy()
if pos_filter != "ALL":
    pool_df = pool_df[pool_df["Position"] == pos_filter]

display_cols = ["Player", "Position", "Salary", "PPG_Season", "Last3_Spark", "Avg_Last3", "Value_Last3_per_$1k"]
pool_view = pool_df[display_cols].copy()

st.data_editor(
    pool_view,
    use_container_width=True,
    hide_index=True,
    disabled=True
)

# -----------------------------
# Main: Value overview table (color-coded)
# -----------------------------
st.markdown("## 📊 Player Value Overview")

value_col = "Value_Last3_per_$1k"

def color_value(val: float) -> str:
    q1 = df[value_col].quantile(0.25)
    q3 = df[value_col].quantile(0.75)
    if val >= q3:
        return "background-color: rgba(0, 200, 0, 0.20)"   # green
    if val <= q1:
        return "background-color: rgba(255, 0, 0, 0.18)"   # red
    return "background-color: rgba(255, 215, 0, 0.18)"     # yellow

table_df = df[["Player", "Position", "Salary", "PPG_Season", "Last3_Spark", "Avg_Last3", value_col]].copy()

styled = (
    table_df.style
    .format({
        "Salary": "${:,.0f}",
        "PPG_Season": "{:.1f}",
        "Avg_Last3": "{:.1f}",
        value_col: "{:.2f}",
    })
    .applymap(color_value, subset=[value_col])
)

st.dataframe(styled, use_container_width=True, hide_index=True)

st.caption(
    "Value (Last 3) per $1k = Avg points (last 3 games) ÷ (Salary / 1000). "
    "Higher values indicate more recent production per dollar."
)
st.markdown("**Legend:** 🟩 High value • 🟨 Medium • 🟥 Low (based on this slate)")
