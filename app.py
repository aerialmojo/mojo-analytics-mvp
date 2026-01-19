import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Mojo Analytics — Demo",
    layout="wide",
    initial_sidebar_state="expanded"
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

# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1, 2])

# -----------------------------
# Roster Builder (Left)
# -----------------------------
with left:
    st.markdown("### 🧩 Roster Builder (DraftKings-style Demo)")
    salary_cap = 50000
    total_salary = 0
    chosen_players = []

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

    c1, c2 = st.columns(2)

    for i, (slot_name, allowed_positions) in enumerate(slots):
        slot_options = df[df["Position"].isin(allowed_positions)]["Player"].tolist()
        slot_options = [p for p in slot_options if p not in chosen_players]

        target_col = c1 if i % 2 == 0 else c2
        with target_col:
            selection = st.selectbox(
                slot_name,
                ["—"] + slot_options,
                key=f"slot_{slot_name}"
            )

        if selection != "—":
            chosen_players.append(selection)
            total_salary += int(df.loc[df["Player"] == selection, "Salary"].iloc[0])

    st.markdown("---")
    st.metric("Salary Used", f"${total_salary:,}")
    st.metric("Remaining", f"${salary_cap - total_salary:,}")

    if total_salary > salary_cap:
        st.error("Over the $50,000 salary cap. Try swapping to lower-cost players.")

# -----------------------------
# Value Table (Right)
# -----------------------------
with right:
    st.markdown("### 📊 Player Value Overview")

    value_col = "Value_per_$1k"

    def color_value(val: float) -> str:
        q1 = df[value_col].quantile(0.25)
        q3 = df[value_col].quantile(0.75)
        if val >= q3:
            return "background-color: rgba(0, 200, 0, 0.20)"   # green
        if val <= q1:
            return "background-color: rgba(255, 0, 0, 0.18)"   # red
        return "background-color: rgba(255, 215, 0, 0.18)"     # yellow

    table_df = df[["Player", "Position", "Salary", "PPG_3", "PPG_Season", value_col]].copy()

    styled = (
        table_df.style
        .format({
            "Salary": "${:,.0f}",
            "PPG_3": "{:.1f}",
            "PPG_Season": "{:.1f}",
            value_col: "{:.2f}",
        })
        .applymap(color_value, subset=[value_col])
    )

    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("**Legend:** 🟩 High value • 🟨 Medium • 🟥 Low (based on this slate)")
