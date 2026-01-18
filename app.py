import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Mojo Analytics — Demo",
    layout="wide"
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
    {"Player": "Jalen Hurts", "Position": "QB", "Salary": 8200, "PPG_3": 24.1, "PPG_Season": 22.8},
    {"Player": "Josh Allen", "Position": "QB", "Salary": 8600, "PPG_3": 23.0, "PPG_Season": 23.5},
    {"Player": "Lamar Jackson", "Position": "QB", "Salary": 8000, "PPG_3": 21.8, "PPG_Season": 22.1},

    {"Player": "Christian McCaffrey", "Position": "RB", "Salary": 9000, "PPG_3": 26.4, "PPG_Season": 24.9},
    {"Player": "Saquon Barkley", "Position": "RB", "Salary": 7600, "PPG_3": 19.8, "PPG_Season": 18.6},

    {"Player": "Tyreek Hill", "Position": "WR", "Salary": 8800, "PPG_3": 25.2, "PPG_Season": 23.7},
    {"Player": "A.J. Brown", "Position": "WR", "Salary": 8100, "PPG_3": 21.4, "PPG_Season": 20.2},
    {"Player": "CeeDee Lamb", "Position": "WR", "Salary": 8300, "PPG_3": 22.1, "PPG_Season": 21.0},

    {"Player": "Travis Kelce", "Position": "TE", "Salary": 6800, "PPG_3": 16.2, "PPG_Season": 15.1},
    {"Player": "Mark Andrews", "Position": "TE", "Salary": 6200, "PPG_3": 14.7, "PPG_Season": 13.9},

    {"Player": "49ers DST", "Position": "DST", "Salary": 3200, "PPG_3": 9.4, "PPG_Season": 8.7},
    {"Player": "Cowboys DST", "Position": "DST", "Salary": 3500, "PPG_3": 10.1, "PPG_Season": 9.1},
]

df = pd.DataFrame(players)
df["Value_per_$1k"] = (df["PPG_Season"] / (df["Salary"] / 1000)).round(2)

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
