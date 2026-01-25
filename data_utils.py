import nflreadpy as nfl
import pandas as pd


def load_player_stats(seasons=None):
    """
    Load NFL player game-level stats for the given seasons
    using nflreadpy (Polars) and convert to pandas.
    """
    if seasons is None:
        seasons = [2023, 2024, 2025]

    # Load Polars DataFrame
    player_stats = nfl.load_player_stats(seasons)

    # Convert Polars → Pandas
    df = player_stats.to_pandas()

        # Select commonly useful columns (offense + defense + penalties)
    keep_cols = [
        # core identity
        "player_display_name",
        "position",
        "recent_team",
        "opponent_team",
        "season",
        "week",

        # offensive production
        "passing_yards",
        "passing_tds",
        "passing_interceptions",   # some defs use this name
        "interceptions",           # old name in your code
        "sacks_suffered",          # sacks taken (for QBs / passers)

        "rushing_yards",
        "rushing_tds",

        "receiving_yards",
        "receiving_tds",

        "fantasy_points",

        # defensive stats (front 7 + DBs)
        "def_tackles",
        "def_tackles_solo",
        "def_tackles_with_assist",
        "def_tackle_assists",
        "def_tackles_for_loss",
        "def_tackles_for_loss_yards",
        "def_fumbles_forced",
        "def_fumbles",
        "def_sacks",
        "def_sack_yards",
        "def_qb_hits",
        "def_interceptions",
        "def_interception_yards",
        "def_pass_defended",
        "def_tds",
        "def_safeties",

        # OL-ish / discipline
        "penalties",
        "penalty_yards",
    ]

    # Keep only columns that exist
    keep_cols = [col for col in keep_cols if col in df.columns]
    df = df[keep_cols]

    return df


def load_team_stats(seasons=None):
    """
    Load NFL team-level weekly stats for the given seasons
    and return as a pandas DataFrame.
    """
    if seasons is None:
        seasons = [2023, 2024, 2025]

    team_stats = nfl.load_team_stats(seasons)
    df = team_stats.to_pandas()

    # Keep commonly useful columns if they exist
    keep_cols = [
        "team",
        "opponent",
        "season",
        "week",
        "passing_yards",
        "rushing_yards",
        "receiving_yards",
        "passing_tds",
        "rushing_tds",
        "receiving_tds",
        "points",
        "total_yards",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    return df
