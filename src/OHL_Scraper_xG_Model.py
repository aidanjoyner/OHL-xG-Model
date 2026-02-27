import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

# Obtain JSON data from leaguestat

api_key = os.getenv("OHL_API_KEY")
BASE = "https://cluster.leaguestat.com/feed/index.php"
COMMON = dict(feed="gc", key=api_key, client_code="ohl", lang_code="en", fmt="json")

def fetch(tab, game_id):
    params = {
        "feed": "gc",
        "key": api_key,
        "client_code": "ohl",
        "game_id": game_id,
        "lang_code": "en",
        "fmt": "json",
        "tab": tab
    }
    res = requests.get(BASE, params=params, timeout=30); res.raise_for_status()
    return res.json()

# Load in all necesdary game data
def load_game(game_id):

    # Save shots and pxp to separate dataframes
    shots = fetch("shots", game_id)
    pxp = fetch("pxpverbose", game_id)
    shots_df = pd.json_normalize(shots["GC"]["Shots"])
    pxp_df = pd.json_normalize(pxp["GC"]["Pxpverbose"])

    # Map players to their ID & teams for stat tracking
    player_map = (
        pxp_df[["player.player_id", "player.first_name", "player.last_name", "player.team_code"]]
        .dropna(subset=["player.player_id"])
        .drop_duplicates(subset=["player.player_id"])
        .rename(columns={
            "player.player_id": "player_id",
            "player.first_name": "first_name",
            "player.last_name": "last_name",
            "player.team_code": "player_team_name"  # renamed directly to "team_name"
        })
        .assign(player_name=lambda d: d["first_name"] + " " + d["last_name"])
        [["player_id", "player_name", "player_team_name"]]
    )

    # Do the same for goalies
    goalie_map = (
        pxp_df[["goalie.player_id", "goalie.first_name", "goalie.last_name", "goalie.team_code"]]
        .dropna(subset=["goalie.player_id"])
        .drop_duplicates(subset=["goalie.player_id"])
        .rename(columns={
            "goalie.player_id": "goalie_id",
            "goalie.first_name": "first_name",
            "goalie.last_name": "last_name",
            "goalie.team_code": "goalie_team_name"  # renamed directly
        })
        .assign(goalie_name=lambda d: d["first_name"] + " " + d["last_name"])
        [["goalie_id", "goalie_name", "goalie_team_name"]]
    )

    # Merge these mappings based on their IDs
    shots_df = (
        shots_df
        .merge(player_map, on="player_id", how="left")
        .merge(goalie_map, on="goalie_id", how="left")
    )

    # Return the shot & pxp data
    return shots_df, pxp_df


# Clean the shot data for our model
def clean_shots(df, game_id, season):

    # Columns from our shot data that we need
    columns = ["period_id", "home", "player_id", "player_name", "player_team_id", "player_team_name", "goalie_id", "goalie_name", "goalie_team_id", "goalie_team_name", "time", "s", "x_location", "y_location", "game_goal_id", 'goal_type_name']
    if 'goal_type_name' not in df.columns:
        df['goal_type_name'] = ''
    
    # Copy, sort & make necesssary data numeric for calculations
    df = df[columns].copy()    
    df = df.sort_values(["period_id", "s"])
    df[["x_location", "y_location", "s", "period_id", "home"]] = (
    df[["x_location", "y_location", "s", "period_id", "home"]]
    .apply(pd.to_numeric, errors="coerce"))

    # Add in basic information 
    df["season"] = season
    df["game_id"] = game_id
    df["is_goal"] = df["game_goal_id"].astype(str).str.len().gt(0).astype(int)
    df["game_seconds"] = (df["period_id"] - 1) * 1200 + df["s"]

    # Add if the shot taken was a rebound (if the last shot was less than 3 seconds ago)
    df["is_rebound"] = ((df["player_team_id"] == df["player_team_id"].shift(1)) & ((df["game_seconds"] - df["game_seconds"].shift(1)) <= 3)).astype(int)

    # Determine the home team goals
    df["home_goals"] = (
        df.assign(home_goal_event=(df["is_goal"] & (df["home"] == 1)))
        .groupby("game_id")["home_goal_event"]
        .transform("cumsum")
        .shift(fill_value=0)
    )

    # Do the same for the visiting team
    df["away_goals"] = (
        df.assign(away_goal_event=(df["is_goal"] & (df["home"] == 0)))
        .groupby("game_id")["away_goal_event"]
        .transform("cumsum")
        .shift(fill_value=0)
    )

    # Map home and away goals to shooting/defending team goals
    df["shooting_team_goals"] = np.where(df["home"] == 1, df["home_goals"], df["away_goals"])
    df["defending_team_goals"] = np.where(df["home"] == 1, df["away_goals"], df["home_goals"])

    # determine the current score difference of the shooting team & defending team, capping it -3 & 3
    df["score_diff"] = (df["shooting_team_goals"] - df["defending_team_goals"]).clip(-3, 3)

    # Set the rink coordinates for distance & angle calculations
    x_min, x_max, y_min, y_max = 0, 600, 0, 300
    mid_x = (x_max + x_min) / 2
    mid_y = (y_max + y_min) / 2
    scale_x = (x_max - x_min) / 2
    scale_y = (y_max - y_min) / 2

    # Standardize x and y coordinates to feet (rink is 85ft wide by 200 feet long)
    df["x_norm"] = -((df["x_location"] - mid_x) / scale_x * 100)
    df["y_norm"] = (df["y_location"] - mid_y) / scale_y * 42.5

    # Set the coordinates for the net's y axis, home & away nets & the net's x axis based on home & away
    y_net, away_net, home_net = 0, -89, 89
    df["net_x"] = np.where(df["home"] == 1, home_net, away_net)

    # Find the x coordinate distance between the net & the shot
    dx = df["net_x"] - df["x_norm"]
    # Find the y coordinate distance between the net & the shot
    dy = df["y_norm"] - y_net
    # Calculate a temporary distance between the shot coordinates and the net
    df["distance"] = np.sqrt(dx**2 + dy**2)

    # Implement a safety net for incorrectly mapped data, where home & away shots are flipped upon data entry & average shot distance is outside the offensive zone
    period_means = df.groupby("period_id")["distance"].mean()
    flipped_periods = period_means[period_means > 80].index.tolist()
    
    if flipped_periods:
        # For flipped periods, swap the net target x coordinate (89 -> -89)
        mask_flip = df["period_id"].isin(flipped_periods)
        df.loc[mask_flip, "net_x"] = df.loc[mask_flip, "net_x"] * -1

        # Recalculate the x distance and overall distance for flipped scenarios
        dx.loc[mask_flip] = df.loc[mask_flip, "net_x"] - df.loc[mask_flip, "x_norm"]
        df.loc[mask_flip, "distance"] = np.sqrt(dx.loc[mask_flip]**2 + dy.loc[mask_flip]**2)

    # Calculate shot angle based on x and y distances
    df["angle"] = np.degrees(np.arctan(np.abs(dy) / np.abs(dx)))

    # Tag empty nets based on the goal type 
    df['is_empty_net'] = df['goal_type_name'].str.contains('EN', na=False).astype(int)
    en_mask = df['is_empty_net'] == 1

    # Remove goalie tags for goals scored on empty nets
    df.loc[en_mask, 'goalie_id'] = 'NA'
    df.loc[en_mask, 'goalie_name'] = 'NA'

    # Select & return relevant columns for analysis
    keep = [
        "season", "game_id", "period_id", "home",
        "player_id", "player_name", "player_team_id", "player_team_name",
        "goalie_id", "goalie_name", "goalie_team_id", "goalie_team_name",
        "time", "game_seconds", "x_norm", "y_norm", "distance", "angle", "shooting_team_goals", "defending_team_goals", "score_diff",
        "is_rebound", "is_empty_net", "is_goal"
    ]
    return df[keep]

# Clean the play-by-play data for penalties & goalie pulls
def clean_pxp(pxp, game_id):

    # Copy the dataframe
    pxp_df = pxp.copy()

    # Add missing columns & fill with zeros
    for col in ["period_id", "s", "home", "team_id", "pp", "minutes"]:
        if col not in pxp_df.columns:
            pxp_df[col] = 0  

    # Assume missing penalties are minors
    if "penalty_class" not in pxp_df.columns:
        pxp_df["penalty_class"] = "Minor"

    # Fill any missing values 
    pxp_df[["period_id", "s", "home", "team_id", "pp", "minutes"]] = (
        pxp_df[["period_id", "s", "home", "team_id", "pp", "minutes"]]
    .apply(pd.to_numeric, errors="coerce")).fillna(0).astype(int)

    # Set gameID and seconds for the event
    pxp_df["game_id"] = game_id
    pxp_df["game_seconds"] = (pxp_df["period_id"] - 1) * 1200 + pxp_df["s"]

    # Create goal dataframe for penalty tracking 
    goals = pxp_df[pxp_df["event"] == "goal"].copy()
    goals["pp"] = goals.get("power_play", 0)
    expected_cols = ["game_id", "home", "team_id", "goal_player_id", "period_id", "game_seconds", "pp"]
    available_cols = [c for c in expected_cols if c in goals.columns]
    goals = goals[available_cols]

    # Create dataframe for all penalties
    pen = pxp_df[(pxp_df["event"] == "penalty") & (pxp_df["pp"] == 1)].copy()
    pen["proj_end_time"] = pen["game_seconds"] + pen["minutes"] * 60
    pen = pen[["game_id", "home", "team_id", "player_id", "period_id", "game_seconds", "penalty_class", "minutes", "pp", "proj_end_time"]]

    # Create dataframe for goalie pulls
    gpull = pxp_df[pxp_df["event"] == "goalie_change"].copy()

    # Determine when goalies were pulled & sort data
    pulled = gpull[gpull["goalie_out_id"].notna() & gpull["goalie_in_id"].isna()].copy()
    pulled = pulled.rename(columns={"game_seconds": "pull_time"})
    pulled["pull_time"] = pulled["pull_time"].astype(float)
    pulled = pulled.sort_values("pull_time")

    # Determine when goalies returned to the net & sort
    returned = gpull[gpull["goalie_in_id"].notna() & gpull["goalie_out_id"].isna()].copy()
    returned = returned.rename(columns={"game_seconds": "return_time"})
    # Create .1 second buffer to ensure shots that occur at the same game second are correctly labelled as empty nets
    returned["return_time"] = returned["return_time"] + 0.1
    returned = returned.sort_values("return_time")    

    # Merge pulled & returned events & clean it
    goalie_intervals = pd.merge_asof(pulled, returned, by="team_id", left_on="pull_time", right_on="return_time", direction="forward", suffixes=("", "_y"))
    goalie_intervals = goalie_intervals[["game_id", "team_id", "period_id", "pull_time", "return_time"]]
    goalie_intervals["return_time"] = goalie_intervals["return_time"].fillna(pxp_df["game_seconds"].max())
    goalie_intervals = goalie_intervals[~goalie_intervals["pull_time"].isin(goals["game_seconds"])] # Change made

    # Return cleaned penalty & goalie intervals
    return pen, goals, goalie_intervals

# Adjust penalty intervals if a goal was scored while on the powerplay
def adjust_penalty_intervals(pen, goals):

    # merge to see if a goal occurred during each penalty window
    merged = pen.merge(
        goals,
        on=["game_id"],
        how="left",
        suffixes=("", "_goal")
    )

    # keep goals scored by opposing team during the penalty & goals on majors
    mask = (
        (merged["team_id"] != merged["team_id_goal"]) &
        (merged["game_seconds_goal"] >= merged["game_seconds"]) &
        (merged["game_seconds_goal"] <= merged["proj_end_time"]) & 
        (merged["penalty_class"] == "Minor")
    )

    # Create a copy of the dataframe to manage powerplay goals cutting off the penalty
    pen_end = merged[mask].copy()

    # Handle dataframes without any penalties
    if pen_end.empty:
        return pen.rename(columns={"game_seconds": "start", "proj_end_time": "end"})

    # Rank active penalties by their start time for each goal.
    pen_end["penalty_rank"] = pen_end.groupby(
        ["game_id", "game_seconds_goal", "team_id"]
    )["game_seconds"].rank("first")
    
    # Only keep the cutoff for Rank 1. Rank 2 (the second penalty in a 5v3) stays active.
    valid_cuts = pen_end[pen_end["penalty_rank"] == 1].copy()

    # take the earliest goal scored during a penalty to cutoff the end time.
    cut = valid_cuts.groupby(
        ["game_id", "team_id", "game_seconds", "proj_end_time"]
    )["game_seconds_goal"].min().reset_index()

    # Create .1 second buffer to ensure shots that occur at the same game second are correctly labelled as on the powerplay
    cut["game_seconds_goal"] = cut["game_seconds_goal"] + 0.1

    # Merge back to the original penalty dataframe
    pen_final = pen.merge(
        cut, 
        on=["game_id", "team_id", "game_seconds", "proj_end_time"], 
        how="left"
    )
    
    # Use the new cut time if it exists. Otherwise, use the projected end time
    pen_final["end"] = pen_final["game_seconds_goal"].fillna(pen_final["proj_end_time"])
    pen_final = pen_final.rename(columns={"game_seconds": "start"})
    
    # Make start and end time floats to work with our buffer
    pen_final["start"] = pen_final["start"].astype(float)
    pen_final["end"] = pen_final["end"].astype(float)

    # Return the final penalty dataframe
    return pen_final[["game_id", "team_id", "home", "period_id", "start", "end"]]

# Combine the penalty & goalie pull data
def combine_strength_intervals(pen_adj, goalie_intervals):

    # Make a copy & rename columns
    pen_adj = pen_adj.copy()
    pen_adj = pen_adj.rename(columns={"pen_start": "start","pen_end": "end"})
   
    # Fill missing data & keep necessary columns
    if "period_id" not in pen_adj:
        pen_adj["period_id"] = np.floor(pen_adj["start"] // 1200 + 1)
    if "home" not in pen_adj:
        pen_adj["home"] = np.nan
    pen_adj["type"] = "penalty"
    pen_adj = pen_adj[["game_id", "team_id", "home", "period_id", "start", "end", "type"]]

    # Make a copy & rename columns
    gpull = goalie_intervals.copy()
    gpull = gpull.rename(columns={"pull_time": "start","return_time": "end"})

    # Fill missing data & keep necessary columns
    if "home" not in gpull:
        gpull["home"] = np.nan
    gpull["type"] = "goalie_pull"
    gpull = gpull[["game_id", "team_id", "home", "period_id", "start", "end", "type"]]

    # Combine penalty and goalie pull intervals
    intervals = pd.concat([pen_adj, gpull], ignore_index=True)


    # Fill missing home with mapped values from penalties
    home_map = pen_adj.dropna(subset=["home"])[["team_id", "home"]].drop_duplicates()
    intervals = intervals.merge(home_map, on="team_id", how="left", suffixes=("", "_pen"))
    intervals["home"] = intervals["home"].fillna(intervals["home_pen"])
    intervals = intervals.drop(columns=["home_pen"])

    # Ensure the data is numeric & sorted
    intervals["start"] = pd.to_numeric(intervals["start"], errors="coerce")
    intervals["end"] = pd.to_numeric(intervals["end"], errors="coerce")
    intervals = intervals.sort_values(["game_id", "start"]).reset_index(drop=True)

    # Return merged & sorted intervals for labelling strength
    return intervals

# Label the strength of each shot based on penalties & goalie pulls
def label_strength2(shots_df, intervals):

    # Handle games without any penalties/pulls
    if intervals.empty:
        df = shots_df.copy()
        # Set base number of skaters for regulation & overtime (period 4)
        base = np.where(df["period_id"] == 4, 3, 5)
        df["home_skaters"] = base
        df["away_skaters"] = base
        
        # Create the shooting & defending team skater columns & set strength state
        df["shooting_team_skaters"] = np.where(df["home"] == 1, df["home_skaters"], df["away_skaters"])
        df["defending_team_skaters"] = np.where(df["home"] == 1, df["away_skaters"], df["home_skaters"])
        df["strength_state"] = df["shooting_team_skaters"].astype(str) + "v" + df["defending_team_skaters"].astype(str)

        # Return the shot data with labelled strengths
        return df.drop(columns=["home_skaters", "away_skaters"], errors="ignore")

    # Create event array
    events = []

    # Iterate through rows of interval data
    for _, row in intervals.iterrows():

        # Create variables for change in home & away data
        h_change = 0
        a_change = 0
        
        # Create tag for if the event was in OT
        is_ot = (row["period_id"] == 4)

        # Adjust skaters if there's a penalty
        if row["type"] == "penalty":

            # Logic for home penalties
            if row["home"] == 1: 
                # If overtime, add a skater to the away team, otherwise take away a home skater
                if is_ot: 
                    a_change = 1 
                else: 
                    h_change = -1    
            # Logic for away penalties
            else: 
                # If overtime, add a skater to the home team, otherwise take away an away skater
                if is_ot: 
                    h_change = 1
                else: 
                    a_change = -1
            
            # Set strength change reversals for when the penalty ends
            h_rev, a_rev = -h_change, -a_change

        # Logic for goalie pulls
        elif row["type"] == "goalie_pull":
            # Add a home skater if the home team pulls their goalie, and add to away otherwise
            if row["home"] == 1: 
                h_change = 1
            else: 
                a_change = 1
            # Set reversals for when the goalies are returned
            h_rev, a_rev = -h_change, -a_change

        # Add event start
        events.append({"game_id": row["game_id"], "game_seconds": row["start"], "h_delta": h_change, "a_delta": a_change})
        # Add event end
        events.append({"game_id": row["game_id"], "game_seconds": row["end"], "h_delta": h_rev, "a_delta": a_rev})

    # Handle and empty dataframes & strength change columns
    if not events:
        event_df = pd.DataFrame(columns=["game_id", "game_seconds", "h_delta", "a_delta"])
    else:
        event_df = pd.DataFrame(events)
    for col in ["h_delta", "a_delta"]:
        if col not in event_df.columns: event_df[col] = 0

    # Sort data based on game & time in seconds
    event_df = pd.DataFrame(events).sort_values(["game_id", "game_seconds"])

   
    # Combine event with shot data & build a full timeline of events
    unique_times = pd.concat([event_df[["game_id", "game_seconds"]],shots_df[["game_id", "game_seconds"]]]).drop_duplicates().sort_values(["game_id", "game_seconds"])
    timeline = unique_times.merge(event_df, on=["game_id", "game_seconds"], how="left").fillna(0)

    # Calculate Cumulative Changes in skater strength
    timeline["h_cumsum"] = timeline.groupby("game_id")["h_delta"].cumsum()
    timeline["a_cumsum"] = timeline.groupby("game_id")["a_delta"].cumsum()

    # Create copy of shot data 
    shots_df = shots_df.copy() 
    # Convert data types
    shots_df["game_id"] = shots_df["game_id"].astype(int)
    shots_df["game_seconds"] = shots_df["game_seconds"].astype(float)
    timeline["game_id"] = timeline["game_id"].astype(int)
    timeline["game_seconds"] = timeline["game_seconds"].astype(float)

    # Short shot & full event data
    shots_df = shots_df.sort_values(["game_id", "game_seconds"])
    timeline = timeline.sort_values(["game_id", "game_seconds"])

    # Merge shot & event data
    merged = pd.merge_asof(
        shots_df, 
        timeline[["game_id", "game_seconds", "h_cumsum", "a_cumsum"]],
        on="game_seconds", 
        by="game_id",
        direction="backward" # Look at state exactly at or before the shot
    )
    
    # Set skater baseline based on whether we're in regulation or overtime
    base_skaters = np.where(merged["period_id"] == 4, 3, 5)

    # Determine the number of skaters for each team, constraining the number o skaters between 3 & 6
    merged["home_skaters"] = (base_skaters + merged["h_cumsum"]).clip(3, 6).astype(int)
    merged["away_skaters"] = (base_skaters + merged["a_cumsum"]).clip(3, 6).astype(int)
    
    # Change the perspective from home/away skaters to shooting/defending team skaters
    merged["shooting_team_skaters"] = np.where(merged["home"] == 1, merged["home_skaters"], merged["away_skaters"])
    merged["defending_team_skaters"] = np.where(merged["home"] == 1, merged["away_skaters"], merged["home_skaters"])
    # Assign a strength state to each shot
    merged["strength_state"] = merged["shooting_team_skaters"].astype(str) + "v" + merged["defending_team_skaters"].astype(str)
    
    # Drop temporary cols to match the original data format
    return merged.drop(columns=["h_cumsum", "a_cumsum", "home_skaters", "away_skaters"], errors="ignore")



# Fully functional scraper to retrieve & clean shot data using all previous functions
def scrape_ohl_data(season_ranges, max_workers=5, save_path=None):
    
    # Load each game & process the data before saving
    def process_game(game_id, season):
       
        # Implement sleep timer to regulate request frequency
        time.sleep(random.uniform(0.5, 1.5))

        # Load shot and play-by-play data
        try:
            shots_raw, pxp = load_game(game_id)
            
            if shots_raw.empty: 
                return None
            
            # Clean our data
            shots = clean_shots(shots_raw, game_id, season)
            pen, goals, pulls = clean_pxp(pxp, game_id)
            
            # Transform our data
            pen_adj = adjust_penalty_intervals(pen, goals)
            intervals = combine_strength_intervals(pen_adj, pulls)
            shots_final = label_strength2(shots, intervals) 

            # Return the final transformed dataframe
            return shots_final

        # Print out any games that could not be loaded correctly
        except Exception as e:
            print(f"Error {game_id}: {e}") 
            return None

    # Create single list of games from each season
    games = []
    for season_name, ids in season_ranges.items():
        for gid in ids:
            games.append((gid, season_name))

    # Display how many games are being scraped with the max number of workers
    total_games = len(games)
    print(f"Starting scrape for {total_games} games with {max_workers} workers...")

    # Initialize list of all shots
    all_shots = []
    
    # Initialize parallel workers
    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        # Submit all tasks to  pool immediately, returning a dictionary to map future to GameID
        futures = {executor.submit(process_game, gid, season): gid for (gid, season) in games}

        # Initialize count of completed & process results as they're completed
        completed_count = 0
        for future in as_completed(futures):
            res = future.result()
            # Add successfully retrieved data to the shot list
            if res is not None:
                all_shots.append(res)

            # Increment completion count & gradually display loading progress
            completed_count += 1
            if completed_count % 50 == 0:
                print(f"Progress: {completed_count}/{total_games} games processed.")

    # Handle retrieval of no data 
    if not all_shots:
        print("No data found.")
        return pd.DataFrame()

    # Merge each game dataframe into one
    shots_all = pd.concat(all_shots, ignore_index=True)

    # Save the data to the desired file path
    if save_path:
        shots_all.to_parquet(save_path)
        print(f"Data saved to {save_path}")

    # Return the requested data
    return shots_all

def predict_xg_for_game(game_id):
    # Load pre-trained components
    ev_model = joblib.load("OHL_xg_modelv1.pkl")
    pp_model = joblib.load("OHL_xg_modelv1_pp.pkl")
    pk_model = joblib.load("OHL_xg_modelv1_pk.pkl")
    en_model = joblib.load("OHL_xg_modelv1_en.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("xg_columns.pkl")

    model_map = {
        "EV": "OHL_xg_modelv1.pkl",
        "PP": "OHL_xg_modelv1_pp.pkl",
        "PK": "OHL_xg_modelv1_pk.pkl",
        "EN": "OHL_xg_modelv1_en.pkl"
    }
    
    groups = {
        "EV": ["5v5", "4v4", "3v3"],
        "PP": ["5v4", "5v3", "4v3"],
        "PK": ["4v5", "3v5", "3v4"],
        "EN": ["6v5", "6v4", "6v3"]
    }

    shots_raw, pxp = load_game(game_id)
    shots = clean_shots(shots_raw, game_id)
    pen, goals, pulls = clean_pxp(pxp, game_id)
    pen_adj = adjust_penalty_intervals(pen, goals)
    intervals = combine_strength_intervals(pen_adj, pulls)
    shots_final = label_strength(shots, intervals)

    for group, strengths in groups.items():
        subset = shots_final[shots_final["strength_state"].isin(strengths)].copy()
        if subset.empty:
            continue
        
        # Load correct model
        model = joblib.load(model_map[group])
        
        X = subset[["period_id", "home", "score_diff", "strength_state", "is_rebound", "distance", "angle"]]
        X = X.dropna(subset=["distance", "angle"])
        X = pd.get_dummies(X, columns=["score_diff"], drop_first=True)
        X = X.reindex(columns=columns, fill_value=0)

        numeric_cols = ["distance", "angle", "period_id"]
        X[numeric_cols] = scaler.transform(X[numeric_cols])

        # Predict
        xg_probs = model.predict_proba(X)[:, 1]
        shots_final.loc[X.index, "xG"] = xg_probs
        
    return shots_final