# Load in xG data
df = pd.read_parquet("ohl_shots_2019_2025_xG.parquet")


def get_teams(series):
        return "/".join(series.unique())

def build_shooter_metrics(df):
    print("Building Shooter Metrics")
    
    # --- STEP 1: VECTORIZED DEFINITIONS (Speed Optimization) ---
    # Create temporary columns for High Danger (xG >= 0.20)
    # We work on a copy so we don't modify the main dataframe outside this function
    df_calc = df.copy()
    is_HD = (df_calc['xG'] >= 0.20)
    
    df_calc['HD_Chance'] = is_HD.astype(int)
    df_calc['HD_Goal']   = (is_HD & (df_calc['is_goal'] == 1)).astype(int)

    # --- STEP 2: AGGREGATIONS ---
    
    # 1. Games Played & Team Names (Trade Logic)
    gp = df_calc.groupby(['player_id', 'player_name', 'season']).agg(
        player_team_name=('player_team_name', lambda x: '/'.join(sorted(x.unique()))), 
        GP=('game_id', 'nunique')
    ).reset_index()

    # 2. Total Stats (Now includes HD counts)
    total = df_calc.groupby(['player_id', 'player_name', 'season']).agg(
        Shots=('xG', 'count'),
        Goals=('is_goal', 'sum'),
        xG=('xG', 'sum'),
        # NEW: High Danger Sums
        HD_Chances=('HD_Chance', 'sum'),
        HD_Goals=('HD_Goal', 'sum')
    ).reset_index()

    # 3. Even Strength
    df_ev = df_calc[df_calc['strength_state'].isin(["5v5", "4v4", "3v3"])]
    ev_stats = df_ev.groupby(['player_id', 'player_name', 'season']).agg(
        Goals_EV=('is_goal', 'sum'),
        xG_EV=('xG', 'sum')
    ).reset_index()

    # 4. Powerplay
    df_pp = df_calc[df_calc['strength_state'].isin(["5v4", "5v3", "4v3"])]
    pp_stats = df_pp.groupby(['player_id', 'player_name', 'season']).agg(
        Goals_PP=('is_goal', 'sum'),
        xG_PP=('xG', 'sum')
    ).reset_index()

    # 5. Merge
    shooters = (gp.merge(total, on=['player_id', 'player_name', 'season'])
                  .merge(ev_stats, on=['player_id', 'player_name', 'season'], how='left')
                  .merge(pp_stats, on=['player_id', 'player_name', 'season'], how='left'))
    
    shooters = shooters.fillna(0)

    # --- STEP 3: DERIVED METRICS ---
    
    # Base Efficiency
    shooters['GAx'] = shooters['Goals'] - shooters['xG']
    shooters['xG_per_Shot'] = shooters['xG'] / shooters['Shots']
    
    # Percentages
    shooters['Shooting_%'] = (shooters['Goals'] / shooters['Shots']) * 100
    shooters['Exp_Shooting_%'] = (shooters['xG'] / shooters['Shots']) * 100
    shooters['Shooting_%_Ax'] = shooters['Shooting_%'] - shooters['Exp_Shooting_%']

    # NEW: High Danger Metrics
    # HD Shooting % (Finishing ability in tight)
    shooters['HD_Sh%'] = (shooters['HD_Goals'] / shooters['HD_Chances'].replace(0, 1)) * 100
    # HD Rate (Net front presence)
    shooters['HD_Chances/GP'] = shooters['HD_Chances'] / shooters['GP']

    # Standard Rates
    shooters['Goals/GP'] = shooters['Goals'] / shooters['GP']
    shooters['xG/GP'] = shooters['xG'] / shooters['GP']
    shooters['Shots/GP'] = shooters['Shots'] / shooters['GP']
    shooters['EV Goals/GP'] = shooters['Goals_EV'] / shooters['GP']
    shooters['EV xG/GP'] = shooters['xG_EV'] / shooters['GP']
    shooters['PP Goals/GP'] = shooters['Goals_PP'] / shooters['GP']
    shooters['PP xG/GP'] = shooters['xG_PP'] / shooters['GP']
    
    # Filter (Min 10 Games)
    shooters = shooters[shooters['GP'] >= 10]

    # --- STEP 4: CURRENT TEAM LOGO LOGIC ---
    latest_games = df_calc.sort_values('game_id', ascending=False)
    current_team = latest_games.drop_duplicates(subset=['player_id'])[['player_id', 'player_team_name']]
    current_team.columns = ['player_id', 'player_curr_team']
    
    shooters = shooters.merge(current_team, on='player_id', how='left')

    return shooters.sort_values('GAx', ascending=False)


def build_goalie_metrics(df):
    print("Building Goalie Metrics")
    
    if 'goalie_name' not in df.columns:
        return pd.DataFrame()

    # --- STEP 1: VECTORIZED DEFINITIONS ---
    df_calc = df.copy()
    
    # Create Masks
    is_HD = (df_calc['xG'] >= 0.20)
    is_MD = (df_calc['xG'] >= 0.06) & (df_calc['xG'] < 0.20)
    is_LD = (df_calc['xG'] < 0.06)

    # Create Integer Columns for Summing
    df_calc['HD_Shot'] = is_HD.astype(int)
    df_calc['HD_Goal'] = (is_HD & (df_calc['is_goal'] == 1)).astype(int)
    
    df_calc['MD_Shot'] = is_MD.astype(int)
    df_calc['MD_Goal'] = (is_MD & (df_calc['is_goal'] == 1)).astype(int)
    
    df_calc['LD_Shot'] = is_LD.astype(int)
    df_calc['LD_Goal'] = (is_LD & (df_calc['is_goal'] == 1)).astype(int)

    # --- STEP 2: AGGREGATIONS ---

    # 1. Games Played & Trade Names
    gp = df_calc.groupby(['goalie_id', 'goalie_name', 'season']).agg(
        goalie_team_name=('goalie_team_name', lambda x: '/'.join(sorted([str(t) for t in x.unique() if pd.notna(t) and t != '']))),
        GP=('game_id', 'nunique')
    ).reset_index()

    # 2. Total Stats (Includes Danger Zones)
    # Note: Added 'goalie_id' to groupby to be safe
    total = df_calc.groupby(['goalie_id', 'goalie_name', 'season']).agg(
        SA=('xG', 'count'),
        GA=('is_goal', 'sum'),
        xGA=('xG', 'sum'),
        # NEW: Danger Zone Sums
        HD_Shots=('HD_Shot', 'sum'),
        HD_Goals=('HD_Goal', 'sum'),
        MD_Shots=('MD_Shot', 'sum'),
        MD_Goals=('MD_Goal', 'sum'),
        LD_Shots=('LD_Shot', 'sum'),
        LD_Goals=('LD_Goal', 'sum')
    ).reset_index()

    # 3. Even Strength
    df_ev = df_calc[df_calc['strength_state'].isin(["5v5", "4v4", "3v3"])]
    ev_stats = df_ev.groupby(['goalie_id', 'goalie_name', 'season']).agg(
        SA_EV=('xG', 'count'),
        GA_EV=('is_goal', 'sum'),
        xGA_EV=('xG', 'sum')
    ).reset_index()

    # 4. Penalty Kill
    df_pk = df_calc[df_calc['strength_state'].isin(["5v4", "5v3", "4v3"])]
    pk_stats = df_pk.groupby(['goalie_id', 'goalie_name', 'season']).agg(
        SA_PK=('xG', 'count'),
        GA_PK=('is_goal', 'sum'),
        xGA_PK=('xG', 'sum')
    ).reset_index()

    # 5. Merge
    goalies = (gp.merge(total, on=['goalie_id', 'goalie_name', 'season'])
                 .merge(ev_stats, on=['goalie_id', 'goalie_name', 'season'], how='left')
                 .merge(pk_stats, on=['goalie_id', 'goalie_name', 'season'], how='left'))
    
    goalies = goalies.fillna(0)

    # --- STEP 3: DERIVED METRICS ---
    
    # Base Stats
    goalies['Sv%'] = ((goalies['SA'] - goalies['GA']) / goalies['SA']) * 100
    goalies['xSv%'] = (1 - (goalies['xGA'] / goalies['SA'])) * 100
    goalies['dSv%'] = goalies['Sv%'] - goalies['xSv%']
    goalies['GSAx'] = goalies['xGA'] - goalies['GA']
    
    # NEW: Danger Zone Save Percentages
    # (Using replace(0, 1) to prevent divide by zero errors)
    goalies['HDSv%'] = (1 - (goalies['HD_Goals'] / goalies['HD_Shots'].replace(0, 1))) * 100
    goalies['MDSv%'] = (1 - (goalies['MD_Goals'] / goalies['MD_Shots'].replace(0, 1))) * 100
    goalies['LDSv%'] = (1 - (goalies['LD_Goals'] / goalies['LD_Shots'].replace(0, 1))) * 100

    # Situational
    goalies['Sv%_EV'] = ((goalies['SA_EV'] - goalies['GA_EV']) / goalies['SA_EV']) * 100
    goalies['GSAx_EV'] = goalies['xGA_EV'] - goalies['GA_EV']
    goalies['Sv%_PK'] = ((goalies['SA_PK'] - goalies['GA_PK']) / goalies['SA_PK']) * 100
    goalies['GSAx_PK'] = goalies['xGA_PK'] - goalies['GA_PK']

    # Rates
    goalies['xGA/GP'] = goalies['xGA'] / goalies['GP'] 
    goalies['GA/GP'] = goalies['GA'] / goalies['GP']
    goalies['GSAx/GP'] = goalies['GSAx'] / goalies['GP']
    goalies['EV GSAx/GP'] = goalies['GSAx_EV'] / goalies['GP']
    goalies['PK GSAx/GP'] = goalies['GSAx_PK'] / goalies['GP']

    # Filter
    goalies = goalies[goalies['GP'] >= 5]

    # --- STEP 4: CURRENT TEAM LOGO LOGIC ---
    latest_games = df_calc.sort_values('game_id', ascending=False)
    current_team = latest_games.drop_duplicates(subset=['goalie_id'])[['goalie_id', 'goalie_team_name']]
    current_team.columns = ['goalie_id', 'goalie_curr_team']
    
    goalies = goalies.merge(current_team, on='goalie_id', how='left')
    
    return goalies.sort_values('GSAx', ascending=False)


def build_team_metrics(df):
    print("Building Team Metrics")
    
    gp = df.groupby(['player_team_name', 'season'])['game_id'].nunique().reset_index()
    gp.columns = ['team_name', 'season', 'GP']
    
    offense = df.groupby(['player_team_name', 'season']).agg(
        xGF=('xG', 'sum'),
        GF=('is_goal', 'sum'),
        SF=('xG', 'count')
    ).reset_index().rename(columns={'player_team_name': 'team_name', 'xG' : 'xGF'})
    
    # 2. Defense (xGA and GA)
    # Group by the defending team (goalie_team_name)
    defense = df.groupby(['goalie_team_name', 'season']).agg(
        xGA=('xG', 'sum'),
        GA=('is_goal', 'sum'),
        SA=('xG', 'count')
    ).reset_index().rename(columns={'goalie_team_name': 'team_name', 'xG' : 'xGA'})

    df_ev = df[df['strength_state'].isin(["5v5", "4v4", "3v3"])]
    
    off_ev = df_ev.groupby(['player_team_name', 'season']).agg(
        xGF_EV=('xG', 'sum'),
        GF_EV=('is_goal', 'sum'),
        SF_EV=('xG', 'count')
    ).reset_index().rename(columns={'player_team_name': 'team_name'})
    
    def_ev = df_ev.groupby(['goalie_team_name', 'season']).agg(
        xGA_EV=('xG', 'sum'),
        GA_EV=('is_goal', 'sum'),
        SA_EV=('xG', 'count')
    ).reset_index().rename(columns={'goalie_team_name': 'team_name'})

    df_pp = df[df['strength_state'].isin(["5v4", "5v3", "4v3"])]
    off_pp = df_pp.groupby(['player_team_name', 'season']).agg(
        xGF_PP=('xG', 'sum'),
        GF_PP=('is_goal', 'sum')
    ).reset_index().rename(columns={'player_team_name': 'team_name'})

    df_pk = df[df['strength_state'].isin(["5v4", "5v3", "4v3"])]
    def_pk = df_pk.groupby(['goalie_team_name', 'season']).agg(
        xGA_PK=('xG', 'sum'),
        GA_PK=('is_goal', 'sum')
    ).reset_index().rename(columns={'goalie_team_name': 'team_name'})
    # 3. Merge
    teams = (gp.merge(offense, on=['team_name', 'season']) 
               .merge(defense, on=['team_name', 'season']) 
               .merge(off_ev, on=['team_name', 'season'], how='left') 
               .merge(def_ev, on=['team_name', 'season'], how='left') 
               .merge(off_pp, on=['team_name', 'season'], how='left') 
               .merge(def_pk, on=['team_name', 'season'], how='left'))

    teams = teams.fillna(0)

    # Derived Metrics
    
    teams['GF%'] = (teams['GF'] / (teams['GF'] + teams['GA'])) * 100
    teams['Goal_Diff'] = teams['GF'] - teams['GA']
    teams['xGF%'] = (teams['xGF'] / (teams['xGF'] + teams['xGA'])) * 100
    teams['xG_Diff'] = teams['xGF'] - teams['xGA']
    teams['SF%_EV'] = (teams['SF_EV'] / (teams['SF_EV'] + teams['SA_EV'])) * 100
    teams['GF_vs_xGF (Shooting)'] = teams['GF'] - teams['xGF'] 
    teams['GA_vs_xGA (Goaltending)'] = teams['xGA'] - teams['GA']
    teams['G_xG_Diff'] = teams['Goal_Diff'] - teams['xG_Diff']
    teams['xGF%_EV'] = (teams['xGF_EV'] / (teams['xGF_EV'] + teams['xGA_EV'])) * 100
    teams['GF%_EV'] = (teams['GF_EV'] / (teams['GF_EV'] + teams['GA_EV'])) * 100
    teams['EVxG_Diff'] = teams['xGF_EV'] - teams['xGA_EV']
    teams['EV_GAx (EV Shooting)'] = teams['GF_EV'] - teams['xGF_EV'] 
    teams['EV_GSAx (EV Goaltending)'] = teams['xGA_EV'] - teams['GA_EV']
    teams['PP_GAx (PP Shooting)'] = teams['GF_PP'] - teams['xGF_PP'] 
    teams['PK_GSAx (PK Goaltending)'] = teams['xGA_PK'] - teams['GA_PK']
    teams['SF%'] = (teams['SF'] / (teams['SF'] + teams['SA'])) * 100
    teams['Sh%'] = (teams['GF'] / teams['SF']) * 100
    teams['Sv%'] = (1 - (teams['GA'] / teams['SA'])) * 100
    teams['PDO'] = teams['Sh%'] + teams['Sv%']
    teams['xGF/GP'] = teams['xGF'] / teams['GP']
    teams['xGA/GP'] = teams['xGA'] / teams['GP']
    
    # per game rates
    teams['GF/GP'] = teams['GF'] / teams['GP']
    teams['GA/GP'] = teams['GA'] / teams['GP']
    teams['xGF/GP'] = teams['xGF'] / teams['GP']
    teams['xGA/GP'] = teams['xGA'] / teams['GP']
    teams['EV xGF/GP'] = teams['xGF_EV'] / teams['GP']
    teams['EV xGA/GP'] = teams['xGA_EV'] / teams['GP']
    teams['EV GF/GP'] = teams['GF_EV'] / teams['GP']
    teams['EV GA/GP'] = teams['GA_EV'] / teams['GP']
    teams['PP xGF/GP'] = teams['xGF_PP'] / teams['GP']
    teams['PP GF/GP'] = teams['GF_PP'] / teams['GP']
    teams['PK xGA/GP'] = teams['xGA_PK'] / teams['GP']
    teams['PK GA/GP'] = teams['GA_PK'] / teams['GP']

    teams = teams[teams['GP'] >= 5]
    
    return teams.sort_values('xGF%_EV', ascending=False)


import os
from datetime import datetime
import requests

# --- CONFIGURATION ---
MASTER_DATA_FILE = 'ohl_shots_with_xg.csv'
EXCEL_REPORT_FILE = 'ohl_analytics_suite_2026.xlsx'
CURRENT_SEASON = 2026


def load_resources():
    print(f"[{datetime.now().time()}] Loading resources...")
    
    # Load Master Data
    if os.path.exists(MASTER_DATA_FILE):
        master_df = pd.read_csv(MASTER_DATA_FILE)
        existing_ids = master_df['game_id'].unique().tolist()
        print(f"   -> Loaded {len(master_df)} rows from master file.")
        print(f"   -> Found {len(existing_ids)} games already processed.")
    else:
        master_df = pd.DataFrame()
        existing_ids = []
        print("   -> No master file found. Starting fresh.")
        
    return master_df, existing_ids

def get_completed_schedule(season_id):
    """
    Fetches the schedule and filters for completed games using 
    the keys found in OHL Season 83 JSON.
    """
    url = f"https://lscluster.hockeytech.com/feed/index.php?feed=modulekit&view=schedule&key=2976319eb44abe94&fmt=json&client_code=ohl&season_id={season_id}"
    
    try:
        response = requests.get(url).json()
        games = response['SiteKit']['Schedule']
        
        completed_ids = []
        for g in games:
            # 1. Check "game_status" (Should be "Final")
            game_status = g.get('game_status', '')
            
            # 2. Check "final" (Should be "1")
            is_final = g.get('final', '0')
            
            # 3. Check "status" (Should be "4" for this API version)
            status_code = g.get('status', '')

            # Logic: If ANY of these say the game is over, we take it.
            if game_status == "Final" or is_final == "1" or status_code == "4":
                completed_ids.append(int(g['game_id']))
                
        return completed_ids
    except Exception as e:
        print(f"   -> Error fetching schedule: {e}")
        return []


def fetch_new_games(existing_ids):
    print(f"[{datetime.now().time()}] Checking for new games...")
    
    # 1. Get the full list of completed games for this season
    # (Make sure you use the correct season_id for 2025-26. It might be roughly 80 or 81 in the OHL API)
    current_season_code = 83 # Example code for 2025-26
    all_completed_games = get_completed_schedule(current_season_code)
    
    # 2. Filter out the ones we already have
    # (Convert to sets for fast subtraction)
    existing_set = set(existing_ids)
    new_game_ids = [gid for gid in all_completed_games if gid not in existing_set]
    
    if not new_game_ids:
        print("   -> Up to date. No new games found.")
        return pd.DataFrame()
    
    print(f"   -> Found {len(new_game_ids)} new games to scrape.")
    
    # 3. Format strictly for your original scraper
    # Your scraper wants: { season_name: [list_of_ids] }
    dynamic_range = {
        2026: new_game_ids
    }
    
    # 4. Call your ORIGINAL scraper
    # It will process only these specific missing games
    df_new = scrape_ohl_data(dynamic_range, max_workers=5)
    
    return df_new


def generate_excel_report(full_df, output_file=EXCEL_REPORT_FILE):
    print(f"[{datetime.now().time()}] Regenerating Excel Leaderboards...")
    
    # 1. Call your existing functions
    # (These are the "Gold Master" functions we wrote earlier)
    df_shooters = build_shooter_metrics(full_df)
    df_goalies = build_goalie_metrics(full_df)
    df_teams = build_team_metrics(full_df)
    
    # 2. Write to Excel
    try:
        with pd.ExcelWriter(output_file) as writer:
            # Tab 1: Shooters
            df_shooters.to_excel(writer, sheet_name='Shooters', index=False)
            
            # Tab 2: Goalies (Only if data exists)
            if not df_goalies.empty:
                df_goalies.to_excel(writer, sheet_name='Goalies', index=False)
                
            # Tab 3: Teams
            df_teams.to_excel(writer, sheet_name='Teams', index=False)
            
        print(f"-> Success! Report saved to: {output_file}")
            
    except Exception as e:
        print(f"-> Error saving Excel report: {e}")
        # Make sure the file isn't open in Excel when you run this!


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load

    master_df, existing_ids = load_resources()
    
    # 2. Get New Data (You plug your scraper in here)
    new_shots = fetch_new_games(existing_ids)
    
    # 3. Process if new data exists
    if not new_shots.empty:
        print(f"   -> Found {len(new_shots)} new shots to process.")
        
        # Calculate xG
        processed_new_shots = assign_xg(new_shots, models, features)
        
        # Merge with Master
        master_df = pd.concat([master_df, processed_new_shots], ignore_index=True)
        master_df_sorted = master_df.sort_values(by='game_id', ascending=False)
        # Save Master
        master_df_sorted.to_csv(MASTER_DATA_FILE, index=False)
        print("   -> Shot CSV saved.")
        
    else:
        print("   -> No new games found.")

    #  Update Excel
    generate_excel_report(master_df)


INPUT_FILE_2 = 'ohl_shots_2019_2025_xG.parquet'  # Your historical file
OUTPUT_FILE_2 = 'OHL_xG_stats.xlsx'

if __name__ == "__main__": # for parquet
    print(f"Loading data from {INPUT_FILE_2}...")
    
    try:
        # Load the parquet (Ensure xG exists)
        df = pd.read_parquet(INPUT_FILE_2)
        
        # CLEANUP: Filter out rows without xG (bad data)
        df = df[df['xG'].notna()]
        
        print(f"Loaded {len(df):,} shots. Generating reports...")

        # Build Metrics
        df_shooters = build_shooter_metrics(df)
        df_goalies = build_goalie_metrics(df)
        df_teams = build_team_metrics(df)

        # Save to Excel
        print(f"Saving to {OUTPUT_FILE_2}...")
        with pd.ExcelWriter(OUTPUT_FILE_2) as writer:
            df_shooters.to_excel(writer, sheet_name='Shooters', index=False)
            df_goalies.to_excel(writer, sheet_name='Goalies', index=False)
            df_teams.to_excel(writer, sheet_name='Teams', index=False)
            
        print("SUCCESS! Historical Stats Updated.")

    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE_2}. Make sure it's in the same folder.")
    except Exception as e:
        print(f"An error occurred: {e}")