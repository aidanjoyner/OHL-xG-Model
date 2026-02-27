# Load the Models
print("Loading models...")
models = {
    "EV": joblib.load('models/xg_model_XGBoostEV.pkl'),
    "PP": joblib.load('models/xg_model_XGBoostPP.pkl'),
    "PK": joblib.load('models/xg_model_XGBoostPK.pkl'),
    "EN": joblib.load('models/xg_model_XGBoostEN.pkl')
}
print("Models loaded successfully.")

# Load Data
# (Replace with your actual file path)
df = pd.read_parquet('ohl_shots_2019_2025.parquet') 

# Ensure rows are sorted correctly for time calculations
df = df.sort_values(['season', 'game_id', 'period_id', 'game_seconds'])

features = ['distance', 'angle', 'is_rebound', 'score_diff', 'game_seconds']


def assign_xg(df, models_dict, features):
    df = df.copy()
    df['xG'] = np.nan
    
    # Define the situations
    buckets = {
        "EV": df["strength_state"].isin(["5v5", "4v4", "3v3"]),
        "PP": df["strength_state"].isin(["5v4", "5v3", "4v3"]),
        "PK": df["strength_state"].isin(["4v5", "3v5", "3v4"]),
        "EN": df["strength_state"].isin(["6v5", "6v4", "6v3"])
    }
    
    print(f"Predicting xG for {len(df)} shots...")
    
    for name, mask in buckets.items():
        if mask.any():
            # Filter data for this situation
            X_subset = df.loc[mask, features]
            
            # Predict
            # (If your PP/PK models were trained with 'strength_value', 
            #  this might error. If so, just add it to the 'features' list above)
            try:
                probs = models_dict[name].predict_proba(X_subset)[:, 1]
                df.loc[mask, 'xG'] = probs
                print(f"  -> {name}: Scored {mask.sum()} shots")
            except Exception as e:
                print(f"  -> ERROR in {name}: {e}")
                
    # Fill any gaps (e.g. rare 3v6 situations) with 0
    df['xG'] = df['xG'].fillna(0)
    
    return df

xG_data = assign_xg(df, models, features)
xG_data.to_parquet("ohl_shots_2019_2025_xG.parquet")
