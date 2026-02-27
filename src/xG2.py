from sklearn.metrics import roc_auc_score, confusion_matrix, brier_score_loss, log_loss
from sklearn.model_selection import RandomizedSearchCV, LeaveOneGroupOut
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

data = pd.read_parquet('ohl_shots_2019_2025.parquet')

data.columns
data

print(data['strength_state'].value_counts())

data = data.dropna(subset=['distance'])

suspicious_ids = [24379, 25638, 25813, 26629] # Shot coordinates messed up for these games, drop them
data = data[~data['game_id'].isin(suspicious_ids)]

df = data[data['is_empty_net'] == 0].copy() # Remove EN goals

# Build strengths, target & model features
EV = ["5v5", "4v4", "3v3"]
PP = ["5v4", "5v3", "4v3"]
PK = ["4v5", "3v5", "3v4"]
EN = ["6v5", "6v4", "6v3"]
target = ["is_goal"]
features = ['distance', 'angle', 'is_rebound', 'score_diff', 'game_seconds']

# Create distinct datasets
ev_df = data[data["strength_state"].isin(EV)].copy()
pp_df = data[data["strength_state"].isin(PP)].copy()
pk_df = data[data["strength_state"].isin(PK)].copy()
en_df = data[data["strength_state"].isin(EN)].copy()

model_dataset = {
    "EV" : ev_df, "PP" : pp_df, "PK" : pk_df, "EN" : en_df
}
final_models = {}

# Model training & eval using CV
for name, df in model_dataset.items():
    print(f"\n{'='*40}")
    print(f"TRAINING MODEL: {name} ({len(df)} shots)")
    print(f"{'='*40}")

    if len(df) < 200:
        print(f"Skipping {name}: Not enough data.")
        continue

    X = df[features]
    y = df[target]
    
    # We need the season column to define the folds for CV
    groups = df['season'] 

    # Define Hyperparameter Grid & weight scaling ratio
    n_goals = y.sum()
    n_misses = len(y) - n_goals
    ratio = float(n_misses / n_goals)   
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9],
        'gamma': [0, 0.1],
        'scale_pos_weight': [1, ratio] 
    }

    xgb_base = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=-1,
        random_state=42
    )


    # Use Leave one out CV. This creates 5 folds: Train on 4 seasons, Test on the remaining 1.
    logo = LeaveOneGroupOut()
    
    search = RandomizedSearchCV(
        xgb_base,
        param_distributions=param_grid,
        n_iter=15,
        scoring='roc_auc',
        cv=logo,             
        verbose=0,
        n_jobs=-1,
        random_state=42
    )

    # We must pass 'groups' here so it knows to split by season
    search.fit(X, y, groups=groups)
    
    print(f"Best Params: {search.best_params_}")
    print(f"Best Season-CV AUC: {search.best_score_:.4f}")

    # We stick to standard CV for calibration to ensure the probabilities 
    # are smoothed across the entire history, not just one season.
    best_xgb = search.best_estimator_
    
    calibrated = CalibratedClassifierCV(
        best_xgb,
        method='isotonic',
        cv=5 
    )
    
    calibrated.fit(X, y)
    
    final_models[name] = calibrated
    
    # Model eval
    probs = calibrated.predict_proba(X)[:, 1]
    ll = log_loss(y, probs)
    brier = brier_score_loss(y, probs)
    auc = roc_auc_score(y, probs)
    print(f"-> Final Calibrated AUC: {auc:.4f}")
    print(f"-> Final Calibrated Brier Score: {brier:.4f}")
    print(f"-> Final Calibrated Log Loss: {ll:.4f}")

print("\nAll models trained!")


# Display final models
final_models

from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

def plot_feature_importance(model, df, features, target='is_goal'):
    # Filter for EV data only
    ev_data = df[df['strength_state'].isin(["5v5", "4v4", "3v3"])].copy()
    X = ev_data[features]
    y = ev_data[target]

    print("Calculating Permutation Importance (this takes a minute)...")
    result = permutation_importance(
        model, X, y, 
        n_repeats=10, 
        random_state=42, 
        n_jobs=-1,
        scoring='roc_auc'
    )

    # Sort and Plot
    sorted_idx = result.importances_mean.argsort()
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
    plt.xlabel("Drop in AUC Score")
    plt.title("What actually matters to your EV Model?")
    plt.grid(axis='x', alpha=0.3)
    plt.show()

# Run it
plot_feature_importance(final_models['EV'], data, features)

import joblib
import os

# Create a folder for organization
if not os.path.exists('models'):
    os.makedirs('models')

# Save each model
print("Saving models...")
for name, model in final_models.items():
    filename = f'models/xg_model_XGBoost{name}.pkl'
    joblib.dump(model, filename)
    print(f"-> Saved {name} model to {filename}")
