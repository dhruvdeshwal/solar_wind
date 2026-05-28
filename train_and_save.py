"""
Run this ONCE in Google Colab after your notebook has finished.
It trains the best models on your full dataset and saves them as .pkl files.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load your already-preprocessed df (run after your notebook cells) ──────
# df is already in memory from your notebook. If running standalone:
# df = pd.read_csv("time_series_60min_singleindex_filtered.csv", parse_dates=[0], index_col=0)

# ── 2. Preprocessing (same as your notebook) ─────────────────────────────────
def preprocess(df):
    if 'DE_solar_generation_actual' in df.columns:
        nulls = df['DE_solar_generation_actual'].isna()
        df.loc[nulls, 'DE_solar_generation_actual'] = df['DE_solar_generation_actual'].shift(24)[nulls]
        df['DE_solar_generation_actual'].fillna(0, inplace=True)
    if 'DE_wind_generation_actual' in df.columns:
        nulls = df['DE_wind_generation_actual'].isna()
        df.loc[nulls, 'DE_wind_generation_actual'] = df['DE_wind_generation_actual'].shift(24)[nulls]
        df['DE_wind_generation_actual'].fillna(df['DE_wind_generation_actual'].mean(), inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

df = preprocess(df)

# ── 3. Solar: Random Forest ───────────────────────────────────────────────────
solar_features = ['DE_solar_profile']
solar_target   = 'DE_solar_generation_actual'

X_s = df[solar_features].values
y_s = df[solar_target].values

solar_scaler = MinMaxScaler()
X_s_scaled   = solar_scaler.fit_transform(X_s)

X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(X_s_scaled, y_s, test_size=0.2, random_state=42)
solar_model = RandomForestRegressor(n_estimators=100, random_state=42)
solar_model.fit(X_s_train, y_s_train)
print(f"Solar RF R² = {r2_score(y_s_test, solar_model.predict(X_s_test)):.4f}")

# ── 4. Wind: SVM ──────────────────────────────────────────────────────────────
wind_features = ['DE_wind_profile', 'DE_wind_onshore_profile', 'DE_wind_offshore_profile']
wind_features = [f for f in wind_features if f in df.columns]
wind_target   = 'DE_wind_generation_actual'

X_w = df[wind_features].values
y_w = df[wind_target].values

wind_scaler = MinMaxScaler()
X_w_scaled  = wind_scaler.fit_transform(X_w)

X_w_train, X_w_test, y_w_train, y_w_test = train_test_split(X_w_scaled, y_w, test_size=0.2, random_state=42)
wind_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
wind_model.fit(X_w_train, y_w_train)
print(f"Wind SVM R² = {r2_score(y_w_test, wind_model.predict(X_w_test)):.4f}")

# ── 5. Compute dataset stats for the UI sliders ───────────────────────────────
stats = {
    "solar_profile":          {"min": float(df['DE_solar_profile'].min()),          "max": float(df['DE_solar_profile'].max()),          "mean": float(df['DE_solar_profile'].mean())},
    "wind_profile":           {"min": float(df['DE_wind_profile'].min()),            "max": float(df['DE_wind_profile'].max()),            "mean": float(df['DE_wind_profile'].mean())},
    "wind_onshore_profile":   {"min": float(df['DE_wind_onshore_profile'].min()),    "max": float(df['DE_wind_onshore_profile'].max()),    "mean": float(df['DE_wind_onshore_profile'].mean())} if 'DE_wind_onshore_profile' in df.columns else {"min":0,"max":1,"mean":0.5},
    "wind_offshore_profile":  {"min": float(df['DE_wind_offshore_profile'].min()),   "max": float(df['DE_wind_offshore_profile'].max()),   "mean": float(df['DE_wind_offshore_profile'].mean())} if 'DE_wind_offshore_profile' in df.columns else {"min":0,"max":1,"mean":0.5},
    "solar_generation_mean":  float(df['DE_solar_generation_actual'].mean()),
    "solar_generation_max":   float(df['DE_solar_generation_actual'].max()),
    "wind_generation_mean":   float(df['DE_wind_generation_actual'].mean()),
    "wind_generation_max":    float(df['DE_wind_generation_actual'].max()),
}

# ── 6. Save everything ────────────────────────────────────────────────────────
bundle = {
    "solar_model":   solar_model,
    "solar_scaler":  solar_scaler,
    "solar_features": solar_features,
    "wind_model":    wind_model,
    "wind_scaler":   wind_scaler,
    "wind_features": wind_features,
    "stats":         stats,
}

with open("models.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("\n✅ Saved models.pkl — download and put it in your app folder!")
print("   from google.colab import files; files.download('models.pkl')")
