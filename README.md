<div align="center">
# ⚡ Solar & Wind Energy Predictor
 
### Real-time Renewable Energy Forecasting using Machine Learning
 
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://solarwind-renewable-energy.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
 
<img src="https://img.shields.io/badge/Solar%20R²-0.97-yellow" /> <img src="https://img.shields.io/badge/Wind%20R²-0.96-blue" /> <img src="https://img.shields.io/badge/LSTM%20Accuracy-98.2%25-brightgreen" />
 
</div>
---
 
## 🌍 Problem Statement
 
Renewable energy generation is **unpredictable** — it varies with weather, seasons, cloud cover, and geography. Grid operators and energy planners need accurate forecasts to balance **supply and demand** without over-relying on fossil fuel backups.
 
> This project builds a production-ready ML system that predicts **Solar and Wind energy output in Megawatts (MW)** — trained on 6 years of real German power grid data.
 
---
 
## 🎥 Demo
 
> 🔗 **[Live App →](https://solarwind-renewable-energy.streamlit.app/)**
 
| ☀️ Solar Prediction | 💨 Wind Prediction |
|---|---|
| Move 1 slider → get MW output | Move 3 sliders → get MW output |
| Condition label (Cloudy / Sunny) | Wind strength indicator |
| % of historical peak shown | Onshore + Offshore profiles |
 
---
 
## 📊 Results
 
| Model | Target | R² Score | RMSE | Notes |
|-------|--------|----------|------|-------|
| **Random Forest** | Solar | **0.97** | — | ✅ Best ML model |
| **Random Forest** | Wind | **0.96** | — | ✅ Best ML model |
| LSTM (Deep Learning) | Solar | — | — | **98.2% accuracy** |
| SVM | Wind | 0.45 | — | Underperformed on full data |
| Decision Tree | Both | ~0.94 | — | Good but overfits |
| Lasso / Ridge | Both | ~0.82 | — | Linear baseline |
 
---
 
## 🏗️ Architecture
 
```
Raw Data (52,000+ hourly records)
         │
         ▼
┌─────────────────────┐
│   Data Preprocessing │  → Null imputation, outlier removal
│   Feature Engineering│  → Correlation analysis, feature importance
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Model Training     │  → Random Forest, SVM, LSTM, Decision Tree
│   Benchmarking       │  → MSE, RMSE, MAE, R² across all models
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Best Model Saved   │  → Serialized as models.pkl (pickle)
│   Hosted on Drive    │  → Auto-downloaded on app start
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Streamlit App      │  → User fills sliders → Real-time MW prediction
│   Deployed on Cloud  │  → No CSV, no login, no setup
└─────────────────────┘
```
 
---
 
## 🤖 Tech Stack
 
| Category | Technology |
|----------|-----------|
| Language | Python 3.10 |
| ML Models | Scikit-learn (Random Forest, SVM, Decision Tree, Lasso, Ridge) |
| Deep Learning | LSTM (Keras / TensorFlow) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Model Storage | Pickle + Google Drive (gdown) |
| Deployment | Streamlit Cloud |
| Version Control | Git + GitHub (Git LFS for large files) |
 
---
 
## 📁 Dataset
 
- **Source:** [Open Power System Data](https://open-power-system-data.org/) — free, open-source
- **Country:** 🇩🇪 Germany — chosen for having **~46% renewable energy share**, highest in Europe
- **Size:** 52,000+ hourly records across **2015–2021**
- **Key Features:** `solar_profile`, `wind_profile`, `wind_onshore_profile`, `wind_offshore_profile`, `solar_generation_actual`, `wind_generation_actual`
---
 
## 🔬 Methodology
 
**1. Data Preprocessing**
- Solar nulls → filled with previous day's value (shift 24h); remaining → filled with 0 (night assumption)
- Wind nulls → filled with column mean
- Removed infinite values and standardized all features
**2. Feature Selection**
- Correlation heatmap analysis
- Feature importance scoring via Random Forest
- Selected top correlated features per energy type
**3. Model Training & Selection**
- Train/test split: 80/20
- Evaluated 5 ML + 1 DL model per energy type
- Selected best model by R² score on test set
**4. Deployment**
- Best models serialized with pickle
- Hosted on Google Drive, auto-downloaded via `gdown`
- Streamlit app loads model → user inputs → real-time prediction
---
 
## 🚀 Run Locally
 
```bash
# Clone the repo
git clone https://github.com/dhruvdeshwal/solar_wind.git
cd solar_wind
 
# Install dependencies
pip install -r requirements.txt
 
# Run the app
streamlit run app.py
```
 
---
 
## 📂 Project Structure
 
```
solar_wind/
├── app.py                  # Streamlit web application
├── train_and_save.py       # Model training + pickle export script
├── requirements.txt        # Python dependencies
└── README.md
```
 
> ⚠️ `models.pkl` (374 MB) is not in the repo. It auto-downloads from Google Drive on first app launch.
 
---
 
## 💡 Key Learnings
 
- Real-world energy data is **noisy and inconsistent** — robust preprocessing is critical
- **LSTM outperforms** traditional ML on sequential energy data but requires historical sequences as input
- **Random Forest** is the best balance of accuracy and simplicity for production deployment
- SVM struggles on **large datasets** (50k+ rows) due to training time and scaling sensitivity
---
 
<div align="center">
Made with ❤️ by **Dhruv Deshwal**
 
[GitHub](https://github.com/dhruvdeshwal) · [Live App](https://your-app-link.streamlit.app)
 
</div>
