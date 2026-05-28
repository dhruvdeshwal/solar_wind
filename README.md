# ⚡ Solar & Wind Energy Predictor

Zero-CSV production app. Users fill sliders → get MW prediction instantly.

## Files
```
solar_wind_app/
├── app.py               ← Streamlit app (user-facing)
├── train_and_save.py    ← Run ONCE in Colab to generate models.pkl
├── models.pkl           ← YOU generate this (not in repo — too large)
├── requirements.txt
└── README.md
```

## Step 1 — Generate models.pkl (run in Colab)

At the end of your notebook, paste and run `train_and_save.py`.
It will save `models.pkl` and prompt you to download it.

```python
# Quick version — paste at end of your Colab notebook:
exec(open('train_and_save.py').read())
from google.colab import files
files.download('models.pkl')
```

## Step 2 — Deploy to Streamlit Cloud

1. Put `models.pkl` in the same folder as `app.py`
2. Push everything to GitHub
3. Go to share.streamlit.io → New app → point to app.py
4. Done — live public URL, no CSV needed ever again

## What users see

- ☀️ Solar tab: one slider (solar profile) → predicted MW + condition label
- 💨 Wind tab: three sliders (wind/onshore/offshore profiles) → predicted MW
- No login, no CSV, no technical knowledge needed
