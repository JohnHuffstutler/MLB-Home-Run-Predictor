# ⚾ Home Run Probability Predictor

This project uses machine learning to predict the probability a batter hits a home run against a specific pitcher on a given day. The model is trained using MLB Statcast data and features a game-level Monte Carlo simulation with support for:

- Pitch-type feature engineering  
- Temperature scaling for better-calibrated probabilities  
- Swing-only risk modeling  
- Pitch-type breakdown of home run likelihood  

---

## 🚀 Features

- **LightGBM Classifier** trained on 3.8 million pitches (2019-2024)
- **Isotonic + Temperature Scaling** to calibrate probability output  
- **Monte Carlo Simulation** to sample multiple PAs per game  
- **Count-weighted pitch sampling** within PAs for realism  
- **Pitch-type breakdown** to identify which pitch is most likely to be hit for a HR  
- **Swing-only risk modeling** — no swing = no HR risk  

---

## 📁 Files

| File                          | Description |
|-------------------------------|-------------|
| `hr_pitch_model_calibrated.joblib` | Main LightGBM + isotonic calibrated model |
| `temp_scaler.joblib`         | Logistic regression for temperature scaling |
| `statcast_2019_2024_with_stats.parquet` | Feature-engineered Statcast data |
| `MLB Prediction Model.ipynb` | Training + evaluation notebook |
| `predict_game_hr_prob.py`    | Game-level Monte Carlo predictor script |

---

## 🧪 How It Works

1. **Train the Model**  
   - Filters unreliable data and clips extreme stats  
   - Trains a LightGBM model with class weighting to reduce false positives  
   - Uses 10% holdout for isotonic calibration  
   - Optionally applies temperature scaling for final calibration  

2. **Run Game-Level Prediction**  
   - Fetches recent pitch data for the pitcher vs batter hand  
   - Samples plate appearances and pitches using realistic pitch count weights  
   - Aggregates predicted HR risk only for swing-type pitches  
   - Reports mean HR probability + 80% confidence interval  
   - Outputs pitch type with highest HR share  

---

## 🧠 Model Goals

> Predict home runs **conservatively**:  
We prioritize **low false positive rate** (e.g., not overpredicting HRs on regular pitches) over high recall. The model aims for:
- True positive HR detection: ~60-70%  
- False positive HR detection: ≤ 5%  

---

## 📦 Dependencies

- Python 3.8+
- `pandas`, `numpy`
- `scikit-learn`
- `lightgbm`
- `pybaseball`
- `joblib`

Install with:

```bash
pip install pandas numpy scikit-learn lightgbm pybaseball joblib
