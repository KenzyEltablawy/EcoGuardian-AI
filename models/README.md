# Models

This folder contains the trained machine learning model.

## Files

- `model.pkl` — Trained Random Forest classifier (300 trees)

## Model Details

- **Algorithm:** Random Forest
- **Number of Trees:** 300
- **Input Features:** 8
- **Training/Test Split:** 80/20
- **Test Accuracy:** ~98%

## How to Load

```python
import joblib
model = joblib.load('models/model.pkl')
