import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

# ── Load dataset ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "housing_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "house_price_model.pkl")

df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())

# ── Encode location ───────────────────────────────────────────────────────────
le = LabelEncoder()
df["location_encoded"] = le.fit_transform(df["location"])

# ── Features & target ─────────────────────────────────────────────────────────
FEATURES = ["location_encoded", "sqft", "bedrooms", "bathrooms", "bhk"]
X = df[FEATURES]
y = df["price"]

# ── Train/test split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Scale features ────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Train model ───────────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)
print(f"\nModel Performance:")
print(f"  MAE : ₹{mae:,.0f}")
print(f"  R²  : {r2:.4f}")

# ── Save artefacts ────────────────────────────────────────────────────────────
payload = {
    "model":     model,
    "scaler":    scaler,
    "encoder":   le,
    "locations": list(le.classes_),
    "features":  FEATURES,
}
with open(MODEL_PATH, "wb") as f:
    pickle.dump(payload, f)

print(f"\nModel saved → {MODEL_PATH}")
print(f"Supported locations: {list(le.classes_)}")
