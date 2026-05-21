"""
House Price Prediction – Flask API
"""
import os
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "house_price_model.pkl")

# ── Load model artefacts ──────────────────────────────────────────────────────
with open(MODEL_PATH, "rb") as f:
    artefacts = pickle.load(f)

model     = artefacts["model"]
scaler    = artefacts["scaler"]
encoder   = artefacts["encoder"]
LOCATIONS = artefacts["locations"]

# ── CORS helper (no flask-cors needed) ───────────────────────────────────────
def _cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.after_request
def after_request(response):
    return _cors(response)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "House Price Prediction API"})


@app.route("/api/locations", methods=["GET"])
def get_locations():
    return jsonify({"locations": LOCATIONS})


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON payload received"}), 400

    # ── Validate inputs ───────────────────────────────────────────────────────
    required = ["location", "sqft", "bedrooms", "bathrooms", "bhk"]
    missing  = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    location  = str(data["location"]).strip()
    if location not in LOCATIONS:
        return jsonify({
            "error": f"Unknown location '{location}'. "
                     f"Choose from: {', '.join(LOCATIONS)}"
        }), 400

    try:
        sqft      = float(data["sqft"])
        bedrooms  = int(data["bedrooms"])
        bathrooms = int(data["bathrooms"])
        bhk       = int(data["bhk"])
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid numeric input: {exc}"}), 400

    # ── Range checks ──────────────────────────────────────────────────────────
    errors = []
    if sqft <= 0 or sqft > 20000:
        errors.append("sqft must be between 1 and 20,000")
    if not 1 <= bedrooms <= 10:
        errors.append("bedrooms must be between 1 and 10")
    if not 1 <= bathrooms <= 10:
        errors.append("bathrooms must be between 1 and 10")
    if not 1 <= bhk <= 10:
        errors.append("BHK must be between 1 and 10")
    if errors:
        return jsonify({"error": "; ".join(errors)}), 400

    # ── Predict ───────────────────────────────────────────────────────────────
    loc_encoded = encoder.transform([location])[0]
    features    = np.array([[loc_encoded, sqft, bedrooms, bathrooms, bhk]])
    features_sc = scaler.transform(features)
    price       = float(model.predict(features_sc)[0])

    # Format price
    def fmt_inr(amount):
        if amount >= 1e7:
            return f"₹{amount/1e7:.2f} Crore"
        if amount >= 1e5:
            return f"₹{amount/1e5:.2f} Lakh"
        return f"₹{amount:,.0f}"

    return jsonify({
        "predicted_price":       round(price, 2),
        "formatted_price":       fmt_inr(price),
        "input": {
            "location":  location,
            "sqft":      sqft,
            "bedrooms":  bedrooms,
            "bathrooms": bathrooms,
            "bhk":       bhk,
        },
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
