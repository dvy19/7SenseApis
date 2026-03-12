import joblib
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

# -----------------------------------
# Initialize Flask App
# -----------------------------------
app = Flask(__name__)
# -----------------------------------
# Load Dataset (Runs Once at Startup)
# -----------------------------------

try:
    df = pd.read_csv("indian_diseases_dataset.csv")
    df.columns = df.columns.str.strip()

    # Keep only required columns
    df=df.dropna()

    # Ensure BMI is numeric
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')



except Exception as e:
    print("Error loading dataset:", e)
    df = pd.DataFrame()


# -----------------------------------
# BMI Category Function
# -----------------------------------
def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


# -----------------------------------
# Core Prediction Logic
# -----------------------------------
def get_top_diseases(user_bmi):
    bmi_df = df[['bmi', 'disease_name']].dropna()

    # Filter BMI range ±1
    filtered = df[(df['bmi'] >= user_bmi - 1.5) &
                  (df['bmi'] <= user_bmi + 1.5)]

    if not filtered.empty:
        return filtered['disease_name'].value_counts().head(5).index.tolist()

    top5 = (
        filtered['disease_name']
        .value_counts()
        .head(5)
        .index
        .tolist()
    )

    return top5


# -----------------------------------
# API Endpoint
# -----------------------------------
@app.route("/getDiseaseViaBmi", methods=["POST"])
def predict():

    data = request.get_json()

    # Input validation
    if not data or "bmi" not in data:
        return jsonify({
            "error": "BMI value is required"
        }), 400

    try:
        user_bmi = float(data.get("bmi"))
    except ValueError:
        return jsonify({
            "error": "Invalid BMI format"
        }), 400

    if user_bmi <= 0:
        return jsonify({
            "error": "BMI must be greater than 0"
        }), 400

    # Get prediction
    diseases = get_top_diseases(user_bmi)

    return jsonify({
        "input_bmi": user_bmi,
        "bmi_category": bmi_category(user_bmi),
        "top_diseases": diseases
    })


# -----------------------------
# Load Saved Models
# -----------------------------
model_category = joblib.load("model_category.pkl")
category_models = joblib.load("category_models.pkl")
mlb = joblib.load("mlb.pkl")


# -----------------------------
# Home Route
# -----------------------------
@app.route("/")
def home():
    return "Two-Stage Disease Prediction API Running"


# -----------------------------
# Prediction Route
# -----------------------------
@app.route("/symptomsPrediction", methods=["POST"])
def predict_disease_via_symptoms():

    try:
        data = request.json

        user_symptoms = data.get("symptoms", [])
        age = data.get("age")
        gender = data.get("gender")
        smoking = data.get("smoking")
        alcohol = data.get("alcohol")
        bmi = data.get("bmi")

        # -------------------------
        # Prepare Symptom Vector
        # -------------------------
        input_symptoms = np.zeros(len(mlb.classes_))
        user_symptoms = [s.lower().strip() for s in user_symptoms]

        for symptom in user_symptoms:
            if symptom in mlb.classes_:
                idx = list(mlb.classes_).index(symptom)
                input_symptoms[idx] = 1

        # -------------------------
        # Encode Lifestyle
        # -------------------------
        gender_val = 1 if gender.lower() == "male" else 0

        smoking_map = {"never": 0, "former": 1, "current": 2}
        alcohol_map = {"none": 0, "occasional": 1, "regular": 2, "heavy": 3}

        smoking_val = smoking_map.get(smoking.lower(), 0)
        alcohol_val = alcohol_map.get(alcohol.lower(), 0)

        # -------------------------
        # Combine Features
        # -------------------------
        input_vector = np.concatenate([
            input_symptoms,
            [age],
            [gender_val],
            [smoking_val],
            [alcohol_val],
            [bmi]
        ])

        input_vector = input_vector.reshape(1, -1)

        # -------------------------
        # Stage 1: Category
        # -------------------------
        predicted_category = model_category.predict(input_vector)[0]

        # -------------------------
        # Stage 2: Disease
        # -------------------------
        disease_model = category_models[predicted_category]
        predicted_disease = disease_model.predict(input_vector)[0]

        return jsonify({
            "predicted_category": predicted_category,
            "predicted_disease": predicted_disease
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# -----------------------------
# Get Medicine Details
# -----------------------------

medicine_df=pd.read_csv("Medicine_Details.csv")

@app.route('/get_medicine', methods=['GET'])
def get_medicine():

    medicine_name = request.args.get("name")

    if not medicine_name:
        return jsonify({"error": "Please provide medicine name"}), 400

    result = medicine_df[medicine_df["medicine_name"].str.contains(medicine_name, case=False, na=False)]

    if result.empty:
        return jsonify({"message": "Medicine not found"})

    medicine_data = result.iloc[0].to_dict()

    return jsonify(medicine_data)

# -----------------------------------
# Run App
# -----------------------------------
if __name__ == "__main__":
    app.run(debug=True)