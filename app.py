from flask import Flask, request, jsonify
import pandas as pd

# -----------------------------------
# Initialize Flask App
# -----------------------------------
app = Flask(__name__)

# -----------------------------------
# Load Dataset (Runs Once at Startup)
# -----------------------------------
try:
    df = pd.read_csv("indian_diseases_dataset.csv")

    # Keep only required columns
    df = df[['BMI', 'Disease']].dropna()

    # Ensure BMI is numeric
    df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')
    df = df.dropna()

except Exception as e:
    print("Error loading dataset:", e)
    df = pd.DataFrame(columns=["BMI", "Disease"])


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

    # Filter BMI range ±1
    filtered = df[(df['BMI'] >= user_bmi - 1) &
                  (df['BMI'] <= user_bmi + 1)]

    if filtered.empty:
        return []

    top5 = (
        filtered['Disease']
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


# -----------------------------------
# Health Check Route (Important for Render)
# -----------------------------------
@app.route("/")
def home():
    return jsonify({
        "status": "API is running"
    })


# -----------------------------------
# Run App
# -----------------------------------
if __name__ == "__main__":
    app.run(debug=True)