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
    df = df[['bmi', 'disease_name']].dropna()

    # Ensure BMI is numeric
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    df = df.dropna()

except Exception as e:
    print("Error loading dataset:", e)
    df = pd.DataFrame(columns=["bmi", "bmi"])


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


#----------------------
#GET TOP DISEASE BY AGE GROUP ONLY
#----------------------

def get_top_diseases_by_age_group(df, age_group):
    """
    Returns top 5 diseases for given age group in JSON serializable format.
    """

    age_group_df = df[df['age_group'] == age_group]

    if age_group_df.empty:
        return None

    top_diseases = (
        age_group_df['disease_category']
        .value_counts()
        .head(5)
    )

    # Convert to list of dictionaries
    result = [
        {"disease": disease, "count": int(count)}
        for disease, count in top_diseases.items()
    ]

    return result

@app.route("/top-diseases-by-age-group", methods=["POST"])
def top_diseases():

    data = request.get_json()

    if not data or "age_group" not in data:
        return jsonify({"error": "age_group is required"}), 400

    age_group = data["age_group"]

    result = get_top_diseases_by_age_group(df, age_group)

    if result is None:
        return jsonify({"message": "No data found"}), 404

    return jsonify({
        "age_group": age_group,
        "top_diseases": result
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