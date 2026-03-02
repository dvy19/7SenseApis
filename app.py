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
# Updated Core Prediction Logic
# -----------------------------------
def get_top_diseases(user_bmi):
    # 1. Try the specific range first (±1.5 for a slightly better reach)
    filtered = df[(df['BMI'] >= user_bmi - 1.5) &
                  (df['BMI'] <= user_bmi + 1.5)]

    if not filtered.empty:
        return filtered['Disease'].value_counts().head(5).index.tolist()

    # 2. Fallback: If range is empty, search by Category
    category = bmi_category(user_bmi)

    # We need to map BMI back to categories in the dataframe to filter
    # This assumes your dataset has enough variety to cover categories
    cat_filtered = df[df['BMI'].apply(bmi_category) == category]

    if not cat_filtered.empty:
        return cat_filtered['Disease'].value_counts().head(5).index.tolist()

    # 3. Final Fallback: Return top 5 most common diseases in the entire dataset
    return df['Disease'].value_counts().head(5).index.tolist()

# -----------------------------------
# API Endpoint (Same as before, logic is inside the function above)
# -----------------------------------
@app.route("/getDiseaseViaBmi", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "bmi" not in data:
        return jsonify({"error": "BMI value is required"}), 400

    try:
        user_bmi = float(data.get("bmi"))
    except ValueError:
        return jsonify({"error": "Invalid format"}), 400

    diseases = get_top_diseases(user_bmi)

    return jsonify({
        "input_bmi": user_bmi,
        "bmi_category": bmi_category(user_bmi),
        "top_diseases": diseases  # This will no longer be empty!
    })

# -----------------------------------
# Run App
# -----------------------------------
if __name__ == "__main__":
    app.run(debug=True)