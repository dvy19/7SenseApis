from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

df = pd.read_csv("indian_diseases_dataset.csv")

df = df[['bmi', 'disease_name']].dropna()
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
df = df.dropna()

def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"


disease_counts = (
    df.groupby(['bmi', 'disease_name'])
      .size()
      .reset_index(name='Count')
)

#print(disease_counts)

@app.route("/bmi&disease", methods=["POST"])
def predict():

    data = request.get_json()
    user_bmi = float(data.get("bmi"))

    # Filter near BMI ±1
    filtered = df[(df['BMI'] >= user_bmi - 1) &
                  (df['BMI'] <= user_bmi + 1)]

    top5 = (
        filtered['Disease']
        .value_counts()
        .head(5)
        .index
        .tolist()
    )

    return jsonify({
        "bmi": user_bmi,
        "top_diseases": top5
    })

if __name__ == "__main__":
    app.run(debug=True)