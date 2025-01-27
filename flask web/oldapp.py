import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify

# Step 1: Simulate Data Collection
# Generating synthetic data for demonstration purposes
data = {
    "Age": np.random.randint(20, 70, 1000),
    "BP": np.random.randint(90, 180, 1000),
    "Oxygen_Level": np.random.randint(85, 100, 1000),
    "Pulse": np.random.randint(60, 100, 1000),
    "Sugar_Status": np.random.choice([0, 1], 1000),
    "Weight": np.random.randint(50, 100, 1000),
    "Height": np.random.randint(150, 200, 1000),
    "Health_Risk": np.random.choice(["Low", "Medium", "High"], 1000)
}
df = pd.DataFrame(data)

# Step 2: Preprocessing
df["Health_Risk"] = df["Health_Risk"].map({"Low": 0, "Medium": 1, "High": 2})
X = df.drop("Health_Risk", axis=1)
y = df["Health_Risk"]

# Step 3: Train a Health Prediction Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Validate the Model
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")

# Step 4: Nutrition Recommendation System
def generate_nutrition_chart(health_risk, weight):
    recommendations = {
        0: "Your health risk is low. Maintain a balanced diet with fruits, vegetables, whole grains, and lean proteins.",
        1: "Your health risk is moderate. Focus on low-sodium and low-sugar meals, with moderate carbs and high protein.",
        2: "Your health risk is high. Follow a heart-friendly diet with minimal saturated fats and more fiber-rich foods."
    }

    indian_dishes = {
        0: ["Chapati with Dal", "Vegetable Curry", "Plain Yogurt", "Fruit Salad"],
        1: ["Brown Rice with Sambar", "Grilled Paneer", "Cucumber Salad", "Green Tea"],
        2: ["Millet Khichdi", "Steamed Vegetables", "Buttermilk", "Apple Slices"]
    }

    return {
        "Recommendation": recommendations[health_risk],
        "Sample Meals": indian_dishes[health_risk]
    }


# Step 5: API with Flask
app = Flask(__name__)

@app.route("/", methods=["POST"])
def predict():
    input_data = request.json
    try:
        input_features = np.array([
            input_data["Age"],
            input_data["BP"],
            input_data["Oxygen_Level"],
            input_data["Pulse"],
            input_data["Sugar_Status"],
            input_data["Weight"],
            input_data["Height"]
        ]).reshape(1, -1)

        health_risk = model.predict(input_features)[0]
        nutrition_chart = generate_nutrition_chart(health_risk, input_data["Weight"])

        return jsonify({
            "Health_Risk": ["Low", "Medium", "High"][health_risk],
            "Nutrition_Chart": nutrition_chart
        })
    except KeyError as e:
        return jsonify({"error": f"Missing key in input data: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "_main_":
    app.run(debug=False)


