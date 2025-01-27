from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)



# Initialize model and scaler
model = RandomForestClassifier(n_estimators=100, random_state=42)
scaler = StandardScaler()

# Generate synthetic data and train model
def init_model():
    # Generate synthetic training data
    n_samples = 1000
    data = {
        "Age": np.random.randint(20, 70, n_samples),
        "BP": np.random.randint(90, 180, n_samples),
        "Oxygen_Level": np.random.normal(97, 2, n_samples).clip(85, 100),
        "Pulse": np.random.normal(75, 10, n_samples).clip(60, 100),
        "Sugar_Status": np.random.choice([0, 1], n_samples),
        "Weight": np.random.normal(70, 15, n_samples).clip(45, 120),
        "Height": np.random.normal(170, 10, n_samples).clip(150, 200)
    }
    
    # Calculate BMI
    data['BMI'] = data['Weight'] / ((data['Height'] / 100) ** 2)
    
    # Generate health risk based on multiple factors
    def calculate_risk(features):
        risk_score = 0
        if features['BMI'] > 30: risk_score += 2
        elif features['BMI'] > 25: risk_score += 1
        if features['BP'] > 140: risk_score += 2
        if features['Oxygen_Level'] < 95: risk_score += 1
        if features['Sugar_Status'] != 0: risk_score += 1
        return 2 if risk_score >= 3 else 1 if risk_score >= 1 else 0

    # Create features and target
    X = np.column_stack([data[col] for col in ['Age', 'BP', 'Oxygen_Level', 'Pulse', 'Sugar_Status', 'Weight', 'Height', 'BMI']])
    y = np.array([calculate_risk(dict(zip(data.keys(), row))) for row in zip(*data.values())])
    
    # Scale features and train model
    global scaler, model
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    model.fit(X_scaled, y)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Calculate BMI
        height_m = data['Height'] / 100
        weight_kg = data['Weight']
        bmi = weight_kg / (height_m ** 2)
        
        # Prepare input features
        features = np.array([[
            data['Age'],
            data['BP'],
            data['Oxygen_Level'],
            data['Pulse'],
            data['Sugar_Status'],
            data['Weight'],
            data['Height'],
            bmi
        ]])
        
        # Scale features and predict
        features_scaled = scaler.transform(features)
        risk_level = model.predict(features_scaled)[0]
        
        # Generate nutrition plan
        nutrition_system = {
            "risk_levels": ["Low", "Medium", "High"],
            "meal_plans": {
                "vegetarian": {
                    0: {
                        "breakfast": ["Oatmeal with fruits", "Whole grain toast with avocado"],
                        "lunch": ["Quinoa bowl with roasted vegetables", "Lentil soup"],
                        "dinner": ["Chickpea curry with brown rice", "Steamed vegetables"],
                        "snacks": ["Mixed nuts", "Greek yogurt with honey"]
                    },
                    # ... Add more meal plans for other risk levels ...
                }
            }
        }
        
        response = {
            "health_risk": nutrition_system["risk_levels"][risk_level],
            "nutrition_plan": {
                "general_advice": "Maintain a balanced diet with regular exercise.",
                "calories_recommendation": f"{int(weight_kg * 24)} - {int(weight_kg * 26)} calories/day",
                "hydration": f"Drink {int(weight_kg * 0.033)} liters of water daily",
                "exercise_recommendation": "30 minutes of moderate exercise 5 times per week",
                "meal_plan": nutrition_system["meal_plans"]["vegetarian"][0]  # Simplified for example
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    init_model()
    app.run(debug=True)

