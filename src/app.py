from utils import db_connect
engine = db_connect()

# your code here

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the Random Forest model that was saved earlier
model = joblib.load('random_forest_diabetes_model.pkl')

@app.route('/')
def index():
    return "Welcome to the Diabetes Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        
        # Extract the features required for prediction
        features = [data['Pregnancies'], data['Glucose'], data['BloodPressure'],
                    data['SkinThickness'], data['Insulin'], data['BMI'], 
                    data['DiabetesPedigreeFunction'], data['Age']]
        
        # Convert the list to a numpy array for model input
        features_array = np.array([features])
        
        # Make the prediction using the preloaded model
        prediction = model.predict(features_array)
        
        # Prepare the response as JSON
        result = {'diabetes_prediction': int(prediction[0])}
        return jsonify(result)
    
    except KeyError as e:
        return jsonify({"error": f"Missing key in the request: {str(e)}"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

