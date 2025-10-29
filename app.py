from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import traceback

# Initialize Flask app
app = Flask(__name__)

# Load model and scaler
try:
    model = pickle.load(open('model/model.pkl', 'rb'))
    scaler = pickle.load(open('model/scaler.pkl', 'rb'))
    print("✅ Model and Scaler loaded successfully.")
except Exception as e:
    print("❌ Error loading model or scaler:", e)


@app.route('/')
def home():
    """Render homepage with prediction form."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle form input and return sales prediction."""
    try:
        # Extract input values from form
        data = [float(x) for x in request.form.values()]
        features = np.array([data])

        # Scale input features
        scaled_features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(scaled_features)
        output = round(prediction[0], 2)

        # Render result page
        return render_template('result.html', prediction_text=f"Predicted Sales: {output}")

    except Exception as e:
        print("❌ Prediction error:", e)
        traceback.print_exc()
        return render_template('error.html', error=str(e))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Optional: API endpoint for programmatic access
    Example JSON input: {"tv": 230.1, "radio": 37.8, "newspaper": 69.2}
    """
    try:
        data = request.get_json(force=True)
        features = np.array([[float(value) for value in data.values()]])
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)
        output = round(prediction[0], 2)
        return jsonify({'predicted_sales': output})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
