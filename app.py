import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load pre-trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/", methods=['POST'])
def predict():
    try:
        # Get data from JSON request
        data = request.get_json()

        # Extract numerical values
        data_values = list(data.values())

        # Reshape data into 2D array
        data_array = np.array(data_values).reshape(1, -1)

        # Make prediction
        prediction = model.predict(data_array)

        # Return response
        return jsonify({'prediction': str(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400
