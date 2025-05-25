from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from utils.preprocess import preprocess_image

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load the model from the root directory
MODEL_PATH = 'resnet50_biopsy_final_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels (update these to match your training labels)
class_names = ['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc']

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    try:
        image = Image.open(file.stream).convert("RGB")
        processed_image = preprocess_image(image)  # Preprocess to (1, 224, 224, 3)
        prediction = model.predict(processed_image)[0]
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        return jsonify({
            'class': predicted_class,
            'confidence': round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
