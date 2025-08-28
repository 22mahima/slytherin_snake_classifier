import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import pickle
import io

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'snake_classifier_model.pkl'
model = None
model_accuracy = 0.0

def load_model():
    global model, model_accuracy
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            model_accuracy = model_data['accuracy']
        print(f"Model loaded successfully. Accuracy: {model_accuracy:.2%}")
    else:
        print("No trained model found. Please run train_model.py first.")

def preprocess_image(image):
    """Preprocess image exactly as in training"""
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to 128x128
    image = image.resize((128, 128))
    
    # Convert to numpy array and flatten
    image_array = np.array(image)
    flattened = image_array.flatten()
    
    # Normalize pixel values to 0-1 range
    normalized = flattened / 255.0
    
    return normalized.reshape(1, -1)

@app.route('/')
def index():
    return render_template('index.html', accuracy=model_accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
        
        if model is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
        # Read and preprocess the image
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image)[0]
        confidence = model.predict_proba(processed_image)[0]
        
        # Get confidence score for the predicted class
        class_confidence = max(confidence)
        
        result = {
            'class': prediction,
            'confidence': float(class_confidence),
            'message': f"This snake is classified as {prediction} with {class_confidence:.1%} confidence."
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

if __name__ == '__main__':
    load_model()
    # Changed port from 5000 to 5001
    app.run(host='0.0.0.0', port=5001, debug=True)
