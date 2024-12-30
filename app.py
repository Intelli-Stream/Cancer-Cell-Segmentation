from flask import Flask, request, render_template
import os
import cv2
import numpy as np
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications import VGG16

app = Flask(__name__)

# Load the VGG16 model for general predictions
vgg_model = VGG16(weights='imagenet', include_top=True)

# Ensure uploads directory exists
os.makedirs('uploads', exist_ok=True)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
        
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_input(np.expand_dims(image, axis=0))
    return image

def simulate_cancer_detection(image_path):
    # This function simulates cancer detection and size estimation
    # Replace this logic with your trained cancer detection model's prediction logic
    
    # Simulated logic (for demo purposes)
    cancer_present = True  # Simulate that cancer is detected
    cancer_size = np.random.uniform(1.0, 5.0)  # Simulate size in cm
    return cancer_present, cancer_size

def predict_image_class(image_path):
    image = preprocess_image(image_path)
    predictions = vgg_model.predict(image)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    
    result = {}
    for prediction in decoded_predictions:
        result[prediction[1]] = prediction[2] * 100
    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    image_path = os.path.join('uploads', file.filename)
    file.save(image_path)
    
    # Simulate cancer detection
    cancer_present, cancer_size = simulate_cancer_detection(image_path)
    
    # Get predictions from VGG model (not directly related to cancer)
    predictions = predict_image_class(image_path)
    
    return render_template('result.html', predictions=predictions, cancer_present=cancer_present, cancer_size=cancer_size)

if __name__ == "__main__":
    app.run(debug=True)
