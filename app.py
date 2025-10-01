from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import pickle
from PIL import Image
import io

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)
svm = model_data['svm']
scaler = model_data['scaler']
use_hog = model_data['use_hog']

def preprocess_image(image_bytes, use_hog):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((64, 64))
    img_array = np.array(img.convert('L'))
    
    if use_hog:
        from skimage.feature import hog
        feat = hog(img_array, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    else:
        feat = img_array.flatten()
    
    feat = scaler.transform([feat])
    return feat

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    img_bytes = file.read()
    
    feat = preprocess_image(img_bytes, use_hog)
    prediction = svm.predict(feat)[0]
    confidence = svm.decision_function(feat)[0]
    
    label = 'Dog' if prediction == 1 else 'Cat'
    conf_pct = abs(confidence) * 100
    
    return jsonify({'label': label, 'confidence': f"{conf_pct:.1f}%"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)