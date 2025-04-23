from flask import Flask, render_template, request
from model import load_model, predict_image
import os

app = Flask(__name__)

# Load the pre-trained model once when the app starts
model = load_model()

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

    # Save the uploaded file
    filename = os.path.join('uploads', file.filename)
    file.save(filename)

    # Run the prediction
    result = predict_image(filename, model)

    return render_template('result.html', result=result)

# 🚫 No app.run() here!
