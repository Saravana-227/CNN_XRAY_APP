from flask import Flask, render_template, request
from model import load_model, predict_image
import os

app = Flask(__name__)

# Load the pre-trained model when app starts
model = load_model()

# Step 1: Create a route to render the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Step 2: Handle file upload and prediction request
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:  # Check if the file is part of the request
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':  # If no file was selected
        return "No selected file"

    # Step 3: Save the uploaded image
    filename = os.path.join('uploads', file.filename)
    file.save(filename)
    
    # Step 4: Make prediction
    result = predict_image(filename, model)
    
    # Return the prediction result
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
