from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Directory to store uploaded images
UPLOAD_FOLDER = 'static/uploads/'  # Update to static/uploads
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = load_model('model.h5')

# Define the image size (should match your model's input size, e.g., 224x224)
IMG_SIZE = (224, 224)

# Function to preprocess the image
def preprocess_image(image_path):
    # Load the image file
    img = load_img(image_path, target_size=IMG_SIZE)
    # Convert the image to an array
    img_array = img_to_array(img)
    # Expand dimensions to match the model input shape
    img_array = np.expand_dims(img_array, axis=0)
    # Rescale pixel values (same as during training)
    img_array = img_array / 255.0
    return img_array

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the upload and prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    # If no file is selected
    if file.filename == '':
        return redirect(request.url)
    
    # Save the uploaded file in the static/uploads/ directory
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Preprocess the image
        img_array = preprocess_image(file_path)
        
        # Predict using the loaded model
        prediction = model.predict(img_array)
        
        # Interpret the result (assuming output is between 0 and 1)
        if prediction[0] > 0.5:
            result = "Coccidiosis Detected"
        else:
            result = "Healthy"
        
        # Render the result page and pass the image URL and result
        return render_template('result.html', result=result, image_url=url_for('static', filename=f'uploads/{filename}'))

if __name__ == '__main__':
    # Ensure the uploads directory exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Run the app
    app.run(debug=True)
