#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import json
import zipfile
from pathlib import Path
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import (
    print_colored, GREEN, BLUE, RED, YELLOW, CYAN, NC
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict leaf disease from an image')
    parser.add_argument('image_path', type=str, help='Path to the leaf image')
    parser.add_argument('--model_zip', type=str, default='leaffliction_model.zip',
                       help='Path to the model zip file')
    parser.add_argument('--output_html', action='store_true',
                       help='Create an HTML output with prediction results')
    
    return parser.parse_args()

def extract_model_files(zip_path):
    """Extract model files from the zip archive."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Paths to model files
    model_path = os.path.join(temp_dir, 'leaf_disease_model.h5')
    class_mapping_path = os.path.join(temp_dir, 'class_mapping.json')
    
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found in zip archive: {zip_path}")
    
    if not os.path.exists(class_mapping_path):
        raise FileNotFoundError(f"Class mapping file not found in zip archive: {zip_path}")
    
    # Load the model
    model = load_model(model_path)
    
    # Load class mapping
    with open(class_mapping_path, 'r') as f:
        class_mapping = json.load(f)
    
    return model, class_mapping, temp_dir

def preprocess_image(image_path):
    """Preprocess an image for prediction."""
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Store the original image
    original_image = image.copy()
    
    # Preprocess the image
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    return original_image, image

def predict_disease(model, image, class_mapping):
    """Predict the disease type from an image."""
    # Make a prediction
    prediction = model.predict(image)
    
    # Get the predicted class index
    predicted_class_idx = np.argmax(prediction[0])
    
    # Get the class name
    predicted_class_name = class_mapping.get(str(predicted_class_idx), "Unknown")
    
    # Get the confidence score
    confidence = prediction[0][predicted_class_idx]
    
    return {
        'class_name': predicted_class_name,
        'confidence': confidence,
        'all_predictions': {class_mapping.get(str(i)): float(score) for i, score in enumerate(prediction[0])}
    }

def display_results(original_image, prediction_results):
    """Display the original image and prediction results."""
    plt.figure(figsize=(10, 6))
    
    # Display the original image
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted: {prediction_results['class_name']}\nConfidence: {prediction_results['confidence']*100:.2f}%")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def create_html_output(original_image, prediction_results, image_path):
    """Create an HTML output with the original image and prediction results."""
    # Create an output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(image_path)), 'prediction_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the original image
    output_image_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_image_path, original_image)
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Leaf Disease Prediction</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                padding: 20px;
                margin-bottom: 20px;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            .prediction-results {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
            }}
            .image-container {{
                flex: 1;
                min-width: 300px;
                text-align: center;
            }}
            .image-container img {{
                max-width: 100%;
                max-height: 400px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .results-container {{
                flex: 1;
                min-width: 300px;
            }}
            .prediction {{
                margin-top: 20px;
                padding: 15px;
                background-color: #e8f4f8;
                border-left: 4px solid #3498db;
                border-radius: 4px;
            }}
            .prediction h2 {{
                margin-top: 0;
                color: #2980b9;
            }}
            .confidence-bar {{
                height: 20px;
                background-color: #ecf0f1;
                border-radius: 10px;
                margin-top: 10px;
                overflow: hidden;
            }}
            .confidence-level {{
                height: 100%;
                background-color: #2ecc71;
                width: {prediction_results['confidence']*100}%;
                border-radius: 10px;
            }}
            .all-predictions {{
                margin-top: 20px;
            }}
            .all-predictions h3 {{
                margin-bottom: 10px;
                color: #34495e;
            }}
            .prediction-item {{
                display: flex;
                justify-content: space-between;
                margin-bottom: 5px;
                padding: 5px 0;
                border-bottom: 1px solid #eee;
            }}
            .footer {{
                margin-top: 30px;
                text-align: center;
                color: #7f8c8d;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Leaf Disease Prediction Results</h1>
            
            <div class="prediction-results">
                <div class="image-container">
                    <h2>Original Image</h2>
                    <img src="{os.path.basename(image_path)}" alt="Leaf Image">
                </div>
                
                <div class="results-container">
                    <div class="prediction">
                        <h2>Prediction: {prediction_results['class_name']}</h2>
                        <p>Confidence: {prediction_results['confidence']*100:.2f}%</p>
                        <div class="confidence-bar">
                            <div class="confidence-level"></div>
                        </div>
                    </div>
                    
                    <div class="all-predictions">
                        <h3>All Predictions</h3>
    """
    
    # Add all predictions, sorted by confidence
    sorted_predictions = sorted(
        prediction_results['all_predictions'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for class_name, confidence in sorted_predictions:
        html_content += f"""
                        <div class="prediction-item">
                            <span>{class_name}</span>
                            <span>{confidence*100:.2f}%</span>
                        </div>
        """
    
    html_content += """
                    </div>
                                    
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated with Leaffliction - Leaf Disease Classification</p>
        </div>
        
        <script>
            // Auto-open the HTML file when created
            window.onload = function() {
                // The page is already open, nothing more needed
                console.log("Prediction results page loaded successfully");
            };
        </script>
    </body>
    </html>
    """
    
    # Save the HTML file
    # Save the HTML file
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    html_path = os.path.join(output_dir, f"prediction_{base_name}.html")
    
    with open(html_path, 'w') as html_file:
        html_file.write(html_content)
    
    return html_path