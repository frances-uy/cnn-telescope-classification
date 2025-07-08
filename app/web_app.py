"""
File: web_app.py
Author: Frances Uy
Date: July 2024
Purpose: Flask application for displaying real-time cloud classification images and results.
         This application reads classification results from a file and displays the latest image
         along with the classification results on a web page. The image is resized to fit within
         specified dimensions, and the results section is centered on the page.
"""

from flask import Flask, render_template_string, send_from_directory
import os
import glob
from PIL import Image
import io

# Flask application instance
app = Flask(__name__)

# Global variables
RESULTS_FILE = "/app/cloud_classification_data/realtime_result.txt"
IMAGE_SAVE_DIR = "/app/static"
MAX_IMAGE_SIZE = (800, 600)  # Maximum width and height for displayed image

# HTML template for rendering the web page
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hilo Cam Classification</title>
    <style>
        body { 
            font-family: Helvetica, sans-serif; 
            line-height: 1.6; 
            padding: 20px; 
            background-color: #2b2b2b; 
            color: white; 
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        h1 {
            text-align: center;
        }
        img { 
            max-width: 100%; 
            height: auto; 
            max-height: 600px;
            display: block;
            margin: 20px auto;
        }
        .results { 
            background-color: #3a3a3a; 
            padding: 15px; 
            border-radius: 5px; 
            font-size: 1.2em;
            text-align: center;
            margin-bottom: 20px;
            width: 100%;
            max-width: 800px;
        }
    </style>
</head>
<body>
    <h1>Hilo Cam Classification</h1>
    <div class="results">
        <h3>Results as of {{ results.date_time_of_retrieval }}:</h2>
        {% if results %}
            <p>File Name: {{ results.file_name }}</p>
            <p>Predicted class: {{ results.predicted_class }}</p>
        {% else %}
            <p>No results available</p>
        {% endif %}
    </div>
    {% if image %}
        <img src="{{ url_for('serve_image', filename=image) }}" alt="Current cloud image">
    {% else %}
        <p>No image available</p>
    {% endif %}
    <script>
        setTimeout(function(){ location.reload(); }, 10000);
    </script>
</body>
</html>
"""

def parse_results_file():
    """
    Parses the results file to extract classification data.

    Returns:
        dict: A dictionary containing the parsed results, or None if the file doesn't exist.
    """
    if not os.path.exists(RESULTS_FILE):
        return None
    results = {}
    with open(RESULTS_FILE, 'r') as file:
        for line in file:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                key = key.strip().lower().replace(" ", "_").replace("/", "_")
                results[key] = value.strip()
    return results

def resize_image(image_path, max_size):
    """
    Resizes the image to fit within the specified maximum size.

    Args:
        image_path (str): The path to the image file.
        max_size (tuple): The maximum width and height for the image.

    Returns:
        bytes: The resized image in byte format.
    """
    with Image.open(image_path) as img:
        img.thumbnail(max_size)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

@app.route('/')
def index():
    """
    The main route that renders the index page with the latest image and results.

    Returns:
        str: The rendered HTML template.
    """
    images = glob.glob(os.path.join(IMAGE_SAVE_DIR, '*.jpg'))
    most_recent_image = max(images, key=os.path.getctime) if images else None
    image_filename = os.path.basename(most_recent_image) if most_recent_image else None
    results = parse_results_file()
    return render_template_string(HTML_TEMPLATE, results=results, image=image_filename)

@app.route('/images/<filename>')
def serve_image(filename):
    """
    Serves the resized image.

    Args:
        filename (str): The name of the image file.

    Returns:
        Response: The image file response with the appropriate MIME type.
    """
    full_path = os.path.join(IMAGE_SAVE_DIR, filename)
    resized_image = resize_image(full_path, MAX_IMAGE_SIZE)
    return app.response_class(resized_image, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

