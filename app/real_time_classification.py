"""
File name: real_time_classification.py

Real Time Classification Script

This script classifies the updating image from http://mkoccamdev-lv1:8080/camera/100000006aa8722a
and outputs Date/Time of Retrieval, File Name, Predicted class,
and Confidence in Classification to file called real_time_classification.txt

Author: Frances Uy
Mentors: Gemini Observatory: Hawi Stecher, Patrick Parks
Date: June-July 2024
"""

# Standard library imports
import os
import time
import base64
from urllib.parse import urljoin, urlparse

# Force tensorflow to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Third party imports
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import requests
from bs4 import BeautifulSoup

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Relative path name of model
model_path = 'cloud_classification_model.keras'

# Input image sizes
IMG_SIZE = (224, 224)

# Defining classification categories
CLASSES = {
    1: 'Clear',
    2: 'Cirrus',
    3: 'Cumulus',
    4: 'Fog/Virga'
}


def load_model_on_cpu(model_path):
    """
    Loads model with CPU

    Args:
        model_path (str): Relative file path of model defined prior.

    Returns:
        tf.keras.models.load_model: Keras model instance.
    """
    with tf.device('/cpu:0'):
        return load_model(model_path)


def load_and_preprocess_image(image_path):
    """
    Loads and preprocesses an image for model prediction.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.array: Preprocessed image array.
    """
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def predict_cloud_coverage(model, image_path):
    """
    Predicts cloud coverage using the given model and image.

    Args:
        model (tf.keras.Model): Loaded Keras model.
        image_path (str): Path to the image file.

    Returns:
        tuple: Predicted class (int) and confidence (float).
    """
    preprocessed_img = load_and_preprocess_image(image_path)
    prediction = model.predict(preprocessed_img, verbose=0)
    predicted_class = np.argmax(prediction) + 1
    confidence = np.max(prediction)
    return predicted_class, confidence

def get_latest_filename(url):
    """
    Retrieves the filename of the second most recent PNG image from the given URL.

    Args:
        url (str): URL to fetch image filenames from.

    Returns:
        str or None: Filename of the second PNG image, or None if not found.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        png_files = [link.get('href') for link in soup.find_all('a') if link.get('href', '').endswith('.png')]
        return png_files[1] if len(png_files) > 1 else None
    except requests.RequestException as e:
        print(f"Error fetching latest filename: {e}")
        return None

def get_image_url(base_url):
    """
    Constructs the full URL for the latest image.

    Args:
        base_url (str): Base URL of the image server.

    Returns:
        str or None: Full URL of the latest image, or None if not available.
    """
    latest_filename = get_latest_filename(base_url)
    return urljoin(base_url, latest_filename) if latest_filename else None

def write_to_file(file_name, predicted_class, class_name, confidence, retrieval_time):
    """
    Writes prediction results to a file.

    Args:
        file_name (str): Name of the image file.
        predicted_class (int): Predicted class number.
        class_name (str): Name of the predicted class.
        confidence (float): Confidence of the prediction.
        retrieval_time (str): Time of image retrieval.
    """
    with open('realtime_result.txt', 'w') as f:
        f.write(f"Date/Time of Retrieval: {retrieval_time}\n")
        f.write(f"File Name: {file_name}\n")
        f.write(f"Predicted class: {predicted_class}: {class_name}\n")
        f.write(f"Confidence in Classification: {confidence:.2f}\n")

def process_image(model, last_processed_file=None):
    """
    Processes the latest image, performs prediction, and logs results.

    Args:
        model (tf.keras.Model): Loaded Keras model.
        last_processed_file (str): Name of the last processed file.

    Returns:
        str: Name of the processed file, or last_processed_file if no new image.
    """
    base_url = "http://mkoccamdev-lv1:8080/camera/100000006aa8722a"
    image_url = get_image_url(base_url)

    if not image_url:
        print("No image URL available at this time.")
        return last_processed_file

    file_name = os.path.basename(urlparse(image_url).path)

    if file_name == last_processed_file:
        print(f"No new image available. Last processed: {file_name}")
        return last_processed_file

    try:
        retrieval_time = time.strftime('%Y-%m-%d %H:%M:%S')
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()

        with open('temp_image.jpg', 'wb') as f:
            f.write(response.content)

        predicted_class, confidence = predict_cloud_coverage(model, 'temp_image.jpg')
        class_name = CLASSES[predicted_class]

        print(f"\nDate/Time of Retrieval: {retrieval_time}")
        print(f"File Name: {file_name}")
        print(f"Predicted class: {predicted_class}: {class_name}")
        print(f"Confidence in Classification: {confidence:.2f}")

        write_to_file(file_name, predicted_class, class_name, confidence, retrieval_time)

        return file_name
    except Exception as e:
        print(f"Error processing image: {e}")
        return last_processed_file

def main():
    print("Starting cloud-based threat classification system")

    try:
        model = load_model_on_cpu(model_path)
        print("cloud_classification_model.keras loaded on CPU")
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    print("Starting real-time inference on images.")
    print("Press Ctrl+C to stop the process.")

    last_processed_file = None
    try:
        while True:
            last_processed_file = process_image(model, last_processed_file)
            time.sleep(20)  # Wait for 20 seconds before the next processing
    except KeyboardInterrupt:
        print("\nProcess stopped by user.")
        print("\nLast classification outputted to realtime_result.txt.")
    finally:
        if os.path.exists('temp_image.jpg'):
            os.remove('temp_image.jpg')

if __name__ == '__main__':
    main()