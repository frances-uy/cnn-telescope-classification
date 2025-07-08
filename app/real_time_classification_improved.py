"""
File: real_time_classification_improved.py
Author: Frances Uy
Date: July 30, 2024
Purpose: This script performs real-time cloud classification using a pre-trained model.
         It fetches images from a specified URL, processes them, and saves the classification results.
"""

import os
import time
import io
from urllib.parse import urljoin, urlparse
import threading
from queue import Queue
import logging
import glob

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import requests
from bs4 import BeautifulSoup
from collections import Counter
from PIL import Image

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Constants
MODEL_PATH = 'cloud_classification_model.keras'
IMG_SIZE = (224, 224)
CLASSES = {
    1: 'Clear',
    2: 'Cirrus',
    3: 'Cumulus',
    4: 'Fog/Virga'
}
NUM_CPUS = 6

# Use a directory in the user's home folder
HOME_DIR = os.path.expanduser("~")
IMAGE_SAVE_DIR = os.path.join(os.path.dirname(__file__), "static")
RESULTS_DIR = "/app/cloud_classification_data"
RESULTS_FILE = os.path.join(RESULTS_DIR, "realtime_result.txt")

# Ensure the image save and results directory exists
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(filename='/app/logs/classification.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

"""
Function: load_model_on_cpu
Purpose: Loads the pre-trained model from the specified path.
Parameters:
    model_path (str): Path to the model file
Returns:
    model: Loaded Keras model
"""
def load_model_on_cpu(model_path):
    logging.info("Loading model...")
    model = load_model(model_path)
    logging.info("Model loaded successfully.")
    return model

"""
Function: preprocess_images
Purpose: Preprocesses input images for the model.
Parameters:
    images: Input images
Returns:
    Preprocessed images
"""
@tf.function
def preprocess_images(images):
    return tf.keras.applications.resnet50.preprocess_input(images)

"""
Function: get_latest_filename
Purpose: Fetches the latest image filename from the specified URL.
Parameters:
    url (str): URL to fetch the image from
Returns:
    str: Latest filename, or None if not found
"""
def get_latest_filename(url):
    try:
        logging.info(f"Fetching latest filename from {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        png_files = [link.get('href') for link in soup.find_all('a') if link.get('href', '').endswith('.png')]
        latest = png_files[1] if len(png_files) > 1 else None
        logging.info(f"Latest filename: {latest}")
        return latest
    except requests.RequestException as e:
        logging.error(f"Error fetching latest filename: {e}")
        return None

"""
Function: get_image_url
Purpose: Constructs the full URL for the latest image.
Parameters:
    base_url (str): Base URL for the images
Returns:
    str: Full URL of the latest image
"""
def get_image_url(base_url):
    latest_filename = get_latest_filename(base_url)
    full_url = urljoin(base_url, latest_filename) if latest_filename else None
    logging.info(f"Full image URL: {full_url}")
    return full_url

"""
Function: write_to_file
Purpose: Writes classification results to a file.
Parameters:
    file_name (str): Name of the image file
    predicted_class (int): Predicted class number
    class_name (str): Name of the predicted class
    retrieval_time (str): Time when the image was retrieved
    image_path (str): Path where the image is saved
"""
def write_to_file(file_name, predicted_class, class_name, retrieval_time, image_path):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        f.write(f"Date/Time of Retrieval: {retrieval_time}\n")
        f.write(f"File Name: {file_name}\n")
        f.write(f"Predicted class: {predicted_class}: {class_name}\n")
        f.write(f"Image saved at: {image_path}\n\n")
    logging.info(f"Results written to file: {RESULTS_FILE}")

"""
Function: delete_old_image
Purpose: Deletes old images, keeping only the most recent one.
"""
def delete_old_image():
    images = glob.glob(os.path.join(IMAGE_SAVE_DIR, '*.jpg'))
    if len(images) > 1:
        images.sort(key=os.path.getmtime, reverse=True)
        for old_image in images[1:]:
            try:
                os.remove(old_image)
                logging.info(f"Deleted old image: {old_image}")
            except Exception as e:
                logging.error(f"Error deleting old image {old_image}: {e}")

"""
Function: compress_image
Purpose: Compresses an image to reduce file size.
Parameters:
    image: PIL Image object
    quality (int): Compression quality (1-95)
Returns:
    PIL Image: Compressed image
"""
def compress_image(image, quality=85):
    img_buffer = io.BytesIO()
    image.save(img_buffer, format='JPEG', quality=quality, optimize=True)
    img_buffer.seek(0)
    return Image.open(img_buffer)

"""
Function: download_and_save_image
Purpose: Downloads an image from a URL and saves it locally.
Parameters:
    image_url (str): URL of the image to download
    file_name (str): Name to save the file as
Returns:
    str: Path where the image is saved, or None if failed
"""
def download_and_save_image(image_url, file_name):
    try:
        logging.info(f"Downloading image from {image_url}")
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        
        # Compress the image
        compressed_image = compress_image(image)
        
        # Change the file extension to .jpg
        file_name = os.path.splitext(file_name)[0] + '.jpg'
        save_path = os.path.join(IMAGE_SAVE_DIR, file_name)
        
        # Delete old images only after new image is ready
        delete_old_image()
        
        # Save the new image
        compressed_image.save(save_path, format='JPEG', quality=85, optimize=True)
        logging.info(f"Compressed image saved to {save_path}")
        return save_path
    except Exception as e:
        logging.error(f"Error downloading and saving image {file_name}: {e}")
        return None

"""
Function: predict_class
Purpose: Predicts the class of an image using the model.
Parameters:
    model: Keras model
    img_array: Preprocessed image array
    result_queue: Queue to store the prediction result
"""
def predict_class(model, img_array, result_queue):
    prediction = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(prediction) + 1
    result_queue.put(predicted_class)

"""
Function: process_image
Purpose: Processes a single image: downloads, saves, and classifies it.
Parameters:
    model: Keras model
    image_url (str): URL of the image to process
Returns:
    tuple: (file_name, predicted_class, class_name, retrieval_time, image_path) or None if failed
"""
def process_image(model, image_url):
    start_time = time.time()
    file_name = os.path.basename(urlparse(image_url).path)
    retrieval_time = time.strftime('%Y-%m-%d %H:%M:%S')

    try:
        logging.info(f"Processing image: {file_name}")
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()

        download_time = time.time()
        logging.info(f"Image download took {download_time - start_time:.2f} seconds")

        image_path = download_and_save_image(image_url, file_name)
        if image_path is None:
            return None

        save_time = time.time()
        logging.info(f"Image save took {save_time - download_time:.2f} seconds")

        img = load_img(io.BytesIO(response.content), target_size=IMG_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_images(img_array)

        preprocess_time = time.time()
        logging.info(f"Image preprocessing took {preprocess_time - save_time:.2f} seconds")

        logging.info("Starting multi-threaded processing...")
        multi_thread_start = time.time()

        threads = []
        result_queue = Queue()

        for _ in range(NUM_CPUS):
            thread = threading.Thread(target=predict_class, args=(model, img_array, result_queue))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        predictions = []
        while not result_queue.empty():
            predictions.append(result_queue.get())

        multi_thread_end = time.time()
        logging.info(f"Multi-threaded prediction took {multi_thread_end - multi_thread_start:.2f} seconds")

        predicted_class = Counter(predictions).most_common(1)[0][0]
        class_name = CLASSES[predicted_class]
        logging.info(f"Final prediction: {predicted_class}: {class_name}")

        end_time = time.time()
        logging.info(f"Total processing time: {end_time - start_time:.2f} seconds")

        return file_name, predicted_class, class_name, retrieval_time, image_path
    except Exception as e:
        logging.error(f"Error processing image {file_name}: {e}")
        return None

"""
Function: main
Purpose: Main function to run the continuous image processing and classification.
"""
def main():
    model = load_model_on_cpu(MODEL_PATH)
    base_url = "http://mkoccamdev-lv1:8080/camera/100000006aa8722a"
    
    while True:
        image_url = get_image_url(base_url)
        if image_url:
            result = process_image(model, image_url)
            if result:
                file_name, predicted_class, class_name, retrieval_time, image_path = result
                write_to_file(file_name, predicted_class, class_name, retrieval_time, image_path)

if __name__ == "__main__":
    main()
