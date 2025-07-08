"""
Author: Frances Uy
Date: July 2024
Purpose: This script performs cloud classification on a set of unlabeled images using a pre-trained
         TensorFlow model. It sorts the images into labeled folders based on predictions and allows
         for manual correction of mis-classifications. Leverages GPU use for speed and efficiency.

NOTE: UNLABELED_PATH and LABELED_PATH can be adjusted to desired locations.

Filename: image_inferencing_average.py
"""

import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tqdm import tqdm
from collections import Counter
import shutil

# Check for GPU availability
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Allow memory growth for the GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Constants
MODEL_PATH = 'cloud_classification_model.keras'
IMG_SIZE = (224, 224)
CLASSES = {
    1: 'Clear',
    2: 'Cirrus',
    3: 'Cumulus',
    4: 'Fog_Virga'
}
UNLABELED_PATH = '/home/frances.uy/pycharm-projects/cloud_classification_cli/cloud_classification_cli/2024-07-22'
LABELED_PATH = '/home/frances.uy/pycharm-projects/cloud_classification_cli/cloud_classification_cli/labeled_dataset_inferencing_average'
NUM_PREDICTIONS = 5  # Number of times to run prediction for each image


def load_model_on_gpu(model_path):
    """
    Loads the TensorFlow model on GPU.

    Args:
    model_path (str): Path to the saved model file.

    Returns:
    tf.keras.Model: Loaded TensorFlow model.
    """
    return load_model(model_path)


@tf.function
def preprocess_images(images):
    """
    Preprocesses images for the ResNet50 model.

    Args:
    images (tf.Tensor): Input images to preprocess.

    Returns:
    tf.Tensor: Preprocessed images.
    """
    return tf.keras.applications.resnet50.preprocess_input(images)


def predict_multiple_times(model, image_path, num_predictions=NUM_PREDICTIONS):
    """
    Makes multiple predictions for a single image and returns the most common prediction
    along with the average confidence.

    Args:
    model (tf.keras.Model): Loaded TensorFlow model.
    image_path (str): Path to the image file.
    num_predictions (int): Number of predictions to make.

    Returns:
    tuple: Most common prediction and average confidence.
    """
    predictions = []
    confidences = []
    for _ in range(num_predictions):
        img = load_img(image_path, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_images(img_array)

        prediction = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(prediction) + 1
        confidence = np.max(prediction)

        predictions.append(predicted_class)
        confidences.append(confidence)

    # Get the most common prediction
    most_common_prediction = Counter(predictions).most_common(1)[0][0]

    # Calculate average confidence
    avg_confidence = sum(confidences) / len(confidences)

    return most_common_prediction, avg_confidence


def process_and_sort_images(model, unlabeled_path, labeled_path):
    """
    Processes and sorts unlabeled images into labeled folders based on model predictions.

    Args:
    model (tf.keras.Model): Loaded TensorFlow model.
    unlabeled_path (str): Path to the folder containing unlabeled images.
    labeled_path (str): Path to the folder where labeled images will be sorted.
    """
    # Create directories for each class
    for class_name in CLASSES.values():
        os.makedirs(os.path.join(labeled_path, class_name), exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(unlabeled_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"No image files found in {unlabeled_path}")
        return

    print(f"Found {len(image_files)} images to process")

    # Process images
    for file_name in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(unlabeled_path, file_name)

        try:
            predicted_class, avg_confidence = predict_multiple_times(model, img_path)

            class_name = CLASSES[predicted_class]
            dst_path = os.path.join(labeled_path, class_name, file_name)

            # Move the image to the appropriate folder
            shutil.move(img_path, dst_path)

            # Log the classification
            log_file = os.path.join(labeled_path, 'classification_log.txt')
            with open(log_file, 'a') as f:
                f.write(f"{file_name},{predicted_class},{class_name},{avg_confidence:.4f}\n")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    print("Processing complete")


def correct_classifications(labeled_path):
    """
    Allows manual correction of misclassified images.

    Args:
    labeled_path (str): Path to the folder containing labeled images.
    """
    while True:
        file_name = input("Enter the filename to correct (or 'done' to finish): ")
        if file_name.lower() == 'done':
            break

        current_class = input("Enter the current (incorrect) class: ")
        correct_class = input("Enter the correct class: ")

        current_path = os.path.join(labeled_path, current_class, file_name)
        correct_path = os.path.join(labeled_path, correct_class, file_name)

        if os.path.exists(current_path):
            shutil.move(current_path, correct_path)
            print(f"Moved {file_name} from {current_class} to {correct_class}")

            # Update log file
            log_file = os.path.join(labeled_path, 'classification_log.txt')
            with open(log_file, 'r') as f:
                lines = f.readlines()
            with open(log_file, 'w') as f:
                for line in lines:
                    if line.startswith(file_name):
                        parts = line.split(',')
                        parts[1] = str(list(CLASSES.keys())[list(CLASSES.values()).index(correct_class)])
                        parts[2] = correct_class
                        f.write(','.join(parts))
                    else:
                        f.write(line)
        else:
            print(f"File {file_name} not found in {current_class}")

    print("Correction process completed.")


def main():
    """
    Main function to execute the cloud classification sorting and correction process.
    """
    print("Starting dataset sorting and correction process")

    # Check contents of the unlabeled folder
    print(f"Contents of {UNLABELED_PATH}:")
    unlabeled_files = os.listdir(UNLABELED_PATH)
    print(f"Number of files: {len(unlabeled_files)}")
    print(f"First few files: {unlabeled_files[:5] if unlabeled_files else 'No files found'}")

    # Sorting phase
    try:
        model = load_model_on_gpu(MODEL_PATH)
        print("Model loaded successfully on GPU")
        process_and_sort_images(model, UNLABELED_PATH, LABELED_PATH)
        print(f"\nSorting complete. Check the '{LABELED_PATH}' folder and 'classification_log.txt' for results.")
    except Exception as e:
        print(f"Error during sorting: {e}")
        return

    # Check contents of labeled folders
    print("\nChecking contents of labeled folders:")
    for class_name in CLASSES.values():
        class_path = os.path.join(LABELED_PATH, class_name)
        files = os.listdir(class_path)
        print(f"{class_name}: {len(files)} files")

    # Correction phase
    print("\nEntering correction phase. You can now correct any misclassifications.")
    correct_classifications(LABELED_PATH)


if __name__ == '__main__':
    main()