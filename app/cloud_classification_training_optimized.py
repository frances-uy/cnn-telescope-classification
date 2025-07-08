"""
File name: cloud_classification_fine_tuning_updated.py

Cloud-Based Threat Classification Model Fine-Tuning Script

This script fine-tunes the previous model called cloud_classification_model.keras
on an expanded dataset

Author: Frances Uy
Mentors: Gemini Observatory: Hawi Stecher, Patrick Parks
Date: June-July 2024
"""

# Standard library imports
import os

# Third-party imports
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

# Model training parameters and constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
NUM_CLASSES = 4

# Filepaths
DATASET_PATH = 'data_2'  # Path to your expanded dataset
ORIGINAL_MODEL_PATH = 'cloud_classification_model.keras'  # Path to your original model
NEW_MODEL_PATH = 'cloud_classification_model_fine_tuned.keras'

# Load the original model
model = load_model(ORIGINAL_MODEL_PATH)
print("Original model loaded.")

# Print the contents of the data directory
print("Contents of expanded data directory:")
for item in os.listdir(DATASET_PATH):
    print(item)

# Data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# Print the length of train and validation generators
print(f"Train generator length: {len(train_generator)}")
print(f"Validation generator length: {len(validation_generator)}")

# Calculate steps per epoch
steps_per_epoch = len(train_generator)
validation_steps = len(validation_generator)

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

# Print dataset composition
print("\nDataset composition:")
for class_name in train_generator.class_indices:
    n_samples = np.sum(train_generator.classes == train_generator.class_indices[class_name])
    print(f"Class {class_name}: {n_samples} samples")


def fine_tune_model(model, epochs, unfreeze_layers=0, learning_rate=0.0001):
    """
    Fine-tune the model with specified parameters.

    Args:
        model (tf.keras.Model): The model to fine-tune.
        epochs (int): Number of epochs to train.
        unfreeze_layers (int): Number of layers to unfreeze from the end.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        tf.keras.callbacks.History: Training history object.
    """

    # Unfreeze specified number of layers
    if unfreeze_layers > 0:
        for layer in model.layers[-unfreeze_layers:]:
            layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Create a tf.data.Dataset that repeats indefinitely
    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, *IMG_SIZE, 3], [None, NUM_CLASSES])
    ).repeat()

    val_dataset = tf.data.Dataset.from_generator(
        lambda: validation_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, *IMG_SIZE, 3], [None, NUM_CLASSES])
    ).repeat()

    history = model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_dataset,
        validation_steps=validation_steps
    )

    return history


def plot_history(history, title):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()


# Fine-tuning schedule
print("\nFine-tuning last few layers...")
history_initial = fine_tune_model(model, epochs=EPOCHS, unfreeze_layers=10, learning_rate=0.0001)
plot_history(history_initial, "Initial Fine-tuning")

print("\nFine-tuning more layers...")
history_more = fine_tune_model(model, epochs=EPOCHS, unfreeze_layers=30, learning_rate=0.00005)
plot_history(history_more, "Extended Fine-tuning")

print("\nFine-tuning all layers...")
history_full = fine_tune_model(model, epochs=EPOCHS, unfreeze_layers=len(model.layers), learning_rate=0.00001)
plot_history(history_full, "Full Model Fine-tuning")

# Save the final model
model.save(NEW_MODEL_PATH)

print("\nFine-tuning complete. Final model saved as", NEW_MODEL_PATH)

# Print class indices for reference
print("\nClass indices:", train_generator.class_indices)