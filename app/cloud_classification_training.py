"""
File name: cloud_classification_training.py

Cloud-Based Threat Classification Model Training Script

This script trains a convolutional neural network using transfer learning
with pre-trained ResNet50 for cloud classification. It includes data augmentation,
progressive unfreezing, decreasing learning rate, L2 regularization,
and visualization of training history.

Author: Frances Uy
Mentors: Gemini Observatory: Hawi Stecher, Patrick Parks
Date: June-July 2024
"""

# Standard library imports
import os

# Third-party imports
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt


# Model training parameters and constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 20
NUM_CLASSES = 4

# Filepaths
DATASET_PATH = '/home/frances.uy/pycharm-projects/cloud_models/data'
MODEL_PATH = 'experiment_5_four.keras'

# Print the contents of the data directory
print("Contents of data directory:")
for item in os.listdir(DATASET_PATH):
    print(item)

# Data generators with augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
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
# Avoids "running out of data" issue during training
steps_per_epoch = len(train_generator)
validation_steps = len(validation_generator)

# Load pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))

# Global spatial average pooling layer
# Reduces the spatial dimensions of the input tensor to a single value per channel
# by computing the average value across all spatial locations for each channel
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a final dense layer for classification with L2 regularization
# The dense layer takes the output from the previous layer (x) as input
# NUM_CLASSES units (neurons) corresponding to the number of classes in the classification task
# The activation function used is 'softmax', which produces a probability distribution over the classes
# L2 regularization is applied to the kernel (weights) of the dense layer with a regularization factor of 0.01
# Prevent overfitting by adding a penalty term to the loss function based on the squared magnitude of the weights
output = Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=l2(0.01))(x)

# Initializing model
model = Model(inputs=base_model.input, outputs=output)

# Freeze all layers initially
for layer in model.layers:
    layer.trainable = False

# Make the final dense layer trainable
# Ensures weights of last layer are trainable
# Layer's weights will be modified based on the gradients computed during backpropagation
model.layers[-1].trainable = True

def train_model(model, epochs, unfreeze_layers=0, learning_rate=0.001):
    """
    Train the model with specified parameters.

    Args:
        model (tf.keras.Model): The model to train.
        epochs (int): Number of epochs to train.
        unfreeze_layers (int): Number of layers to unfreeze from the end.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        tf.keras.callbacks.History: Training history object.
    """

    if unfreeze_layers > 0:
        for layer in model.layers[-unfreeze_layers:]:
            layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Create a tf.data.Dataset that repeats indefinitely
    # Avoids "running out of data" issue during training
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

    # Save the model after training
    model.save(MODEL_PATH)

    return history

def plot_history(history, title):
    """
    Plot and save the training history.

    Args:
        history (tf.keras.callbacks.History): Training history object.
        title (str): Title for the plot.
    """

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

# Initial training (only final dense layer)
print("Training final dense layer...")
history_initial = train_model(model, epochs=EPOCHS, learning_rate=0.001)
plot_history(history_initial, "Initial Training")

# Unfreeze and train last ResNet block
print("Training last ResNet block...")
history_last_block = train_model(model, epochs=EPOCHS, unfreeze_layers=17, learning_rate=0.0001)
plot_history(history_last_block, "Last Block Training")

# Unfreeze and train last two ResNet blocks
print("Training last two ResNet blocks...")
history_last_two_blocks = train_model(model, epochs=EPOCHS, unfreeze_layers=36, learning_rate=0.00001)
plot_history(history_last_two_blocks, "Last Two Blocks Training")

# Unfreeze and train all layers
print("Fine-tuning all layers...")
history_full = train_model(model, epochs=EPOCHS, unfreeze_layers=len(model.layers), learning_rate=0.000001)
plot_history(history_full, "Full Model Fine-tuning")

print("Training complete. Final model saved as", MODEL_PATH)

# Print class indices for reference
print("Class indices:", train_generator.class_indices)