# File: live_cell_segmentation.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import matplotlib.pyplot as plt

def load_data(data_dir: str, img_size: Tuple[int, int] = (256, 256)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess the image data from the specified directory.

    Parameters:
    data_dir (str): The directory where the data is stored.
    img_size (Tuple[int, int]): The desired size of the images.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing training data, validation data, and validation labels.
    """
    images = []
    masks = []

    for filename in os.listdir(os.path.join(data_dir, 'images')):
        img = load_img(os.path.join(data_dir, 'images', filename), target_size=img_size, color_mode='grayscale')
        img = img_to_array(img) / 255.0
        images.append(img)

        mask = load_img(os.path.join(data_dir, 'masks', filename), target_size=img_size, color_mode='grayscale')
        mask = img_to_array(mask) / 255.0
        masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)

    train_images, val_images, train_masks, val_masks = train_test_split(images, masks, test_size=0.2, random_state=42)

    return train_images, val_images, val_masks

def unet_model(input_shape: Tuple[int, int, int] = (256, 256, 1)) -> tf.keras.Model:
    """
    Define the U-Net model architecture.

    Parameters:
    input_shape (Tuple[int, int, int]): Shape of the input images.

    Returns:
    tf.keras.Model: Compiled U-Net model.
    """
    inputs = layers.Input(shape=input_shape)

    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    u1 = layers.UpSampling2D((2, 2))(p3)
    c4 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u1)

    u2 = layers.UpSampling2D((2, 2))(c4)
    c5 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u2)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c5)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def train_model(model: tf.keras.Model, train_data: np.ndarray, val_data: np.ndarray, epochs: int = 10) -> None:
    """
    Train the U-Net model.

    Parameters:
    model (tf.keras.Model): U-Net model to be trained.
    train_data (np.ndarray): Training data.
    val_data (np.ndarray): Validation data.
    epochs (int): Number of epochs to train the model.
    """
    model.fit(train_data, epochs=epochs, validation_data=val_data)

def evaluate_and_visualize(model: tf.keras.Model, val_data: np.ndarray, val_labels: np.ndarray) -> None:
    """
    Evaluate the model and visualize the results.

    Parameters:
    model (tf.keras.Model): Trained U-Net model.
    val_data (np.ndarray): Validation data.
    val_labels (np.ndarray): Validation labels.
    """
    loss, accuracy = model.evaluate(val_data, val_labels)
    print(f'Validation loss: {loss}')
    print(f'Validation accuracy: {accuracy}')

    predictions = model.predict(val_data[:5])
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))

    for i in range(5):
        axes[0, i].imshow(val_data[i].reshape(256, 256), cmap='gray')
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')

        axes[1, i].imshow(val_labels[i].reshape(256, 256), cmap='gray')
        axes[1, i].set_title('Ground Truth')
        axes[1, i].axis('off')

        axes[2, i].imshow(predictions[i].reshape(256, 256), cmap='gray')
        axes[2, i].set_title('Predicted')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_dir = 'path/to/data'
    train_data, val_data, val_labels = load_data(data_dir)

    model = unet_model()
    train_model(model, train_data, val_data)

    model.save('live_cell_segmentation_model.h5')
    evaluate_and_visualize(model, val_data, val_labels)
