import gzip
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Define a function to load the MNIST dataset from a gzipped file
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 28, 28, 1).astype(np.float32)
    return data / 255.0

# Define a function to load the MNIST labels from a gzipped file
def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data.astype(np.int64)

# Load the MNIST training data
train_images = load_mnist_images('dataset/train-images-idx3-ubyte.gz')
train_labels = load_mnist_labels('dataset/train-labels-idx1-ubyte.gz')

# Build a deep learning model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 output units for 10 classes (0-9)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=35, batch_size=64, validation_split=0.2)

# Save the trained model if needed
model.save('mnist_digit_recognition_model.h5')
