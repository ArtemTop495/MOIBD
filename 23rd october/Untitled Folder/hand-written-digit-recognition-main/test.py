import gzip
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

# Define a function to load the T10K test data from a gzipped file
def load_t10k_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 28, 28, 1).astype(np.float32)
    return data / 255.0

# Define a function to load the T10K test labels from a gzipped file
def load_t10k_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data.astype(np.int64)

# Load the T10K test data
t10k_images = load_t10k_images('dataset/t10k-images-idx3-ubyte.gz')
t10k_labels = load_t10k_labels('dataset/t10k-labels-idx1-ubyte.gz')

# Load your trained model
model = tf.keras.models.load_model('mnist_digit_recognition_model.h5')  # Update with your model's filename

# Evaluate the model on the T10K test data
predictions = model.predict(t10k_images)
predicted_labels = np.argmax(predictions, axis=1)

# Generate a classification report
report = classification_report(t10k_labels, predicted_labels)

print("Classification Report:\n")
print(report)
