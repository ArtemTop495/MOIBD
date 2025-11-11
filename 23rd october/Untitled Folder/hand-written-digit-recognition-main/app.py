import streamlit as st
import tensorflow as tf
import numpy as np

st.cache(allow_output_mutation=True)

# Load the trained model
model = tf.keras.models.load_model('mnist_digit_recognition_model.h5')

# Function to preprocess the uploaded image
def preprocess_image(file):
    image = tf.image.decode_image(file.read(), channels=1)
    image = tf.image.resize(image, (28, 28))
    image = np.expand_dims(image, axis=0) / 255.0
    return image

# Streamlit UI
st.title('MNIST Digit Recognition')
st.write('Upload an image of a handwritten digit for recognition.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    if st.button('Predict'):
        image = preprocess_image(uploaded_file)
        prediction = model.predict(image)
        predicted_label = np.argmax(prediction[0])
        st.write(f'Predicted digit: {predicted_label}')
