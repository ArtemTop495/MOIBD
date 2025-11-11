# MNIST Digit Recognition with TensorFlow & Streamlit

This project demonstrates the development of a digit recognition system using TensorFlow and Streamlit. The repository includes model training, a Streamlit web app for digit recognition, and testing scripts for evaluation.

## Features

- **Model Training**: Train a convolutional neural network (CNN) model on the MNIST dataset to recognize handwritten digits.

- **Streamlit Web App**: Create an interactive web app that allows users to upload images of handwritten digits for recognition using the trained model.

- **Testing and Evaluation**: Evaluate the model's performance on the MNIST test dataset, generating a detailed classification report.

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/hand-written-digit-recognition.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Train the Model: Run the model training script to train your own digit recognition model or use the provided pre-trained model.

   ```bash
   python model.py
   ```

4. Launch the Streamlit Web App: Start the Streamlit app to enable real-time digit recognition.

   ```bash
   streamlit run app.py
   ```

5. Test the Model: To evaluate the model's performance, run the testing script and view the classification report.

   ```bash
   python test.py
   ```

## Usage

- Access the Streamlit app through your web browser and upload an image of a handwritten digit for recognition.

- Explore the model training and evaluation code for further insights.

## Contributing

Contributions are welcome! Feel free to open issues, submit pull requests, or provide suggestions for improvements.

## License

This project is licensed under the MIT License.

## Acknowledgments

- The MNIST dataset used in this project is sourced from [MNIST Database](http://yann.lecun.com/exdb/mnist/).

- Special thanks to the TensorFlow and Streamlit communities for their invaluable resources.

- This projject was created by [Harsh Bafna](https://github.com/harshbafnaa/).
