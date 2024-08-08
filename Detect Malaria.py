import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Title and description
st.title("Malaria Detection App")
st.write("Upload an image to detect if it has malaria parasites.")

# Function to load and preprocess the image
def load_image(image_file):
    img = Image.open(image_file)
    return img

def preprocess_image(image):
    image = image.resize((64, 64))  # Resize to match model's input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to load the model
@st.cache_resource
def load_model():
    model_path = r"C:\Users\Lenovo\OneDrive\Desktop\Deep Learning\malaria_model.h5"
  # Update this path to your model file
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# Upload the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = load_image(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Predict using the model
    prediction = model.predict(processed_image)

    # Display the prediction
    if prediction[0][0] > 0.785:
        st.write("The image is predicted to have  malaria parasites.",prediction[0][0])
    else:
        st.write("The image is predicted to be free of malaria parasites.",prediction[0][0])
