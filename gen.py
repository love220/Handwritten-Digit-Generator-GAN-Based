import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu  # Importing option menu

def load_model(file_path):
    return tf.keras.models.load_model(file_path)

# Load models
generator = load_model('C:/Users/Lenovo/Desktop/saved_models/generator_model.h5')
discriminator = load_model('C:/Users/Lenovo/Desktop/saved_models/discriminator_model.h5')

# Function to generate an image
def generate_image(generator, latent_dim=100):
    noise = np.random.randn(1, latent_dim)
    generated_image = generator.predict(noise).reshape(28, 28)
    generated_image = (generated_image - generated_image.min()) / (generated_image.max() - generated_image.min())  # Normalize
    return generated_image

# Streamlit UI
st.set_page_config(page_title="Handwritten Digit Generator", layout="wide")
st.title("Handwritten Digit Generator")

# Sidebar Navigation with Buttons
selected_tab = option_menu(
    menu_title="Navigation",
    options=["Generate", "Visualize"],
    icons=["magic", "bar-chart"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "20px", "background-color": "#f0f2f6"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "10px",  # Added more margin between buttons
            "border-radius": "20px",
            "padding": "12px",
            "border": "2px solid #6c757d",
        },
        "nav-link-selected": {"background-color": "#6c757d", "color": "white"},
    }
)

if selected_tab == "Generate":
    st.write("")  # Adds an empty space
    st.write("")  # Adds another empty space for better separation
    if st.button("Generate Digit"):
        image = generate_image(generator)
        fig, ax = plt.subplots(figsize=(0.5, 0.5))  # Small image size
        ax.imshow(image, cmap='gray', interpolation='nearest')  # Prevent blurring
        ax.axis('off')
        st.pyplot(fig)
        
        # Evaluate with Discriminator
        image_tensor = np.expand_dims(image, axis=[0, -1])  # Add batch and channel dimensions
        prediction = discriminator.predict(image_tensor)[0, 0]
        st.write(f"Discriminator Score: {prediction:.4f}")

        # Store prediction for visualization
        st.session_state['last_prediction'] = prediction

elif selected_tab == "Visualize":
    st.write("")  # Adds an empty space
    st.write("")  # Adds another empty space for better separation
    st.header("Discriminator Confidence Visualization")
    if 'last_prediction' in st.session_state:
        prediction = st.session_state['last_prediction']
        fig, ax = plt.subplots(figsize=(3, 2))  # Smaller graph size
        ax.bar(["Fake", "Real"], [1 - prediction, prediction], color=["red", "green"])
        ax.set_title("Discriminator Confidence")
        ax.set_ylabel("Probability")
        st.pyplot(fig)
    else:
        st.write("Generate a digit first to visualize the results.")
