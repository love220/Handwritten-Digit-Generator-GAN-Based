## ✍️ Handwritten Digit Generator – GAN-Based ML Project with Streamlit Deployment

### 📌 Project Description:

The **Handwritten Digit Generator** is an AI-powered application that uses a **Generative Adversarial Network (GAN)** to synthesize realistic images of handwritten digits, similar to those in the MNIST dataset. This project demonstrates the creative side of machine learning by generating new image data from scratch using two key models:

1. **Generator Model**  
   - Takes random noise as input and generates fake digit images  
   - Trained to "fool" the discriminator by producing realistic-looking digits

2. **Discriminator Model**  
   - Takes in images (real or fake) and classifies them as real (from dataset) or fake (from generator)  
   - Trained to distinguish between real and generated digits

These two models are trained together in an adversarial loop, improving the quality of generated digits over time.

---

### 🧠 Machine Learning Workflow:
- Preprocessing of MNIST handwritten digit dataset
- Building and training the **GAN** (Generator + Discriminator)
- Saving the trained models using **Keras/TF Save** or **Pickle**
- Deploying the generator via **Streamlit** with a simple and interactive UI

---

### 🌐 Deployment:
The application is deployed as a **Streamlit web app** using a Python script named `gen.py`. The user interface allows users to:
- Click a button to generate a random digit
- View the generated image instantly in the app
- Optionally control the noise vector seed for reproducibility

---

### 📁 Folder Structure:
```
C:\ml_deployment
│
├── gen.py                   # Streamlit app to generate digits
├── generator_model.h5       # Trained generator model
├── discriminator_model.h5   # Trained discriminator (for training only)
├── requirements.txt         # Python dependencies
└── utils/                   # (Optional) helper functions or plotting
```

---

### 🔥 Key Features:
- Realistic handwritten digit generation with GAN
- Simple, interactive UI for digit creation
- Fully offline and lightweight interface
- Educational showcase of adversarial training in ML

---

### 🛠️ Technologies Used:
- Python
- TensorFlow / Keras (for GAN implementation)
- NumPy, Matplotlib (for image display and processing)
- Streamlit (for real-time deployment)
- HDF5 / Pickle (for model saving)
