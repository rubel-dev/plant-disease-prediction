# plant-disease-prediction
# 🌱 Plant Disease Recognition System (96% Accuracy)

An end-to-end **Plant Disease Recognition System** built using **Deep Learning** and deployed with **Streamlit**.  
The system allows users to upload a plant leaf image and instantly predicts the disease.

 **Live Demo:** https://clj4fi74itioovzvwxwnd9.streamlit.app/  
 **Model Accuracy:** **96%**

---

##  Features
- Plant leaf image upload
- Automatic disease prediction
- Simple and user-friendly Streamlit interface
- Fast inference with trained deep learning model
- Fully deployed end-to-end system

---

##  Model Details
- **Problem Type:** Multi-class classification
- **Architecture:** CNN 
- **Accuracy:** 96%
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam

> Model trained and evaluated on plant disease image dataset.

---

## 🛠 Tech Stack
- **Python**
- **TensorFlow / Keras**
- **Streamlit**
- **NumPy**
- **Pillow**
- **OpenCV** 

---

##  System Workflow
1. User uploads a plant leaf image  
2. Image is resized and normalized  
3. Trained CNN model processes the image  
4. Disease class is predicted  
5. Result is displayed on the web app  

---

## ⚙️ Environment Setup (Conda)

###  Create Conda Environment
```bash
conda create -n tensorflow_env python=3.9 -y

conda activate tensorflow_env
streamlit run app.py
