import streamlit as st
import tensorflow as tf
import numpy as np

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_plant_disease_model.h5")  # convert to .h5 for TF 2.10

model = load_model()

def model_prediction(uploaded_file):
    image = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = input_arr / 255.0
    input_arr = np.expand_dims(input_arr, axis=0)  # (1,128,128,3)

    preds = model.predict(input_arr)
    return int(np.argmax(preds, axis=1)[0])

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox('Select page',['Home', 'About','Disease Detection'])

if(app_mode == 'Home'):
    st.header('PLANT DISEASE DETECTION SYSTEM')
    img = 'img.jpg'
    st.image(img, use_column_width=True)
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
        padding: 30px;
        border-radius: 18px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
    ">

    <h1 style="text-align:center; color:#2e7d32;">
    🌿 Plant Disease Recognition System 🔍
    </h1>

    <p style="font-size:18px; text-align:center; color:#444;">
    Empowering farmers and researchers with intelligent plant disease detection.  
    Upload a plant image and let our AI identify potential diseases in seconds.
    </p>

    <hr style="border:1px solid #c8e6c9;">

    <h3 style="color:#1b5e20;"> How It Works</h3>

    <ol style="font-size:16px; color:#333;">
    <li><b> Upload Image:</b> Navigate to the <b>Disease Recognition</b> page and upload a plant image.</li>
    <li><b> AI Analysis:</b> Our deep learning model examines the image for disease patterns.</li>
    <li><b> Results:</b> Instantly receive predictions and actionable insights.</li>
    </ol>

    <h3 style="color:#1b5e20;"> Why Choose Us?</h3>

    <ul style="font-size:16px; color:#333;">
    <li><b> High Accuracy:</b> Powered by state-of-the-art CNN models.</li>
    <li><b> User-Friendly:</b> Clean and intuitive interface built with Streamlit.</li>
    <li><b> Fast & Efficient:</b> Get reliable results in seconds.</li>
    </ul>

    <h3 style="color:#1b5e20;"> Get Started</h3>

    <p style="font-size:16px; color:#333;">
    Click on the <b>Disease Recognition</b> page from the sidebar to begin your analysis  
    and experience the power of AI-driven plant healthcare 🌱
    </p>

    <h3 style="color:#1b5e20;">ℹ About This Project</h3>

    <p style="font-size:16px; color:#333;">
    Learn more about the project vision, technology stack, and future roadmap  
    on the <b>About</b> page.
    </p>

    </div>
    """, unsafe_allow_html=True)
elif (app_mode == 'About'):
    st.header('About')
    st.markdown("""
<div style="
    background: #f9fff7;
    padding: 25px;
    border-radius: 16px;
    border-left: 6px solid #4caf50;
">

<h3 style="color:#2e7d32;"> About the Dataset</h3>

<p style="font-size:16px; color:#333;">
This dataset is recreated using <b>offline data augmentation</b> from the original plant disease dataset.
The original dataset is publicly available on a GitHub repository.
</p>

<p style="font-size:16px; color:#333;">
It contains approximately <b>87,000 RGB images</b> of healthy and diseased crop leaves,
categorized into <b>38 distinct classes</b>.
</p>

<p style="font-size:16px; color:#333;">
The dataset is split using an <b>80:20 ratio</b> for training and validation while preserving
the original directory structure.  
A separate test directory containing <b>33 images</b> is created for prediction and evaluation.
</p>

<h4 style="color:#1b5e20;"> Dataset Structure</h4>

<ul style="font-size:16px; color:#333;">
  <li><b>Train:</b> 70,295 images</li>
  <li><b>Validation:</b> 17,572 images</li>
  <li><b>Test:</b> 33 images</li>
</ul>

</div>
""", unsafe_allow_html=True)

elif (app_mode =='Disease Detection'):
    st.header('Disease Detection')
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    if test_image is not None:
        st.image(test_image, use_container_width=True)

    if(st.button('Predict')):
        if test_image is None:
            st.warning("Please upload an image first.")
            st.stop()   
        st.write('Prediction')
        result_index = model_prediction(test_image)
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success(f"✅ Model Prediction: **{class_name[result_index]}**")