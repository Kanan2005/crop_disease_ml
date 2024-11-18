import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("crop_disease_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(64,64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.markdown(
    """
    <h1 style='color: #1E90FF; text-align: center;'>PLANT DISEASE RECOGNITION SYSTEM</h1>
    """,
    unsafe_allow_html=True
)

    #image_path = "home_page.jpeg"
    #st.image(image_path, width=800) 
    #st.image(image_path, use_container_width=True)  # Updated parameter
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help Farmers in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated by first combining 2 freely available datasets on kaggle.com. One of them has 20k+ images and other had 87k+ images. We took the top 35 classes of crop diseases for efficient working of model.
                #### Content
                1. train (11558 images)
                2. validation (2344 images)

                #### About Team
                This is our term project by Group#1 done under the guidance of Mr. Rakesh Bairathi Sir. Team Members:
                1. Kanan Agarwal
                2. Devashish Tushar
                3. Rama Swarnkar
                4. Rakshita Agrawal
                5. Rahul Bairwa

                """)

#Prediction Page
elif(app_mode == "Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_container_width=True)

        # Predict button
        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction")

            # Get the prediction
            result_index = model_prediction(test_image)

            # Reading Labels
            class_name = [
                'American Bollworm - Cotton', 'Aphid - Wheat', 'Apple Scab - Apple',
                'Bacterial Blight - Cotton', 'Bacterial Spot - Pepper, bell',
                'Bacterial Spot - Tomato', 'Black Rot - Apple', 'Black Rot - Grape',
                'Black Rust - Wheat', 'Blast - Rice', 'Common Rust - Corn',
                'Ear Rot - Maize', 'Early Blight - Tomato', 'Healthy - Apple',
                'Healthy - Cherry', 'Healthy - Corn', 'Healthy - Cotton', 'Healthy - Grape',
                'Healthy - Maize', 'Healthy - Pepper, bell', 'Healthy - Potato',
                'Healthy - Strawberry', 'Healthy - Sugarcane', 'Healthy - Tomato',
                'Healthy - Wheat', 'Late Blight - Potato', 'Leaf Blight - Grape',
                'Leaf Scorch - Strawberry', 'Mite - Wheat', 'Mosaic - Sugarcane',
                'Powdery Mildew - Cherry', 'Red Bug - Cotton', 'Stem Borer - Maize',
                'Stem Fly - Wheat', 'Yellow Rust - Sugarcane'
            ]

            # Treatment Measures Dictionary
            treatment_measures = {
                'American Bollworm - Cotton': 'Use insecticides like cypermethrin or natural predators.',
                'Aphid - Wheat': 'Apply neem oil or insecticidal soaps; encourage beneficial insects.',
                'Apple Scab - Apple': 'Use fungicides and ensure proper pruning for air circulation.',
                'Bacterial Blight - Cotton': 'Apply copper-based bactericides and ensure crop rotation.',
                'Bacterial Spot - Pepper, bell': 'Use disease-free seeds and apply fixed copper fungicides.',
                'Bacterial Spot - Tomato': 'Apply streptomycin or copper fungicides; avoid overhead watering.',
                'Black Rot - Apple': 'Remove infected branches and apply sulfur-based sprays.',
                'Black Rot - Grape': 'Prune infected vines and use fungicides containing mancozeb.',
                'Black Rust - Wheat': 'Use resistant varieties and apply fungicides like propiconazole.',
                'Blast - Rice': 'Use seed treatment and apply systemic fungicides like tricyclazole.',
                'Common Rust - Corn': 'Use resistant hybrids and apply fungicides like azoxystrobin.',
                'Ear Rot - Maize': 'Harvest early and store grains in a dry, cool place.',
                'Early Blight - Tomato': 'Apply chlorothalonil or copper fungicides and remove infected leaves.',
                'Healthy - Apple': 'No treatment needed.',
                'Healthy - Cherry': 'No treatment needed.',
                'Healthy - Corn': 'No treatment needed.',
                'Healthy - Cotton': 'No treatment needed.',
                'Healthy - Grape': 'No treatment needed.',
                'Healthy - Maize': 'No treatment needed.',
                'Healthy - Pepper, bell': 'No treatment needed.',
                'Healthy - Potato': 'No treatment needed.',
                'Healthy - Strawberry': 'No treatment needed.',
                'Healthy - Sugarcane': 'No treatment needed.',
                'Healthy - Tomato': 'No treatment needed.',
                'Healthy - Wheat': 'No treatment needed.',
                'Late Blight - Potato': 'Use fungicides like mancozeb and destroy infected plants.',
                'Leaf Blight - Grape': 'Apply sulfur fungicides and prune affected leaves.',
                'Leaf Scorch - Strawberry': 'Improve soil drainage and apply potassium-rich fertilizers.',
                'Mite - Wheat': 'Use acaricides and avoid excessive nitrogen fertilizers.',
                'Mosaic - Sugarcane': 'Destroy infected plants and control aphid vectors.',
                'Powdery Mildew - Cherry': 'Use sulfur-based fungicides and prune to increase air circulation.',
                'Red Bug - Cotton': 'Apply carbaryl or encourage natural predators like spiders.',
                'Stem Borer - Maize': 'Use light traps and neem-based pesticides.',
                'Stem Fly - Wheat': 'Apply systemic insecticides like chlorpyrifos.',
                'Yellow Rust - Sugarcane': 'Use resistant varieties and fungicides like propiconazole.'
            }

            # Display the predicted class and treatment
            predicted_disease = class_name[result_index]
            treatment = treatment_measures.get(predicted_disease, "Treatment information not available.")
            st.success(f"Model is Predicting it's a **{predicted_disease}**")
            st.info(f"**Treatment Measures:** {treatment}")
