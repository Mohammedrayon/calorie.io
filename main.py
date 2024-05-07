import numpy as np
import tensorflow as tf
import streamlit as st
import os
import json
from PIL import Image

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/final_food_classification_model.h5"

# load pretrained model
model = tf.keras.models.load_model(model_path)

# loading the class name
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Load the JSON file
with open('calories.json', 'r') as file:
    data = json.load(file)

# function for load_and_preprocess_image
def load_and_preprocess_image(image_path, img_size=(224, 224)):
    # load image
    img = Image.open(image_path)
    # resize image
    img = img.resize(img_size)
    # convert to array
    img_array = np.array(img)
    # adding dimension
    img_array = np.expand_dims(img_array, axis=0)
    # normalize
    img_array = img_array.astype('float32') / 255.
    return img_array


# function for predict the class
def predict_image_class(image_path, model, class_indices, img_size=(224, 224)):
    preprocessed_image = load_and_preprocess_image(image_path, img_size)
    prediction = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    print(predicted_class_index)
    predicted_class = class_indices[str(predicted_class_index)]
    return predicted_class


# streamlit app
st.title("Calorie Estimation")
# upload the image
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((224, 224))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(uploaded_image, model, class_indices)
            if prediction is not None:
                st.success(f'Prediction: {str(prediction)}')
                # Display the JSON data in Streamlit
                if prediction in data:
                    st.write(f"**Calorie of {prediction} per 100g:**")
                    st.write(f"  - Calories: {data[prediction]['calories']}")
                    st.write(f"  - Protein: {data[prediction]['protein']}g")
                    st.write(f"  - Fat: {data[prediction]['fat']}g")
                    st.write(f"  - Carbohydrates: {data[prediction]['carbohydrates']}g")
                else:
                    st.warning(f"No nutritional data found for '{prediction}'.")
            else:
                st.error("Failed to predict. The uploaded image may not contain a food item.")


