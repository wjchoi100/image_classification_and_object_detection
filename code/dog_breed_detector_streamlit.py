# Reference source:
# https://medium.com/mlearning-ai/live-webcam-with-streamlit-f32bf68945a4
# https://discuss.streamlit.io/t/new-component-streamlit-webrtc-a-new-way-to-deal-with-real-time-media-streams/8669/23?page=2
# https://www.youtube.com/watch?v=QelZLAreOdU
# https://docs.streamlit.io/library/api-reference/widgets/st.camera_input
# https://stackoverflow.com/questions/70932538/how-to-center-the-title-and-an-image-in-streamlit

import numpy as np
import streamlit as st
import cv2
import pickle
from PIL import Image
import io
import tempfile
from tensorflow.keras.models import load_model

# import model
model = load_model('xception2.h5')

# import dog breeds
with open('/Users/woojongchoi/Desktop/dsi/capstone/breed_categories.pickle', 'rb') as f:
    loaded_list = pickle.load(f)
class_labels = loaded_list

# title
st.markdown(
    """
    <style>
    .stApp {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("🐾 Dog Breed Detector 🐾")

# background
image = Image.open("background.png")
st.image(image, width=700)

# Providing options
option = st.radio("Choose an option:", ("Option 1: Upload a picture of your dog", "Option 2: Take a picture of your dog using webcam"))

# function to identify dog breed
def classify_dog_breeds(image_path, model, class_labels, top_n=3):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (299, 299))
    img_array = np.array(img)
    img_array_norm = img_array / 255.0

    preds = model.predict(np.expand_dims(img_array_norm, axis=0))
    predicted_class_index = np.argmax(preds)
    predicted_class_label = class_labels[predicted_class_index]
    predicted_class_label = predicted_class_label.replace('_', ' ').title()

    top_predictions = []

    top_indices = np.argsort(preds[0])[::-1][:top_n]
    top_probabilities = preds[0][top_indices]
    top_labels = [class_labels[i] for i in top_indices]

    for i in range(len(top_labels)):
        breed = top_labels[i].replace('_', ' ').title()
        percentage = round(float(top_probabilities[i]), 2)
        top_predictions.append(f'{breed}: {percentage}%')

    return predicted_class_label, top_predictions


# Option 1: Uploaded Image
if option == "Option 1: Upload a picture of your dog":
    uploaded_file = st.file_uploader("Upload a picture of your dog:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(uploaded_file.read())
            image_path = temp_file.name

        predicted_class, predictions = classify_dog_breeds(image_path, model, class_labels, top_n=3)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.image(image, caption='Uploaded Image.', width=500)
        
        st.write("Your dog is a :")
        st.write(predicted_class)
        for prediction in predictions:
            st.write(prediction)
        st.write("________________")


# Option 2: Take a Picture using Webcam
elif option == "Option 2: Take a picture of your dog using webcam":
    webcam_image = st.camera_input(label="Click the 'Take a Picture' button to capture your dog's image:")

    if webcam_image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(webcam_image.read())
            image_path = temp_file.name

        predicted_class, predictions = classify_dog_breeds(image_path, model, class_labels, top_n=3)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.image(image, caption='Picture taken', width=500)
        
        st.write("Your dog is a :")
        st.write(predicted_class)
        for prediction in predictions:
            st.write(prediction)
        st.write("________________")
