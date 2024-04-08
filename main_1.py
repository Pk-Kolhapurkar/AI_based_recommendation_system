import streamlit as st
import mysql.connector
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Define function for feature extraction
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Define function for recommendation
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return True
    except:
        return False

# Function to show dashboard content
def show_dashboard():
    st.header("Fashion Recommender System")

    # Load ResNet model for image feature extraction
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model.trainable = False
    model = tensorflow.keras.Sequential([
        model,
        GlobalMaxPooling2D()
    ])

    feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
    filenames = pickle.load(open('filenames.pkl', 'rb'))

    # File upload section
    uploaded_file = st.file_uploader("Choose an image")
    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file):
            # Display the uploaded image
            display_image = Image.open(uploaded_file)
            st.image(display_image)

            # Feature extraction
            features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)

            # Recommendation
            indices = recommend(features, feature_list)

            # Display recommended products
            col1, col2, col3, col4, col5 = st.beta_columns(5)
            with col1:
                st.image(filenames[indices[0][0]])
            with col2:
                st.image(filenames[indices[0][1]])
            with col3:
                st.image(filenames[indices[0][2]])
            with col4:
                st.image(filenames[indices[0][3]])
            with col5:
                st.image(filenames[indices[0][4]])

        else:
            st.header("Some error occurred in file upload")

# Main Streamlit app
def main():
    # Give title to the app
    st.title("Fashion Recommender System")

    # Show dashboard
    show_dashboard()

# Run the main app
if __name__ == "__main__":
    main()
