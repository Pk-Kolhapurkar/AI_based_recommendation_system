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
from chatbot import Chatbot  # Assuming you have a chatbot module

# Function to authenticate user credentials from the database
def authenticate_user(username, password):
    try:
        # Connect to the MySQL database
        connection = mysql.connector.connect(
            host="sql6.freemysqlhosting.net",
            database="sql6697353",
            user="sql6697353",
            password="wvfSQJbmMs"
        )

        # Create a cursor object to execute SQL queries
        cursor = connection.cursor() #cursor allows you to execute SQL queries on the connected database.

        # Query to check if the username and password match
        query = "SELECT * FROM users WHERE username = %s AND password = %s"  ##The %s placeholders are used for parameterized queries to prevent SQL injection.
        cursor.execute(query, (username, password))
        user = cursor.fetchone()  # Fetch the first row whrere the data matched

        # If user exists, return True (authentication successful)
        if user:
            return True
        else:
            return False

    except mysql.connector.Error as error:
        st.error(f"Error: {error}")
        return False

    finally:
        # Close the cursor and database connection
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()

# Define your login function
def login():
    st.header("Login Page")
    # Add your login code here
    username = st.text_input("Username")
    password = st.text_input("Password", type="password") #type="password" argument to hide the entered characters.
    if st.button("Login"):
        # Authenticate user with provided credentials
        if authenticate_user(username, password): # if user exist run below code
            st.session_state.logged_in = True     #Session state allows you to persist information across Streamlit sessions.
            query_params = st.experimental_get_query_params()  #Query parameters are typically used to pass information between different pages or components in a web application.
            query_params["page"] = ["Dashboard"]
            st.experimental_set_query_params(**query_params)
            # Redirect to dashboard
            st.success("Login successful!") #displays a success message 
        else:
            st.error("Invalid username or password")

# Define your registration function
def register():
    st.header("Registration Page")
    # Add your registration code here
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    email = st.text_input("Email")

    if st.button("Register"):
        # Validate input
        if not username or not password or not email:
            st.error("Please enter username, password, and email.")
        else:
            try:
                # Connect to the MySQL database
                connection = mysql.connector.connect(
                    host="sql6.freemysqlhosting.net",
                    database="sql6697353",
                    user="sql6697353",
                    password="wvfSQJbmMs"
                )

                # Create a cursor object to execute SQL queries
                cursor = connection.cursor()

                # Check if username or email already exists
                cursor.execute("SELECT * FROM users WHERE username = %s OR email = %s", (username, email))
                result = cursor.fetchone() fech fitch first row 
                if result:
                    st.error("Username or email already exists. Please choose different ones.")
                else:
                    # Insert new user into database
                    cursor.execute("INSERT INTO users (username, password, email) VALUES (%s, %s, %s)", (username, password, email))
                    connection.commit() #Committing the transaction makes the changes permanent in the database.
                    st.success("Registration successful. You can now log in.")
            except mysql.connector.Error as error:
                st.error(f"Error: {error}")
            finally:
                # Close the cursor and database connection
                if 'connection' in locals() and connection.is_connected():
                    cursor.close()
                    connection.close()


# Define function for feature extraction
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)  #This line adds an extra dimension to the image array. It's often necessary because deep learning models typically expect input data in batches. By adding an extra dimension at the beginning, we convert the image data into a batch with a single image.
    preprocessed_img = preprocess_input(expanded_img_array)  #. The preprocess_input function prepares the image data in the format expected by the model.
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Define function for recommendation
def recommend(features, feature_list):  ##takes two parameters: features (the features extracted from the uploaded image) and feature_list (a list of features of all products in the database)
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')  ##Brute-force search is simple to implement and conceptually easy to understand. It exhaustively searches through all data points to find the nearest neighbors. While this approach may be computationally expensive for large datasets, it ensures accuracy because it considers all points without any assumptions about the underlying data distribution.  NO hyperparameters tunning neeeded .
    neighbors.fit(feature_list)  fit model on features of all images
    distances, indices = neighbors.kneighbors([features]) ##This method returns the distances and indices of the nearest neighbors of features in feature_list.
    return indices

# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f: #The with statement ensures that the file is properly closed after writing, even if an error occurs during the write operation.
            f.write(uploaded_file.getbuffer())  # writes the contents of the uploaded file (retrieved using uploaded_file.getbuffer()) to the opened file 'f'.
        return True
    except:
        return False

# Function to show dashboard content
def show_dashboard():
    st.header("Fashion Recommender System")
    chatbot = Chatbot()
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

    # Chatbot section
    user_question = st.text_input("Ask a question:")
    if user_question:
        bot_response, recommended_products = chatbot.generate_response(user_question)
        st.write("Chatbot:", bot_response)

        # Display recommended products
        for result in recommended_products:
            pid = result['corpus_id']
            product_info = chatbot.product_data[pid]
            st.write("Product Name:", product_info['productDisplayName'])
            st.write("Category:", product_info['masterCategory'])
            st.write("Article Type:", product_info['articleType'])
            st.write("Usage:", product_info['usage'])
            st.write("Season:", product_info['season'])
            st.write("Gender:", product_info['gender'])
            st.image(chatbot.images[pid])

# Main Streamlit app
def main():
    # Give title to the app
    st.title("Fashion Recommender System")

    # Sidebar navigation
    page = st.sidebar.radio("Navigation", ["Login", "Register", "Dashboard"])

    # Check if user is logged in
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # Show login page if user is not logged in
    if not st.session_state.logged_in:
        if page == "Login":
            login()

    # Show registration page if selected in the sidebar
    if page == "Register":
        register()

    # Show dashboard if selected in the sidebar and user is logged in
    if page == "Dashboard" and st.session_state.logged_in:
        show_dashboard()

# Run the main app
if __name__ == "__main__":
    main()
