import tensorflow
from tensorflow.keras.preprocessing import image #module is used for loading and processing images
from tensorflow.keras.layers import GlobalMaxPooling2D   #for dimensionality reduction in convolutional neural networks (CNNs).
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input   #ResNet50 is a pre-trained CNN architecture commonly used for image classification tasks
import numpy as np
from numpy.linalg import norm
import os              #for interacting with the operating system, such as navigating file paths.
from tqdm import tqdm  ###used to display progress bars for iterative tasks.###
import pickle  #which is used for serializing and deserializing Python objects.

# Load pre-trained ResNet50 model

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False  #Freezes the weights of the ResNet50 model so that they are not updated during training.

'''
Loads the ResNet50 model pre-trained on the ImageNet dataset, excluding the top classification layer. 
The input shape of the images is specified as (224, 224, 3)
'''

# Create a Sequential model with ResNet50 as base and GlobalMaxPooling2D layer

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
#####the ResNet50 model is followed by a GlobalMaxPooling2D layer.##############
#print(model.summary())

# Function to extract features from an image using the model
def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)          #
    expanded_img_array = np.expand_dims(img_array, axis=0) #used to prepare the image array for batch processing.In many deep learning frameworks, including TensorFlow and Keras, models are typically trained and evaluated using batches of data rather than individual samples.
    preprocessed_img = preprocess_input(expanded_img_array)#Preprocesses the image array according to the requirements of the ResNet50 model.

    # Extract features using the model and normalize the result

    result = model.predict(preprocessed_img).flatten()
    '''
    Model Prediction (model.predict(preprocessed_img)): This part of the code passes the preprocessed image through the trained model to obtain the output. For feature extraction, the model is typically used up to a certain layer (in this case, before the final classification layer). The output of this prediction step is a feature vector representing the image.

    Flattening (flatten()): The output from the model prediction is usually in the form of a multi-dimensional array or tensor. However, for many applications (including storing features or feeding them into further layers), it's convenient to have a one-dimensional representation of the features. Flattening the output tensor collapses all of its dimensions into a single dimension, resulting in a one-dimensional array.

    For example, if the output tensor has dimensions (batch_size, height, width, channels), flattening it will result in a one-dimensional array of length batch_size * height * width * channels. This makes it easier to handle and manipulate the features, especially when storing them in a database or feeding them into a machine learning model.

    In summary, the line result = model.predict(preprocessed_img).flatten() is used to obtain a one-dimensional feature vector representing the preprocessed image, which can then be further processed or stored for various machine learning tasks such as similarity search, clustering, or classification.
    '''
    normalized_result = result / norm(result)

    return normalized_result

# List all filenames in the 'images' directory

filenames = []  #collect all images from 'images'

for file in os.listdir('images'):
    filenames.append(os.path.join('images',file)) #Appends the file path to the filenames list.

feature_list = []   # Empty list to store extracted features

# Loop through each image file, extract features, and append to the feature_list

for file in tqdm(filenames):   #displaying a progress bar using tqdm
    feature_list.append(extract_features(file,model)) #file means image
# Save the extracted features and filenames to pickle files
pickle.dump(feature_list,open('embeddings.pkl','wb'))   # Save features
pickle.dump(filenames,open('filenames.pkl','wb'))       # save filenames
