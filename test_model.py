import tensorflow as tf
import cv2
from keras.utils import img_to_array
import numpy as np

# Load the model
model = tf.keras.models.load_model('/Users/takisvoutos/Desktop/mura_project/model.h5')

# Load and preprocess the image
image = cv2.imread('/Users/takisvoutos/Desktop/mura_project/positive.png')
image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
image = image.astype('float') / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# Make prediction on the image
prediction = model.predict(image)[0]

# Define threshold for classification
threshold = 0.5
# Classify the image based on the prediction and threshold
if prediction >= threshold:
    print("Abnormality detected")
else:
    print("Abnormality not detected")