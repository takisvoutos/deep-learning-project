import pandas as pd
import os
import cv2
import numpy as np
import visualkeras
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, BatchNormalization
from keras.layers import Activation,SpatialDropout2D,AvgPool2D
from keras.layers import MaxPool2D,Dropout,GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D,Flatten,Dropout,Dense
from keras.optimizers import Adam
import scipy
from keras.applications import MobileNetV2
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import  confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from keras.utils import img_to_array

train_csv = "/Users/takisvoutos/Desktop/mura_project/MURA-v1.1/train_image_paths.csv"
valid_csv = "/Users/takisvoutos/Desktop/mura_project/MURA-v1.1/valid_image_paths.csv"

# Load the CSV files
train_df = pd.read_csv(train_csv, names=['filename'])
valid_df = pd.read_csv(valid_csv, names=['filename'])

print(train_df.head())

# Add a 'target' column to both the training and validation DataFrames
# If the word 'positive' is in the filename, the x-ray is abnormal (target=1), else normal (target=0)
train_df['label'] = train_df['filename'].apply(lambda x: 1 if 'positive' in x else 0)
valid_df['label'] = valid_df['filename'].apply(lambda x: 1 if 'positive' in x else 0)

def rename(path):
  return os.path.join('/Users/takisvoutos/Desktop/mura_project',path)

train_df['filename'] = train_df['filename'].map(rename)
valid_df['filename'] = valid_df['filename'].map(rename)

print(train_df.dtypes)

train_df = pd.DataFrame({"filename": train_df['filename'], "label": [str(label) for label in train_df['label']]})
valid_df = pd.DataFrame({"filename": valid_df['filename'], "label": [str(label) for label in valid_df['label']]})

print(train_df['label'].value_counts())

print(valid_df['label'].value_counts())

print(train_df.head())

num_epochs = 10

batch_size = 128

validation_ratio = 0.1

h = 224
w = 224
c = 3
image_dim = (h,w,c)

print(image_dim)

class Generators:
    def __init__(self, train_df, valid_df):
        self.batch_size = batch_size
        self.img_size = image_dim

        _train_datagen = ImageDataGenerator(
            rescale = 1/255,
            zoom_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
            fill_mode='nearest'
        )
        self.train_generator = _train_datagen.flow_from_dataframe(
            dataframe=train_df,
            x_col="filename",
            y_col="label",
            class_mode='binary',
            color_mode = 'rgb',
            batch_size=self.batch_size,
            shuffle=True,
            seed=1,
            target_size=self.img_size[:2]
        )
        print('Train generator created')

        _valid_datagen = ImageDataGenerator(rescale = 1/255)
        self.valid_generator = _valid_datagen.flow_from_dataframe(
            dataframe=valid_df,
            x_col="filename",
            y_col='label',
            class_mode='binary',
            color_mode = 'rgb',
            batch_size=self.batch_size,
            shuffle=False,
            seed=1,
            target_size=self.img_size[:2] 
        )
        print('Valid generator created')


generators = Generators(train_df, valid_df)

print(generators.train_generator.next()[0].shape)

plt.imshow(generators.train_generator.next()[0][0], 'gray')

train_df.head()

print(len(train_df['filename']))
print(len(train_df['label']))
print(len(valid_df['filename']))
print(len(valid_df['label']))

import gc

gc.collect()

print(train_df.shape)


# MOBILENET

os.environ['REQUESTS_CA_BUNDLE'] = '/Users/takisvoutos/Downloads/cacert.pem'

# Load the pre-trained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(h, w, c))

# Unfreeze the last few layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Replace Flatten with GlobalAveragePooling to reduce model parameters
x = base_model.output
x = Dropout(0.5)(x)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

# Define the model
model2 = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model2.compile(optimizer=Adam(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Add learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)



print(model2.summary())

vis = visualkeras.layered_view(model2, legend=True)
vis.show()


tf.keras.utils.plot_model(model2, show_shapes=True,)

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)


with tf.device('/device:GPU:0'):
# Train the model
    history = model2.fit( generators.train_generator,
                          validation_data=generators.valid_generator,
                          callbacks=[early_stopping, lr_scheduler],
                          epochs=10,
                        )
    
with tf.device('/device:GPU:0'):
# Train the model
    history = model2.fit( generators.train_generator,
                          validation_data=generators.valid_generator,
                          callbacks=[ lr_scheduler],
                          epochs=10,
                        )

    
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

y_pred = model2.predict(generators.valid_generator)


# Post-process predicted outputs to obtain binary class labels
y_obs = []
for i in y_pred:
    if i > 0.5:
        y_obs.append(1)
    else:
        y_obs.append(0)

# Obtain true labels
y_true = generators.valid_generator.classes

# Print a sample of true and predicted labels
print("True Labels:", y_true[:20])
print("Predicted Labels:", y_obs[:20])

len(generators.valid_generator.classes)

# Calculate accuracy score
accuracy = accuracy_score(y_true, y_obs)
print("Accuracy:", accuracy)

 
# Save the model
model2.save('model.h5')


