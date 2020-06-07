import tensorflow as tf
from tensorflow import keras
import pickle
import os
import numpy as np
from NozPiker_Funcs import main as NZ
import pandas as pd
import glob

print("Script is begining")

# Define Data Path
DATA_DIR = os.path.join(os.path.dirname(os.getcwd()),'data/build_data/')

MRC_PATHS = glob.glob(DATA_DIR + '*mrcs')

print(len(MRC_PATHS))

store = {}

for MRC in MRC_PATHS:
    size = NZ.GetImageSet(MRC,MRC.split('/')[-1].split('_')[0],store)

print("All mrcs loaded")

train_images, train_labels, class_names = NZ.CreateTrainingData(store,size)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(size,size)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(len(class_names))
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)
