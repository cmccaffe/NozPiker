import tensorflow as tf
from tensorflow import keras
import pickle
import os
import numpy as np
from NozPiker_Funcs import main as NZ
import pandas as pd
import glob

# Define Data Path
DATA_DIR = os.path.join(os.path.dirname(os.getcwd()),'data/build_data/')

MRC_PATHS = glob.glob(DATA_DIR + '*mrcs')

store = {}

for MRC in MRC_PATHS:
    size = NZ.GetImageSet(MRC,MRC.split('/')[-1].split('_')[0],store)

train_images, train_labels, class_names = NZ.CreateTrainingData(store,size)

pickle.dump(train_images, open(DATA_DIR + "train_images.pkl", "w"))
pickle.dump(train_labels, open(DATA_DIR + "train_labels.pkl", "w"))
pickle.dump(class_names, open(DATA_DIR + "class_names.pkl", "w"))
