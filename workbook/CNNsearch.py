import tensorflow as tf
from tensorflow import keras
import pickle
import os
import numpy as np
from NozPiker_Funcs import main as NZ
import pandas as pd
import glob
import matplotlib.pyplot as plt
import time

def GetImageSet(mrc,store):
    from mrcfile import open
    name = mrc.split('/')[-1].split('_')[0]
    ID = mrc.split('/')[-1].split('_')[1]
    with open(mrc,'r') as m:
        data = []
        for i in range(m.data.shape[0]):
            data.append(m.data[i,:,:])
    size = len(data[-1][:,0])
    for image in range(len(data)):
        store[(name, ID, image)] = data[image].astype('uint8')
    return size

def CreateTrainingData(store, size, percent_test):
    from numpy import array, where, zeros, min, max, resize, transpose, floor
    from tensorflow_core.image import resize_with_crop_or_pad
    import random
    # Get Dict keys
    class_names = list(set([key[0] for key in store]))
    # Pre-allocate arrays
    length = 30
    shape = (size,size)
    test_length = int(len(store)*percent_test)
    test_set = dict(random.sample(store.items(), test_length))
    train_set = {key: store[key] for key in store if key not in list(test_set.keys())}
    test_images = []
    test_labels = []
    for k, v in test_set.items():
        array = v
        array = resize(array,(array.shape[0],array.shape[1],1))
        array = resize_with_crop_or_pad(array,size,size)
        array = transpose(array,(2,0,1))
        test_images.append(array)
        test_labels.append(class_names.index(k[0]))

    #test_images = np.asarray(test_images)
    test_images = np.concatenate(test_images)
    #test_images = test_images.astype('uint8')
    test_labels = np.asarray(test_labels)
    
    train_images = []
    train_labels = []
    for k, v in train_set.items():
        array = v
        array = resize(array,(array.shape[0],array.shape[1],1))
        array = resize_with_crop_or_pad(array,size,size)
        array = transpose(array,(2,0,1))
        train_images.append(array)
        train_labels.append(class_names.index(k[0]))

    #train_images = np.asarray(train_images)    
    train_images = np.concatenate(train_images)
    #train_images = train_images.astype('uint8')
    train_labels = np.asarray(train_labels)
        
    
    return train_images, train_labels, test_images, test_labels, class_names

# Define Data Path
DATA_DIR = '/home/cns-mccafferty/NozPiker/data/tetts_F15_massSpec/'
MRC_PATHS = glob.glob(DATA_DIR + '*mrcs')

store = {}
MXSZ = 0
for MRC in MRC_PATHS:
    size = GetImageSet(MRC,store)
    if size > MXSZ:
        MXSZ = size
        
train_images, train_labels, test_images, test_labels, class_names = CreateTrainingData(store, MXSZ, 0.3)

train_images = np.asarray(train_images).reshape((7413, 200, 200))
train_images = np.asarray(train_images).reshape((7413, 200, 200, 1))
test_images = np.asarray(test_images).reshape((3177, 200, 200))
test_images = np.asarray(test_images).reshape((3177, 200, 200, 1))


checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(DATA_DIR)),'checkpoints/cp.ckpt')
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


denselayers = [0, 1, 2]
layersizes = [64, 128, 256]
convlayers = [1, 2, 3]

for denselayer in denselayers:
    for layersize in layersizes:
        for convlayer in convlayers:
            NAME = "{}_conv_{}_nodes_{}_dense_{}".format(convlayer, layersize, denselayer, int(time.time()))
            print(NAME)
            
            model = keras.models.Sequential()

            model.add(keras.layers.Conv2D(layersize, (3,3),input_shape=[784,784,1]))
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

            for l in range(convlayer-1):
                model.add(keras.layers.Conv2D(layersize, (3,3)))
                model.add(keras.layers.Activation('relu'))
                model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

            model.add(keras.layers.Flatten())

            for l in range(denselayer):
                model.add(keras.layers.Dense(layersize))
                model.add(keras.layers.Activation('relu'))
                
            model.add(keras.layers.Dense(len(class_names)))
            model.add(keras.layers.Activation('sigmoid'))
            
            tensorboard = tf.keras.callbacks.TensorBoard(log_dir=DATA_DIR+'logs/{}'.format(NAME))
            
            model.compile(optimizer = 'adam',
                  loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics = ['accuracy'])
            
            model.fit(train_images,
              train_labels,
              epochs=5,
              validation_data = (test_images,test_labels),
              callbacks=[cp_callback])
