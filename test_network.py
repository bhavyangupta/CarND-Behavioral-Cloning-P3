#!/usr/bin/python 

import numpy as np
import cv2
import csv
import sys

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

folders = ['data', 'corners', 'track2', 'recovery1', 'recovery2', 'recovery3', 'reverse_track_1']

images = []
measurements = []

def readData(folder):
    lines = [] 
    with open(folder + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    
    
    for line in lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        relative_path = folder + '/IMG/' + filename
        image = cv2.cvtColor(cv2.imread(relative_path), cv2.COLOR_BGR2RGB)
        flipped_image = cv2.flip(image, 1)
        images.append(image)
        images.append(flipped_image)
        measurements.append(float(line[3]))
        measurements.append(-1.0 * float(line[3]))

for folder in folders:
    readData(folder)

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
# Normalize 
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape = (160, 320, 3)))

# Convolution 1
model.add(Convolution2D(6, 5, 5, dim_ordering = 'tf', input_shape = (160, 320, 3)))

# Activation 1
model.add(Activation('relu'))

# Max Pooling 1
model.add(MaxPooling2D())

# Convolution 2
model.add(Convolution2D(16, 5, 5, dim_ordering = 'tf'))

# Activation 2
model.add(Activation('relu'))

# Max Pooling 2
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

model.summary()

model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 5)
model.save('trained_model/testNetwork.h5')
