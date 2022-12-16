import numpy as np
import matplotlib.pyplot as plt
import keras
from anaconda_navigator.api.external_apps.detectors import folders
from keras.layers import *
from keras.models import *
from keras.preprocessing import image

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])

# model.summary()

train_datagen = image.ImageDataGenerator(
    rescale = 1./255,
    shear_range= 0.2,
    zoom_range=0.2,
    horizontal_flip= True,
)

test_dataset = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'brain_tumor_dataset/train',
    target_size = (224,224),
    batch_size=32,
    class_mode='binary'
)

# print(train_generator.class_indices)

validation_generator = test_dataset.flow_from_directory(
'brain_tumor_dataset/test',
    target_size = (224,224),
    batch_size=32,
    class_mode='binary'
)

hist = model.fit(
    train_generator,
    steps_per_epoch=4,
    epochs=8,
    validation_data=validation_generator,
    validation_steps=2
)

print("Training data: ", model.evaluate(train_generator))

print("Testing data: ", model.evaluate(validation_generator))

