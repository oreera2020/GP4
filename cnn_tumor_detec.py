import os
import random

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import keras
from anaconda_navigator.api.external_apps.detectors import folders
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
from tqdm import tqdm

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

# Store the data in X_train, y_train variables by iterating over the batches
validation_generator.reset()
x_valid, y_valid = next(validation_generator)
for i in tqdm(range(int(len(validation_generator)/32)-1)): #1st batch is already fetched before the for loop.
  img, label = next(validation_generator)
  x_valid = np.append(x_valid, img, axis=0 )
  y_valid = np.append(y_valid, label, axis=0)
print(x_valid.shape, y_valid.shape)

labels = ["No", "Yes"]
y_hat = model.predict(x_valid)
no_of_indices = 15
random_indices = np.random.choice(
    x_valid.shape[0], size=no_of_indices, replace=False)
# Plot a random sample of 15 test images, with their predicted labels and ground truth
figure = plt.figure(figsize=(20, 13))
sub_title = "Random samples of 15 test images, with their predicted labels and ground truth"
figure.suptitle(sub_title, fontsize=20)
for i in range(no_of_indices):
    rand_index = random_indices[i]

    # Display each image
    ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_valid[rand_index]))

    # Set the title for each image
    prediction_val = y_hat[rand_index][0]
    predict_index = 0 if (prediction_val < 0.5) else 1
    true_index = y_valid[rand_index]
    prediction = labels[predict_index]
    truth = labels[int(true_index)]
    title_color = "blue" if predict_index == true_index else "red"
    ax_title = "Prediction: {} ({:.2f})\nGround Truth: {}".format(
        prediction, prediction_val, truth)
    ax.set_title(ax_title, color=title_color)
plt.show()