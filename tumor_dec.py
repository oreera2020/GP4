import cv2
import keras
import pandas as pd
from IPython.core.display import SVG
from PIL import Image
import os
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing import image
from keras.utils import model_to_dot
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.utils import shuffle
from tabulate import tabulate
from sklearn.preprocessing import LabelBinarizer


def get_x_y(path1, path2):
    # Convert jpeg images to array representation
    # data = []
    # # labels for training
    # y = []

    # # pre process the no pictures
    # for filename in os.scandir(path1):
    #     if filename.is_file():
    #         # print(filename.path)
    #         img = Image.open(filename.path)
    #         img = img.resize(size=(32, 32))
    #         img = img.convert('L')
    #         data.append(np.array(img).flatten())
    #         # Label 0 means that there is no tumor detected
    #         y.append(0)
    #         del img
    #
    # # Preprocess the yes pictures
    # for filename in os.scandir(path2):
    #     if filename.is_file():
    #         # print(filename.path)
    #         img = Image.open(filename.path)
    #         img = img.resize(size=(32, 32))
    #         img = img.convert('L')
    #         data.append(np.array(img).flatten())
    #         # Label 1 means that there is a tumor detected
    #         y.append(1)
    #         del img
    #
    # # Convert the array of data into a numpy array
    # x = np.array(data)

    data = []
    # labels for training
    y = []

    # pre process the no pictures
    for filename in os.scandir(path1):
        if filename.is_file():
            # print(filename.path)
            img = cv2.imread(filename.path)
            img = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
            img = img / 255.0
            data.append(img)
            # Label 0 means that there is no tumor detected
            y.append(0)

    # Preprocess the yes pictures
    for filename in os.scandir(path2):
        if filename.is_file():
            # print(filename.path)
            # print(filename.path)
            img = cv2.imread(filename.path)
            img = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
            img = img / 255.0
            data.append(img)
            # Label 1 means that there is a tumor detected
            y.append(1)

    # Convert the array of data into a numpy array
    x = np.array(data)
    y = np.array(y)

    x, y = shuffle(x, y)

    return x, y


def split_data(X, y, test_size=0.2):
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size, shuffle=True)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, shuffle=True)

    return X_train, y_train, X_val, y_val, X_test, y_test


def cnn():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=(240, 240, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def best_finder(x_data, y_data):
    model_params = {
        'Artificial Neural Network': {
            'model': MLPClassifier(max_iter=700),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (125,), (150,)],
                'learning_rate': ['constant', 'invscaling', 'adaptive']
            }
        },
        'Support Vector Machine': {
            'model': svm.SVC(),
            'params': {
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
            }
        },
        'Naive Bayes': {
            'model': GaussianNB(),
            'params': {
                'priors': [None],
                'var_smoothing': [0, .2, .4, .6, .8, 1, 1e-9]
            }
        },
        'AdaBoost': {
            'model': AdaBoostClassifier(n_estimators=100, random_state=0),
            'params': {
                'algorithm': ['SAMME', 'SAMME.R'],
                'learning_rate': [0.1, 0.2, 0.3, 0.4, 1, 2],
                'n_estimators': [1, 10, 50, 100]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [50, 100, 250],
                'min_samples_leaf': [1, 50, 100]
            }
        },
    }

    scores = []
    best_par = []
    for model_name, mp in model_params.items():
        df = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False, scoring='f1')
        df.fit(x_data, y_data)
        best_par.append(df.best_params_)
        scores.append({
            'model': model_name,
            'best_params': df.best_params_,
            'top_score': df.best_score_,
        })

    display = pd.DataFrame(scores)
    print(tabulate(display, headers=["ML Trained Model", "Its Best Set of Parameters", "Its F1-Score on "
                                                                                       "the 5-fold Cross "
                                                                                       "Validation on "
                                                                                       "Training Data"]))
    return best_par


def best_test_finder(x_test, y_test, best_params):
    model_params = {
        'Artificial Neural Network': {
            'model': MLPClassifier(max_iter=700),
            'params': {
                'hidden_layer_sizes': [best_params[0]['hidden_layer_sizes']],
                'learning_rate': [best_params[0]['learning_rate']]
            }
        },
        'Support Vector Machine': {
            'model': svm.SVC(),
            'params': {
                'kernel': [best_params[1]['kernel']]
            }
        },
        'Naive Bayes': {
            'model': GaussianNB(),
            'params': {
                'priors': [best_params[2]['priors']],
                'var_smoothing': [best_params[2]['var_smoothing']]
            }
        },
        'AdaBoost': {
            'model': AdaBoostClassifier(n_estimators=100, random_state=0),
            'params': {
                'algorithm': [best_params[3]['algorithm']],
                'learning_rate': [best_params[3]['learning_rate']],
                'n_estimators': [best_params[3]['n_estimators']]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [best_params[4]['n_estimators']],
                'min_samples_leaf': [best_params[4]['min_samples_leaf']]
            }
        },
    }

    scores = []
    for model_name, mp in model_params.items():
        df = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False, scoring='f1')
        df.fit(x_test, y_test)
        scores.append({
            'model': model_name,
            'best_params': df.best_params_,
            'top_score': df.best_score_,
        })

    display = pd.DataFrame(scores)
    # print(best_par)
    print(tabulate(display, headers=["ML Trained Model", "Its Best Set of Parameters", "Its F1-Score on Testing Data"]))

    best_model = scores[0]
    for model in scores[1:]:
        if model['top_score'] > best_model['top_score']:
            best_model = model

    print("\nBased on the F1-scores from the table above, the " + best_model['model'] + " Model should be chosen to "
                                                                                        "predict machine failure for future data.")


# Print Image from directory in SciView
# img = mpimg.imread(r'C:\Users\alexa\Documents\Senior Year\B Term\Intro to AI\GP4\brain_tumor_dataset\no\1 no.jpeg')
# imgplot = plt.imshow(img)
# plt.show()

def main():
    print("A")
    x, y = get_x_y(r"/Users/sultanadedeji/PycharmProjects/Fall2022CS4341/GP4/brain_tumor_dataset/no",
                   r"/Users/sultanadedeji/PycharmProjects/Fall2022CS4341/GP4/brain_tumor_dataset/yes")

    # rus = RandomUnderSampler(random_state=42)
    # X_res, y_res = rus.fit_resample(x, y)
    #
    # pd.set_option('display.max_rows', None)
    #
    # print(X_res)
    # new_shape = np.reshape(x[200], (32, 32))
    # plt.imshow(new_shape)
    # plt.show()

    # x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, train_size=0.7, random_state=42)

    # five_k_val = best_finder(x_train, y_train)
    # print("\n")
    # best_test_finder(x_test, y_test, five_k_val)

    print("A")
    x_train, x_test, x_val, y_val, y_train, y_test = split_data(x, y, test_size=0.2)
    our_cnn = cnn()
    our_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # our_cnn.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_val, y_val))
    #
    # SVG(model_to_dot(our_cnn, show_shapes=True).create(prog='dot', format='svg'))
    # score = our_cnn.evaluate(x_test, y_test, verbose=0)
    # print('\n', 'Test accuracy:', score[1])
    train_datagen = image.ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    test_dataset = image.ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        'brain_tumor_dataset/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    # print(train_generator.class_indices)

    validation_generator = test_dataset.flow_from_directory(
        'brain_tumor_dataset/test',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    hist = our_cnn.fit(
        train_generator,
        steps_per_epoch=4,
        epochs=8,
        validation_data=validation_generator,
        validation_steps=2
    )

    print("Training data: ", our_cnn.evaluate(train_generator))

    print("Testing data: ", our_cnn.evaluate(validation_generator))


if __name__ == "__main__":
    main()
