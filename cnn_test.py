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
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer


def get_x_y(path1, path2):
    # Convert jpeg images to array representation
    data = []
    # labels for training
    y = []

    # pre process the no pictures
    for filename in os.scandir(path1):
        if filename.is_file():
            # print(filename.path)
            img = Image.open(filename.path)
            img = img.resize(size=(32, 32))
            img = img.convert('L')
            data.append(img)
            # Label 0 means that there is no tumor detected
            y.append(0)
            # del img

    # Preprocess the yes pictures
    for filename in os.scandir(path2):
        if filename.is_file():
            # print(filename.path)
            img = Image.open(filename.path)
            img = img.resize(size=(32, 32))
            img = img.convert('L')
            data.append(img)
            # Label 1 means that there is a tumor detected
            y.append(1)
            # del img

    # Convert the array of data into a numpy array
    x = np.array(data)
    y = np.array(y)

    return x, y

def split_data(X, y, test_size=0.2):
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size, shuffle=True)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, shuffle=True)

    return X_train, y_train, X_val, y_val, X_test, y_test


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
        df = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False, scoring='accuracy')
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
        df = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False, scoring='accuracy')
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
def calc_accuracy(predictions, labels):
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            correct += 1
    accuracy = correct / len(predictions)
    return accuracy

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


    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Load the training data
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    #
    # # Preprocess the data
    # x_train = x_train.reshape((-1, 28, 28, 1))
    # x_test = x_test.reshape((-1, 28, 28, 1))
    # x_train = x_train.astype('float32') / 255
    # x_test = x_test.astype('float32') / 255
    #
    # # Train the model
    # model.fit(x_train, y_train, epochs=5, batch_size=64)
    # test_loss, test_acc = model.evaluate(x_test, y_test)
    # print('Test loss:', test_loss)
    # print('Test accuracy:', test_acc)
    # predictions = model.predict(x_test)
    # print(predictions)

    predictions = model.predict(x)
    predictions = [int(round(p)) for p in predictions]

    accuracy = calc_accuracy(predictions, y)
    print(accuracy)


if __name__ == "__main__":
    main()
