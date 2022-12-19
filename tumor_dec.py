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
from tqdm import tqdm


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
            data.append(np.array(img).flatten())
            # Label 0 means that there is no tumor detected
            y.append(0)
            del img

    # Preprocess the yes pictures
    for filename in os.scandir(path2):
        if filename.is_file():
            # print(filename.path)
            img = Image.open(filename.path)
            img = img.resize(size=(32, 32))
            img = img.convert('L')
            data.append(np.array(img).flatten())
            # Label 1 means that there is a tumor detected
            y.append(1)
            del img

    # Convert the array of data into a numpy array
    x = np.array(data)

    return x, y


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
    x, y = get_x_y(r"brain_tumor_dataset/no",
                   r"brain_tumor_dataset/yes")

    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(x, y)
    #
    # pd.set_option('display.max_rows', None)
    #
    # print(X_res)
    # new_shape = np.reshape(x[200], (32, 32))
    # plt.imshow(new_shape)
    # plt.show()

    x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, train_size=0.7, random_state=42)

    five_k_val = best_finder(x_train, y_train)
    print("\n")
    best_test_finder(x_test, y_test, five_k_val)

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

    hist = model.fit(
        train_generator,
        steps_per_epoch=4,
        epochs=8,
        validation_data=validation_generator,
        validation_steps=2
    )

    print("Training data: ", model.evaluate(train_generator))

    print("Testing data: ", model.evaluate(validation_generator))

    # Store the data in X_train, y_train variables by iterating over the batches
    validation_generator.reset()
    x_valid, y_valid = next(validation_generator)
    for i in tqdm(range(int(len(validation_generator) / 32) - 1)):  # 1st batch is already fetched before the for loop.
        img, label = next(validation_generator)
        x_valid = np.append(x_valid, img, axis=0)
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


if __name__ == "__main__":
    main()
