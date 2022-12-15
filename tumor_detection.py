import pandas as pd
from PIL import Image
import os
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import image as mpimg, pyplot as plt
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from tabulate import tabulate
from PIL import Image
import numpy as np


# def augment(x, y):
#
#     return x, y


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
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'C': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
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
                'kernel': [best_params[1]['kernel']],
                'C': [best_params[1]['C']]
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


def main():
    x, y = get_x_y(r"brain_tumor_dataset\no",
                   r"brain_tumor_dataset\yes")

    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(x, y)

    # pd.set_option('display.max_rows', None)
    # print(X_res)

    # Print Image from directory in SciView
    # img = mpimg.imread(r'C:\Users\alexa\Documents\Senior Year\B Term\Intro to AI\GP4\brain_tumor_dataset\no\1 no.jpeg')
    # imgplot = plt.imshow(img)
    # plt.show()

    new_shape = np.reshape(x[200],(32,32))
    plt.imshow(new_shape)
    plt.show()

    x_train, x_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, train_size=0.7, random_state=42)

    five_k_val = best_finder(x_train, y_train)
    print("\n")
    best_test_finder(x_test, y_test, five_k_val)


if __name__ == "__main__":
    main()
