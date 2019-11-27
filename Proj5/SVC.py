#!/usr/bin/python
__author__ = "Samuel Steinberg"
__date__ = "November 23rd, 2019"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
import sys
import warnings
warnings.filterwarnings("ignore")

# handles ionosphere dataset
def process_ionosphere(scaler, param_grid):
    data = pd.read_csv("DataFiles\\ionosphere.data", header=None)
    #replace string vals with 1 or 0 identifier
    data[data.columns[-1]] = data[data.columns[-1]].replace("g", 1)
    data[data.columns[-1]] = data[data.columns[-1]].replace("b", 0)
    # normalize and split into train/test/validate
    data_normalized = scaler.fit_transform(data)
    train, test = train_test_split(data_normalized, test_size=0.2)
    train = train[70:]
    valid = train[:70]
    #needs this conversion because sklearn converted it to np array
    _test = pd.DataFrame(data=test[1:, 1:], index=test[1:, 0], columns=test[0, 1:])
    _train = pd.DataFrame(data=train[1:, 1:], index=train[1:, 0], columns=train[0, 1:])
    _valid = pd.DataFrame(data=valid[1:, 1:], index=valid[1:, 0], columns=valid[0, 1:])
    train_X, train_Y = split(_train)
    valid_X, valid_Y = split(_valid)
    test_X, test_Y = split(_test)
    #perform svc and print metrics
    SVC = svm.SVC()
    train_Y = conv_back(train_Y)
    test_Y = conv_back(test_Y)
    valid_Y = conv_back(valid_Y)
    SVC.fit(train_X, train_Y)
    search_grid(SVC, train_X, train_Y, valid_X, valid_Y, param_grid)

#handles sat, same flow as ionosphere
def process_sat(scaler, param_grid):
    train_data = pd.read_csv("DataFiles\\sat.trn", header=None, delim_whitespace=True)
    train_normalized = scaler.fit_transform(train_data)
    actual = np.array((train_data[train_data.columns[-1]]))
    for i in range(len(train_normalized)):
        train_normalized[i][36] = actual[i]
    _train = pd.DataFrame(data=train_normalized[1:, 1:], index=train_normalized[1:, 0],
                          columns=train_normalized[0, 1:])
    train_X, train_Y = split(_train)
    SVC = svm.SVC()
    SVC.fit(train_X, train_Y)
    test_data = pd.read_csv("DataFiles\\sat.tst", header=None, delim_whitespace=True)
    test_normalized = scaler.fit_transform(test_data)
    test_actual = np.array((test_data[test_data.columns[-1]]))
    for i in range(len(test_normalized)):
        test_normalized[i][36] = test_actual[i]
    _test = pd.DataFrame(data=test_normalized[1:, 1:], index=test_normalized[1:, 0],
                         columns=test_normalized[0, 1:])
    test_X, test_Y = split(_test)
    SVC = svm.SVC()
    SVC.fit(test_X, test_Y)
    search_grid(SVC, train_X, train_Y, test_X, test_Y, param_grid)

#handles vowels, same flow as ionosphere
def process_vowels(scaler, param_grid):
    data = pd.read_csv("DataFiles\\vowel-context.data", header=None, delim_whitespace=True)
    data = data.drop([0, 1, 2], axis=1)
    data_normalized = scaler.fit_transform(data)
    actual = np.array((data[data.columns[-1]]))
    for i in range(len(data_normalized)):
        data_normalized[i][10] = actual[i]
    #
    train, test = train_test_split(data_normalized, test_size=0.3)
    train = train[300:]
    valid = train[:300]
    _test = pd.DataFrame(data=test[1:, 1:], index=test[1:, 0], columns=test[0, 1:])
    _train = pd.DataFrame(data=train[1:, 1:], index=train[1:, 0], columns=train[0, 1:])
    _valid = pd.DataFrame(data=valid[1:, 1:], index=valid[1:, 0], columns=valid[0, 1:])
    train_X, train_Y = split(_train)
    valid_X, valid_Y = split(_valid)
    test_X, test_Y = split(_test)
    SVC = svm.SVC()
    SVC.fit(train_X, train_Y)
    search_grid(SVC, train_X, train_Y, valid_X, valid_Y, param_grid)

def search_grid(SVC, train_X, train_Y, valid_X, valid_Y, param_grid):
    scores = ['precision', 'recall']
    for score in scores:
        gs = GridSearchCV(SVC, param_grid, cv=5, scoring='%s_macro' % score, iid=True)
        gs.fit(train_X, train_Y)
        print(gs.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = gs.cv_results_['mean_test_score']
        stds = gs.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, gs.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = valid_Y, gs.predict(valid_X)
        print(classification_report(y_true, y_pred))
        print()

#confusion matrix and metric calculations
def confusion_matrix(predictions, actual, option):
    conf = [0 for i in range(4)]
    print(predictions)
    for i in range(len(predictions)):
        # true negative
        if predictions[i] == actual[i] and actual[i] == 0:
            conf[0] += 1
        # false positive
        if predictions[i] != actual[i] and actual[i] == 1:
            conf[1] += 1
        # false negative
        if predictions[i] != actual[i] and actual[i] == 0:
            conf[2] += 1
        # true positive
        if predictions[i] == actual[i] and actual[i] == 1:
            conf[3] += 1
    if (conf[3] + conf[2]) <= 0:
        TPR = 0
    else:
        TPR = conf[3] / (conf[3] + conf[2])
    if (conf[3] + conf[1]) <= 0:
        PPV = 0
    else:
        PPV = conf[3] / (conf[3] + conf[1])
    if (conf[0] + conf[1]) <= 0:
        TNR = 0
    else:
        TNR = conf[0] / (conf[0] + conf[1])
    if (PPV + TPR) > 0:
        F1Score = 2 * PPV * TPR / (PPV + TPR)
    else:
        F1Score = 0

    accuracy = ((conf[0] + conf[3]) / (conf[0] + conf[3] + conf[2] + conf[1])) * 100
    print("Accuracy: {0:.2f}%".format(accuracy))
    print("TPR: {0:.2f}%".format(TPR * 100))
    print("PPV: {0:.2f}%".format(PPV * 100))
    print("TNR: {0:.2f}%".format(TNR * 100))
    print("F1 Score: {0:.2f}%".format(F1Score * 100))
    conf = np.array(conf).reshape((2, 2))
    conf_df = pd.DataFrame(conf, ["Neg", "Pos"], ["Neg", "Pos"])
    print(conf_df)
    f = _accuracy(predictions, actual)
    print("Accuracy is: ", f)
    plotPredictions(predictions, actual, option)

def _accuracy(predictions, actual):
    count = 0
    for i in range(len(predictions)):
        if predictions[i] == actual[i]:
            count += 1
    return ((count/len(predictions))*100)

def plotPredictions(predictions, actual, option):
    import os
    if option == 1: file_name = "\\ion.png"
    elif option == 2: file_name = "\\vowels.png"
    else: file_name = "\\sat.png"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(range(len(actual)), actual, c='b', label='Actual')
    ax.scatter(range(len(predictions)), predictions, c='r', label='Predicted')
    plt.xlabel("Instances")
    plt.ylabel("Classifier Value")
    plt.title("Predictions vs. Actual")
    plt.legend(loc='best')
    path = "C:\\Users\\Sam\\Documents\\Senior Year\\cs425\\Project4\\BackPropagation\\P5" + "\\Plotting"
    try:
        os.mkdir(path)
        print("Created Directory!")
    except OSError:
        print("Directory exists....Appending file...")
        pass
    fig.savefig(path + file_name)
    plt.close()

#split into prediction matrix and actual classifications
def split(_set):
    x = _set.iloc[:, :-1].values
    y = _set.iloc[:, -1].values
    return x, y

# Convert back values for ionosphere
def conv_back(arr):
    for i in range(len(arr)):
        if arr[i] > 0:
            arr[i] = 1
        else:
            arr[i] = 0
    return arr

# Main, ask for what to run and set up.
if __name__ == '__main__':
    set_name = input("Enter ionosphere, sat, or vowel: ")
    if set_name == "ionosphere" or set_name == "sat" or set_name == "vowel":
        pass
    else:
        sys.stderr.write("Could not resolve option. Exiting...")
        sys.exit(1)

    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)
    scaler = StandardScaler()
    param_grid = {'C': [1, 2, 3, 4, 5, 10, 15, 20, 25, 50],
                  'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
                  'gamma': [.0001, .001, .01, .1, 1, 10, 100]
                  }
    if set_name == "ionosphere":
        process_ionosphere(scaler, param_grid)
    elif set_name == "sat":
        process_sat(scaler, param_grid)
    else:
        process_vowels(scaler, param_grid)