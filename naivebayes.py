# First let's import our libraries
import graphviz as graphviz
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold


def read_data_from_file(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint(8).reshape(shape))


# Homework idea: Crowss validation 10 iterasyon yap her iterasyondan %10 test fakat o test datası farklı bölgeden
if __name__ == "__main__":
    # loaded minst database and for start training neigbors

    """raw_train = read_data_from_file("train-images-idx3-ubyte")
    train_data = np.reshape(raw_train, (60000, 28 * 28))
    train__label = read_data_from_file("train-labels-idx1-ubyte")

    raw_test = read_data_from_file("t10k-images-idx3-ubyte")
    test_data = np.reshape(raw_test, (7, 28 * 28))
    test_label = read_data_from_file("t10k-labels-idx1-ubyte")

    idx = (train__label == 2) | (train__label ==3 ) | (train__label == 8)

    X = train_data[idx]
    Y = train__label[idx]
    """

    data = datasets.load_digits()
    # samples in data
    print(data.data.shape)
    print(data.keys())
    # target value?
    print("target")
    print(data.target.shape)
    print(data.feature_names)
    # print(data.DESCR)

    # conver frema to pandas
    data_df = pd.DataFrame(data.data)

    data_df.columns = data.feature_names
    data_df["PRICE"] = data.target

    # taking data from minst and choose percentage for each test data
    (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data.data), data.target, test_size=0.1,
                                                                      random_state=46)

    (trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size=0.1,
                                                                    random_state=84)

    print("training data points: {}".format(len(trainLabels)))
    print("validation data points: {}".format(len(valLabels)))
    print("testing data points: {}".format(len(testLabels)))
    # initialize the values of k for our k-Nearest Neighbor classifier along with the
    # list of accuracies for each value of k
    count = 0
    accuracies = []
    accuracies_name = []
    # var_smoothing = 1e-9
    for var_smoothing in np.arange(1e-9, 1e-5, 1e-9):
        modelGNB = GaussianNB(var_smoothing=var_smoothing)
        # y_prediction = modelGNB.fit(trainData, trainLabels).predict(testData)
        modelGNB.fit(trainData, trainLabels)
        score = modelGNB.score(valData, valLabels)
        accuracies.append(score)
        accuracies_name.append(var_smoothing)
        count = count + 1

    i = np.argmax(accuracies)
    print("name: %s with var_smoothing achieved highest accuracy of %.2f%% on validation data" % (
        str(i), accuracies[i] * 100))

    print(accuracies_name[i])
    print(count)
