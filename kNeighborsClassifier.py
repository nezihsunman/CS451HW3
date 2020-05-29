# First Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

import torch
import torchvision

if __name__ == "__main__":

    # loaded minst database and for start training neigbors with API
    n_epochs = 3
    batch_size_train = 60000
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)
    data = enumerate(train_loader)
    batch_idx, (example_data, example_targets) = next(data)
    print(example_data.shape)

    X_train = example_data.reshape(batch_size_train, 1 * 28 * 28)

    # loaded minst database and for start training neigbors
    # data = datasets.load_digits()
    """# samples in data
    print(data.data.shape)
    print(data.keys())
    # target value?
    print("target")
    print(data.target.shape)
    print(data.feature_names)
    """
    """# taking data from minst and choose percentage for each test data
        (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data.data), data.target, test_size=0.1,
                                                                          random_state=46)"""

    # split date for train test and validation
    (trainData, testData, trainLabels, testLabels) = train_test_split(X_train,
                                                                      example_targets,
                                                                      test_size=0.1)

    (trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size=0.1)

    # print train validate and test data

    print("training data points: {}".format(len(trainLabels)))
    print("validation data points: {}".format(len(valLabels)))
    print("testing data points: {}".format(len(testLabels)))

    # initialize the values of k for our k-Nearest Neighbor classifier along with the
    # list of accuracies for each parameter

    kVals = range(1, 10, 1)

    accuracies = []
    accuraciesName = []
    # Distance better than uniform that's why comment out
    # uniformList = ['uniform', 'distance']

    uniformList = ['distance']
    # All algorithm is same
    # algorithmList = ['auto', 'ball_tree', 'kd_tree', 'brute']
    algorithmList = ['auto']
    # loop over various parameters for the k-Nearest Neighbor classifier
    leaf_size = range(1, 2, 1)

    for k in kVals:
        for leaf in leaf_size:
            for weights in uniformList:
                # train the k-Nearest Neighbor classifier with the current value of `k`
                model = KNeighborsClassifier(n_neighbors=k, weights=weights, leaf_size=leaf)
                # Giving train data  of k-Nearest Neighbor classifier
                model.fit(trainData, trainLabels)
                # Testing validation data for best parameter
                score = model.score(valData, valLabels)
                print("k=%d, accuracy=%.2f%% for %s and %s" % (k, score * 100, weights, leaf))
                name = 'k: ' + str(k) + ' weights: ' + str(weights) + ' leaf_size: ' + str(leaf)

                accuracies.append(score)
                accuraciesName.append(name)

    # find the value of k that has the largest accuracy
    i = np.argmax(accuracies)
    print("name: %s achieved highest accuracy of %.2f%% on validation data for uniform" % (
        accuraciesName[i], accuracies[i] * 100))

    # test data
    splitParameter = accuraciesName[i].split()
    model = KNeighborsClassifier(n_neighbors=int(splitParameter[1]), weights=splitParameter[3],
                                 leaf_size=int(splitParameter[5]))
    model.fit(trainData, trainLabels)
    predictions = model.predict(testData)

    print("EVALUATION ON TESTING DATA for KNN")
    print(classification_report(testLabels, predictions))

    print("Confusion matrix for KNN")
    print(confusion_matrix(testLabels, predictions))
