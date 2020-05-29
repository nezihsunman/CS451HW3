# First let's import our libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import tree
import torch
import torchvision

# Homework idea: Crowss validation 10 iterasyon yap her iterasyondan %10 test fakat o test datası farklı bölgeden
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
    """data = datasets.load_digits()
    # samples in data
    print(data.data.shape)
    print(data.keys())
    # target value?
    print("target")
    print(data.target.shape)
    print(data.feature_names)
    # print(data.DESCR)
    """
    # taking data from minst and choose percentage for each test data
    (trainData, testData, trainLabels, testLabels) = train_test_split(X_train, example_targets, test_size=0.1)

    (trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size=0.1)

    print("training data points: {}".format(len(trainLabels)))
    print("validation data points: {}".format(len(valLabels)))
    print("testing data points: {}".format(len(testLabels)))

    # İnitial Parameter

    criterion = ['gini', 'entropy']
    splitter = ['best', 'random']
    max_depth = range(5, 20, 1)
    ccp_alpha = np.arange(0, 0.01, 1e-2)
    # Start Tree Clasifierr

    accuracies = []
    accuracies_name = []

    for criterionP in criterion:
        for splitterP in splitter:
            for max_depth_value in max_depth:
                for ccp_alpha_value in ccp_alpha:
                    # Initialization of DecisionTreeClassifier
                    clf = tree.DecisionTreeClassifier(max_depth=max_depth_value, splitter=splitterP,
                                                      criterion=criterionP, ccp_alpha=ccp_alpha_value)
                    # Giving train data  of DecisionTreeClassifier
                    clf.fit(trainData, trainLabels)
                    # Testing validation data for best parameter
                    score = clf.score(valData, valLabels)

                    accuracies.append(score)

                    name = 'criterionP: ' + str(criterionP) + ' splitterP: ' + str(
                        splitterP) + ' max_depth_value: ' + str(max_depth_value) + ' ccp_alpha_value: ' + str(
                        ccp_alpha_value)
                    print(name + ' Accuracy= %.2f%% ' % (score * 100))
                    accuracies_name.append(name)

    i = np.argmax(accuracies)
    print("name: %s achieved highest accuracy of %.2f%% on validation data for uniform" % (
        accuracies_name[i], accuracies[i] * 100))

    splitParameter = accuracies_name[i].split()
    model = tree.DecisionTreeClassifier(max_depth=float(splitParameter[5]), splitter=splitParameter[3],
                                        criterion=splitParameter[1], ccp_alpha=float(splitParameter[7]))
    model.fit(trainData, trainLabels)
    predictions = model.predict(testData)

    print("EVALUATION ON TESTING DATA for DecisionTreeClassifier ")
    print(classification_report(testLabels, predictions))

    print("Confusion matrix for DecisionTreeClassifier")
    print(confusion_matrix(testLabels, predictions))
