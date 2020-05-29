# First let's import our libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import torch
import torchvision

# Homework idea: Crowss validation 10 iterasyon yap her iterasyondan %10 test fakat o test datası farklı bölgeden
if __name__ == "__main__":
    # loaded minst database and for start training neigbors
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
    """
    """# taking data from minst and choose percentage for each test data
    (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data.data), data.target, test_size=0.1,
                                                                      random_state=46)"""

    # taking data from MINST and choose percentage for each test data
    (trainData, testData, trainLabels, testLabels) = train_test_split(X_train, example_targets, test_size=0.1)

    (trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels, test_size=0.1)

    print("training data points: {}".format(len(trainLabels)))
    print("validation data points: {}".format(len(valLabels)))
    print("testing data points: {}".format(len(testLabels)))

    # initialize the values of paremeter for our GaussianNB classifier along with the
    # list of accuracies for each value of parameter
    count = 0
    accuracies = []
    accuracies_name = []
    for var_smoothing in np.arange(1e-15, 1e-1, 1e-3):
        modelGNB = GaussianNB(var_smoothing=var_smoothing)
        # Giving train data  of GaussianNB
        modelGNB.fit(trainData, trainLabels)
        # Testing validation data for best parameter
        score = modelGNB.score(valData, valLabels)
        accuracies.append(score)
        accuracies_name.append(var_smoothing)
        count = count + 1

        print("var_smoothing=%.15f, accuracy=%.2f%%" % (var_smoothing, score * 100))

    i = np.argmax(accuracies)
    print("For NaiveBayes name: %s with var_smoothing achieved highest accuracy of %.2f%% on validation data" % (
        str(i), accuracies[i] * 100))

    print(accuracies_name[i])

    model = GaussianNB(var_smoothing=accuracies_name[i])

    model.fit(trainData, trainLabels)
    predictions = model.predict(testData)

    print("EVALUATION ON TESTING DATA for Naïve Bayes")
    print(classification_report(testLabels, predictions))

    print("Confusion matrix for Naïve Bayes")
    print(confusion_matrix(testLabels, predictions))
