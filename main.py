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

    """" 10 farklı parçaya bölüm her bir iterasyonda farklı bir kısma test uygulamaya çalıştım

    test_data = np.array(data.data)

    kf = KFold(n_splits=10)
    # for train_index, test_index in kf.split(np.array(data.data)):
    # print("TRAIN:", train_index, "TEST:", test_index)
    print(kf)


    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
    sss.get_n_splits(test_data)
    for train_index, test_index in sss.split(test_data):
        X_train, X_test = test_data[train_index], test_data[test_index]
        y_train, y_test = test_data[train_index], test_data[test_index]
    """

    # show the sizes of each data split

    print("training data points: {}".format(len(trainLabels)))
    print("validation data points: {}".format(len(valLabels)))
    print("testing data points: {}".format(len(testLabels)))
    # initialize the values of k for our k-Nearest Neighbor classifier along with the
    # list of accuracies for each value of k

    kVals = range(1, 20, 1)
    accuracies = []
    accuraciesName = []
    uniformList = ['uniform', 'distance']
    algorithmList = ['auto', 'ball_tree', 'kd_tree', 'brute']
    # loop over various values of `k` for the k-Nearest Neighbor classifier

    for k in range(1, 15, 1):

        for weights in uniformList:
            for algorithm in algorithmList:
                # train the k-Nearest Neighbor classifier with the current value of `k`

                model = KNeighborsClassifier(n_neighbors=k, weights=weights, algorithm=algorithm)
                model.fit(trainData, trainLabels)
                # model.fit(X, Y)
                # evaluate the model and update the accuracies list
                score = model.score(valData, valLabels)
                print("k=%d, accuracy=%.2f%% for %s and %s" % (k, score * 100, weights, algorithm))
                name = 'k: ' + str(k) + ' weights: ' + str(weights) + ' algorithm: ' + str(algorithm);

                accuracies.append(score)
                accuraciesName.append(name)

    # find the value of k that has the largest accuracy

    i = np.argmax(accuracies)
    print("name: %s achieved highest accuracy of %.2f%% on validation data for uniform" % (
        accuraciesName[i], accuracies[i] * 100))

    # re-train our classifier using the best k value and predict the labels of the
    # test data
    splitParameter = accuraciesName[i].split()
    model = KNeighborsClassifier(n_neighbors=int(splitParameter[1]), weights=splitParameter[3],
                                 algorithm=splitParameter[5])
    model.fit(trainData, trainLabels)
    predictions = model.predict(testData)
    # print(predictions[1])

    # show a final classification report demonstrating the accuracy of the classifier
    # for each of the digits

    print("EVALUATION ON TESTING DATA")
    print(classification_report(testLabels, predictions))

    print("Confusion matrix")
    print(confusion_matrix(testLabels, predictions))

    # loop over a few random digits

    for i in np.random.randint(0, high=len(testLabels), size=(5,)):
        # grab the image and classify it
        image = testData[i]
        prediction = model.predict([image])[0]
        # convert the image for a 64-dim array to an 8 x 8 image compatible with OpenCV,
        # then resize it to 32 x 32 pixels so we can see it better
        ##         image = image.reshape((64, 64))
        ##         image = exposure.rescale_intensity(image, out_range=(0, 255))
        ##         image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)

        # show the prediction

        imgdata = np.array(image, dtype='float')
        pixels = imgdata.reshape((8, 8))
        plt.imshow(pixels, cmap='gray')
        plt.annotate(prediction, (3, 3), bbox={'facecolor': 'white'}, fontsize=16)
        print("i think tha digit is : {}".format(prediction))
        # cv2.imshow("image", image)
        plt.show()
    # cv2.waitKey(0)

    # END of knn

    # Start bayesin network
    modelGNB = GaussianNB()
    y_prediction = modelGNB.fit(trainData, trainLabels).predict(testData)

    print("For Bayesian Number of mislabeled points out of a total %d points : %d" % (
        testData.shape[0], (testLabels != y_prediction).sum()))

    # Start Tree Clasifierr

    clf = tree.DecisionTreeClassifier()
    y_prediction_for_tree = clf.fit(trainData, trainLabels).predict(testData)
    print("FOR TREE Number of mislabeled points out of a total %d points : %d" % (
        testData.shape[0], (testLabels != y_prediction_for_tree).sum()))
