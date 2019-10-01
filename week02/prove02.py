from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import user_interface as ui
import statistics
from collections import Counter


class HardCodedClassifier:
    def fit(self, data_train, data_test):
        print("Training complete!")

    def predict(self, data_test, data_train, targets_test):
        neighbors = self.knn(3, data_train, data_test, targets_test)
        predictions = []
        for row in targets_test[neighbors]:
            counter = Counter(row)
            predictions.append(counter.most_common(1)[0][0])
        return predictions

    def getDist(self, x, y):
        dist = 0
        for col in x:
            print(col)
            # dist += np.sqrt((x[col] - y[col])**2)
        return dist

    # where x is data_test, y is data_train
    def knn(self, k, train_data, test_data, test_targets):

        nInputs = np.shape(test_data)[0]
        closest = np.zeros(nInputs)

        for n in range(nInputs):
            print(n)
            distances = np.sum((test_data-train_data[n, :])**2, axis=1)

            indices = np.argsort(distances, axis=0)

            classes = np.unique(test_targets[indices[:k]])
            print(len(classes))
            if len(classes) == 1:
                closest[n] = np.unique(classes)
            else:
                counts = np.zeros(max(classes)+1)
                for i in range(k):
                    counts[test_targets[indices[i]]] += 1
                closest[n] = np.max(counts)
        print(len(closest))
        return closest

        # neighbors = []
        # for s in test_data:
        #     distances = []
        #     for r in train_data:
        #         distances.append(self.getDist(s, r))
        #     distances2 = np.array(distances)

        #     sorted_dist = np.argsort(distances2)
        #     smallest = distances2[sorted_dist[:k]]

        #     smallest_index = sorted_dist[:k]
        #     neighbors.append(smallest_index)
        # return neighbors


def main(dataset, split_size):
    # Show the data (the attributes of each instance)
    data = dataset.data

    # Show the target values (in numeric format) of each instance
    target = dataset.target

    # Show the actual target names that correspond to each number
    names = dataset.target_names

    # Split the data into a training set and a testing set
    data_train, data_test, targets_train, targets_test = train_test_split(
        data, target, test_size=split_size)

    hc_classifier = HardCodedClassifier()
    hc_pred = hc_classifier.knn(3, data_train, data_test, targets_test)
    hc_accuracy = accuracy_score(targets_test, hc_pred)
    print("Accuracy of KNN classifier: {}%".format(
        round(hc_accuracy * 100, 2)))

    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(data_train, targets_train)
    knn_pred = knn_classifier.predict(data_test)
    knn_accuracy = accuracy_score(targets_test, knn_pred)
    print("Accuracy of KNN classifier: {}%".format(
        round(knn_accuracy * 100, 2)))


if __name__ == "__main__":
    print("USING THE IRIS DATSET: ")
    main(datasets.load_iris(), 0.7)

    # print("\nUSING A USER SELECTED DATASET: ")
    # dataset = ui.selectDataset()
    # split_size = ui.selectDataSplit()
    # main(dataset, split_size)
