from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import user_interface as ui


class HardCodedClassifier:
    def fit(self, data_train, target_train):
        print("Training complete!")

    def predict(self, data_test):
        return [0 for d in data_test]


def main(dataset, split_size):
    # Show the data (the attributes of each instance)
    # print("\nDATA:")
    # print(dataset.data)
    data = dataset.data

    # Show the target values (in numeric format) of each instance
    # print("\nTARGET:")
    # print(dataset.target)
    target = dataset.target

    # Show the actual target names that correspond to each number
    # print("\nTARGET NAMES:")
    # print(dataset.target_names)

    # Split the data into a training set and a testing set
    data_train, data_test, targets_train, targets_test = train_test_split(
        data, target, test_size=split_size)

    # Train the model based on the training data
    classifier = GaussianNB()
    classifier.fit(data_train, targets_train)

    # Model makes a predicition using what it learned from the classifier
    targets_predicted = classifier.predict(data_test)
    # print(targets_test)
    # print(targets_predicted)

    # Get the accuracy of the prediction
    accuracy = accuracy_score(targets_test, targets_predicted)
    print("Accuracy of Gaussian classifier: {}%".format(
        round(accuracy * 100, 2)))

    hc_classifier = HardCodedClassifier()
    hc_classifier.fit(data_train, targets_train)
    hc_pred = hc_classifier.predict(data_test)

    hc_accuracy = accuracy_score(targets_test, hc_pred)
    print("Accuracy of HardCoded classifier: {}%".format(
        round(hc_accuracy * 100, 2)))


if __name__ == "__main__":
    print("USING THE IRIS DATSET: ")
    main(datasets.load_iris(), 0.7)

    print("\nUSING A USER SELECTED DATASET: ")
    dataset = ui.selectDataset()
    split_size = ui.selectDataSplit()
    main(dataset, split_size)
