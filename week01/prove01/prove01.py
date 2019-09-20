from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


class HardCodedClassifier:
    def fit(self, data_train, target_train):
        print("Training complete!")

    def predict(self, data_test):
        return [0 for d in data_test]


def selectDataset():
    # Continue to get user input until a return statement is reached
    while True:
        user_input = input(
            "Please enter a dataset to use (or type '-list' to see options)> ")

        # list all options
        if user_input == "-list":
            print("Available datasets are \n'iris'" +
                  "\n'digits'\n'wine'\n'breast cancer'")

        # Load a classification dataset from sklearn
        elif user_input == "iris":
            print(user_input)
            return datasets.load_iris()
        elif user_input == "digits":
            return datasets.load_digits()
        elif user_input == "wine":
            return datasets.load_wine()
        elif user_input == "breast cancer":
            return datasets.load_breast_cancer()

        else:
            print("Invalid input. Please try again.")


def main(dataset):
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
        data, target, test_size=0.7)

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
    main(datasets.load_iris())

    print("\nUSING A USER SELECTED DATASET: ")
    dataset = selectDataset()
    main(dataset)
