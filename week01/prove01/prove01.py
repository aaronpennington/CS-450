from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()

# Show the data (the attributes of each instance)
# print("\nDATA:")
# print(iris.data)
data = iris.data

# Show the target values (in numeric format) of each instance
# print("\nTARGET:")
# print(iris.target)
target = iris.target

# Show the actual target names that correspond to each number
# print("\nTARGET NAMES:")
# print(iris.target_names)

# Split the data into a training set and a testing set
data_train, data_test, targets_train, targets_test = train_test_split(
    data, target, test_size=0.7, random_state=0)

# Train the model based on the training data
classifier = GaussianNB()
classifier.fit(data_train, targets_train)

# Model makes a predicition using what it learned from the classifier
targets_predicted = classifier.predict(data_test)
# print(targets_test)
# print(targets_predicted)

# Get the accuracy of the prediction
accuracy = accuracy_score(targets_test, targets_predicted) * 100
print("Accuracy: {}%".format(round(accuracy, 2)))


class HardCodedClassifier:
    def fit(self, data_train, target_train):
        print("Training complete!")

    def predict(self, data_test):
        return [0 for d in data_test]


hc_classifier = HardCodedClassifier()
hc_classifier.fit(data_train, targets_train)
hc_pred = hc_classifier.predict(data_test)

hc_accuracy = accuracy_score(targets_test, hc_pred) * 100
print("Accuracy: {}%".format(round(hc_accuracy, 2)))
