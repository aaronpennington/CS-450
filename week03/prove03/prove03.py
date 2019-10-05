from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn import datasets
import pandas as pd
import numpy as np


def car_dataset():
    # CAR EVALUATION DATASET
    # Columns:
    # buying: vhigh, high, med, low.
    # maint: vhigh, high, med, low.
    # doors: 2, 3, 4, 5more.
    # persons: 2, 4, more.
    # lug_boot: small, med, big.
    # safety: low, med, high.
    #
    # No missing data here.

    car_eval_names = ["buying", "maint", "doors",
                      "persons", "lug_boot", "safety", "acceptability"]

    data = pd.read_csv("car.data", header=None, skipinitialspace=True,
                       names=car_eval_names, na_values=["?"])

    # Let's do some label encoding!
    #
    data.buying = data.buying.astype('category')
    data["buying_cat"] = data.buying.cat.codes

    data.maint = data.maint.astype('category')
    data["maint_cat"] = data.maint.cat.codes

    data.doors = data.doors.astype('category')
    data["doors_cat"] = data.doors.cat.codes

    data.persons = data.persons.astype('category')
    data["persons_cat"] = data.persons.cat.codes

    data.lug_boot = data.lug_boot.astype('category')
    data["lug_boot_cat"] = data.lug_boot.cat.codes

    data.safety = data.safety.astype('category')
    data["safety_cat"] = data.safety.cat.codes

    data = data.drop(columns=["buying", "maint", "doors",
                              "persons", "lug_boot", "safety"])

    # Time for some testing
    #
    X = data.drop(columns=["acceptability"]).values
    y = data["acceptability"].values.flatten()
    return X, y


def auto_mpg_dataset():
    # AUTO MPG DATASET
    # Columns:
    # 1. mpg: continuous
    # 2. cylinders: multi-valued discrete
    # 3. displacement: continuous
    # 4. horsepower: continuous
    # 5. weight: continuous
    # 6. acceleration: continuous
    # 7. model year: multi-valued discrete
    # 8. origin: multi-valued discrete
    # 9. car name: string (unique for each instance)
    #
    # Note that the first column is our target!
    # Horsepower has missing rows!!

    mpg_names = ["mpg", "cylinders", "displacement", "horsepower",
                 "weight", "acceleration", "model_year", "origin", "car_name"]
    data = pd.read_csv("auto-mpg.data", header=None, delim_whitespace=True,
                       skipinitialspace=True, names=mpg_names, na_values=["?"])

    # Since only 6 entries of the horsepower attribute are missing,
    # we can safely remove those 6 rows without losing too much
    # data.
    data = data.dropna()

    # There are about a billion different car names, so we will label encode
    # those.
    data.car_name = data.car_name.astype('category')
    data["car_name_cat"] = data.car_name.cat.codes
    data = data.drop(columns=["car_name"])

    X = data.drop(columns=["mpg"]).values
    y = data["mpg"].values.flatten()
    return X, y


def student_dataset():
    # AUTO MPG DATASET
    # Columns:
    # Too many to list... But you can find them here:
    # https://archive.ics.uci.edu/ml/datasets/Student+Performance#
    #
    # The good news is that there is no missing data :)

    # We will use the headers from the file instead of declaring
    # them ourselves.
    data = pd.read_csv("student-mat.csv", header=0,
                       sep=";", na_values=["?"])

    # Better get some ice for this one-HOT encoding!!!
    data = pd.get_dummies(data, columns=["school", "sex", "address",
                                         "famsize", "Pstatus",
                                         "schoolsup", "famsup", "paid",
                                         "activities", "nursery", "higher",
                                         "internet", "romantic"])

    # We're also gonna do some LABEL ENCODING
    data.Mjob = data.Mjob.astype('category')
    data["Mjob_cat"] = data.Mjob.cat.codes

    data.Fjob = data.Fjob.astype('category')
    data["Fjob_cat"] = data.Fjob.cat.codes

    data.reason = data.reason.astype('category')
    data["reason_cat"] = data.reason.cat.codes

    data.guardian = data.guardian.astype('category')
    data["guardian_cat"] = data.guardian.cat.codes

    data = data.drop(columns=["Mjob", "Fjob", "reason",
                              "guardian"])

    # Time for some testing
    #
    X = data.drop(columns=["G3"]).values
    y = data["G3"].values.flatten()
    return X, y


def main():
    # Test the CAR Dataset
    X, y = car_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy of KNN classifier on Car Dataset: {}%".format(
        round(accuracy * 100, 2)))

    # Test the AUTO-MPG dataset
    auto_mpg_X, auto_mpg_y = auto_mpg_dataset()
    am_X_train, am_X_test, am_y_train, am_y_test = train_test_split(
        auto_mpg_X, auto_mpg_y, test_size=0.2)

    regr = KNeighborsRegressor(n_neighbors=3)
    regr.fit(am_X_train, am_y_train)
    am_predictions = regr.predict(am_X_test)

    am_accuracy = r2_score(am_y_test, am_predictions)
    print("Accuracy of KNN Regression on Auto MPG Dataset: {}%".format(
        round(am_accuracy * 100, 2)))

    # STUDENT dataset
    student_X, student_y = student_dataset()
    student_X_train, student_X_test, student_y_train,\
        student_y_test = train_test_split(
            auto_mpg_X, auto_mpg_y, test_size=0.2)

    regr2 = KNeighborsRegressor(n_neighbors=3)
    regr2.fit(student_X_train, student_y_train)
    student_predictions = regr2.predict(student_X_test)

    student_accuracy = r2_score(student_y_test, student_predictions)
    print("Accuracy of KNN Regression on Student Dataset: {}%".format(
        round(student_accuracy * 100, 2)))


if __name__ == "__main__":
    main()
