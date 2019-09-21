from sklearn import datasets


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


def selectDataSplit():
    while True:
        user_input = float(input("Enter the amount of data to train > "))
        if user_input < 0.9 and user_input > 0:
            return user_input
        else:
            print("Please enter an amount between 0 and 0.9")
