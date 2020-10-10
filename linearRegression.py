# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score


# prediction function for testing dataset
def get_regression_prediction(input_features, intercept, slop):
    predicted_values = input_features * slop + intercept

    return predicted_values


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # read scv file
    # check first five rows
    data = pd.read_csv("FuelConsumptionCo2.csv")
    data.head()

    # select some features from the table
    data = data[["ENGINESIZE", "CO2EMISSIONS"]]

    # visualize the data on a scatter plot
    # ENGINESIZE vs CO2EMISSIONS
    plt.scatter(data["ENGINESIZE"], data["CO2EMISSIONS"], color="blue")
    plt.xlabel("ENGINESIZE")
    plt.ylabel("CO2EMISSIONS")
    plt.show()

    # Divide the data into training and testing data
    # 80% for training
    train = data[:int(len(data) * 0.8)]
    test = data[int(len(data) * 0.8):]

    # Training the model
    regr = linear_model.LinearRegression()

    train_x = np.array(train[["ENGINESIZE"]])
    train_y = np.array(train[["CO2EMISSIONS"]])

    regr.fit(train_x, train_y)

    # the coefficients
    # slop
    print("coefficients : ", regr.coef_)
    # intercept
    print("intercept :", regr.intercept_)

    # Based on the coefficients plotting the regression line
    plt.scatter(train["ENGINESIZE"], train["CO2EMISSIONS"], color="blue")
    plt.plot(train_x, regr.coef_ * train_x + regr.intercept_, '-r')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.show()

    # predicting emission for future car
    new_engine_size = 3.5
    estimated_emission = get_regression_prediction(new_engine_size, regr.intercept_[0], regr.coef_[0][0])
    print("Estimate emission :", estimated_emission)

    # checking accuracy
    test_x = np.array(test[["ENGINESIZE"]])
    test_y = np.array(test[["CO2EMISSIONS"]])
    test_y_ = regr.predict(test_x)

    print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
    print("Mean sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
    print("R2-score: %.2f" % r2_score(test_y_, test_y))

