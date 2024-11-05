import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Auxiliary functions
# The linear regression functions compute the value of w which minimise the error (demonstration in the slides)

def linear_regression(x, t):
        w_num=0
        w_denom=0
        for i in range(len(x)):
            w_num += x[i]*t[i]
            w_denom += x[i]**2
        w = w_num / w_denom
        return w

def linear_regression_intercept(x, t):
    x_bar = np.mean(x)
    t_bar = np.mean(t)
    w1_num=0
    w1_denom=0
    for i in range(len(x)):
        w1_num += (x[i] - x_bar) * (t[i] - t_bar)
        w1_denom += (x[i] - x_bar) ** 2
    w1 = w1_num / w1_denom
    w0 = t_bar - w1 * x_bar
    return w1, w0

def multi_dimensional_linear_regression_intercept(X, t):
    X = np.c_[np.ones(X.shape[0]), X]
    X = X.astype(float)
    W= np.linalg.pinv(X) @ t.T
    return W

def mean_squared_error(y, t):
    return np.mean((y - t) ** 2)


### Main function 

def main(pourcentage_subset = 0.1, n_subset = 5):
    ### Turkish data
    # TASK 1 : Get data
    turkish_data = pd.read_csv("turkish-se-SP500vsMSCI.csv", header=None)

    # Column 0: SP500 (predictor)
    # Column 1: MSCI (target)
    x_turkish = turkish_data[0].values  # Predictor variable (SP500)
    t_turkish = turkish_data[1].values    # Target variable (MSCI)

    # TASK 2.1: One-dimensional regression without intercept on Turkish stock data
    plt.figure(figsize=(10, 6))
    L_err_train_turkish = []
    L_err_test_turkish = []

    for i in range(n_subset):  # Compare n_subset random subsets
        randomSubset = np.random.permutation(len(x_turkish))[:round(pourcentage_subset * len(x_turkish))]
        x_subset = x_turkish[randomSubset]
        t_subset = t_turkish[randomSubset]
        
        # Fit model without intercept
        w = linear_regression(x_subset, t_subset)
        
        # TASK 2.2 Plot subset data and regression line
        plt.scatter(x_subset, t_subset, alpha=0.5, label="Data Subset")
        plt.plot(x_subset, w*x_subset, label="Subset Fit", linewidth=1)
        L_err_train_turkish.append(mean_squared_error(t_subset, w*x_subset))
        L_err_test_turkish.append(mean_squared_error(t_turkish, w*x_turkish))

    plt.xlabel("SP500")
    plt.ylabel("MSCI")
    plt.title("Linear Fit on Random {}% Subsets of Turkish Data (No Intercept)".format(int(pourcentage_subset*100)))
    plt.legend()
    plt.show()

    ### TASK 2.3: One-dimensional problem with intercept on the Motor Trends car data, using columns mpg and weight

    car_data = pd.read_csv("mtcarsdata-4features.csv")
    car_data = car_data.values

    # Column 4: weight (predictor)
    # Column 1: mpg (target)
    x_car = car_data[:, 4]
    t_car = car_data[:, 1]

    plt.figure(figsize=(10, 6))
    L_err_train_lin_car = []
    L_err_test_lin_car = []

    for i in range(n_subset):  # Compare five random subsets
        randomSubset = np.random.permutation(len(x_car))[:round(pourcentage_subset * len(x_car))]
        x_subset = x_car[randomSubset]
        t_subset = t_car[randomSubset]
        
        # Fit model without intercept
        w1,w0 = linear_regression_intercept(x_subset, t_subset)
        
        plt.scatter(x_subset, t_subset, alpha=0.5, label="Data Subset")
        plt.plot(x_car, w1*x_car + w0, label="Subset Fit", linewidth=1)
        L_err_train_lin_car.append(mean_squared_error(t_subset, w1*x_subset + w0))
        L_err_test_lin_car.append(mean_squared_error(t_car, w1*x_car + w0))

    plt.xlabel("weight")
    plt.ylabel("mpg")
    plt.title("Linear Fit on Random {}% Subsets of car Data with intercept)".format(int(pourcentage_subset*100)))
    plt.legend()
    plt.show()

    # TASK 2.4 Multi-dimensional problem on the complete MTcars data, using all four columns (predict mpg with the other three columns)

    # Column 2,3,4: disp, hp, weight (predictor)
    # Column 2: mpg (target)
    x_car = car_data[:, 2:5]
    t_car = car_data[:, 1]

    L_err_train_multi_car = []
    L_err_test_multi_car = []

    for i in range(n_subset): 
        randomSubset = np.random.permutation(len(t_car))[:round(pourcentage_subset * len(t_car))]
        x_subset = x_car[randomSubset][:]
        t_subset = t_car[randomSubset]
        
        W = multi_dimensional_linear_regression_intercept(x_subset, t_subset)

        # Add the first column of ones to the x_subset matrix
        X_subset = np.c_[np.ones(x_subset.shape[0]), x_subset]
        X_car = np.c_[np.ones(x_car.shape[0]), x_car]

        L_err_train_multi_car.append(mean_squared_error(t_subset, X_subset@W))
        L_err_test_multi_car.append(mean_squared_error(t_car, X_car@W))

    # Print the results in a table
    tab_error = {"Error_train_turkish":L_err_train_turkish, "Error_test_turkish":L_err_test_turkish, "Error_train_lin_car":L_err_train_lin_car,
                  "Error_test_lin_car":L_err_test_lin_car, "Error_train_multi_car":L_err_train_multi_car, "Error_test_multi_car":L_err_test_multi_car}

    df = pd.DataFrame(tab_error)
    print(df)

    print("Average mean square error on the training data for turkish stocks : ",np.mean(L_err_train_turkish))
    print("Average mean square error on the test data for turkish stocks : ",np.mean(L_err_test_turkish))
    print("Average mean square error on the training data for one-dimensional problem car: ",np.mean(L_err_train_lin_car))
    print("Average mean square error on the test data for one-dimensional problem car : ",np.mean(L_err_test_lin_car))
    print("Average mean square error on the training data for multi_dimensional problem car: ",np.mean(L_err_train_multi_car))
    print("Average mean square error on the test data for multi_dimensional problem car: ",np.mean(L_err_test_multi_car))

    return tab_error
    

        

if __name__ == "__main__":
    main(pourcentage_subset = 0.1, n_subset = 10)
