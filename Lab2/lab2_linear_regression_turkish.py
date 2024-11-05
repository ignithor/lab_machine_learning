import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Helper function for linear regression
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

# Helper function to compute predictions
def predict(X, beta, intercept=True):
    if intercept:
        # Add a column of ones if we have an intercept
        X = np.c_[np.ones(X.shape[0]), X]
    return X @ beta

# Helper function to compute MSE
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)



def main(pourcentage_random_subset = 0.5):
    # TASK 1 : Get data
    turkish_data = pd.read_csv("turkish-se-SP500vsMSCI.csv", header=None)

    # Column 0: SP500 (predictor)
    # Column 1: MSCI (target)
    x_turkish = turkish_data[0].values  # Predictor variable (SP500)
    t_turkish = turkish_data[1].values    # Target variable (MSCI)

    # TASK 2.1: One-dimensional regression without intercept on Turkish stock data
    plt.figure(figsize=(10, 6))
    for i in range(5):  # Compare five random subsets
        randomSubset = np.random.permutation(len(x_turkish))[:int(pourcentage_random_subset * len(x_turkish))]
        x_subset = x_turkish[randomSubset]
        type(x_subset)
        t_subset = t_turkish[randomSubset]
        
        # Fit model without intercept
        w = linear_regression(x_subset, t_subset)
        
        # TASK 2.2 Plot subset data and regression line
        plt.scatter(x_subset, t_subset, alpha=0.5, label="Data Subset")
        plt.plot(x_turkish, w*x_turkish, label="Subset Fit", linewidth=1)

    plt.xlabel("SP500")
    plt.ylabel("MSCI")
    plt.title("Linear Fit on Random 10% Subsets of Turkish Data (No Intercept)")
    plt.legend()
    plt.show()

    ### TASK 2.3: One-dimensional problem with intercept on the Motor Trends car data, using columns mpg and weight

    car_data = pd.read_csv("mtcarsdata-4features.csv")
    car_data = car_data.values

    # Column 5: weight (predictor)
    # Column 2: mpg (target)
    x_car = car_data[:, 4]
    t_car = car_data[:, 1]

    plt.figure(figsize=(10, 6))
    for i in range(1):  # Compare five random subsets
        randomSubset = np.random.permutation(len(x_car))[:int(pourcentage_random_subset * len(x_car))]
        x_subset = x_car[randomSubset]
        t_subset = t_car[randomSubset]
        
        # Fit model without intercept
        w1,w0 = linear_regression_intercept(x_subset, t_subset)
        # y_pred = predict(x_turkish, w, intercept=False)
        
        # TASK 2.2 Plot subset data and regression line
        plt.scatter(x_subset, t_subset, alpha=0.5, label="Data Subset")
        plt.plot(x_car, w1*x_car + w0, label="Subset Fit", linewidth=1)

    plt.xlabel("weight")
    plt.ylabel("mpg")
    plt.title("Linear Fit on Random 10% Subsets of car Data (No Intercept)")
    plt.legend()
    plt.show()

    # Load dataset from CSV file

    # # Task 3: Testing Regression Model with Different Splits
    # results = []

    # for i in range(10):  # Repeat 10 times
    #     # Train-test split (5% train, 95% test)
    #     indices = np.random.permutation(len(x_turkish))
    #     train_size = int(0.05 * len(indices))
    #     train_indices = indices[:train_size]
    #     test_indices = indices[train_size:]
        
    #     # Select training and test data
    #     X_train, y_train = x_turkish[train_indices], t_turkish[train_indices]
    #     X_test, y_test = x_turkish[test_indices], t_turkish[test_indices]
        
    #     # Fit model without intercept
    #     beta = linear_regression(X_train, y_train, intercept=False)
        
    #     # Compute MSE for training and test sets
    #     train_mse = mean_squared_error(y_train, predict(X_train, beta, intercept=False))
    #     test_mse = mean_squared_error(y_test, predict(X_test, beta, intercept=False))
    #     results.append({"Split": i + 1, "Train MSE": train_mse, "Test MSE": test_mse})

    # # Display results in table format
    # results_df = pd.DataFrame(results)
    # print(results_df)

    # # Plot MSEs for comparison
    # results_df.plot(x='Split', y=['Train MSE', 'Test MSE'], kind='bar', figsize=(10, 6))
    # plt.title("Train vs Test MSE on Different Splits")
    # plt.xlabel("Split")
    # plt.ylabel("Mean Squared Error")
    # plt.show()



if __name__ == "__main__":
    main()
