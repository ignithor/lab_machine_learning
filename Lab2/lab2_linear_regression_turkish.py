import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Helper function for linear regression
def linear_regression(x, t, intercept=True):
    if intercept:
        # Add a column of ones for the intercept term
        X = np.c_[np.ones(X.shape[0]), X]
        # Solve for beta coefficients using the normal equation
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
    else:
        w_num=0
        w_denom=0
        for i in range(len(x)):
            w_num += x[i]*t[i]
            w_denom += x[i]**2
        w = w_num / w_denom
    return w

# Helper function to compute predictions
def predict(X, beta, intercept=True):
    if intercept:
        # Add a column of ones if we have an intercept
        X = np.c_[np.ones(X.shape[0]), X]
    return X @ beta

# Helper function to compute MSE
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)



def main(pourcentage_random_subset = 0.1):
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
        t_subset = t_turkish[randomSubset]
        
        # Fit model without intercept
        w = linear_regression(x_subset, t_subset, intercept=False)
        # y_pred = predict(x_turkish, w, intercept=False)
        
        # TASK 2.2 Plot subset data and regression line
        plt.scatter(x_subset, t_subset, alpha=0.5, label="Data Subset")
        plt.plot(x_turkish, w*x_turkish, label="Subset Fit", linewidth=1)

    plt.xlabel("SP500")
    plt.ylabel("MSCI")
    plt.title("Linear Fit on Random 10% Subsets of Turkish Data (No Intercept)")
    plt.legend()
    plt.show()

    ### TASK 2.3: One-dimensional problem with intercept on the Motor Trends car data, using columns mpg and weight

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
        t_subset = t_turkish[randomSubset]
        
        # Fit model without intercept
        w = linear_regression(x_subset, t_subset, intercept=False)
        # y_pred = predict(x_turkish, w, intercept=False)
        
        # TASK 2.2 Plot subset data and regression line
        plt.scatter(x_subset, t_subset, alpha=0.5, label="Data Subset")
        plt.plot(x_turkish, w*x_turkish, label="Subset Fit", linewidth=1)

    plt.xlabel("SP500")
    plt.ylabel("MSCI")
    plt.title("Linear Fit on Random 10% Subsets of Turkish Data (No Intercept)")
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
