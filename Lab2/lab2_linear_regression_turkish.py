import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data without headers
turkish_data = pd.read_csv("turkish-se-SP500vsMSCI.csv", header=None)

# Assuming the Turkish dataset format is as follows:
# Column 0: SP500 (predictor)
# Column 1: MSCI (target)
X_turkish = turkish_data[[0]].values  # Predictor variable (SP500)
y_turkish = turkish_data[1].values    # Target variable (MSCI)

# Helper function for linear regression
def linear_regression(X, y, intercept=True):
    if intercept:
        # Add a column of ones for the intercept term
        X = np.c_[np.ones(X.shape[0]), X]
    # Solve for beta coefficients using the normal equation
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta

# Helper function to compute predictions
def predict(X, beta, intercept=True):
    if intercept:
        # Add a column of ones if we have an intercept
        X = np.c_[np.ones(X.shape[0]), X]
    return X @ beta

# Helper function to compute MSE
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Task 2.1: One-dimensional regression without intercept on Turkish stock data
# Plot different random 10% subsets
plt.figure(figsize=(10, 6))
for i in range(5):  # Compare five 10% random subsets
    # Select 10% of data randomly
    indices = np.random.choice(len(X_turkish), size=int(0.1 * len(X_turkish)), replace=False)
    X_subset = X_turkish[indices]
    y_subset = y_turkish[indices]
    
    # Fit model without intercept
    beta = linear_regression(X_subset, y_subset, intercept=False)
    y_pred = predict(X_turkish, beta, intercept=False)
    
    # Plot subset data and regression line
    plt.scatter(X_subset, y_subset, alpha=0.3, label="Data Subset")
    plt.plot(X_turkish, y_pred, label="Subset Fit", linewidth=1)

plt.xlabel("SP500")
plt.ylabel("MSCI")
plt.title("Linear Fit on Random 10% Subsets of Turkish Data (No Intercept)")
plt.legend()
plt.show()

# Task 3: Testing Regression Model with Different Splits
results = []

for i in range(10):  # Repeat 10 times
    # Train-test split (5% train, 95% test)
    indices = np.random.permutation(len(X_turkish))
    train_size = int(0.05 * len(indices))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Select training and test data
    X_train, y_train = X_turkish[train_indices], y_turkish[train_indices]
    X_test, y_test = X_turkish[test_indices], y_turkish[test_indices]
    
    # Fit model without intercept
    beta = linear_regression(X_train, y_train, intercept=False)
    
    # Compute MSE for training and test sets
    train_mse = mean_squared_error(y_train, predict(X_train, beta, intercept=False))
    test_mse = mean_squared_error(y_test, predict(X_test, beta, intercept=False))
    results.append({"Split": i + 1, "Train MSE": train_mse, "Test MSE": test_mse})

# Display results in table format
results_df = pd.DataFrame(results)
print(results_df)

# Plot MSEs for comparison
results_df.plot(x='Split', y=['Train MSE', 'Test MSE'], kind='bar', figsize=(10, 6))
plt.title("Train vs Test MSE on Different Splits")
plt.xlabel("Split")
plt.ylabel("Mean Squared Error")
plt.show()
