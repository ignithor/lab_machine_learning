import pandas as pd
import numpy as np

def validate_data_size(train_data, test_data):
    # Check dimensions
    d = train_data.shape[1] - 1  # number of features in training data
    c = test_data.shape[1] - 1  # number of features in test data (target included if present)

    if not (c == d or c == d - 1):
        raise ValueError("Test set must have either d or d+1 columns.")
        
def train_naive_bayes(train_data, num_values, a):

    # Split features and target
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]

    # Find the different possible target values
    classes = np.unique(y_train)
    n_features = X_train.shape[1]

    # Calculate prior probabilities P(c)
    prior_probs = {}
    for c in classes:
        prior_probs[c] = np.mean(y_train == c)

    # Calculate likelihoods P(attribute x = value y | class c)
    likelihoods = {}
    for c in classes:
        class_data = X_train[y_train == c]
        class_likelihoods = {}
        for feature in range(n_features):
            unique_values = np.unique(X_train[:, feature])
            v = num_values[feature]  # number of possible values for this feature
            class_likelihoods[feature] = {}
            for value in unique_values:
                count = np.sum(class_data[:, feature] == value)
                # Apply Laplace smoothing
                class_likelihoods[feature][value] = (count + a) / (class_data.shape[0] + a * v)
            # Handle unseen values with smoothing
            for value in range(v):
                if value not in class_likelihoods[feature]:
                    class_likelihoods[feature][value] = a / (class_data.shape[0] + a * v)
        likelihoods[c] = class_likelihoods

    return prior_probs, likelihoods, classes

def predict(test_data, prior_probs, likelihoods, classes, target_included):
    if target_included:
        test_data = test_data[:, :-1]

    predictions = []
    for x in test_data:
        posteriors = {}
        for c in classes:
            # Calculate posterior P(c | x) = P(c) * P(x | c)
            posterior = prior_probs[c]
            for feature in range(test_data.shape[1]):
                value = x[feature]
                # Use smoothed likelihoods
                if value in likelihoods[c][feature]:
                    posterior *= likelihoods[c][feature][value]
                else:
                    # Unseen value in training data, use Laplace smoothing
                    v = len(likelihoods[c][feature])
                    posterior *= 1 / v
            posteriors[c] = posterior
        
        # Predict class with maximum posterior
        predictions.append(max(posteriors, key=posteriors.get))
    
    return np.array(predictions)

def error_rate(predictions, actual):
    return np.mean(predictions != actual)

def main(print_msg = True ,a = 1):
    # Load dataset from CSV file
    df = pd.read_csv('dataset_laplace.csv')
    if df.isnull().any().any():
        raise ValueError("Dataset contains NaN values.")
    dataset = df.values

    # Get number of possible values for each feature for Laplace smoothing
    num_values = []
    for j in range(dataset.shape[1]):
        num_values.append(int(dataset[0,j]))
    dataset = np.delete(dataset, 0, axis=0)

    # Split dataset into training and test set
    train_indices = np.random.choice(dataset.shape[0], size=10, replace=False)
    train_set = dataset[train_indices]
    test_set = np.delete(dataset, train_indices, axis=0)

    # Validate size of training and test set
    validate_data_size(train_set, test_set)

    # Train Naive Bayes Classifier with Laplace smoothing
    prior_probs, likelihoods, classes = train_naive_bayes(train_set, num_values, a)

    # Make predictions
    if test_set.shape[1] == train_set.shape[1]:
        # Test set includes target values
        actual = test_set[:, -1]
        test_set_without_target = test_set[:, :-1]
        predictions = predict(test_set_without_target, prior_probs, likelihoods, classes, False)
        err_rate = error_rate(predictions, actual)
        if print_msg:
            print(f"Predictions: {predictions}")
            print(f"Actual: {actual}")
            print(f"Error Rate: {err_rate:.2f}")
        return(err_rate)
    else:
        # Test set does not include target values
        predictions = predict(test_set, prior_probs, likelihoods, classes, True)
        if print_msg:
            print(f"Predictions: {predictions}")
        return(0)

if __name__ == "__main__":
    print_msg = False
    for a in [0.1, 1, 2]:
        L_error = []
        print(f"Laplace smoothing parameter: {a}")
        for i in range(10000):
            L_error.append(main(print_msg,a))
        print("Average error rate: ", np.mean(L_error))
