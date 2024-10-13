import pandas as pd
import numpy as np
import math

def validate_data(train_data, test_data):
    # Check dimensions
    d = train_data.shape[1] - 1  # number of features in training data
    c = test_data.shape[1] - 1  # number of features in test data (target included if present)

    if not (c == d or c == d - 1):
        raise ValueError("Test set must have either d or d+1 columns.")
    
    # Check for entries less than 1
    # nan_check = train_data.isnull().any().any() or test_data.isnull().any().any()
    nan_check = np.isnan(train_data)
    print(nan_check)
    # if nan_check:
    #     raise ValueError(" Dataset contains NaN values.")
    # return(True)
        

def train_naive_bayes(train_data):
    # Split features and target
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]

    classes = np.unique(y_train)
    n_classes = len(classes)
    n_features = X_train.shape[1]

    # Calculate prior probabilities P(c)
    prior_probs = {c: np.mean(y_train == c) for c in classes}

    # Calculate likelihoods P(x_i | c)
    likelihoods = {}
    for c in classes:
        class_data = X_train[y_train == c]
        class_likelihoods = {}
        for feature in range(n_features):
            unique_values = np.unique(X_train[:, feature])
            class_likelihoods[feature] = {}
            for value in unique_values:
                count = np.sum(class_data[:, feature] == value)
                class_likelihoods[feature][value] = count / class_data.shape[0]
        likelihoods[c] = class_likelihoods

    return prior_probs, likelihoods, classes

def predict(test_data, prior_probs, likelihoods, classes):
    X_test = test_data[:, :-1] if test_data.shape[1] > 1 else test_data

    predictions = []
    for x in X_test:
        posteriors = {}
        for c in classes:
            # Calculate posterior P(c | x) = P(c) * P(x | c)
            posterior = prior_probs[c]
            for feature in range(X_test.shape[1]):
                value = x[feature]
                if value not in likelihoods[c][feature]:
                    raise ValueError(f"Value {value} for feature {feature} not seen in training data.")
                posterior *= likelihoods[c][feature][value]
            posteriors[c] = posterior
        
        # Predict class with maximum posterior
        predictions.append(max(posteriors, key=posteriors.get))
    
    return np.array(predictions)

def error_rate(predictions, actual):
    return np.mean(predictions != actual)

def main():
    # Load dataset from CSV file
    df = pd.read_csv('dataset.csv')
    # validate_data(df,df)
    print(df)
    dataset = df.values
    print(dataset)
    # np.isnan(dataset)
    for i in dataset:
        for j in i:
            if math.isnan(j):
                print("trouvé")
            print(j)
    # print(np.isnan(dataset).any())


    # Split dataset into training and test set
    # np.random.seed(0)  # For reproducibility
    train_indices = np.random.choice(dataset.shape[0], size=10, replace=False)
    train_set = dataset[train_indices]
    test_set = np.delete(dataset, train_indices, axis=0)

    # Validate data
    # validate_data(train_set, test_set)

    # Train Naive Bayes Classifier
    prior_probs, likelihoods, classes = train_naive_bayes(train_set)

    # Make predictions
    if test_set.shape[1] == train_set.shape[1]:
        actual = test_set[:, -1]
        test_set_without_target = test_set[:, :-1]
        predictions = predict(test_set_without_target, prior_probs, likelihoods, classes)
        err_rate = error_rate(predictions, actual)
        print(f"Predictions: {predictions}")
        print(f"Actual: {actual}")
        print(f"Error Rate: {err_rate:.2f}")
    else:
        predictions = predict(test_set, prior_probs, likelihoods, classes)
        print(f"Predictions: {predictions}")

if __name__ == "__main__":
    main()
