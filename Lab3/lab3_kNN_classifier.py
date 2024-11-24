import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt


def check_input(train_X, test_X, k):
    if not train_X.shape[1] == test_X.shape[1]:
        raise ValueError("Train and test data have different number of features")
    if k > train_X.shape[0] or k <= 0:
        raise ValueError("Invalid k value")


def kNN_classifier(k, train_X, train_y, test_X, test_y=None):
    """
    Perform k-Nearest Neighbors classification.
    """
    n_test = test_X.shape[0]
    predictions = []

    for i in range(n_test):
        distances = np.linalg.norm(train_X - test_X[i], axis=1)
        nearest_neighbors = np.argsort(distances)[:k] # Get indices of k nearest neighbors
        neighbor_labels = train_y[nearest_neighbors]
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        predicted_label = unique[np.argmax(counts)]
        predictions.append(predicted_label)
    
    predictions = np.array(predictions)
    error_rate = None
    if test_y is not None:
        error_rate = np.mean(predictions != test_y)
    
    return predictions, error_rate


def normalize_features(X):
    """
    Normalize features to the [0, 1] range using min-max scaling.
    """
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    return (X - min_vals) / (max_vals - min_vals)


def evaluate_knn_for_tasks(k_values, train_X, train_y, test_X, test_y, classes):
    """
    Evaluate KNN for multiple k values and binary tasks (class vs. others).
    """
    results = {}
    
    for cls in classes:
        binary_train_y = (train_y == cls).astype(int)
        binary_test_y = (test_y == cls).astype(int)
        results[cls] = {}
        
        for k in k_values:
            predictions, error_rate = kNN_classifier(k, train_X, binary_train_y, test_X)
            cm = compute_confusion_matrix(binary_test_y, predictions)
            print(f"Confusion Matrix for class {cls} with k={k}:\n", cm)
            metrics = compute_classification_metrics(cm)
            
            results[cls][k] = {
                "confusion_matrix": cm,
                "classification_metrics": metrics
            }
    
    return results


def compute_confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix for binary classification.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positive
    tn = np.sum((y_true == 0) & (y_pred == 0))  # True Negative
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positive
    fn = np.sum((y_true == 1) & (y_pred == 0))  # False Negative
    return np.array([[tp, fn], [fp, tn]])


def compute_classification_metrics(cm):
    """
    Compute precision, recall and accuracy from confusion matrix.
    """
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return {"precision": precision, "recall": recall, "accuracy": accuracy}


def leave_one_out_cross_validation(k, X, y):
    """
    Perform Leave-One-Out Cross-Validation (LOOCV) for KNN.
    """
    n_samples = X.shape[0]
    errors = 0

    for i in range(n_samples):
        train_X = np.delete(X, i, axis=0)
        train_y = np.delete(y, i)
        test_X = X[i].reshape(1, -1)
        test_y = y[i]
        
        prediction, _ = kNN_classifier(k, train_X, train_y, test_X, np.array([test_y]))
        if prediction[0] != test_y:
            errors += 1
    
    return errors / n_samples


def plot_results(summary, k_values):
    """
    Plot accuracy results across different k values.
    """
    mean_accuracies = [summary[k]["mean_accuracy"] for k in k_values]
    std_accuracies = [summary[k]["std_accuracy"] for k in k_values]

    plt.figure(figsize=(10, 6))
    plt.errorbar(k_values, mean_accuracies, yerr=std_accuracies, fmt='o-', capsize=5)
    plt.title("KNN Accuracy vs. k Values (with Error Bars)")
    plt.xlabel("k Value")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.show()


def summarize_results(results):
    """
    Summarize results with average and standard deviation of accuracy across tasks.
    """
    summary = {}
    for k in results[0].keys():
        accuracies = [results[digit][k]["classification_metrics"]["accuracy"] for digit in results.keys()]
        summary[k] = {
            "mean_accuracy": np.mean(accuracies),
            "std_accuracy": np.std(accuracies)
        }
    return summary


def main():
    dataset_choice = int(input("Choose dataset mnist : 0 or wine : 1): "))
    # Wine dataset
    if dataset_choice:
        wine = load_wine()
        X, y = wine.data, wine.target
        X = normalize_features(X)  # Normalize features to [0, 1]
        
        # Split into train and test sets (80/20 split)
        split_idx = int(0.8 * X.shape[0])
        
        # Permutation for shuffling the dataset
        permutation = np.random.permutation(len(y))
        X = X[permutation]
        y = y[permutation]
        
        train_X, test_X = X[:split_idx], X[split_idx:]
        train_y, test_y = y[:split_idx], y[split_idx:]
        classes = np.unique(y)
        # avoid k= 3n to avoid ties
        k_values = [1, 2, 4, 5, 10, 16, 20, 31, 40, 50]
        
        # If we want to run LOOCV
        
        # check_input(train_X, test_X, min(k_values))
        # print("Running Leave-One-Out Cross-Validation (LOOCV)...")
        # loocv_results = {}
        # for k in k_values:
        #     loocv_error = leave_one_out_cross_validation(k, train_X, train_y)
        #     loocv_results[k] = 1 - loocv_error
        #     print(f"k={k}: LOOCV Accuracy = {1 - loocv_error:.4f}")
        # plt.plot(k_values, list(loocv_results.values()), marker='o', label="LOOCV Accuracy")
        # plt.title("LOOCV Accuracy vs. k for Wine Dataset")
        # plt.xlabel("k")
        # plt.ylabel("Accuracy")
        # plt.grid()
        # plt.show()
        
    # MNIST dataset
    else:
        print("Loading MNIST dataset...")
        (train_X, train_y), (test_X, test_y) = mnist.load_data()
        train_X = train_X.reshape(train_X.shape[0], -1) / 255.0
        test_X = test_X.reshape(test_X.shape[0], -1) / 255.0
        
        # Use a subset of MNIST for faster computation 80-20 split
        train_X = train_X[:1200]
        train_y = train_y[:1200]
        test_X = test_X[:300]
        test_y = test_y[:300]
        classes = list(range(10)) # Digits 0-9
        # avoid k = 10n to avoid ties
        k_values = [1, 2, 3, 5, 11, 15, 21, 31, 41, 51]
        
    check_input(train_X, test_X, min(k_values))
    
    # Evaluate KNN on tasks
    results = evaluate_knn_for_tasks(k_values, train_X, train_y, test_X, test_y, classes)
    
    # Summarize and plot
    summary = summarize_results(results)
    plot_results(summary, k_values)

    # Build a dictionary to hold table data
    table_data = {"k": k_values}

    for cls in classes:  # Add a column for each digit
        table_data[cls] = [
            results[cls][k]["classification_metrics"]["accuracy"]
            for k in k_values
        ]

    # Create a DataFrame
    results_table = pd.DataFrame(table_data)

    # Print the table
    print("\nAccuracy Table:")
    print(results_table)

    # Save the table to a CSV file
    results_table.to_csv("accuracy_table.csv", index=False)
    
    return 0
    
if __name__ == "__main__":
    main()
