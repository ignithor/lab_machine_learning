import numpy as np
from tensorflow.keras.datasets import mnist
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
        nearest_neighbors = np.argsort(distances)[:k]
        neighbor_labels = train_y[nearest_neighbors]
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        predicted_label = unique[np.argmax(counts)]
        predictions.append(predicted_label)
    
    predictions = np.array(predictions)
    error_rate = None
    if test_y is not None:
        error_rate = np.mean(predictions != test_y)
    
    return predictions, error_rate

def one_vs_all_labels(y, target_class):
    """
    Transform labels into binary for one-vs-all classification.
    """
    return (y == target_class).astype(int)

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
    Compute precision, recall, F1-score, and accuracy from confusion matrix.
    """
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return {"precision": precision, "recall": recall, "f1_score": f1_score, "accuracy": accuracy}

def evaluate_knn_for_tasks(k_values, train_X, train_y, test_X, test_y, classes):
    """
    Evaluate KNN for multiple k values and binary tasks (digit vs. others).
    """
    results = {}
    
    for digit in classes:
        binary_train_y = one_vs_all_labels(train_y, digit)
        binary_test_y = one_vs_all_labels(test_y, digit)
        results[digit] = {}
        
        for k in k_values:
            predictions, _ = kNN_classifier(k, train_X, binary_train_y, test_X)
            cm = compute_confusion_matrix(binary_test_y, predictions)
            metrics = compute_classification_metrics(cm)
            
            results[digit][k] = {
                "confusion_matrix": cm,
                "classification_metrics": metrics
            }
    
    return results

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

def main():
    k_values = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]
    classes = list(range(10))  # Digits 0-9
    
    # Load MNIST data
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = train_X.reshape(train_X.shape[0], -1) / 255.0
    test_X = test_X.reshape(test_X.shape[0], -1) / 255.0

    # Optionally use a subset for faster computation
    train_X = train_X[:2000]
    train_y = train_y[:2000]
    test_X = test_X[:500]
    test_y = test_y[:500]
    
    check_input(train_X, test_X, min(k_values))
    
    # Evaluate KNN on tasks
    results = evaluate_knn_for_tasks(k_values, train_X, train_y, test_X, test_y, classes)
    
    # Summarize results
    summary = summarize_results(results)
    for k, stats in summary.items():
        print(f"k={k}: Mean Accuracy={stats['mean_accuracy']:.3f}, Std. Deviation={stats['std_accuracy']:.3f}")
    
    # Plot results
    plot_results(summary, k_values)

if __name__ == "__main__":
    main()
