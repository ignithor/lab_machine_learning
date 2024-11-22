import numpy as np
from tensorflow.keras.datasets import mnist

def check_input(train_X, test_X, k):
    if not train_X.shape[1] == test_X.shape[1]:
        raise ValueError("Train and test data have different number of features")
    if k > train_X.shape[0] or k <= 0:
        raise ValueError("Invalid k value")

def kNN_classifier(k, train_X, train_y, test_X, test_y=None):
    """
    Perform k-Nearest Neighbors classification.
    
    Args:
        k (int): Number of neighbors to consider.
        train_X (numpy.ndarray): Training data, shape (n_train, n_features).
        train_y (numpy.ndarray): Training labels, shape (n_train,).
        test_X (numpy.ndarray): Test data, shape (n_test, n_features).
        test_y (numpy.ndarray, optional): Test labels, shape (n_test,).
        
    Returns:
        numpy.ndarray: Predicted labels for the test data.
        float: Error rate, if test_y is provided. Otherwise, None.
    """
    n_test = test_X.shape[0]
    predictions = []

    for i in range(n_test):
        # Compute L2 distance between test sample and all training samples
        distances = np.linalg.norm(train_X - test_X[i], axis=1)
        
        # Get the indices of the k-nearest neighbors
        nearest_neighbors = np.argsort(distances)[:k]
        
        # Get the labels of the k-nearest neighbors
        neighbor_labels = train_y[nearest_neighbors]
        
        # Determine the most common label (majority voting)
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        predicted_label = unique[np.argmax(counts)]
        predictions.append(predicted_label)
    
    predictions = np.array(predictions)
    
    # Compute error rate if test_y is provided
    error_rate = None
    if test_y is not None:
        error_rate = np.mean(predictions != test_y)
    
    return predictions, error_rate

def main():
    k = 3
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    
    # Flatten the images into 1D arrays
    train_X = train_X.reshape(train_X.shape[0], -1)  # shape (n_train, 784)
    test_X = test_X.reshape(test_X.shape[0], -1)    # shape (n_test, 784)
    
    # Normalize the pixel values to range [0, 1]
    train_X = train_X / 255.0
    test_X = test_X / 255.0
    
    check_input(train_X, test_X, k)
    
    # Perform KNN classification
    predictions, error_rate = kNN_classifier(k, train_X, train_y, test_X[:100], test_y[:100])  # Use a subset for faster computation
    
    print("Predictions:", predictions)
    if error_rate is not None:
        print("Error rate:", error_rate)

if __name__ == "__main__":
    main()
