from tensorflow.keras.datasets import mnist

# from sklearn.datasets import load_wine
# wine = load_wine()
# X, y = wine.data, wine.target

def check_input(train_X, test_X, k):
    if not train_X.shape[1] == test_X.shape[1]:
        raise ValueError("Train and test data have different number of features")
    if k > train_X.shape[0] or k <= 0:
        raise ValueError("Invalid k value")


def kNN_classifier(k, train_X, train_y, test_X, test_y):
    pass



def main():
    k=3
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    check_input(train_X, test_X, k)
    print(train_X.shape)
    print(train_y.shape)
    print(test_X.shape)
    print(test_y.shape)
    
    


if __name__ == "__main__":
    main()
