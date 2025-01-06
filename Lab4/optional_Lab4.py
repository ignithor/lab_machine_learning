import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape to (samples, 28, 28, 1) to include the channel dimension
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Number to compare
number1 = 1
number2 = 5

# Filter the dataset to include only digits 1 and 8
train_filter = (y_train == number1) | (y_train == number2)
test_filter = (y_test == number1) | (y_test == number2)

x_train_filtered = x_train[train_filter]
y_train_filtered = y_train[train_filter]
x_test_filtered = x_test[test_filter]
y_test_filtered = y_test[test_filter]

# Define the autoencoder
latent_dim = 32
input_shape = (28, 28, 1)

# Simplified Encoder: One hidden layer
encoder_input = layers.Input(shape=input_shape)
x = layers.Flatten()(encoder_input)
latent_space = layers.Dense(latent_dim, activation='sigmoid')(x)  # Single hidden layer

encoder = models.Model(encoder_input, latent_space, name="encoder")

# Simplified Decoder: One layer reconstructing the input
decoder_input = layers.Input(shape=(latent_dim,))
x = layers.Dense(28 * 28, activation='sigmoid')(decoder_input)  # Output layer reconstructing input
decoder_output = layers.Reshape((28, 28, 1))(x)

decoder = models.Model(decoder_input, decoder_output, name="decoder")


# Autoencoder
autoencoder_input = encoder_input
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)

autoencoder = models.Model(autoencoder_input, decoded, name="autoencoder")

# Compile the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
history = autoencoder.fit(
    x_train_filtered, x_train_filtered,
    epochs=100,
    batch_size=128,
    validation_data=(x_test_filtered, x_test_filtered)
)

# Evaluate the autoencoder
loss = autoencoder.evaluate(x_test_filtered, x_test_filtered)
print(f"Test loss: {loss}")

# Visualize the original and reconstructed images

# Pick some samples from the test set
n_samples = 10
samples = x_test_filtered[:n_samples]
reconstructions = autoencoder.predict(samples)

plt.figure(figsize=(20, 4))
for i in range(n_samples):
    # Original images
    ax = plt.subplot(2, n_samples, i + 1)
    plt.imshow(samples[i].squeeze(), cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Reconstructed images
    ax = plt.subplot(2, n_samples, i + 1 + n_samples)
    plt.imshow(reconstructions[i].squeeze(), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

plt.show()

# Confusion Matrix
# Calculate reconstruction errors
reconstruction_errors = np.mean(np.square(x_test_filtered - autoencoder.predict(x_test_filtered)), axis=(1, 2, 3))

# Set a threshold for classification (tune based on results)
threshold = np.percentile(reconstruction_errors, 95)

# Predict reconstructed vs non-reconstructed
predictions = reconstruction_errors < threshold
true_labels = (y_test_filtered == 1).astype(int)  # Treat '1' as one class, all else as another

# Confusion Matrix
cm = confusion_matrix(true_labels, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Reconstructed", "Reconstructed"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# PCA for Latent Space Visualization
latent_features = encoder.predict(x_test_filtered)
pca = PCA(n_components=2)
latent_2d = pca.fit_transform(latent_features)

# Plot the 2D PCA projection of latent features
plt.figure(figsize=(8, 6))
colors = ['red' if label == number1 else 'blue' for label in y_test_filtered]
plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=colors, marker='+', s=50)
plt.legend([f'Digit {number1}', f'Digit {number2}'], loc="upper right")
plt.title(f"Features for Recognizing '{number1}' vs '{number2}'")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid()
plt.show()
