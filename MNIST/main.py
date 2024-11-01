import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# Load and prepare the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, Y = mnist.data, mnist.target.astype(int)
X = X / 255.0  # Normalize the data

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# One-hot encode labels for Y_train
one_hot_encoder = OneHotEncoder(sparse_output=False)
Y_train_one_hot = one_hot_encoder.fit_transform(np.array(Y_train).reshape(-1, 1))

def relu(z):
    return np.maximum(0, z)

def d_relu(z):
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

class NeuralNetwork:
    def __init__(self, X, Y, neurons):
        self.input = X
        self.Y = Y
        self.weightOne = np.random.randn(self.input.shape[1], neurons) * np.sqrt(2. / self.input.shape[1])
        self.weightTwo = np.random.randn(neurons, 10) * np.sqrt(2. / neurons)
        self.output = np.zeros(self.Y.shape)
        self.learning_rate = 0.01

    def feedforward(self):
        self.layer1 = relu(np.dot(self.input, self.weightOne))
        self.output = softmax(np.dot(self.layer1, self.weightTwo))

    def backpropagation(self):
        error = self.output - self.Y  # Cross-entropy error
        d_weightTwo = np.dot(self.layer1.T, error)  # Gradient for second weight matrix
        d_weightOne = np.dot(self.input.T, np.dot(error, self.weightTwo.T) * d_relu(self.layer1))  # Gradient for first weight matrix

        # Gradient clipping
        np.clip(d_weightOne, -1, 1, out=d_weightOne)
        np.clip(d_weightTwo, -1, 1, out=d_weightTwo)

        # Update weights
        self.weightOne -= d_weightOne * self.learning_rate
        self.weightTwo -= d_weightTwo * self.learning_rate

    def predict(self, X):
        layer1 = relu(np.dot(X, self.weightOne))
        output = softmax(np.dot(layer1, self.weightTwo))
        return np.argmax(output, axis=1)  # Return class with highest probability

# Initialize and train the network
nn = NeuralNetwork(X_train, Y_train_one_hot, neurons=128)

# Training loop
for i in range(10000):
    nn.feedforward()
    nn.backpropagation()
    loss = cross_entropy_loss(Y_train_one_hot, nn.output)
    train_pred = nn.predict(X_train)
    train_accuracy = accuracy_score(Y_train, train_pred)
    print(f"Epoch {i}, Loss: {loss:.4f}, Training Accuracy: {train_accuracy * 100:.2f}%")

# Test the network
Y_pred = nn.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
