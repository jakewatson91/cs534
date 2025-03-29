
import numpy as np
import json
from itertools import product

def sigmoid(z):
    return 1 / (1+ np.exp(-z))

def forward(X, w):
    # print("X: ", X.shape)
    # print("w: ", w.shape)
    z = np.dot(X, w)
    # print("Z: ", z.shape)
    y_pred = sigmoid(z)
    # print("y_pred: ", y_pred.shape)
    return y_pred

def classify(y_pred):
    # y_pred = forward(X, w)
    return (y_pred >= 0.5).astype(int)

def loss_fn(X, Y, y_pred, w):
    eps = 1e-8
    # print("Y: ", Y.shape)
    loss = -(Y * (np.log(y_pred) + eps) + (1 - Y) * np.log(1 - y_pred + eps))
    return np.mean(loss)

def gradient(X, Y, y_pred, w):
   dz = y_pred - Y
#    print(dz.shape)
   dw = np.dot(X.T, dz)
   return dw

def train(X, Y, iterations, lr):
    w = np.random.randn(X.shape[1], 1)
    loss = 0
    for _ in range(iterations):
        y_probs = forward(X, w)
        y_pred = classify(y_probs)
        loss += loss_fn(X, Y, y_pred, w)
        w -= gradient(X, Y, y_pred, w) * lr

    return w, np.mean(loss), y_pred, Y

def test(X, Y, w):
    y_pred = classify(forward(X, w))
    accuracy = sum(y_pred == Y) / len(Y)
    return accuracy

def get_example_elements(json_example):
    """
    Load the data
    :param json_example:
    :return:
    """
    train = json_example["train"]
    test = json_example["test"]
    pairs = []
    for pair in train:
        input = pair["input"]
        output = pair["output"]

        example = {}
        example["input"] = np.array(input)
        example["output"] = output
        pairs.append(example)

    pairs_test = []
    for pair in test:
        input = pair["input"]
        output = pair["output"]

        example = {}
        example["input"] = np.array(input)
        example["output"] = output
        pairs_test.append(example)
    return pairs, pairs_test

# TODO change which file you load for each
filename = 'data_0.json'
with open('data/' + filename, 'r') as file:
    train_data = json.load(file)
    pairs_train, pairs_test = get_example_elements(train_data)
    X = []
    Y = []
    for pair in pairs_train:
        input = pair["input"]
        output = pair["output"]
        input_flat = input.flatten()
        X.append(input_flat)
        Y.append(output)
    # print(X)
    X = np.array(X)
    # We need Y to be a matrix so we can do matix multiplication
    Y = np.array(Y).reshape(-1,1)
    # Train
    # TODO what iterations and lr work??
    # # how does it do on the training data?

    X_test = []
    Y_test = []
    for pair in pairs_test:
        input = pair["input"]
        output = pair["output"]
        input_flat = input.flatten()
        X_test.append(input_flat)
        Y_test.append(output)
    X_test = np.array(X_test)
    # We need Y to be a matrix so we can do matix multiplication
    Y_test = np.array(Y_test).reshape(-1,1)
    # how does it do on test data?

if __name__ == "__main__":
    lrs = [0.5, 0.1, 1e-2]
    iterations = [1, 10, 100]

    for lr, iteration in product(lrs, iterations):
        weights, _, y_pred, Y = train(X, Y, iteration, lr)
        print(f"\nLearning rate: {lr}, Iterations: {iteration}")
        print(f"Prediction: {y_pred.flatten()}")
        print(f"Actual: {Y.flatten()}")
        print(f"Train Accuracy: {sum(y_pred == Y) / len(Y)}")
        # print(weights)
        accuracy = test(X_test, Y_test, weights)
        print(f"Test Accuracy: {accuracy}")

        # import matplotlib.pyplot as plt

        # # Plot sigmoid curve for current weights
        # x_vals = np.linspace(-10, 10, 200).reshape(-1, 1)
        # y_vals = sigmoid(x_vals @ weights[:1])  # using first weight for 1D curve

        # plt.figure()
        # plt.plot(x_vals, y_vals)
        # plt.title(f"Sigmoid Curve (lr={lr}, iters={iteration})")
        # plt.xlabel("Input")
        # plt.ylabel("Sigmoid Output")
        # plt.grid(True)
        # plt.savefig(f"plots/{filename}_{lr}_{iteration}.png")
        # plt.show()
