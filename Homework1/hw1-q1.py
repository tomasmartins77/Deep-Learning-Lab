#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import time
import utils

class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1 a
        # calculate prediction
        predicted_label = self.predict(x_i)

        # if prediction is incorrect
        if predicted_label != y_i:
            self.W[y_i] += x_i  # increase weight of gold class
            self.W[predicted_label] -= x_i  # decrease weight of incorrect class

        return


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001, l2_penalty=0.0, **kwargs):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.2 (a,b)
        scores = self.W @ x_i  # compute the scores
        
        exp_scores = np.exp(scores)
        P_w = exp_scores / np.sum(exp_scores)  # compute the probabilities

        e_y = np.zeros(self.W.shape[0])
        e_y[y_i] = 1
        
        self.W += learning_rate * (np.outer(e_y - P_w, x_i) - l2_penalty * self.W)   # update the weights
        return

class MLP(object):
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        self.n_classes = n_classes
        self.units = [n_features,  hidden_size, n_classes]

        self.W = [np.random.normal(loc=0.1, scale=0.1, size=(
            self.units[i+1], self.units[i])) for i in range(0, len(self.units)-1)]
        
        self.B = [np.zeros(self.units[i+1])
                  for i in range(0, len(self.units)-1)]

    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def softmax(self, x):
        """Softmax function for output probabilities."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability trick
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes.
        # Q1.3 (a)
        z1 = np.dot(X, self.W[0].T) + self.B[0] # Hidden layer pre-activation
        a1 = self.relu(z1) # Hidden layer ReLU activation

        z2 = np.dot(a1, self.W[1].T) + self.B[1] # Output layer pre-activation
        probabilities = self.softmax(z2) # Output layer Softmax activation

        return np.argmax(probabilities, axis=1) # Predicted class label


    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001, **kwargs):
        """
        Dont forget to return the loss of the epoch.
        """
        # Q1.3 (b)

        total_loss = 0
        n_examples = X.shape[0]

        for i in range(n_examples):
            # Forward pass
            x_i = X[i:i+1]  # Single example (1, n_features)
            z1 = np.dot(x_i, self.W[0].T) + self.B[0]  # Hidden layer pre-activation
            a1 = self.relu(z1)  # Hidden layer ReLU activation

            z2 = np.dot(a1, self.W[1].T) + self.B[1]   # Output layer pre-activation
            probabilities = self.softmax(z2)  # Output layer Softmax activation

            # Compute loss (cross-entropy)
            target = np.zeros(self.n_classes)
            target[y[i]] = 1
            loss = -target @ np.log(probabilities.T + 1e-8)  # Cross-entropy loss
            loss = np.sum(loss)
            total_loss += loss

            # Backward pass
            # Output layer
            grad_z2 = probabilities - target  # Gradient of loss z2
            grad_w2 = np.dot(grad_z2.T, a1)  # Gradient of loss W2
            grad_b2 = grad_z2  # Gradient of loss B2

            # Hidden layer
            grad_a1 = np.dot(grad_z2, self.W[1])  # Gradient of loss a1
            grad_z1 = grad_a1 * (z1 > 0)  # Backprop through ReLU
            grad_w1 = np.dot(grad_z1.T, x_i)  # Gradient of loss W1
            grad_b1 = grad_z1  # Gradient of loss B1

            # Update weights and biases
            self.W[1] -= learning_rate * grad_w2
            self.B[1] -= learning_rate * grad_b2.mean(axis=0)   
            self.W[0] -= learning_rate * grad_w1
            self.B[0] -= learning_rate * grad_b1.mean(axis=0)

        return total_loss / n_examples


def plot(epochs, train_accs, val_accs, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_loss(epochs, loss, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_w_norm(epochs, w_norms, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('W Norm')
    plt.plot(epochs, w_norms, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=100,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    parser.add_argument('-l2_penalty', type=float, default=0.0,)
    parser.add_argument('-data_path', type=str, default='intel_landscapes.npz',)
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_dataset(data_path=opt.data_path, bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    weight_norms = []
    valid_accs = []
    train_accs = []

    start = time.time()

    print('initial train acc: {:.4f} | initial val acc: {:.4f}'.format(
        model.evaluate(train_X, train_y), model.evaluate(dev_X, dev_y)
    ))
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate,
                l2_penalty=opt.l2_penalty,
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        elif opt.model == "logistic_regression":
            weight_norm = np.linalg.norm(model.W)
            print('train acc: {:.4f} | val acc: {:.4f} | W norm: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1], weight_norm,
            ))
            weight_norms.append(weight_norm)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs, filename=f"Q1-{opt.model}-accs.pdf")
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss, filename=f"Q1-{opt.model}-loss.pdf")
    elif opt.model == 'logistic_regression':
        plot_w_norm(epochs, weight_norms, filename=f"Q1-{opt.model}-w_norms.pdf")
    with open(f"Q1-{opt.model}-results.txt", "w") as f:
        f.write(f"Final test acc: {model.evaluate(test_X, test_y)}\n")
        f.write(f"Training time: {minutes} minutes and {seconds} seconds\n")


if __name__ == '__main__':
    main()
