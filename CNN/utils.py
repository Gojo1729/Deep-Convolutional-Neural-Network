import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os

img_folder = "./images"


# region plotting functions
def plot_loss_curve(train_loss, val_loss):
    plt.plot(train_loss, "b", linewidth=3.0, label="Training Loss")
    plt.plot(val_loss, "r", linewidth=3.0, label="Validation Loss")
    plt.xlabel("Iteration", fontsize=16)
    plt.ylabel("Accuracy rate", fontsize=16)
    plt.legend()
    plt.title("Training vs Validation loss", fontsize=16)
    plt.savefig(f"{img_folder}/train_val_loss.png")
    plt.show()


def plot_accuracy_curve(accuracy_history, val_accuracy_history):
    plt.plot(accuracy_history, "b", linewidth=3.0, label="Training accuracy")
    plt.plot(val_accuracy_history, "r", linewidth=3.0, label="Validation accuracy")
    plt.xlabel("Iteration", fontsize=16)
    plt.ylabel("Accuracy rate", fontsize=16)
    plt.legend()
    plt.title("Training vs Validation Accuracy", fontsize=16)
    plt.savefig(f"{img_folder}/train_val_accuracy.png")
    plt.show()


def plot_learning_curve(loss_history):
    plt.plot(loss_history, "b", linewidth=3.0, label="Cross entropy")
    plt.xlabel("Iteration", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend()
    plt.title("Learning Curve", fontsize=16)
    plt.savefig("learning_curve.png")
    plt.show()


def minmax_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val) / (max_val - min_val)
    return x


def visualize_filters(images, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        img = images[i][0]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{i} activation")
        ax.axis("off")
    plt.show()


def visualize_activations(images, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        img = images[i]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{i} activation")
        ax.axis("off")
    plt.show()


def plot_histogram(layer_name, layer_weights):
    plt.hist(layer_weights, density=True)
    plt.title(f"Histogram of {layer_name} layer weights")
    plt.xlabel("Weight values")
    plt.ylabel("Relative distribution")
    plt.show()


# endregion


# region metrics
def accuracy(y_predicted, y_true):
    # print(y_predicted)
    # print(y_true)
    return y_predicted == y_true


# endregion


# region batch gen
def generate_batches(x: np.array, y: np.array, batch_size: int):
    """
    :param x - features array with (n, ...) shape
    :param y - one hot ground truth array with (n, k) shape
    :batch_size - number of elements in single batch
    ----------------------------------------------------------------------------
    n - number of examples in data set
    k - number of classes
    """
    for i in range(0, x.shape[0], batch_size):
        yield (
            x.take(indices=range(i, min(i + batch_size, x.shape[0])), axis=0),
            y.take(indices=range(i, min(i + batch_size, y.shape[0])), axis=0),
        )


# endregion

# region BB


def displayBoundingBoxes(contours, rows=2, cols=6):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    kernel = np.array([[0, -1, 0], [-1, 6, -1], [0, -1, 0]])

    for i, ax in enumerate(axes.flat):
        img = contours[i]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{i} bounding box")
        ax.axis("off")
    plt.show()


# endregion


def save_model(model, model_name: str, path: str = "./models/"):
    import time

    model_name = f"{model_name}_{time.strftime('%Y%m%d%H%M%S')}.pkl"
    with open(f"{path}/{model_name}", "wb") as saved_model:
        pickle.dump(model, saved_model, pickle.HIGHEST_PROTOCOL)


# region load and save model
def load_model(model_path: str = "./models/exp1_10k_train_2_20231127191523.pkl"):
    from layers.conv import ConvLayer
    from layers.dense import DenseLayer
    from layers.flatten import Flatten
    from layers.maxpool import MaxPool
    from layers.relu import ReluLayer

    with open(model_path, "rb") as inp_model:
        model = pickle.load(inp_model)
        return model


# endregion


# region preprocess
def preprocess_image(img):
    kernel = np.array([[0, -1, 0], [-1, 6, -1], [0, -1, 0]])
    img = cv2.resize(img, (55, 55), cv2.INTER_CUBIC, fx=0.1, fy=0.1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # img = cv2.dilate(img, np.ones((2, 2), np.uint8), iterations=1)
    # img = cv2.filter2D(img, -1, kernel)

    return img


# endregion
