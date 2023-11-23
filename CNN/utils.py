import matplotlib.pyplot as plt
import numpy as np

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
    return ((y_predicted == y_true).mean()) * 100


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
