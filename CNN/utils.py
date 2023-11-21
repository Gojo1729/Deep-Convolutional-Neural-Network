import matplotlib.pyplot as plt


def plot_histogram(layer_name, layer_weights):
    plt.hist(layer_weights, density=True)
    plt.title(f"Histogram of {layer_name} layer weights")
    plt.xlabel("Weight values")
    plt.ylabel("Relative distribution")
    plt.show()
