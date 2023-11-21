import numpy as np
from layer import Layer


class Softmax:
    def __init__(self):
        pass

    def forward(self, x):
        exponents = np.exp(x - np.max(x))
        return exponents / np.sum(exponents)


class CategoricalCrossEntropy:
    def __init__(self):
        pass

    def forward(self, softmax_y, y_true):
        # y_true is a single dimension class label
        samples = len(y_true)
        softmax_y = np.clip(softmax_y, 1e-7, 1 - 1e-7)
        predicted_prob = softmax_y[range(samples), y_true]

        # for the rest of the classes y_true would be 0 so, no need to consider them
        loss = -np.log(predicted_prob)

        return loss


class Softmax_Categorical_CrossEntropy(Layer):
    def __init__(self, layer_name):
        super().__init__(layer_name)
        self.activation = Softmax()
        self.loss = CategoricalCrossEntropy()
        self.cached_output = None

    def forward(self, y_predicted, y_true):
        """
        https://jaykmody.com/blog/stable-softmax/
        """
        self.cached_output = self.activation.forward(y_predicted)
        # print(f"Softmax output {self.cached_output}, shape {self.cached_output.shape}")
        return self.loss.forward(self.cached_output, y_true)

    def backward(self, y_softmax, y_true):
        """
        @param y_softmax -> cached softmax predictions
        @param y_true -> categorical y labels
        """
        samples = len(y_true)
        # print(f"y softmax shape {y_softmax.shape}")
        gradients = y_softmax
        gradients[range(samples), y_true] -= 1
        # gradients /= samples

        return gradients


# def regularized_softmax(x):
#     """
#     https://jaykmody.com/blog/stable-softmax/
#     """
#     return softmax(x - np.max(x))
