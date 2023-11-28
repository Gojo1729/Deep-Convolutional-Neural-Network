from baselayer import Layer


class Flatten(Layer):
    def __init__(self, layer_name):
        super().__init__(layer_name)
        self.input_shape = None

    def forward(self, input_activations):
        self.input_shape = input_activations.shape
        return input_activations.ravel().reshape(self.input_shape[0], -1)

    def backward(self, global_gradients, learning_rate):
        return global_gradients.reshape(self.input_shape)
