class Layer:
    """
    Base class for all the layers in the model.

    """

    def __init__(self, layer_name):
        self.layer_name = layer_name

    def forward(self, input_activations):
        pass

    def backward(self):
        pass
