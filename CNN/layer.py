class Layer:
    """
    Base class for all the layers in the model.

    """

    def __init__(self, layer_name):
        self.layer_name = layer_name

    def forward(self, input_activations):
        print(f"Performing forward pass for the layer {self.layer_name}")
        pass

    def backward(self):
        print(f"Performing backward pass for the layer {self.layer_name}")
        pass
