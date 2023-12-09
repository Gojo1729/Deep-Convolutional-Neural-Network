class Layer:
    """
    Base class for all the layers in the model.

    """

    def __init__(self, layer_name):
        self.layer_name = layer_name
        self.layer_log = ""
        self.dashes = "-" * 10 + "\n"
        self.init = True

    def forward(self, input_activations):
        print(f"Performing forward pass for the layer {self.layer_name}")
        pass

    def backward(self):
        print(f"Performing backward pass for the layer {self.layer_name}")
        pass

    def layer_info(self):
        return f"Layer Name -> {self.layer_name}\n"

    def reset_weights(self):
        self.init = True
