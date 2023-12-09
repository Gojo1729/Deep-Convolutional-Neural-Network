import numpy as np
from baselayer import Layer


class DenseLayer(Layer):
    def __init__(
        self, layer_name, input_nodes, output_nodes, glorot_init=True, debug=False
    ):
        super().__init__(layer_name)
        self.debug = debug
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.weights = None
        self.bias = None
        # we need to cache the input and output from this layer for the purpose of backprop
        self.cached_input = None
        self.cached_output = None
        self.glorot_activation = glorot_init

    def init_params(self):
        print(f"Init params {self.layer_name}")
        """
        Gloro/Xavier initialization for weights for faster convergence
        https://pyimagesearch.com/2021/05/06/understanding-weight-initialization-for-neural-networks/
        """
        limit = np.sqrt(2 / float(self.input_nodes + self.output_nodes))
        if self.glorot_activation:
            self.weights = np.random.normal(
                0.0, limit, size=(self.input_nodes, self.output_nodes)
            )
        else:
            self.weights = np.random.randn(self.input_nodes, self.output_nodes)

        """ 
        No initialization is required for bias,
        https://cs231n.github.io/neural-networks-2/#:~:text=Initializing%20the%20biases
        """
        self.bias = np.zeros((1, self.output_nodes))

        # we need to cache the input and output from this layer for the purpose of backprop
        self.cached_input = None
        self.cached_output = None
        self.init = False

    def reset_weights(self):
        self.init = True
        self.weights, self.bias = None, None

    # overriden
    def forward(self, input_activations):
        """

        @param input_activations - Output of previous layer
        @return softmax of logits

        Dense layer forward prop
        1. Flatten the input
        2. Dot product with weights and add the bias
        3. Cache the input and logits for backpop
        4. Apply softmax on logits and return it
        """

        # init the weights for the first epoch
        if self.init:
            self.init_params()

        shape = input_activations.shape
        layer_input = input_activations
        layer_logits = np.dot(layer_input, self.weights) + self.bias

        self.cached_input = layer_input
        self.cached_output = layer_logits
        if self.debug:
            print(
                f"Cached output {self.cached_output}, ip shape {layer_input.shape}, op shape {layer_logits.shape}"
            )
        return self.cached_output

    # overriden
    def backward(self, global_gradient, current_lr):
        """
        @param global_gradient, gradients from the previous layer
        @param current_lr


        @returns The gradient of Loss w.r.t to the input to this layer
        i.e the input_activations recieved during forward pass.

        Dense layer backward prop
        1. Calculate the gradients w.r.t to the weights
        2. Calculate the gradients w.r.t to the received activations
        and return it for usage in other previous layers.

        """
        output_grad = global_gradient
        samples = global_gradient.shape[0]

        assert output_grad.shape[1] == self.weights.T.shape[0]
        assert self.cached_input.T.shape[1] == output_grad.shape[0]

        input_grad = np.dot(output_grad, self.weights.T) / samples
        weight_grad = np.dot(self.cached_input.T, output_grad) / samples

        if self.debug:
            print(
                f"weights shape {self.weights.shape}, weight grad shape {weight_grad.shape}, input_shape {self.cached_input.shape}, input_grad shape {input_grad.shape}, "
            )

        self.weights -= current_lr * weight_grad
        self.bias -= current_lr * np.sum(output_grad, axis=0, keepdims=True)

        return input_grad

    def layer_info(self):
        layer_log = f"Layer Name -> {self.layer_name}\n"
        # weights shape
        layer_log += f"Weights shape -> {self.weights.shape}\n"

        return layer_log
