import numpy as np
from baselayer import Layer


class ConvLayer(Layer):
    def __init__(self, layer_name, n_filters=8, filter_size=3, stride=1, debug=True):
        super().__init__(layer_name)
        np.random.seed(42)
        self.filter_size = filter_size
        self.stride = stride
        self.n_filters = n_filters
        self.debug = debug
        self.cached_input_activations = None

    def get_patch(self, image, width_idx, height_idx, filter_size):
        return image[
            :,
            :,
            height_idx : height_idx + filter_size,
            width_idx : width_idx + filter_size,
        ]

    def forward(self, input_activations):
        """
        @param input_activations - output of previous layer, will be of shape
        N x C x H x W
        Conv forward

        1. Get the patch of the size similar to conv filter
        2. Perform convolution on that patch and store in an numpy array
        3. Do the above for all the filters
        4. Append all the filters output and return it for next layer

        """
        self.cached_input_activations = input_activations
        first_activation = True
        n_samples, input_channels, input_width, input_height = input_activations.shape

        if self.debug:
            print(f"{n_samples=}, {input_channels=}, {input_width=}, {input_height=}")

        variance = 2 / (input_channels + self.n_filters)
        self.conv_filter = np.random.normal(
            0.0,
            np.sqrt(variance),
            size=(self.n_filters, input_channels, self.filter_size, self.filter_size),
        )

        output_shape = (int((input_width - self.filter_size) / self.stride)) + 1
        output_activations = np.zeros([])

        for conv_filter in self.conv_filter:
            filter_output = np.zeros((n_samples, output_shape, output_shape))
            height_idx = 0
            filter_y = 0

            while (height_idx + self.filter_size) <= input_height:
                width_idx = 0
                filter_x = 0
                while (width_idx + self.filter_size) <= input_width:
                    patch = self.get_patch(
                        input_activations, width_idx, height_idx, self.filter_size
                    )

                    conv = np.sum(conv_filter * patch, axis=(1, 2, 3))
                    if self.debug:
                        print(
                            f"{height_idx=}, {width_idx=}, {filter_output.shape=}, {patch.shape=}, {conv.shape=}"
                        )

                    filter_output[:, filter_y, filter_x] = conv
                    width_idx += self.stride
                    filter_x += 1

                height_idx += self.stride
                filter_y += 1
                # filter_output = filter_output.reshape(output_shape, output_shape)

            if first_activation:
                output_activations = filter_output
                output_activations = np.expand_dims(output_activations, axis=1)
                first_activation = False

            else:
                filter_output = np.expand_dims(filter_output, axis=1)
                if self.debug:
                    print(f"{output_activations.shape=}, {filter_output.shape=}")
                output_activations = np.append(
                    output_activations, filter_output, axis=1
                )

            if self.debug:
                print(f"{output_activations.shape=}, {filter_output.shape=}")

        self.cached_output_activations = output_activations
        return output_activations

    def backward(self, global_gradients, learning_rate):
        """
        Backward propagation for the convolutional filters

        https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf

        dL_input = global_gradients * filter
        dL_filter = global_gradients * local_gradients

        """
        (
            n_samples,
            input_channels,
            input_width,
            input_height,
        ) = self.cached_input_activations.shape
        first_activation = True
        output_gradients = np.zeros_like(
            self.cached_input_activations, dtype=np.float64
        )
        local_gradients = np.zeros_like(self.conv_filter)

        for c_id, conv_filter in enumerate(self.conv_filter):
            # print(f"entering loop {c_id}")
            height_idx = 0
            filter_y = 0
            # filter_grad = np.zeros(shape=(self.n_filters, self.filter_size, self.filter_size))

            while height_idx + self.filter_size <= input_height:
                width_idx = 0
                filter_x = 0

                while width_idx + self.filter_size <= input_width:
                    patch = self.get_patch(
                        self.cached_input_activations,
                        width_idx,
                        height_idx,
                        self.filter_size,
                    )
                    global_patch = global_gradients[:, c_id, filter_y, filter_x][
                        :, np.newaxis, np.newaxis, np.newaxis
                    ]
                    prod_sum = np.sum((global_patch * patch), axis=0)
                    local_gradients[np.newaxis, c_id] += prod_sum
                    temp = (global_patch) * conv_filter
                    output_gradients[
                        :,
                        :,
                        height_idx : height_idx + self.filter_size,
                        width_idx : width_idx + self.filter_size,
                    ] += temp

                    width_idx += self.stride
                    filter_x += 1

                height_idx += self.stride
                filter_y += 1

            # print(f"{local_gradients.shape=}, {filter_grad.shape=}")

            # local_gradients[c_id] = filter_grad[c_id]

        if self.debug:
            print(f"{local_gradients.shape=}")
        self.conv_filter -= learning_rate * (local_gradients / n_samples)

        return output_gradients
