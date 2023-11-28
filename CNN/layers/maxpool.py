from baselayer import Layer
import numpy as np


class MaxPool(Layer):  # max pooling layer using pool size equal to 2
    def __init__(self, layer_name, stride=2, pool_size=2):
        super().__init__(layer_name)
        self.cached_input = None
        self.pool_stride = stride
        self.pool_size = pool_size
        self.mask_caches = {}

    def forward(self, activation):
        self.cached_input = activation

        n_samples, n_channels, h_prev, w_prev = activation.shape
        downsampled_h = (
            (h_prev - self.pool_size) // self.pool_stride
        ) + 1  # compute output dimensions after the max pooling
        downsampled_w = ((w_prev - self.pool_size) // self.pool_stride) + 1

        downsampled = np.zeros((n_samples, n_channels, downsampled_h, downsampled_w))

        for channel_i in range(n_channels):
            height_idx = d_y = 0
            while height_idx + self.pool_size <= h_prev:
                width_idx = d_x = 0
                while width_idx + self.pool_size <= w_prev:
                    patch = activation[
                        :,
                        channel_i,
                        height_idx : height_idx + self.pool_size,
                        width_idx : width_idx + self.pool_size,
                    ]
                    downsampled[:, channel_i, d_y, d_x] = np.max(patch, axis=(1, 2))
                    width_idx += self.pool_stride
                    d_x += 1

                height_idx += self.pool_stride
                d_y += 1

        return downsampled

    def backward(self, global_gradients, learning_rate):
        n_samples, n_channels, h_prev, w_prev = self.cached_input.shape

        output_gradient = np.zeros_like(self.cached_input)  # initialize derivative

        for c in range(n_channels):
            height_idx = out_y = 0
            while height_idx + self.pool_size <= h_prev:
                width_idx = out_x = 0
                while width_idx + self.pool_size <= w_prev:
                    patch = self.cached_input[
                        :,
                        c,
                        height_idx : height_idx + self.pool_size,
                        width_idx : width_idx + self.pool_size,
                    ]  # obtain index of largest
                    (n, x, y) = np.unravel_index(
                        np.nanargmax(patch), patch.shape
                    )  # value in patch
                    output_gradient[
                        :, c, height_idx + x, width_idx + y
                    ] += global_gradients[:, c, out_y, out_x]
                    width_idx += self.pool_stride
                    out_x += 1
                height_idx += self.pool_stride
                out_y += 1

        return output_gradient
