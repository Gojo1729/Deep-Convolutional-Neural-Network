from typing import List
from layer import Layer
import time
import numpy as np
from activations import Softmax_Categorical_CrossEntropy
from utils import accuracy, generate_batches


class Sequential:
    def __init__(self, layers: List[Layer]):
        self.layers = layers
        self.cce_loss: Softmax_Categorical_CrossEntropy = (
            Softmax_Categorical_CrossEntropy("Softmax_CCE")
        )
        self.train_accuracy = []
        self.validation_accuracy = []
        self.test_accuracy = []

        self.train_loss = []
        self.validation_loss = []
        self.test_loss = []

    def train(
        self, train_data, validation_data, test_data, epochs, batch_size, debug=True
    ):
        x_train, y_train = train_data
        x_validation, y_validation = validation_data
        x_test, y_test = test_data
        train_acc, validation_acc, test_acc = 0, 0, 0

        for epoch in range(epochs):
            epoch_start = time.time()
            y_predictions = []
            train_loss, validation_loss, test_loss = 0, 0, 0

            # for every tenth of epochs print the metrics

            permuted_idx = np.random.permutation(range(x_train.shape[0]))
            x_permuted = x_train[permuted_idx]
            y_permuted = y_train[permuted_idx]

            for idx, (x_batch, y_batch) in enumerate(
                generate_batches(x_permuted, y_permuted, batch_size)
            ):
                y_predicted = self._forward(x_batch)

                # print(
                #     f"Xbatch size {x_batch.shape}, y_predicted_shape {y_predicted.shape}, y_true shape {y_batch.shape}"
                # )
                train_loss += self.cce_loss.forward(y_predicted, y_batch)

                global_grad = self.cce_loss.backward(
                    self.cce_loss.cached_output, y_batch
                )

                # print(f"global grad shape {global_grad.shape}")
                self._backward(global_grad, 1e-02)

                y_predictions.append(np.argmax(y_predicted))

            train_acc = accuracy(y_predictions, y_permuted)
            self.train_accuracy.append(train_acc)

            self.train_loss.append(train_loss / len(y_permuted))

            if epoch:
                print("-" * 10)
                print(f"Epoch {epoch}, Time {time.time() - epoch_start} seconds")
                print(
                    f"Train Accuracy {train_acc}, Validation accuracy {validation_acc}, Test accuracy {test_acc}"
                )

                print(
                    f"Train Loss {train_loss}, Validation loss {validation_loss}, Test loss {test_loss}"
                )
                print("-" * 10)

    def _forward(self, x_batch):
        prev_layer_activation = x_batch
        for layer in self.layers:
            prev_layer_activation = layer.forward(prev_layer_activation)
            # print(
            #     f"ypred_shape of layer {layer.layer_name}, shape {prev_layer_activation.shape}"
            # )

        return prev_layer_activation

    def _backward(self, global_grad, learning_rate):
        """
        Make sure you call this after softmax cross entropy loss backward
        """
        prev_layer_grad = global_grad

        for layer in reversed(self.layers):
            prev_layer_grad = layer.backward(prev_layer_grad, learning_rate)
