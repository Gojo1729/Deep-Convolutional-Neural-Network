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
        self.train_loss = []
        self.validation_loss = []

    def train(
        self, train_data, validation_data, epochs, batch_size, learning_rate, debug=True
    ):
        x_train, y_train = train_data
        train_acc, validation_acc = 0, 0
        self.train_loss = []
        for epoch in range(epochs):
            print("-" * 10)
            print(f"Epoch {epoch+1}")
            epoch_start = time.time()
            y_predictions = []
            epoch_loss = 0
            permuted_idx = np.random.permutation(range(x_train.shape[0]))
            x_permuted = x_train[permuted_idx]
            y_permuted = y_train[permuted_idx]

            for idx, (x_batch, y_batch) in enumerate(
                generate_batches(x_permuted, y_permuted, batch_size)
            ):
                print(
                    f"\r Epoch Progress {((idx/(x_train.shape[0]/batch_size)) * 100):.2f} %",
                    end="",
                )
                y_predicted = self._forward(x_batch)
                epoch_loss += self.cce_loss.forward(y_predicted, y_batch).mean()
                global_grad = self.cce_loss.backward(
                    self.cce_loss.cached_output, y_batch
                )
                self._backward(global_grad, learning_rate)
                y_predictions = np.append(y_predictions, np.argmax(y_predicted, axis=1))

            # train metrics
            train_acc = accuracy(y_predictions, y_permuted)
            self.train_accuracy.append(train_acc)
            self.train_loss.append(epoch_loss)

            # validation metrics
            validation_acc, validation_loss = self._validate(validation_data)
            self.validation_accuracy.append(validation_acc)
            self.validation_loss.append(validation_loss)

            # epoch metrics
            print(f"\nTime {time.time() - epoch_start} seconds")
            print(f"Train Accuracy {train_acc}, Validation accuracy {validation_acc}")
            print(f"Train Loss {epoch_loss}, Validation loss {validation_loss}")
            print("-" * 10)

    def _forward(self, x_batch):
        prev_layer_activation = x_batch
        for layer in self.layers:
            prev_layer_activation = layer.forward(prev_layer_activation)
        return prev_layer_activation

    def _backward(self, global_grad, learning_rate):
        """
        Make sure you call this after softmax cross entropy loss backward
        """
        prev_layer_grad = global_grad
        for layer in reversed(self.layers):
            prev_layer_grad = layer.backward(prev_layer_grad, learning_rate)

    def test(self, test_data):
        x_test, y_test = test_data
        test_accuracy, test_loss = 0, 0
        y_pred_test = self._forward(x_test)
        test_loss = self.cce_loss.forward(y_pred_test, y_test)
        test_accuracy = accuracy(np.argmax(y_pred_test, axis=1), y_test)

        return test_accuracy, test_loss.mean()

    def _validate(self, validation_data):
        x_validate, y_validate = validation_data
        validation_acc, valid_loss = 0, 0
        y_pred_validate = self._forward(x_validate)
        valid_loss = self.cce_loss.forward(y_pred_validate, y_validate)
        validation_acc = accuracy(np.argmax(y_pred_validate, axis=1), y_validate)

        return validation_acc, valid_loss.mean()
