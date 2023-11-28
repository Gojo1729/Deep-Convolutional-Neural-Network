from typing import List
from baselayer import Layer
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
            permuted_idx = np.random.permutation(range(x_train.shape[0]))
            x_permuted = x_train[permuted_idx]
            y_permuted = y_train[permuted_idx]
            correct_predictions, epoch_loss = 0.0, 0.0
            n_samples = x_train.shape[0]
            for idx, (x_batch, y_batch) in enumerate(
                generate_batches(x_permuted, y_permuted, batch_size)
            ):
                batch_loss = 0
                y_predicted = self._forward(x_batch)
                # assert y_predicted.shape == y_predicted.shape
                batch_loss = self.cce_loss.forward(y_predicted, y_batch)
                epoch_loss += batch_loss
                global_grad = self.cce_loss.backward(
                    self.cce_loss.cached_output, y_batch
                )
                self._backward(global_grad, learning_rate)

                batch_predictions = np.sum(np.argmax(y_predicted, axis=1) == y_batch)
                correct_predictions += batch_predictions
                print(
                    f"\rEpoch Progress {((idx/(x_train.shape[0]/batch_size)) * 100):.2f} %, batch {idx}, batch loss {batch_loss}, batch accuracy {(batch_predictions/x_batch.shape[0]) * 100}",
                    end="",
                )

            # train metrics

            total_epoch_loss = epoch_loss / (n_samples // batch_size)
            total_epoch_accuracy = (correct_predictions / n_samples) * 100
            self.train_accuracy.append(total_epoch_accuracy)
            self.train_loss.append(total_epoch_loss)

            # validation metrics
            validation_acc, validation_loss = self._validate(validation_data)
            self.validation_accuracy.append(validation_acc)
            self.validation_loss.append(validation_loss)

            # epoch metrics
            print(f"\nTime {time.time() - epoch_start} seconds")
            print(
                f"Train Accuracy {total_epoch_accuracy}, Validation accuracy {validation_acc}"
            )
            print(f"Train Loss {total_epoch_loss}, Validation loss {validation_loss}")
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
        test_accuracy = (np.argmax(y_pred_test, axis=1) == y_test).mean() * 100

        return test_accuracy, test_loss

    def _validate(self, validation_data):
        x_validate, y_validate = validation_data
        validation_acc, valid_loss = 0, 0
        y_pred_validate = self._forward(x_validate)
        valid_loss = self.cce_loss.forward(y_pred_validate, y_validate)
        validation_acc = (np.argmax(y_pred_validate, axis=1) == y_validate).mean() * 100

        return validation_acc, valid_loss
