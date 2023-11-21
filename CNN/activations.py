import numpy as np


def softmax(x):
    exponents = np.exp(x)
    return exponents / np.sum(exponents)


def regularized_softmax(x):
    """
    https://jaykmody.com/blog/stable-softmax/
    """
    return softmax(x - np.max(x))
