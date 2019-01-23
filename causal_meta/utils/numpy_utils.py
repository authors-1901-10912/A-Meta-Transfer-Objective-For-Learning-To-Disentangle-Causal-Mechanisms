import numpy as np

def normalize(x):
    return (x - np.mean(x)) / np.std(x)

def softmax(x, axis=-1):
    exponent = np.exp(x)
    exponent /= np.sum(exponent, axis=axis, keepdims=True)
    return exponent

def random_choice_matrix(p):
    """Generate random samples from the columns of a 2D array

    Parameters
    ----------
    p: np.ndarray
        2D matrix whose columns correspond to individual probability
        vectors, with shape `K x N`. All the columns must sum to 1.

    Returns
    -------
    samples: np.ndarray
        Vector of samples, with shape `N`. All the samples are integers
        between 0 and `K - 1`.
    """
    N = p.shape[1]
    cumsum = np.cumsum(p, axis=0)
    assert np.allclose(cumsum[-1], 1.)
    u = np.random.rand(1, N)
    return np.argmax(u < cumsum, axis=0)
