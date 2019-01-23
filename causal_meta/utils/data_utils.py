import torch
import torch.nn as nn
import numpy as np
import scipy
from scipy import interpolate
try:
    import matplotlib.pyplot as plt
except Exception: 
    plt = None

class RandomSplineSCM(nn.Module): 
    def __init__(self, input_noise=False, output_noise=True, 
                 span=6, num_anchors=10, order=3, range_scale=1.): 
        super(RandomSplineSCM, self).__init__()
        self._span = span
        self._num_anchors = num_anchors
        self._range_scale = range_scale
        self._x = np.linspace(-span, span, num_anchors)
        self._y = np.random.uniform(-range_scale * span, range_scale * span, 
                                    size=(num_anchors,))
        self._spline_spec = interpolate.splrep(self._x, self._y, k=order)
        self.input_noise = input_noise
        self.output_noise = output_noise
    
    def forward(self, X, Z=None):
        if Z is None: 
            Z = self.sample(X.shape[0])
        if self.input_noise: 
            X = X + Z
        X_np = X.detach().cpu().numpy().squeeze()
        _Y_np = interpolate.splev(X_np, self._spline_spec)
        _Y = torch.from_numpy(_Y_np).view(-1, 1).float().to(X.device)
        if self.output_noise:
            Y = _Y + Z
        else: 
            Y = _Y
        return Y
        
    def sample(self, N): 
        return torch.normal(torch.zeros(N), torch.ones(N)).view(-1, 1)
    
    def plot(self, X, title="Samples from the SCM", label=None, show=True): 
        Y = self(X)
        if show:
            plt.figure()
            plt.title(title)
        plt.scatter(X.squeeze().numpy(), Y.squeeze().numpy(), marker='+', label=label)
        if show:
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()

def generate_data_categorical(num_samples, pi_A, pi_B_A):
    """Sample data using ancestral sampling
    
    x_A ~ Categorical(pi_A)
    x_B ~ Categorical(pi_B_A[x_A])
    """
    N = pi_A.shape[0]
    r = np.arange(N)
    
    x_A = np.dot(np.random.multinomial(1, pi_A, size=num_samples), r)
    x_Bs = np.zeros((num_samples, N), dtype=np.int64)
    for i in range(num_samples):
        x_Bs[i] = np.random.multinomial(1, pi_B_A[x_A[i]], size=1)
    x_B = np.dot(x_Bs, r)
    
    return np.vstack((x_A, x_B)).T.astype(np.int64)

def generate_data_multivariate_normal(num_samples, mean_A, cov_A, beta_0, beta_1, cov_B_A):
    """ Sample data using ancestral sampling
    
    x_A ~ MultivariateNormal(mean_A, cov_A)
    x_B ~ MultivariateNormal(beta_1 * x_A + beta_0, cov_B_A)
    """
    dim = mean_A.shape[0]
    A = np.random.multivariate_normal(mean_A, cov_A, size=num_samples)  # (num_samples, dim)
    noise = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), size=num_samples)
    scaled_noise =  np.matmul(noise, np.transpose(scipy.linalg.sqrtm(cov_B_A)))  # (num_samples, dim)
    B = np.matmul(A, np.transpose(beta_1)) + beta_0 + scaled_noise

    return np.stack([A, B]).astype(np.float64)  # (2, num_samples, dim)
