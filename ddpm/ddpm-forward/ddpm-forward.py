import numpy as np

def get_alpha_bar(betas):
    """
    Compute cumulative product of (1 - beta).
    Returns list of floats rounded to 6 decimals.
    """
    betas = np.asarray(betas, dtype=np.float64)
    alpha_bar = np.cumprod(1 - betas)
    return np.round(alpha_bar, 6)

def forward_diffusion(x_0, t, betas, epsilon):
    """
    Returns: tuple of (np.ndarray x_t, np.ndarray epsilon) with same shape as x_0
    """
    x_0 = np.array(x_0)
    alpha_bar = get_alpha_bar(betas)
    alpha_bar = alpha_bar[t - 1]
    epsilon = np.asarray(epsilon, dtype=np.float64)
    x_t = np.sqrt(alpha_bar) * x_0 + np.sqrt(1 - alpha_bar) * epsilon
    x_t = np.round(x_t, 6)
    return x_t.tolist()