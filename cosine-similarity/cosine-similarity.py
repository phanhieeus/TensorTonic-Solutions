import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    a = np.array(a)
    b = np.array(b)
    if (np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2))) == 0: return 0
    return np.linalg.vecdot(a, b) / (np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2)))