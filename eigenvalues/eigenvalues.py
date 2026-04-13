import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    try:
        matrix = np.array(matrix)
        eig_values = np.linalg.eigvals(matrix)
        return eig_values
    except:
        return None