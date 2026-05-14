import numpy as np

def compute_advantage(states, rewards, V, gamma):
    """
    Returns: A (NumPy array of advantages)
    """
    # Write code here
    G = np.array(rewards, dtype=float)
    for i in range(len(V) - 2, -1, -1):
        G[i] += G[i+1] * gamma
    A = G - V
    return A