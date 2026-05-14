import numpy as np

def q_learning_update(Q, s, a, r, s_next, alpha, gamma):
    """
    Returns: updated Q-table Q_new
    """
    # Write code here
    next_Q = np.array(Q[s_next])
    TD_error = r + gamma * (np.max(next_Q)) - Q[s][a]
    Q[s][a] = Q[s][a] + alpha * TD_error
    Q[s][a] = float(Q[s][a])
    return Q