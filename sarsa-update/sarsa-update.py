import numpy as np
def sarsa_update(q_table, state, action, reward, next_state, next_action, alpha, gamma):
    """
    Perform one SARSA update and return the updated Q-table.
    """
    # Write code here
    q_table = np.array(q_table, dtype=float)
    TD_error = reward + gamma * q_table[next_state, next_action] - q_table[state, action]
    q_table[state, action] += alpha * TD_error
    return q_table.tolist()