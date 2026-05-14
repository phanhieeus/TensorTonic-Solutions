import numpy as np
def replay_buffer_sample(buffer, batch_size, seed):
    """
    Sample a batch of transitions from the replay buffer.
    """
    # Write code here
    # rng = np.random.default_rng(seed=seed)
    np.random.seed(seed)
    indices = np.random.choice(len(buffer), size=batch_size, replace=False)
    return [buffer[i] for i in indices]