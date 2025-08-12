import numpy as np


def compute_rms_envelope(x, block_size):
    """
    Compute the RMS (Root Mean Square) envelope of an audio signal.

    Parameters:
        x (ndarray): The input audio signal as a 1D numpy array.
        block_size (int): The size of each block to compute the RMS value.

    Returns:
        ndarray: A 1D numpy array containing the RMS values for each block.

    Notes:
        - The input signal is divided into non-overlapping blocks of size `block_size`.
        - The RMS value for each block is calculated as the square root of the mean
          of the squared values within the block.
        - If the length of the input signal is not a multiple of `block_size`,
          the remaining samples are ignored.
    """
    # Calculate the number of complete blocks in the input signal
    num_blocks = len(x) // block_size

    # Initialize an array to store the RMS values for each block
    rms_values = np.zeros(num_blocks)

    # Iterate over each block and compute the RMS value
    for i in range(num_blocks):
        start = i * block_size  # Start index of the current block
        frame = x[start:start + block_size]  # Extract the block
        rms_values[i] = np.sqrt(np.mean(frame ** 2))  # Compute RMS value

    return rms_values
