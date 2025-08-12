import numpy as np
import noise


def generate_perlin_noise_sequence(length, scale=0.3, seed=None):
    """
    Generates a normalized sequence of Perlin noise values.

    Args:
        length (int): The number of noise values to generate.
        scale (float, optional): The scale factor for the noise. Default is 0.3.
        seed (int, optional): Seed for the random number generator. Default is None.

    Returns:
        np.ndarray: A 1D array of normalized Perlin noise values in the range [0, 1].
    """

    # Set the random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Generate random offsets for each point
    offsets = np.random.uniform(0, 100, size=length)

    # Generate Perlin noise values
    raw_values = np.array([noise.pnoise1(i * scale + offsets[i]) for i in range(length)])

    # Normalize the values to the range [0, 1]
    normalized = (raw_values - raw_values.min()) / (raw_values.max() - raw_values.min())

    # Return the normalized values
    return normalized
