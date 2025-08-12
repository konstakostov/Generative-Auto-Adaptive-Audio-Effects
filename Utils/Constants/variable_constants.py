"""
DELAY_STATES_TIMES: dict
    A dictionary defining delay times for different states of audio processing.

    Keys:
        - "short" (list of float): Delay times ranging from 20ms to 79ms,
          generated in increments of 2ms.
        - "medium" (list of float): Delay times ranging from 80ms to 349ms,
          generated in increments of 10ms.
        - "long" (list of float): Delay times ranging from 350ms to 1500ms,
          generated in increments of 40ms.

    Notes:
        - All delay times are represented in seconds (converted from milliseconds).
        - The `round` function ensures precision up to 3 decimal places.
"""

# Define the delay times for different states
DELAY_STATES_TIMES = {
    # Short State (20ms - 79ms), Generated 31 values
    "short": [round(0.001 * x, 3) for x in range(20, 80, 2)],
    # Medium State (80ms - 349ms), Generated 28 values
    "medium": [round(0.001 * x, 3) for x in range(80, 350, 10)],
    # Long State (350ms - 1500ms), Generated 30 values
    "long": [round(0.001 * x, 3) for x in range(350, 1501, 40)],
}

