import numpy as np

class MarkovChainDelay:
    """
    A class to represent a Markov Chain for adaptive delay effects.

    Attributes:
        states (list): A list of possible states in the Markov Chain.
        transition_matrix (numpy.ndarray): A square matrix representing the transition probabilities between states.
        delay_times (dict): A dictionary mapping each state to a list of possible delay times.
        current_state (str): The current state of the Markov Chain.

    Methods:
        next_state(modulation_factor=0.0):
            Determines the next state based on the current state and transition probabilities,
            with optional modulation to adjust randomness.

        get_delay_time():
            Retrieves a random delay time from the set of delay times associated with the current state.
    """

    def __init__(self, states, transition_matrix, delay_times):
        """
        Initializes the MarkovChain with states, a transition matrix, and delay times.

        Args:
            states (list): A list of possible states.
            transition_matrix (numpy.ndarray): A square matrix of transition probabilities.
            delay_times (dict): A dictionary mapping states to lists of delay times.
        """
        self.states = states
        self.transition_matrix = transition_matrix
        self.delay_times = delay_times
        self.current_state = np.random.choice(self.states)

    def next_state(self, modulation_factor=0.0):
        """
        Determines the next state in the Markov Chain.

        Args:
            modulation_factor (float, optional): A factor to adjust the randomness of state transitions.
                Values closer to 1 increase uniformity, while values closer to 0 follow the transition matrix.
                Defaults to 0.0.

        Returns:
            str: The next state in the Markov Chain.
        """
        current_index = self.states.index(self.current_state)
        probabilities = self.transition_matrix[current_index]

        # Apply modulation to the transition probabilities
        modulated_probabilities = probabilities * (1 - modulation_factor)
        modulated_probabilities += modulation_factor / len(self.states)
        modulated_probabilities /= modulated_probabilities.sum()

        # Select the next state based on the modulated probabilities
        self.current_state = np.random.choice(self.states, p=modulated_probabilities)
        return self.current_state

    def get_delay_time(self):
        """
        Retrieves a random delay time for the current state.

        Returns:
            float: A randomly selected delay time from the current state's delay time set.
        """
        delay_time_set = self.delay_times[self.current_state]

        return np.random.choice(delay_time_set)
