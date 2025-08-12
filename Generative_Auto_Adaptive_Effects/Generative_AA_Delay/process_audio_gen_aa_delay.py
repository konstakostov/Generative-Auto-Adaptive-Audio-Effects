import numpy as np
import soundfile as sf

from Utils.Constants.variable_constants import DELAY_STATES_TIMES
from Generative_Auto_Adaptive_Effects.Generative_AA_Delay.markov_chain_delay import MarkovChainDelay
from Generative_Auto_Adaptive_Effects.Generative_AA_Delay.adaptive_basic_delay import adaptive_basic_delay
from Generative_Auto_Adaptive_Effects.Generative_AA_Delay.plots_delay import plot_waveforms_delay


def process_audio_with_generative_adaptive_delay(
        input_signal,
        output_filename,
        sampling_rate=44_100,
        frame_size=8192,
        gain_delay=0.4,
        cutoff_frequency=1.0,
        display_window_start=0,
        display_window_count=0,
        waveform_plot=None,
        transition_matrix_path=None,
):
    """
    Process an audio signal using a generative adaptive delay effect based on a Markov chain.

    Parameters:
        input_signal (ndarray): The input audio signal as a 1D numpy array.
        output_filename (str): The file path to save the processed audio signal.
        sampling_rate (int): The sampling rate of the audio signal (default: 44,100 Hz).
        frame_size (int): The size of each frame for processing (default: 8192 samples).
        gain_delay (float): The gain applied to the delayed signal (default: 0.4).
        cutoff_frequency (float): The cutoff frequency for the delay effect in Hz (default: 1.0).
        display_window_start (int): The starting index of the display window for plotting (default: 0).
        display_window_count (int): The number of frames to display in the plot (default: 0).
        waveform_plot (str or None): The file path to save the waveform plot, or None to skip plotting.
        transition_matrix_path (str): The file path to the transition matrix for the Markov chain.

    Returns:
        ndarray: The processed audio signal.

    Raises:
        ValueError: If `transition_matrix_path` is not provided.

    Notes:
        - The transition matrix is loaded from the specified file path and used to define
          the state transitions and delay times for the Markov chain.
        - The processed audio signal is saved to the specified output file.
        - If `waveform_plot` is provided, the input and output waveforms are plotted and saved.
    """
    if transition_matrix_path is None:
        raise ValueError("transition_matrix_path must be provided")

    # Load the transition matrix from the specified file
    transition_matrix = np.load(transition_matrix_path)

    # Define the states for the Markov chain
    states = ["short", "medium", "long"]

    # Use predefined delay times for each state from DELAY_STATES_TIMES
    delay_times = DELAY_STATES_TIMES

    # Initialize the Markov chain with states, transition matrix, and delay times
    markov_chain = MarkovChainDelay(states, transition_matrix, delay_times)

    # Apply the adaptive basic delay effect
    output_signal, window_info = adaptive_basic_delay(
        input_signal,
        sampling_rate,
        frame_size,
        markov_chain,
        gain_delay,
        cutoff_frequency,
        display_window_start,
        display_window_count
    )

    # Save the processed audio signal to the output file
    sf.write(output_filename, output_signal, int(sampling_rate))
    print(f"Processed signal saved to {output_filename}")

    # Plot the input and output waveforms if a plot file path is provided
    if waveform_plot:
        plot_waveforms_delay(input_signal, output_signal, sampling_rate, window_info, waveform_plot)

    return output_signal, window_info
