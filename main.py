import os

from Audio_Effects.Delay_Effect.apply_delay_effect import apply_delay_effect
from Generative_Auto_Adaptive_Effects.Generative_AA_Delay.process_audio_gen_aa_delay import process_audio_with_generative_adaptive_delay
from Utils.save_window_data import save_window_data_to_txt
from Utils.Constants.path_constants import (
    CLEAN_AUDIO_DIR,
    DELAYED_AUDIO_DIR,
    DELAY_TEXT_FILES_DIR,
    MATRICES_DELAY_DIR,
    METADATA_DELAY_DIR,
)
from Utils.Process_Audio_Input.process_mic_input import process_mic_input
from Utils.Process_Audio_Input.process_wav_input import process_wav_input

# Input audio file name to be processed
input_audio_file_name = "sample_guitar_01.wav"
# Extracts the base name (without extension) from the input audio file
base_name = os.path.splitext(input_audio_file_name)[0]
# Output audio file name after processing, with a suffix indicating the effect/version
output_audio_file_name = f"{base_name}_005.wav"

# Name and path for the transition matrix used in generative adaptive delay
transition_matrix_name = "matrix_delay_run_2025.08.04.16.44.41_001.npy"  # Transition matrix file name
transition_matrix_path = f"{MATRICES_DELAY_DIR}/{transition_matrix_name}"  # Full path to the transition matrix

# Reference to the generative adaptive delay effect function
generative_adaptive_delay_effect = process_audio_with_generative_adaptive_delay

if __name__ == "__main__":
    """
    Main execution block:
    - Applies basic delay effect to audio files.
    - Processes microphone input with generative adaptive delay.
    - Processes WAV input with generative adaptive delay.
    """

    # # Apply basic delay effect to audio files in the specified directories
    # apply_basic_delay = apply_delay_effect(CLEAN_AUDIO_DIR, DELAYED_AUDIO_DIR, METADATA_DELAY_DIR)
    # (Commented out: Applies a basic delay effect to all audio files in the CLEAN_AUDIO_DIR and saves the results.)

    # # Process microphone input with generative adaptive delay effect
    # generative_delay_to_mic, window_info = process_mic_input(
    #     input_file_name=input_audio_file_name,  # Input file name for the microphone recording
    #     output_file_name=output_audio_file_name,  # Output file name for the processed audio
    #     audio_effect=generative_adaptive_delay_effect,  # Generative adaptive delay effect function
    #     transition_matrix_path=transition_matrix_path,  # Path to the transition matrix
    #     display_window_count=10,  # Number of windows to display
    # )
    # (Commented out: Processes microphone input using the generative adaptive delay effect.)

    # Process WAV input with generative adaptive delay effect
    generative_delay_to_wav, window_info = process_wav_input(
        input_file_name=input_audio_file_name,  # Input WAV file name
        output_file_name=output_audio_file_name,  # Output file name for the processed audio
        audio_effect=generative_adaptive_delay_effect,  # Generative adaptive delay effect function
        transition_matrix_path=transition_matrix_path,  # Path to the transition matrix
        display_window_count=10,  # Number of windows to display
    )

    # Generate the output text file name for saving window data
    output_txt_file_name = os.path.splitext(output_audio_file_name)[0] + ".txt"  # Derive the text file name from the output audio file name
    window_data_txt_path = f"{DELAY_TEXT_FILES_DIR}/{output_txt_file_name}"  # Full path to the text file for saving window data

    # Save the window data to a text file
    save_window_data_to_txt(
        window_info['window_data'],  # Window data to save
        window_data_txt_path,  # Path to the output text file
        display_window_start=0,  # Start index for displaying window data
        display_window_end=10  # End index for displaying window data
    )
