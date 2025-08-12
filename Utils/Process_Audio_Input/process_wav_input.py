import numpy as np
import soundfile as sf

from Utils.Constants.path_constants import DELAY_WAV_OUTPUTS_DIR, AUDIO_INPUTS_DIR

def process_wav_input(
    input_file_name,
    output_file_name,
    audio_effect,
    input_file_path=AUDIO_INPUTS_DIR,
    output_file_path=DELAY_WAV_OUTPUTS_DIR,
    **effect_kwargs,
):
    """
    Processes a WAV audio file by applying a specified audio effect.

    Args:
        input_file_name (str): The name of the input WAV file.
        output_file_name (str): The name of the output WAV file.
        audio_effect (callable): A function that applies an audio effect to the input signal.
        input_file_path (str, optional): The directory path where the input file is located. Defaults to AUDIO_INPUT_DIR.
        output_file_path (str, optional): The directory path where the output file will be saved. Defaults to DELAY_MIC_OUTPUTS_DIR.
        **effect_kwargs: Additional keyword arguments to pass to the `audio_effect` function.

    Returns:
        np.ndarray: The processed audio signal.
    """
    # Construct the full paths for the input and output files
    input_file = f"{input_file_path}/{input_file_name}"
    output_file = f"{output_file_path}/{output_file_name}"

    # Read the input WAV file
    input_signal, sampling_rate = sf.read(input_file)

    # If the input signal has multiple channels, use only the first channel
    if input_signal.ndim > 1:
        input_signal = input_signal[:, 0]

    # Convert the input signal to 32-bit floating point
    input_signal = input_signal.astype(np.float32)

    # Apply the audio effect to the input signal
    output_signal, window_info = audio_effect(
        input_signal=input_signal,
        output_filename=output_file,
        sampling_rate=sampling_rate,
        **effect_kwargs,
    )

    # Return the processed audio signal
    return output_signal, window_info