import soundfile as sf

from Utils.Constants.path_constants import AUDIO_INPUTS_DIR, DELAY_MIC_OUTPUTS_DIR
from Utils.Process_Audio_Input.mic_recorder import MicRecorder
from Utils.wait_for_spacebar import wait_for_spacebar


def process_mic_input(
        input_file_name,
        output_file_name,
        audio_effect,
        sr=44_100,
        input_file_path=AUDIO_INPUTS_DIR,
        output_file_path=DELAY_MIC_OUTPUTS_DIR,
        **effect_kwargs,
):
    """
    Processes microphone input by recording audio, saving the raw input,
    applying a specified audio effect, and saving the processed output.

    Args:
        input_file_name (str): The name of the file to save the raw recorded audio.
        output_file_name (str): The name of the file to save the processed audio.
        audio_effect (callable): A function that applies an audio effect to the input signal.
        sr (int, optional): The sampling rate for recording. Defaults to 44,100 Hz.
        input_file_path (str, optional): The directory path where the raw input file will be saved. Defaults to AUDIO_INPUT_DIR.
        output_file_path (str, optional): The directory path where the processed output file will be saved. Defaults to DELAY_MIC_OUTPUTS_DIR.
        **effect_kwargs: Additional keyword arguments to pass to the `audio_effect` function.

    Returns:
        np.ndarray or None: The processed audio signal, or None if no audio was recorded.
    """
    sampling_rate = sr
    recorder = MicRecorder(sampling_rate)

    # Construct the full paths for the input and output files
    input_file = f"{input_file_path}/{input_file_name}"
    output_file = f"{output_file_path}/{output_file_name}"

    # Prompt the user to start recording
    print("Press spacebar to start recording")
    wait_for_spacebar()

    # Start recording audio
    recorder.start_recording()

    # Prompt the user to stop recording
    wait_for_spacebar()

    # Stop recording and retrieve the recorded audio signal
    recorder.stop_recording()
    input_signal = recorder.get_recorded_audio()

    output_signal = None
    window_info = None

    # Process the recorded audio if it contains data
    if len(input_signal) > 0:
        # Save the raw recorded audio to the input file
        sf.write(input_file, input_signal, sampling_rate)
        print(f"Original signal saved to {input_file}")

        # Apply the audio effect to the recorded signal and save the processed output
        output_signal, window_info = audio_effect(
            input_signal=input_signal,
            output_filename=output_file,
            sampling_rate=sampling_rate,
            **effect_kwargs,
        )

    return output_signal, window_info