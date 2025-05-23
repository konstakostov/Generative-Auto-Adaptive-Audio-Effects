import numpy as np


class AudioEffect:
    """
    Base class for audio effects.

    Attributes:
        sample_rate (int): The audio sample rate in Hz.
        parameters (dict): Additional effect parameters.
    """

    def __init__(self, sample_rate=44_100, **parameters):
        """
        Initialize the AudioEffect.

        Args:
            sample_rate (int): The audio sample rate in Hz.
            **parameters: Additional effect parameters.
        """

        self.sample_rate = sample_rate
        self.parameters = parameters

    def process(self, input_audio):
        """
        Process the input audio signal.

        This method should be overridden in subclasses to apply the effect.

        Args:
            input_audio (np.ndarray): The input audio signal.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """

        raise NotImplementedError("Subclasses must implement the process method.")


class DelayEffect(AudioEffect):
    def __init__(self, sample_rate=44_100, delay_time=0.5, gain=0.5):
        """
        Initialize the DelayEffect.

        Args:
            sample_rate (int): The audio sample rate in Hz.
            delay_time (float): Delay time in seconds.
            gain (float): Gain applied to the delayed signal.
        """

        super().__init__(sample_rate=sample_rate, delay_time=delay_time, gain=gain)
        self.delay_time = delay_time
        self.gain = gain

    def process(self, input_audio):
        """
        Apply a delay effect to the input audio.

        Args:
            input_audio (np.ndarray): The input audio signal.

        Returns:
            np.ndarray: The audio signal with the delay effect applied.
        """

        delay_in_samples = int(self.delay_time * self.sample_rate)
        output_audio = np.copy(input_audio)

        if delay_in_samples > 0:
            output_audio[delay_in_samples:] += self.gain * input_audio[:-delay_in_samples]

        return output_audio
