from scipy.signal import butter, filtfilt


def lowpass_filter(data, cutoff_freq, sample_rate, order=4):
    """
    Apply a Butterworth lowpass filter to smooth data.

    Parameters:
        data (ndarray): Signal to be filtered
        cutoff_freq (float): Cutoff frequency in Hz
        sample_rate (float): Sample rate of signal in Hz
        order (int): Filter order, controls steepness of cutoff

    Returns:
        ndarray: Filtered signal

    Notes:
        Higher order values create steeper cutoffs but may introduce ringing
    """
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)[:2]

    return filtfilt(b, a, data)
