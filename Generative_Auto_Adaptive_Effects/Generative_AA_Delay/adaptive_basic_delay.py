import numpy as np
from Generative_Auto_Adaptive_Effects.Generative_AA_Delay.lowpass_filter_delay import lowpass_filter

def adaptive_basic_delay(
        x,
        sample_rate,
        block_size,
        markov_chain,
        gain,
        cutoff_freq=20.0,
        display_window_start=0,
        display_window_count=10,
):
    """
    Applies an adaptive delay effect to an input signal using a Markov chain to determine delay times.

    Parameters:
        x (numpy.ndarray): The input audio signal.
        sample_rate (float): The sample rate of the audio signal in Hz.
        block_size (int): The size of each processing block in samples.
        markov_chain (object): A Markov chain object that determines delay times based on normalized RMS values.
        gain (float): The gain applied to the delayed signal.
        cutoff_freq (float, optional): The cutoff frequency for the lowpass filter applied to RMS values. Default is 20.0 Hz.
        display_window_start (int, optional): The starting index of the display window for processed data. Default is 0.
        display_window_count (int, optional): The number of display windows to show. Default is 10.

    Returns:
        tuple:
            numpy.ndarray: The processed audio signal with the adaptive delay effect applied.
            dict: A dictionary containing metadata about the processed signal, including:
                - 'start_time': The start time of the displayed windows in seconds.
                - 'end_time': The end time of the displayed windows in seconds.
                - 'extended_end_time': The extended end time considering delay effects in seconds.
                - 'block_size': The size of each processing block in samples.
    """
    # Initialize lists for window data
    input_values = []  # Stores input values for display windows
    rms_values = []  # Stores RMS values for display windows
    normalized_rms_values = []  # Stores normalized RMS values for display windows
    current_states = []  # Stores current Markov chain states for display windows
    next_states = []  # Stores next Markov chain states for display windows
    delay_times = []  # Stores delay times for display windows
    output_values = []  # Stores output values for display windows
    window_times = []  # Stores window times for display windows

    # Calculate the number of blocks and the end of the display window
    num_blocks = len(x) // block_size
    display_window_end = min(display_window_start + display_window_count, num_blocks)

    # Calculate the maximum delay in samples and initialize the output signal
    max_delay_samples = int(np.ceil(max(markov_chain.delay_times["long"]) * sample_rate))
    output_length = len(x) + max_delay_samples
    y = np.zeros(output_length, dtype=x.dtype)
    y[:len(x)] = x

    # Compute RMS values for each block
    all_rms_values = []
    for i in range(num_blocks):
        block_start = i * block_size
        block_end = min(block_start + block_size, len(x))
        frame = x[block_start:block_end]
        block_rms = np.sqrt(np.mean(frame ** 2)) if len(frame) > 0 else 0
        all_rms_values.append(block_rms)

    # Apply a lowpass filter to RMS values if conditions are met
    all_rms_values = np.array(all_rms_values)
    if len(all_rms_values) > 15 and cutoff_freq > 0:
        filtered_rms_values = lowpass_filter(all_rms_values, cutoff_freq, sample_rate / block_size, order=4)
    else:
        filtered_rms_values = all_rms_values

    # Normalize the filtered RMS values
    rms_min = min(filtered_rms_values)
    rms_max = max(filtered_rms_values) + 1e-6

    # Process each block and apply the adaptive delay effect
    for i in range(num_blocks):
        block_start = i * block_size
        block_end = min(block_start + block_size, len(x))
        window_time = block_start / sample_rate
        block_middle = block_start + (block_end - block_start) // 2
        input_value = x[block_middle] if block_middle < len(x) else 0
        raw_rms = filtered_rms_values[i]
        norm_rms = (raw_rms - rms_min) / (rms_max - rms_min)
        current_state = markov_chain.current_state
        next_state = markov_chain.next_state(norm_rms)
        delay_time = markov_chain.get_delay_time()
        delay_samples = int(np.ceil(delay_time * sample_rate))
        for n in range(block_start, block_end):
            if n >= delay_samples:
                y[n] += gain * x[n - delay_samples]
        output_value = y[block_middle] if block_middle < len(y) else 0
        if display_window_start <= i < display_window_end:
            window_times.append(window_time)
            input_values.append(input_value)
            rms_values.append(raw_rms)
            normalized_rms_values.append(norm_rms)
            current_states.append(current_state)
            next_states.append(next_state)
            delay_times.append(delay_time)
            output_values.append(output_value)

    # Print window data after processing all blocks
    window_range = f"{display_window_start} to {display_window_end - 1}"
    print(f"\nWindows {window_range} of processed data:")
    print(
        f"{'Window Time (s)':<15} "
        f"{'Input Value':<15} "
        f"{'RMS Value':<15} "
        f"{'Normalized RMS':<15} "
        f"{'Current State':<15} "
        f"{'Next State':<15} "
        f"{'Delay Time':<15} "
        f"{'Output Value':<15}"
    )
    print("-" * 120)

    def format_number(value, threshold=0.0001):
        """
        Formats a number for display, removing trailing zeros for small values.

        Parameters:
            value (float): The number to format.
            threshold (float): The threshold below which scientific notation is used.

        Returns:
            str: The formatted number as a string.
        """
        if abs(value) >= threshold:
            return f"{value:.4f}".rstrip('0').rstrip('.')
        else:
            return f"{value:.4e}"

    # Display the selected window data
    if window_times:
        for i in range(len(window_times)):
            time_str = format_number(window_times[i])
            input_str = format_number(input_values[i])
            rms_str = format_number(rms_values[i])
            norm_rms_str = format_number(normalized_rms_values[i])
            delay_str = format_number(delay_times[i])
            output_str = format_number(output_values[i])
            print(
                f"{time_str:<15} "
                f"{input_str:<15} "
                f"{rms_str:<15} "
                f"{norm_rms_str:<15} "
                f"{current_states[i]:<15} "
                f"{next_states[i]:<15} "
                f"{delay_str:<15} "
                f"{output_str:<15}"
            )
        total_time = window_times[-1] + (block_size / sample_rate) - window_times[0]
        max_display_delay = max(delay_times) if delay_times else 0
        extended_end_time = window_times[-1] + (block_size / sample_rate) + max_display_delay
        print(
            f"\nThe displayed windows represent {total_time:.4f} seconds of the processed signal "
            f"(from {window_times[0]:.4f}s to {window_times[-1] + (block_size / sample_rate):.4f}s)."
        )
        print(f"With delay effects visible until {extended_end_time:.4f}s")
    else:
        print("No windows selected for display (window_times is empty).")

    # Return the processed signal and metadata
    return y, {
        'start_time': window_times[0] if window_times else 0,
        'end_time': window_times[-1] + (block_size / sample_rate) if window_times else 0,
        'extended_end_time': (window_times[-1] + (block_size / sample_rate) + max(
            delay_times)) if window_times and delay_times else 0,
        'block_size': block_size,
        'window_data': {
            'window_times': window_times,
            'input_values': input_values,
            'rms_values': rms_values,
            'normalized_rms_values': normalized_rms_values,
            'current_states': current_states,
            'next_states': next_states,
            'delay_times': delay_times,
            'output_values': output_values,
        }
    }
