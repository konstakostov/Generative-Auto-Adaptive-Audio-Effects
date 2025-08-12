import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _plot_waveform(time, original, processed, output_path, legend_handles):
    """
    Helper function to plot waveforms of original and processed audio signals.

    Parameters:
        time (ndarray): The time axis for the plot.
        original (ndarray): The original audio signal.
        processed (ndarray): The processed audio signal.
        output_path (str): The file path to save the plot.
        legend_handles (list): List of Line2D objects for the plot legend.

    Notes:
        - The function creates a plot with the given signals and saves it to the specified path.
        - The plot includes a legend, grid, and labeled axes.
    """
    plt.figure(figsize=(10, 8))
    plt.plot(
        time,
        original,
        label="Input Audio Signal",
        color="red",
        linestyle="solid",
        linewidth=1.0,
        alpha=1.0)
    plt.plot(
        time,
        processed,
        label="Delayed Input Audio Signal",
        color="blue",
        linestyle="solid",
        linewidth=1.0,
        alpha=0.7)
    plt.legend(handles=legend_handles)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_waveforms_delay(
        original,
        processed,
        sample_rate,
        window_info=None,
        output_filename="waveform_comparison.png",
):
    """
    Generate comparison plots of original and processed audio waveforms.

    Parameters:
        original (ndarray): The original audio signal.
        processed (ndarray): The processed audio signal.
        sample_rate (float): The sample rate of the audio signals in Hz.
        window_info (dict, optional): Dictionary containing start_time, end_time, and extended_end_time for detailed view.
        output_filename (str, optional): Path to save the waveform comparison plots. Default is "waveform_comparison.png".

    Notes:
        - Two plots are generated: an overlay of the full signals and a detailed view of a specific window.
        - The detailed view uses the window range specified in `window_info` or a default range.
        - The plots are saved as separate files with "_overlay" and "_detail" suffixes.
    """
    # Determine the minimum length of the two signals
    min_len = min(len(original), len(processed))
    time_axis = np.linspace(0, min_len / sample_rate, num=min_len)
    output_base = output_filename.rsplit('.', 1)[0]
    output_ext = output_filename.split('.')[-1]

    # Define legend handles for the plots
    legend_handles = [
        Line2D([0], [0], color="red", lw=2, label="Input Audio Signal"),
        Line2D([0], [0], color="blue", lw=2, label="Delayed Input Audio Signal"),
    ]

    # Generate the overlay plot
    _plot_waveform(
        time_axis,
        original[:min_len],
        processed[:min_len],
        f"{output_base}_overlay.{output_ext}",
        legend_handles
    )

    # Determine the time range for the detailed view
    if window_info:
        start_time = window_info['start_time']
        end_time = window_info.get('extended_end_time', window_info['end_time'])
    else:
        block_size = 8192
        start_time = 0
        end_time = (10 * block_size) / sample_rate

    # Calculate start and end indices for the detailed view
    start_idx = int(start_time * sample_rate)
    end_idx = int(end_time * sample_rate)

    # Clip indices to the lengths of the signals
    orig_end_idx = min(end_idx, len(original))
    proc_end_idx = min(end_idx, len(processed))

    # Generate time axes for the detailed view
    orig_time = np.linspace(start_time, start_time + (orig_end_idx - start_idx) / sample_rate, orig_end_idx - start_idx)
    proc_time = np.linspace(start_time, start_time + (proc_end_idx - start_idx) / sample_rate, proc_end_idx - start_idx)

    # Use the shorter time axis for both signals
    detail_len = min(len(orig_time), len(proc_time))
    _plot_waveform(
        orig_time[:detail_len],
        original[start_idx:start_idx + detail_len],
        processed[start_idx:start_idx + detail_len],
        f"{output_base}_detail.{output_ext}",
        legend_handles
    )

    print(f"Waveform plots saved as {output_base}_overlay.{output_ext} and {output_base}_detail.{output_ext}")
