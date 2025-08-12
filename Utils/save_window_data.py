def save_window_data_to_txt(window_data, txt_path, display_window_start, display_window_end):
    def format_number(value, threshold=0.0001):
        if abs(value) >= threshold:
            return f"{value:.4f}".rstrip('0').rstrip('.')
        else:
            return f"{value:.4e}"

    lines = []
    header = (
        f"\nWindows {display_window_start} to {display_window_end - 1} of processed data:\n"
        f"{'Window Time (s)':<15} {'Input Value':<15} {'RMS Value':<15} "
        f"{'Normalized RMS':<15} {'Current State':<15} {'Next State':<15} "
        f"{'Delay Time':<15} {'Output Value':<15}\n"
        + "-" * 120
    )
    lines.append(header)
    wt = window_data['window_times']
    if wt:
        for i in range(len(wt)):
            line = (
                f"{format_number(wt[i]):<15} "
                f"{format_number(window_data['input_values'][i]):<15} "
                f"{format_number(window_data['rms_values'][i]):<15} "
                f"{format_number(window_data['normalized_rms_values'][i]):<15} "
                f"{window_data['current_states'][i]:<15} "
                f"{window_data['next_states'][i]:<15} "
                f"{format_number(window_data['delay_times'][i]):<15} "
                f"{format_number(window_data['output_values'][i]):<15}"
            )
            lines.append(line)
    else:
        lines.append("No windows selected for display (window_times is empty).")
    with open(txt_path, "w") as f:
        for line in lines:
            f.write(line + "\n")

    print("Saved window data to text file")