import numpy as np
import matplotlib.pyplot as plt

def time_shift(n, signal, k):
    """
    Shifts a discrete-time signal by k units.

    Args:
        n (np.ndarray): The original time indices.
        signal (np.ndarray): The original signal values.
        k (int): The amount of shift. A positive k shifts the signal to the right.

    Returns:
        tuple: A tuple containing the new time indices and the shifted signal.
    """
    shifted_n = n + k
    
    plt.stem(shifted_n, signal)
    plt.title(f'Time Shift (k={k})')
    plt.xlabel('n (time index)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

    return shifted_n, signal

def time_scale(n, signal, k):
    """
    Scales the time axis of a discrete-time signal by a factor k.
    
    If k > 1, it performs decimation (downsampling).
    If k < 1, it performs interpolation (upsampling).
    
    Note: For non-integer k, this is a conceptual example for discrete signals.
    A more robust implementation would involve interpolation.

    Args:
        n (np.ndarray): The original time indices.
        signal (np.ndarray): The original signal values.
        k (float): The scaling factor.

    Returns:
        tuple: A tuple containing the new time indices and the scaled signal.
    """
    scaled_n = n[::int(k)]
    scaled_signal = signal[::int(k)]
    
    plt.stem(scaled_n, scaled_signal)
    plt.title(f'Time Scaling (k={k})')
    plt.xlabel('n (time index)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

    return scaled_n, scaled_signal

def signal_addition(n, signal1, signal2):
    """
    Adds two discrete-time signals. The signals must have the same length.

    Args:
        n (np.ndarray): The time indices.
        signal1 (np.ndarray): The first signal.
        signal2 (np.ndarray): The second signal.

    Returns:
        np.ndarray: The resulting signal from the addition.
    """
    if len(signal1) != len(signal2):
        raise ValueError("Signals must have the same length for addition.")
    
    added_signal = signal1 + signal2
    
    plt.stem(n, added_signal)
    plt.title('Signal Addition')
    plt.xlabel('n (time index)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()
    
    return added_signal

def signal_multiplication(n, signal1, signal2):
    """
    Performs point-wise multiplication of two discrete-time signals.

    Args:
        n (np.ndarray): The time indices.
        signal1 (np.ndarray): The first signal.
        signal2 (np.ndarray): The second signal.

    Returns:
        np.ndarray: The resulting signal from the multiplication.
    """
    if len(signal1) != len(signal2):
        raise ValueError("Signals must have the same length for multiplication.")
    
    multiplied_signal = signal1 * signal2
    
    plt.stem(n, multiplied_signal)
    plt.title('Signal Multiplication')
    plt.xlabel('n (time index)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

    return multiplied_signal

# Example Usage:
n = np.arange(-5, 6)
signal_a = np.where(n >= 0, 1, 0)  # Unit step signal
signal_b = np.where(np.abs(n) <= 2, 1, 0) # Rectangular pulse

# Time Shift
time_shift(n, signal_a, 2)

# Time Scale (Decimation)
time_scale(n, signal_a, 2)

# Signal Addition
signal_addition(n, signal_a, signal_b)

# Signal Multiplication
signal_multiplication(n, signal_a, signal_b)