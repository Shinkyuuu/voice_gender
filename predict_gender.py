
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import threading
import time
from collections import deque
import aubio

buf_len = 100
audio_buffer = deque([0.0] * buf_len)
curr_index = 0
fs = 44100.0
pitch_detection = None

def callback(indata, frame_count, time_info, flag):
    global pitch_detection

    if status:
        print(f"Error: {status}")
    else:
        # Extract the mono channel if multiple channels are present
        # normalized_data = indata / np.max(np.abs(indata))
        
        # Perform FFT on the normalized amplitudes
        fft_result = np.fft.fft(indata)

        # Calculate the corresponding frequencies
        frequencies = np.fft.fftfreq(len(fft_result), 1.0 / fs)

        # Calculate the squared magnitudes of FFT coefficients
        magnitude_squared = np.abs(fft_result)**2

        # Calculate the mean frequency
        mean_frequency = np.sum(frequencies * magnitude_squared) / np.sum(magnitude_squared)

        print(mean_frequency)

def save_buffer_to_file(filename, fs):
    """Save the current buffer content to a WAV file."""
    global audio_buffer

    #write(filename, fs, audio_buffer)

def periodic_save(filename, interval, stop_event):
    """Periodically save the buffer to a file."""
    global fs

    while not stop_event.is_set():
        save_buffer_to_file(filename, fs)
        time.sleep(interval)

def start_continuous_recording(write_period, channels=1):
    global fs
    stop_event = threading.Event()
    save_thread = threading.Thread(target=periodic_save, args=("output.wav", write_period, stop_event))

    try:
        with sd.InputStream(samplerate=fs, channels=channels, callback=callback):
            print("Recording started. Press Ctrl+C to stop.")
            save_thread.start()
            while True:
                time.sleep(0.1)  # Keep the main thread alive.
    except KeyboardInterrupt:
        print("Recording stopped by user.")
    finally:
        stop_event.set()
        save_thread.join()


start_continuous_recording(2)
