import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import wave
import time


import librosa
import numpy as np
from threading import Thread
from scipy.stats import skew


def extract_features():
    y, sr = librosa.load('output.wav')
    Nfft = 256
    # stft = librosa.stft(y, n_fft=Nfft, window=sig.windows.hamming)
    f0, voiced_flag, voiced_probs = librosa.pyin(y,
                                             fmin=librosa.note_to_hz('C2'),
                                             fmax=librosa.note_to_hz('C7'))
    fun_freq = [i for i in f0 if not np.isnan(i)]
    print(np.mean(fun_freq) * 0.001)

    spec = np.abs(np.fft.rfft(y))
    freq = librosa.fft_frequencies(sr = sr, n_fft=Nfft)
    print("freq1")
    print(freq)
    freq2 = np.fft.rfftfreq(len(y), d=1 / (sr * 0.001))
    print("freq2")
    print(freq2)
    spec = np.abs(spec)
    amp = spec / spec.sum()
    #mean = np.mean(freq)
    mean = (freq * amp).sum()
    sd = np.sqrt(np.sum(amp * ((freq - mean) ** 2)))
    amp_cumsum = np.cumsum(amp)
    median = freq[len(amp_cumsum[amp_cumsum <= 0.5]) + 1]
    mode = freq[amp.argmax()]
    Q25 = freq[len(amp_cumsum[amp_cumsum <= 0.25]) + 1]
    Q75 = freq[len(amp_cumsum[amp_cumsum <= 0.75]) + 1]
    IQR = Q75 - Q25
    z = amp - amp.mean()
    w = amp.std()
    skew = ((z ** 3).sum() / (len(spec) - 1)) / w ** 3
    kurt = ((z ** 4).sum() / (len(spec) - 1)) / w ** 4

    print("Q25: " + str(Q25))
    print("Q75: " + str(Q75))
    print("skew: " + str(skew))
    print("kurt: " + str(kurt))
    print("mean: " + str(mean))
    print("median: " + str(median))
    print("mode: " + str(mode))

    # while i < 200:        
    #     y, sr = librosa.load('output.wav')
    #     librosa.display.waveshow(y, sr=sr, color="blue")

    #     # print(y.shape[0])
    #     # print(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    #     # time.sleep(1)
    #     i += 1


def listen(threshold=60):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    p = pyaudio.PyAudio()

    wf = wave.open('output.wav', 'wb')

    # Set the wave file parameters
    wf.setframerate(RATE)
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)


    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        output=True,
        frames_per_buffer=CHUNK
    )

    stream.start_stream()

    fig, ax = plt.subplots()

    x = np.arange(0, 2 * CHUNK, 2)
    line, = ax.plot(x, np.random.rand(CHUNK))
    ax.set_ylim(-10000, 10000)
    ax.set_xlim(0, CHUNK)

    plt.show(block=False)

    i = 0
    while i < 70:
        data = stream.read(CHUNK)
        # Convert to numpy array
        audio_data = np.frombuffer(data, dtype=np.int16)
        # Calculate the amplitude (volume) of the audio chunk
        amplitude = np.abs(audio_data).mean()
        print(amplitude)
        # Check for silence
        if amplitude > threshold:
            wf.writeframes(data)

        amps = np.frombuffer(data, np.int16)
        line.set_ydata(amps)
        plt.draw()
        fig.canvas.flush_events()
        i += 1

    stream.stop_stream()

# t1 = Thread(target=listen)
# t2 = Thread(target=extract_features)

# t1.start()
# t2.start()
# t1.join()
# t2.join()


import numpy as np
import scipy.signal
import librosa

def specan_python(audio_file):
    # Load the audio file
    y, sr = librosa.load(audio_file)
    
    # Calculate duration
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mean_freq = np.mean(spectral_centroid)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bandwidth = np.mean(spectral_bandwidth)
    
    # Calculate the RMS energy
    rms = librosa.feature.rms(y=y)
    rms_amplitude = np.mean(rms)
    
    # Peak frequency
    # This is a simplified approach; actual peak frequency calculation might require peak detection on the spectrum
    D = np.abs(librosa.stft(y))  # STFT of y
    mean_freq = np.mean(np.mean(D, axis=1)) * sr / len(D)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    mean_f0 = np.nanmean(f0)

    peak_freq = np.argmax(np.mean(D, axis=1)) * sr / len(D)
    
    # Spectral entropy
    spectral_entropy = scipy.stats.entropy(np.mean(D, axis=1))
    
    # Signal to Noise Ratio (SNR) - This is a very basic and not accurate estimation of SNR
    snr = 10 * np.log10(np.mean(y**2) / np.mean((y - np.mean(y))**2))
    
    # Wiener Entropy and ACI are not directly available in librosa and require custom implementation
    # These placeholders are for demonstration purposes
    wiener_entropy = None
    aci = None
    
    results = {
        "Duration (s)": duration,
        "Mean Freq (Hz)": mean_f0,
        "Bandwidth (Hz)": bandwidth,
        "RMS Amplitude": rms_amplitude,
        "Spectral Entropy": spectral_entropy,
        "Signal-to-Noise Ratio (SNR)": snr,
        "Wiener Entropy": wiener_entropy,
        "Acoustic Complexity Index (ACI)": aci
    }
    
    return results

# audio_file = 'output.wav'
# results = specan_python(audio_file)
# print(results)

listen()