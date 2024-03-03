import pyaudio
import struct
import math
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time
from scipy.io.wavfile import write

THRESHOLD = 0 # dB
RATE = 44100
INPUT_BLOCK_TIME = 1 # 30 ms
INPUT_FRAMES_PER_BLOCK = int(RATE * INPUT_BLOCK_TIME)
INPUT_FRAMES_PER_BLOCK_BUFFER = int(RATE * INPUT_BLOCK_TIME)

def get_rms(block):
    return np.sqrt(np.mean(np.square(block)))

class AudioHandler(object):
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = self.open_mic_stream()
        self.threshold = THRESHOLD
        self.plot_counter = 0

    def stop(self):
        self.stream.close()

    def find_input_device(self):
        device_index = None
        for i in range( self.pa.get_device_count() ):
            devinfo = self.pa.get_device_info_by_index(i)
            print('Device %{}: %{}'.format(i, devinfo['name']))

            for keyword in ['mic','input']:
                if keyword in devinfo['name'].lower():
                    print('Found an input: device {} - {}'.format(i, devinfo['name']))
                    device_index = i
                    return device_index

        if device_index == None:
            print('No preferred input found; using default input device.')

        return device_index

    def open_mic_stream( self ):
        device_index = self.find_input_device()

        stream = self.pa.open(  format = self.pa.get_format_from_width(2,False),
                                channels = 1,
                                rate = RATE,
                                input = True,
                                input_device_index = device_index)

        stream.start_stream()
        return stream

    def processBlock(self, snd_block):
        f, t, Sxx = signal.spectrogram(snd_block, RATE)
        zmin = Sxx.min()
        zmax = Sxx.max()
        plt.pcolormesh(t, f, Sxx, cmap='RdBu', norm=LogNorm(vmin=zmin, vmax=zmax))
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.axis([t.min(), t.max(), f.min(), f.max()])
        plt.colorbar()
        plt.savefig('data/spec{}.png'.format(self.plot_counter), bbox_inches='tight')
        plt.close()
        write('data/audio{}.wav'.format(self.plot_counter),RATE,snd_block)
        self.plot_counter += 1

    def listen(self):
        try:
            print("start"), self.stream.is_active(), self.stream.is_stopped()
            #raw_block = self.stream.read(INPUT_FRAMES_PER_BLOCK, exception_on_overflow = False)

            total = 0
            t_snd_block = []
            while total < INPUT_FRAMES_PER_BLOCK:
                while self.stream.get_read_available() <= 0:
                  print('waiting')
                  time.sleep(0.01)
                while self.stream.get_read_available() > 0 and total < INPUT_FRAMES_PER_BLOCK:
                    raw_block = self.stream.read(self.stream.get_read_available(), exception_on_overflow = False)
                    count = len(raw_block) / 2
                    total = total + count
                    print("done", total,count)
                    format = '%dh' % (count)
                    t_snd_block.append(np.fromstring(raw_block,dtype=np.int16))
            snd_block = np.hstack(t_snd_block)
        except Exception as e:
            print('Error recording: {}'.format(e))
            return

        self.processBlock(snd_block)

if __name__ == '__main__':
    audio = AudioHandler()
    for i in range(0,5):
        audio.listen()
