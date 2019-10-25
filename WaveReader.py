from __future__ import division, print_function, absolute_import
from scipy import signal

import numpy
import struct
import warnings
import librosa
import collections
from guppy import hpy
from operator import itemgetter
from PIL import  Image
import matplotlib.pyplot as plt
import sys, os
import numpy as np
class WavFileWarning(UserWarning):
    pass


class WaveReader:
    def __init__(self,filename,mmap=True,debug=False):

        if hasattr(filename, 'read'):
            self.fid = filename
            mmap = False
        else:
            self.fid = open(filename, 'rb')

        self.mmap = mmap

        self._debug = debug
        self._riff_start = 4
        self._fmt_start = 12  # pointer to FMT
        self._fmt_size = 20
        self._data_start = 36  # pointer to data
        self._four_byte = 4
        self._one_bit = 1


        self.fsize = self._read_riff_chunk()
        self.format_chunk_size = 0
        self.data_chunk_size = 0
        self.compression = 0x0001
        self.number_of_channels = 1
        self.sampling_rate = 0
        self.average_bytes_per_second = 0     #sample rate * block align
        self.block_align = 0      # significant bits per sample / (8 * Number of channels)
        self.significant_bits = 8   # Number of bit to define each sample
        self.data_start = 0
        self.data_end = 0
        self._read_fmt_chunk()


        data = self._read_data_chunk()



    def _read_riff_chunk(self):
        str1 = self.fid.read(self._riff_start)
        if(self._debug):
            print("The RIFF is at 0x{}".format(str1.hex()))
        if str1 == b'RIFX':
            _big_endian = True
        elif str1 != b'RIFF':
            raise ValueError("Not a WAV file.")
        fmt = '<I'
        fsize = struct.unpack(fmt, self.fid.read(self._four_byte))[0] + self._four_byte*2
        str2 = self.fid.read(self._four_byte)
        if (self._debug):
            print("The WAVE is at 0x{}".format(str2.hex()))
        if (str2 != b'WAVE'):
            raise ValueError("Not a WAV file.")
        if str1 == b'RIFX':
            _big_endian = True
        return fsize


    def _read_fmt_chunk(self):
        self.fid.seek(0,os.SEEK_SET)
        self.fid.seek(self._fmt_start)
        chunk_id = self.fid.read(self._four_byte)
        if chunk_id == b'fmt ':
            res = struct.unpack('<ihHIIHH', self.fid.read(self._fmt_size))
            size, comp, noc, rate, sbytes, ba, bits = res
            if (self._debug):
                print("The Chunk Data size is {} at {} 4 bytes".format(size, hex(size)))
                print("The Compression code is {} at {} 2 bytes".format(comp, hex(comp)))
                print("The Number of Channel is {} at {} 2 bytes".format(noc, hex(noc)))
                print("The sampling rate is {} at 0x{} 4 bytes".format(rate, hex(rate)))
                print("The Average bytes per second is {} at {} 2 bytes".format(sbytes, hex(sbytes)))
                print("The Block Assign is at {} 0x{} 2 bytes".format(ba, hex(ba)))
                print("The Significant bits per sample is {} at 0x{} 2 bytes".format(bits, hex(bits)))

            if (self.compression != 1 or size > 16):
                if (self.compression == 3):
                    global _ieee
                    _ieee = True
                    # warnings.warn("IEEE format not supported", WavFileWarning)
                else:
                    warnings.warn("Unfamiliar format bytes", WavFileWarning)
                if (size > 16):
                    self.fid.read(size - 16)

            self.format_chunk_size = size
            self.compression = comp
            self.number_of_channels = noc
            self.sampling_rate = rate
            self.average_bytes_per_second = sbytes
            self.block_align = ba
            self.significant_bits = bits
            self.data_start = self.fid.tell()

    def _read_data_chunk(self):
        self.fid.seek(0, os.SEEK_SET)
        self.fid.seek(self._data_start)  # Offsetting the pointer of 12
        chunk_id = self.fid.read(self._four_byte)
        if chunk_id == b'data':
            fmt = '<i'
            size = struct.unpack(fmt, self.fid.read(self._four_byte))[0]
            if (self._debug):
                print("The Data size is {} at {} 4 bytes".format(size, hex(size)))
            self.data_chunk_size = size

            bytes = self.significant_bits // (8*self._one_bit)
            if self.significant_bits == (8*self._one_bit):
                dtype = 'u1'
            else:

                dtype = '<'
                if self.compression == 1:
                    dtype += 'i%d' % bytes
                else:
                    dtype += 'f%d' % bytes
            self.data_start = self.fid.tell()
            if (self._debug):
                print("The data type format is {}".format(dtype))
                print('The data starts at {}'.format(self.data_start))
                print('The data ends at {}'.format(self.data_chunk_size))


    def get_next_data(self,chunk_size):
        self.fid.seek(0, os.SEEK_SET)
        start = self.data_start
        size = self.data_chunk_size
        end = start + chunk_size
        outputstr = []
        while end < size:
            outputstr.append("=")
            print(outputstr)
            data = numpy.memmap(self.fid, dtype='<i2', mode='c', offset=start, shape=(chunk_size // 2,))
            self.fid.seek(end)
            start = start + chunk_size
            end = start + chunk_size
            yield data

    def get_header(self):
        return self.sampling_rate, self.data_chunk_size/(2*self.sampling_rate), self.data_chunk_size

    """
    def print_data(self):
        self.fid.seek(0, os.SEEK_SET)
        self.fid.seek(44)
        cursor_pos = self.data_start
        while cursor_pos < self.data_chunk_size:
            val = struct.unpack('<h', self.fid.read(2))[0]
            #print(val)
            #print(int.from_bytes(val, byteorder='big'))
            break
    """
    def log_specgram(self,audio, sample_rate, window_size=1, step_size=10, eps=1e-10):
        # f, t, spec = signal.spectrogram(audio, fs=sample_rate,window='hann', nperseg=nperseg, noverlap=noverlap,detrend=False,scaling='spectrum',mode='magnitude')
        f, t, spec = signal.spectrogram(audio, fs=sample_rate, window='hann', nperseg=window_size,
                                        noverlap=50, scaling='spectrum', mode='magnitude')
        return 80 + 20 * np.log10((spec.T.astype(np.float32) + eps) / np.max(spec)), f, t



    def get_sepctrogram(self):
        self.global_chunk = 10240 * 10
        self.local_chunk = int(self.global_chunk / 10)
        data = self.get_next_data(self.global_chunk)
        fs, tim, total_samples = self.get_header()
        freq = np.linspace(0, fs / 2, int(self.local_chunk / 2) +1)
        timeline = np.linspace(0, tim, int(total_samples / self.global_chunk))
        img_array = np.zeros((int(self.local_chunk / 2) + int(1), int(total_samples / self.global_chunk)),dtype=np.dtype(np.int8))
        i = 0
        for chunk in data:
            spec, f, t = self.log_specgram(audio=chunk, sample_rate=fs, window_size=self.local_chunk)
            img_array[:, i] = np.average(spec, axis=0)
            i = i + 1
        return img_array,freq,timeline


if __name__ == "__main__":
   reader = WaveReader("test.wav")
   img_array,f,t = reader.get_sepctrogram()
   plt.figure(figsize=(20,5))
   plt.pcolormesh(t,f,img_array,cmap="jet")
   plt.show()

