from __future__ import division, print_function, absolute_import
from scipy import signal

import numpy
import struct
import math
from itertools import chain
import warnings
import matplotlib.pyplot as plt
import sys, os
import numpy as np
class WavFileWarning(UserWarning):
    pass


class WaveReader:

    """
    This class is written to read large wav files and compute spectrogram on them
    Don't change any of the pointer values in the code, this might lead to error

    The reader only handles mono at the moment, stereo will be added in the future,
    compressed data format are also not supported by this reader.

    Author: Kamal SELVAM  --> kamal.selvam@expleogroup.com
    """
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
        self._read_data_chunk()

    def _read_riff_chunk(self):
        str1 = self.fid.read(self._riff_start)
        if(self._debug):
            print("The RIFF is at 0x{}".format(str1.hex()))
        if str1 == b'RIFX':
            _big_endian = True
        elif str1 != b'RIFF':
            raise ValueError("Not a WAV file.")
        fmt = '<I'
        fsize = struct.unpack(fmt, self.fid.read(self._four_byte))[0] + (self._four_byte*2)
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
            self.global_data_y = []
            self.global_data_x = []

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


    def get_all_data(self):
        self.fid.seek(0, os.SEEK_SET)
        dat_1 = np.memmap(self.fid, dtype='<i2', mode='c', offset=42, shape=(self.data_chunk_size//2,))
        print(len(dat_1))
        n = np.ceil(dat_1.size / 10)
        from scipy.interpolate import interp1d
        f = interp1d(np.linspace(0, 1, dat_1.size), dat_1, 'linear')
        return f(np.linspace(0, 1, n))

    def get_next_data(self,chunk_size):
        self.fid.seek(0, os.SEEK_SET)
        start = self.data_start
        size = self.data_chunk_size
        end = start + chunk_size
        t_start = 0
        outputstr = []
        while start+chunk_size < size:
            data = numpy.memmap(self.fid, dtype='<i2', mode='c', offset=start, shape=(chunk_size // 2,))
            n = np.ceil(data.size / 10)
            from scipy.interpolate import interp1d
            datax= np.linspace(t_start, t_start + len(data), len(data), endpoint=False)
            import matplotlib.pyplot as plt

            f = signal.resample(data, int(len(data)/100))
            xnew = np.linspace(t_start, t_start + len(data), int(len(data)/100), endpoint=False)


            #print(t_start)
            f[0] = data[0]
            f[-1] = data[-1]
            #plt.plot(datax,data)
            #plt.plot(xnew,f)
            #plt.show()

            print(datax[0],datax[-1],xnew[0],xnew[-1])

            self.global_data_y.extend(f)
            self.global_data_x.extend(xnew)
            self.fid.seek(end)
            t_start = t_start + len(data)
            start = start + int(chunk_size)
            end = start + int(chunk_size)
            yield data


    def get_patch_data(self,start,chunk_size,y_start):
        self.fid.seek(0, os.SEEK_SET)
        end = start + chunk_size
        datay = numpy.memmap(self.fid, dtype='<i2', mode='c', offset=start, shape=(chunk_size // 2,))
        datax = [i for i in range(y_start,y_start+(chunk_size//2))]
        self.fid.seek(end)
        return datax,datay


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
        f, t, spec = signal.spectrogram(audio, fs=sample_rate, window='hann', nperseg=window_size, noverlap=50, scaling='spectrum', mode='magnitude')
        return 80 + 20 * np.log10((spec.T.astype(np.float32) + eps) / np.max(spec)), f, t



    def get_sepctrogram(self,global_chunk):
        fs, tim, total_samples = self.get_header()


        self.global_chunk = global_chunk
        self.local_chunk = int(self.global_chunk / 10)
        data = self.get_next_data(self.global_chunk)
        freq = np.linspace(0, fs / 2, int(self.local_chunk / 2) +1)
        timeline = np.linspace(0, tim, int(total_samples / self.global_chunk))
        img_array = np.zeros((int(self.local_chunk / 2) + int(1), int((total_samples/self.global_chunk)/2)),dtype=np.dtype(np.int8))


        i = 0
        for chunk in data:
            spec, f, t = self.log_specgram(audio=chunk, sample_rate=fs, window_size=self.local_chunk)
            try:
                img_array[:, i] = np.average(spec, axis=0)
            except:
                pass
            i = i + 1
            print(len(chunk))
        return img_array,freq,timeline


    def get_data(self):
        start = 44
        chunk = 2
        start_y = 0
        # reading first point
        sdatax, sdatay = reader.get_patch_data(start, chunk, start_y)

        total_chunk_remain = reader.data_chunk_size - 4

        start = 46
        chunk = 10
        offset = 20
        start_y = 1

        A = 0
        Next_A = 0
        max_area_point = (0, 0)
        sampled = [[sdatax[0], sdatay[0]]]

        while start + chunk < total_chunk_remain:
            datax, datay = reader.get_patch_data(start, chunk, start_y)
            splits = len(datay) / 2
            bucket_c_start = int(splits)
            bucket_c_end = len(datay)
            bucket_b_start = 0
            bucket_b_end = int(splits - 1)
            print(start)
            avg_x = 0
            avg_y = 0
            # print(bucket_c_start, bucket_c_end, bucket_b_start, bucket_b_end)
            len_ran = 0
            while bucket_c_start < bucket_c_end:
                avg_x += datax[bucket_c_start]
                avg_y += datay[bucket_c_start]
                len_ran += 1
                bucket_c_start += 1

            avg_x /= len_ran
            avg_y /= len_ran
            point_ax = sdatax[0]
            point_ay = sdatay[0]
            max_area = - 1
            while bucket_b_start < bucket_b_end:
                area = math.fabs(
                    (point_ax - avg_x) * (datay[bucket_b_start] - point_ay) - (point_ax - datax[bucket_b_start]) * (
                                avg_y - point_ay)) * 0.5
                if area > max_area:
                    max_area = area
                    max_area_point_x = datax[bucket_b_start]
                    max_area_point_y = datay[bucket_b_start]
                    sdatax = [max_area_point_x]  # Next a is this b
                    sdatay = [max_area_point_y]
                bucket_b_start += 1000

            sampled.append([max_area_point_x, max_area_point_y])
            start_y = start_y + int((offset / 2))
            start = start + offset

        end_byte =  reader.data_chunk_size - 2
        edatax, edatay = reader.get_patch_data(end_byte, 2, (reader.data_chunk_size//2)-1)
        end = reader.data_chunk_size
        sampled.append([edatax[0],edatay[0]])
        print(edatax[0],edatay[0])
        return sampled



if __name__ == "__main__":
   reader = WaveReader("test.wav")
   #sampled = reader.get_all_data()
   from scipy.io import wavfile
   fs,data= wavfile.read("Digital-Seashore.wav")
   print(data)
   img_array,f,t = reader.get_sepctrogram(global_chunk=20480)
   x = [  ((x / 48000) / 60) for x in reader.global_data_x]
   plt.plot(x,reader.global_data_y)
   plt.show()
   #x, y = zip(*sampled)
   #plt.plot(x,y)
   #plt.plot(data,'r')
   #print(len(x),len(data))
   #plt.show()

   #plt.figure(figsize=(20,5))
   #plt.pcolormesh(img_array,cmap="jet")
   #plt.colorbar()
   #plt.show()

   #f, t, Sxx = signal.spectrogram(data, fs=fs, window='hann', nperseg=512,noverlap=50, scaling='spectrum', mode='magnitude')
   #eps = 1e-10
   #ser = 80 + 20 * np.log10((Sxx.astype(np.float32) + eps) / np.max(Sxx))
   #plt.pcolormesh(t, f, ser,cmap="jet")
   #plt.colorbar()
   #plt.ylabel('Frequency [Hz]')
   #plt.xlabel('Time [sec]')
   #plt.show()


