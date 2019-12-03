from __future__ import division, print_function, absolute_import
from scipy import signal
from scipy import fft, fftpack
from scipy.signal import blackman
from multiprocessing import Process, Manager
import os
import os
import numpy
import struct
import math
from itertools import chain
import warnings
import matplotlib.pyplot as plt
import sys, os
import numba
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

    def get_next_data(self,chunk_size):
        self.fid.seek(0, os.SEEK_SET)
        #print(self.data_start)
        start = self.data_start
        size = self.data_chunk_size
        end = start + chunk_size
        while start+chunk_size < size:
            data = numpy.memmap(self.fid, dtype='<i2', mode='c', offset=start, shape=(chunk_size // 2,))
            self.fid.seek(end)
            start = start + int(chunk_size)
            end = start + int(chunk_size)
            data = fft(data)

            yield data

    def get_header(self):
        return self.sampling_rate, self.data_chunk_size/(2*self.sampling_rate), self.data_chunk_size, self.data_start


    def close(self):
        self.fid.close()

    @staticmethod
    def sliding_window(data, size, stepsize=1, padded=False, axis=-1, copy=True):
        if axis >= data.ndim:
            raise ValueError(
                "Axis value out of range"
            )

        if stepsize < 1:
            raise ValueError(
                "Stepsize may not be zero or negative"
            )

        if size > data.shape[axis]:
            raise ValueError(
                "Sliding window size may not exceed size of selected axis"
            )

        shape = list(data.shape)
        shape[axis] = numpy.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
        shape.append(size)

        strides = list(data.strides)
        strides[axis] *= stepsize
        strides.append(data.strides[axis])

        strided = numpy.lib.stride_tricks.as_strided(
            data, shape=shape, strides=strides
        )

        if copy:
            return strided.copy()
        else:
            return strided

    @staticmethod
    def get_spectro(filename,start,end,chunk_size,pid,sdi,mod_chk,height):
        fid = open(filename, 'rb')
        fid.seek(0, os.SEEK_SET)
        start = start
        size = end
        end = start + chunk_size
        mat = np.zeros((mod_chk,height))
        i = 0
        while start + chunk_size <= size:
            data = numpy.memmap(fid, dtype='<i2', mode='c', offset=start, shape=(chunk_size // 2,))
            fid.seek(end)
            start = start + int(chunk_size)
            end = start + int(chunk_size)
            w = blackman(len(data))
            data = fft(data * w)
            data = np.abs(data[1:len(data)//2])
            #print(len(data),height)
            window = int(np.rint((len(data)/height)))
            a = numpy.array(data)
            t = WaveReader.sliding_window(a, window,window)
            strided = np.mean(t,axis=1)
            try:
                mat[i,:] = strided
            except:
                mat=np.zeros((mod_chk,len(strided)))
                mat[i, :] = strided
            i = i +1

        print(pid,mod_chk,i)
        fid.close()
        sdi.update({pid:mat})


def spectrogram(filename,L,H,tstart,tend):
    sampling_frequency, tim, total_byte, data_start = reader.get_header()
    tstat_offset = (tstart * sampling_frequency)*2
    tend_offset = (tend * sampling_frequency) *2

    if tend_offset < total_byte:
        total_byte = tend_offset

    data_start = data_start + tstat_offset
    data_end = 44 + total_byte
    total_byte = data_end - data_start

    number_of_processor = 4
    chunk_size_req = int(total_byte / L)
    real_chunk_size = fftpack.helper.next_fast_len(chunk_size_req)
    real_L = int(total_byte / real_chunk_size)

    if (real_chunk_size % 2 == 0):
        pass
    else:
        real_chunk_size = real_chunk_size + 1

    chunk_per_proc = int((total_byte) / number_of_processor)
    mod_chk = chunk_per_proc // real_chunk_size
    per_proc = mod_chk * real_chunk_size
    reader.close()

    startp = [data_start]
    mod_ini = ((per_proc)%real_chunk_size)
    endp = [data_start+per_proc+mod_ini]
    idxl = [0]
    mod_chk_list = [int((endp[0]-startp[0])/real_chunk_size)]

    for i in range(1, number_of_processor-1):
        if (endp[i-1] < data_end):
            startp.append(endp[i - 1])
            endp.append(endp[i - 1] + mod_chk*real_chunk_size)
            idxl.append(i)
            mod_chk_list.append(mod_chk)

    mod_last = ((data_end-endp[-1])//real_chunk_size)
    startp.append(endp[-1])
    endp.append(data_end)
    idxl.append(i+1)
    mod_chk_list.append(mod_last)


    procs = []
    manager = Manager()
    sdi = manager.dict()


    #print(real_chunk_size/4,H)
    if (H > real_chunk_size/4):
        print("Cant create Spectrogram")
        print(H,real_chunk_size/4)
    else:

        for start, end, mod, idx in zip(startp, endp, mod_chk_list,idxl):
            print(start, end, real_chunk_size, idx, sdi, mod, H)
            proc = Process(target=WaveReader.get_spectro, args=(filename, start, end, real_chunk_size, idx, sdi, mod, H,))
            procs.append(proc)
            proc.start()

        # complete the processes
        for proc in procs:
            proc.join()

        spec = sdi[0].T
        if number_of_processor > 1:
            for i in range(1, number_of_processor):
                spec = np.hstack((spec, sdi[i].T))
                pass
        eps = 1e-10
        newspec = 80 + 20 * np.log10((spec.astype(np.float32) + eps) / np.max(spec))

        time_dim = newspec.shape[1]
        freq_dim = newspec.shape[0]

        #time_vector = [x for x in range(0,int(total_byte/2*sampling_frequency))]
        time_vector = np.linspace(0,tim,time_dim)
        freq_vector = np.linspace(0,sampling_frequency/2,freq_dim)

        return newspec, time_vector, freq_vector
    return None, None, None

if __name__ == "__main__":
   import timeit
   start= timeit.timeit()
   filename = "test.wav"
   reader = WaveReader(filename)
   L = 2200
   H = 2000
   tstart = 1600   #(in seconds)
   tend = 1800

   ns, time, freq = spectrogram(filename,L,H,tstart,tend)

   if ns is not None:
       end = timeit.timeit()
       #print(newspec.shape)
       print(end-start)
       plt.figure(figsize=(20,5))
       plt.pcolormesh(ns,cmap="jet")
       plt.colorbar()
       plt.show()
