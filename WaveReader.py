from __future__ import division, print_function, absolute_import

import numpy
import struct
import warnings
import librosa
import collections
from guppy import hpy
from operator import itemgetter
import sys, os



class WavFileWarning(UserWarning):
    pass

class WaveReader:

    def __init__(self,filename,mmap=True):

        if hasattr(filename, 'read'):
            self.fid = filename
            mmap = False
        else:
            self.fid = open(filename, 'rb')

        self.mmap = mmap
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
        str1 = self.fid.read(4)
        print("The RIFF is at 0x{}".format(str1.hex()))
        if str1 == b'RIFX':
            _big_endian = True
        elif str1 != b'RIFF':
            raise ValueError("Not a WAV file.")

        fmt = '<I'
        fsize = struct.unpack(fmt, self.fid.read(4))[0] + 8
        str2 = self.fid.read(4)
        print("The WAVE is at 0x{}".format(str2.hex()))
        if (str2 != b'WAVE'):
            raise ValueError("Not a WAV file.")
        if str1 == b'RIFX':
            _big_endian = True
        return fsize


    def _read_fmt_chunk(self):
        self.fid.seek(0,os.SEEK_SET)
        self.fid.seek(12)            #Offsetting the pointer of 12
        chunk_id = self.fid.read(4)
        if chunk_id == b'fmt ':
            res = struct.unpack('<ihHIIHH', self.fid.read(20))
            size, comp, noc, rate, sbytes, ba, bits = res

            print("The Chunk Data size is {} at {} 4 bytes".format(size, hex(size)))
            print("The Compression code is {} at {} 2 bytes".format(comp, hex(comp)))
            print("The Number of Channel is {} at {} 2 bytes".format(noc, hex(noc)))
            print("The sampling rate is {} at 0x{} 4 bytes".format(rate, hex(rate)))
            print("The Average bytes per second is {} at {} 2 bytes".format(sbytes, hex(sbytes)))
            print("The Block Assingn is at {} 0x{} 2 bytes".format(ba, hex(ba)))
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
        self.fid.seek(36)  # Offsetting the pointer of 12
        chunk_id = self.fid.read(4)
        print(chunk_id)
        if chunk_id == b'data':
            fmt = '<i'
            size = struct.unpack(fmt, self.fid.read(4))[0]
            print("The Data size is {} at {} 4 bytes".format(size, hex(size)))
            self.data_chunk_size = size

            bytes = self.significant_bits // 8
            if self.significant_bits == 8:
                dtype = 'u1'
            else:

                dtype = '<'
                if self.compression == 1:
                    dtype += 'i%d' % bytes
                else:
                    dtype += 'f%d' % bytes

            print("The data type format is {}".format(dtype))


            self.data_start = self.fid.tell()

            print('The data starts at {}'.format(self.data_start))
            print('The data ends at {}'.format(self.data_chunk_size))


    def get_next_data(self):
        self.fid.seek(0, os.SEEK_SET)
        start = self.data_start
        size = self.data_chunk_size
        chunk_size = 2
        end = 0
        while end < size:
            print(start, end, chunk_size)
            data = numpy.memmap(self.fid, dtype='<i2', mode='c', offset=start, shape=(chunk_size // 2,))
            end = start + chunk_size
            self.fid.seek(end)
            start = start + chunk_size
            yield data




    def print_data(self):
        self.fid.seek(0, os.SEEK_SET)
        self.fid.seek(44)
        cursor_pos = self.data_start
        while cursor_pos < self.data_chunk_size:
            val = struct.unpack('<h', self.fid.read(2))[0]

            print(val)
            #print(int.from_bytes(val, byteorder='big'))
            break



if __name__ == "__main__":
   reader = WaveReader("test.wav")
   reader.print_data()

   data = reader.get_next_data()
   print(data)
   for d in data:
       print(d[0])
       break