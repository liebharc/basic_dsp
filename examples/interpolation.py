from scipy import signal
import numpy
import matplotlib.pyplot as plt
import ctypes
import struct
import time

# 
# A small example how to use basic_dsp in a different language.
#

class VecResult(ctypes.Structure):
    _fields_ = [("resultCode", ctypes.c_int),
                ("result", ctypes.c_void_p)]

lib = ctypes.WinDLL('basic_dsp.dll')

new64Proto = ctypes.WINFUNCTYPE (
    ctypes.c_void_p, # Return type.
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_ulong,
    ctypes.c_double)

new64 = new64Proto (("new64", lib))

getValue64Proto = ctypes.WINFUNCTYPE (
    ctypes.c_double, # Return type.
    ctypes.c_void_p,
    ctypes.c_ulong)

getValue = getValue64Proto (("get_value64", lib))

setValue64Proto = ctypes.WINFUNCTYPE (
    ctypes.c_void_p, # Return type.
    ctypes.c_void_p,
    ctypes.c_ulong,
    ctypes.c_double)

setValue = setValue64Proto (("set_value64", lib))

interpftProto = ctypes.WINFUNCTYPE (
    VecResult, # Return type.
    ctypes.c_void_p, 
    ctypes.c_ulong)

interpft = interpftProto (("interpft64", lib))

interpolateProto = ctypes.WINFUNCTYPE (
    VecResult, # Return type.
    ctypes.c_void_p, 
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_ulong
    ctypes.c_double)

interpolate = interpolateProto (("interpolate64", lib))

vec = new64(
        ctypes.c_int(0),
        ctypes.c_int(0),
        ctypes.c_double(0.0),
        ctypes.c_ulong(20),
        ctypes.c_double(1.0))

x = numpy.linspace(0, 10, 20, endpoint=False)
y1 = numpy.cos(-x**2/6.0)
f = signal.resample(y1, 100)

for i in range(0, 19):
    setValue(vec, i, y1[i])
# vec_res = interpft(vec, 100)
vec_res = interpolate(vec, 1, 0.35, 100, 0)
assert(vec_res.resultCode == 0)
vec = vec_res.result
y2 = [0.0] * 100
for i in range(0, 99):
    y2[i] = getValue(vec, i)
xnew = numpy.linspace(0, 10, 100, endpoint=False)
plt.plot(x, y1, 'go-', xnew, f, '.-', xnew, y2, '--', 10, y1[0], 'ro')
plt.legend(['data', 'resampled scipy', 'resampled basic_dsp'], loc='best')
plt.show()