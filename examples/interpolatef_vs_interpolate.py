from scipy import signal
import numpy
import matplotlib.pyplot as plt
import ctypes
import struct
import time

#
# An example which compares interpolatef, interpolate and numpy resample
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
    ctypes.c_ulong,
    ctypes.c_double)

interpolate = interpolateProto (("interpolate64", lib))

interpolatefProto = ctypes.WINFUNCTYPE (
    VecResult, # Return type.
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_ulong)

interpolatef = interpolatefProto (("interpolatef64", lib))

vec = new64(
        ctypes.c_int(0),
        ctypes.c_int(0),
        ctypes.c_double(0.0),
        ctypes.c_ulong(20),
        ctypes.c_double(1.0))

num = 20
x = numpy.linspace(0, 30, num=num, endpoint=False)
y1 = numpy.cos(-x*2/6.0)
y4 = signal.resample(y1, 5 * num)

for i in range(0, num-1):
    setValue(vec, i, y1[i])
vec_res = interpolate(vec, 0, 0, 5 * num, 0.5)
assert(vec_res.resultCode == 0)
vec = vec_res.result

y2 = [0.0] * 5 * num
for i in range(0, 5 * num-1):
    y2[i] = getValue(vec, i)

vec = new64(
        ctypes.c_int(0),
        ctypes.c_int(0),
        ctypes.c_double(0.0),
        ctypes.c_ulong(20),
        ctypes.c_double(1.0))
for i in range(0, num-1):
    setValue(vec, i, y1[i])

vec_res = interpolatef(vec, 0, 0, 5, -0.5, 32)
assert(vec_res.resultCode == 0)
vec = vec_res.result

y3 = [0.0] * 5 * num
for i in range(0, 5 * num - 1):
    y3[i] = getValue(vec, i)

xnew = numpy.linspace(0, 30, 5 * num, endpoint=False)
plt.plot(x, y1, 'go-', xnew, y3, '.-', xnew, y2, '--', xnew, y4, 'ro')
plt.legend(['data', 'interpolatef (0.5 shift)', 'interpolate (0.5 shift)', 'numpy (0.0 shift)'], loc='best')
plt.show()
