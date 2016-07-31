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

getValue64 = getValue64Proto (("get_value64", lib))

offset64Proto = ctypes.WINFUNCTYPE (
    VecResult, # Return type.
    ctypes.c_void_p, 
    ctypes.c_double)

offset64 = offset64Proto (("real_offset64", lib))

vec = new64(
        ctypes.c_int(0),
        ctypes.c_int(0),
        ctypes.c_double(0.0),
        ctypes.c_ulong(100000),
        ctypes.c_double(1.0))
val = getValue64(vec, ctypes.c_ulong(0))
print('At the start: vec[0] = {}'.format(val))
start = time.clock()
iterations = 100000
toNs = 1e9 / iterations
increment = 5.0
for x in range(0, iterations): 
    vecRes = offset64(vec, ctypes.c_double(increment))
    vec = vecRes.result
end = time.clock()
print('{} ns per iteration, each iteration has {} samples'.format((end - start) * toNs, iterations))
print('Result code: {} (0 means no error)'.format(vecRes.resultCode))
vecRes = offset64(vec, ctypes.c_double(5.0))
vec = vecRes.result
val = getValue64(vec, ctypes.c_ulong(0))
print('After {} iterations of increment by {}: vec[0] = {}'.format(iterations + 1, increment, val))
