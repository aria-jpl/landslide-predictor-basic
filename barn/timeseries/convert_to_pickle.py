import sys

import numpy as np

from scipy.interpolate import interp1d

import pickle

if len(sys.argv) != 3:
    print("Usage: %s inputFilePath outputFilePath" % sys.argv[0] , file=sys.stderr)
    sys.exit(-1)

inputFilePath = sys.argv[1]
outputFilePath = sys.argv[2]

displacementInMeter = np.load(inputFilePath)
print("displacement in meters dtype", displacementInMeter.dtype)
print("displacement in meters shape", displacementInMeter.shape)

# convert to millimeter
print("convert displacement to millimeters")
displacementInMillimeter = displacementInMeter * 1000.0

# construct interp function
z_in = displacementInMillimeter

time_in = np.linspace(0, 1, z_in.shape[1])
print("input time intervals", time_in)

# and interpolate the time dim to 101
func = interp1d(time_in, z_in, axis=1)

time_out = np.linspace(0, 1, 101)
print("output time intervals", time_out)

z_out = func(time_out)
# make sure it is float32 not float64
z_out = np.float32(z_out)
print("interpolated dtype", z_out.dtype)
print("interpolated shape", z_out.shape)

# as required by landslide predictor
d = {"data":z_out}

print("dump interpolated as pickle to", outputFilePath)
with open(outputFilePath, "wb") as f:
    pickle.dump(d, f)
