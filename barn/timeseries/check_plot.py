import sys

import numpy

import pickle

import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt

if len(sys.argv) != 4:
    print("Usage: %s npyFilePath pickleFilePath outputFilePath" % sys.argv[0] , file=sys.stderr)
    sys.exit(-1)


npyFilePath = sys.argv[1]
pickleFilePath = sys.argv[2]
outputFilePath = sys.argv[3]

# read in displaceent from npy file
displacement = numpy.load(npyFilePath)
print(displacement.dtype)
print(displacement.shape)

# pick a middle point
idx = int(displacement.shape[0]/2)
print("idx", idx)

x0 = numpy.linspace(0, 1, displacement.shape[1])
print("x0", x0)
y0 = displacement[idx,:] * 1000.0 # to millimeter
print("y0", y0)

# read in interpolated from pickle file
with open(pickleFilePath, "rb") as f:
    interpolated = pickle.load(f)
interpolated = interpolated["data"]
x1 = numpy.linspace(0, 1, interpolated.shape[1])
print("x1", x1)
y1 = interpolated[idx,:]
print("y1", y1)

#plt.axis([0, 1, -10, 10])
#plt.xlabel("lon")
plt.ylabel("displacement in millimeter")
plt.title("comparison of original displacement and interpolated one")

plt.plot(x0, y0, marker=".", linewidth=1)
plt.plot(x1, y1, marker=".", linewidth=1)

plt.savefig(outputFilePath, dpi=100)
