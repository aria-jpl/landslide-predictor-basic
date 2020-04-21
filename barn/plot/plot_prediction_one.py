import os, sys
import numpy

import pickle

import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt

if len(sys.argv) != 4:
    print("Usage: %s lonlatFilePath predictionFilePath outputFilePath" % sys.argv[0] , file=sys.stderr)
    sys.exit(-1)


lonlatFilePath = sys.argv[1]
predictionFilePath = sys.argv[2]
outputFilePath = sys.argv[3]

lonlat = numpy.load(lonlatFilePath)
print("lonlat dtype", lonlat.dtype)
print("lonlat shape", lonlat.shape)

prediction = numpy.fromfile(predictionFilePath, numpy.int32)
print("prediction dtype", prediction.dtype)
print("prediction shape", prediction.shape)
print(prediction)

lon = []
lat = []
for idx in prediction:
    lon.append(lonlat[idx, 0])
    lat.append(lonlat[idx, 1])
print(min(lon), max(lon))
print(min(lat), max(lat))

#plt.axis([13.7, 14.2, 42.1, 42.5])
plt.xlabel("lon")
plt.ylabel("lat")
plt.title("predicted location")

plt.scatter(lon, lat, marker=".", linewidth=1)

plt.savefig(outputFilePath, dpi=100)
