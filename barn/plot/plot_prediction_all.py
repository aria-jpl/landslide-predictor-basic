import os, sys
import re

import numpy

import pickle

import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt

PREDICTION_FILE_PATTERN = r'\w+\.(\d{4})\.displacement\.pred'


def get_lonlat_for_one_part(fileNamePrefix, lonlatDirPath, predictionDirPath, partName):
    lonlatFilePath = os.path.join(lonlatDirPath, ".".join([fileNamePrefix, partName, "lonlat", "npy"]))
    predictionFilePath = os.path.join(predictionDirPath, ".".join([fileNamePrefix, partName, "displacement", "pred"]))

    print("lonlat file", lonlatFilePath)
    lonlat = numpy.load(lonlatFilePath)
    print("lonlat dtype", lonlat.dtype)
    print("lonlat shape", lonlat.shape)

    print("prediction file", predictionFilePath)
    prediction = numpy.fromfile(predictionFilePath, numpy.int32)
    print("prediction dtype", prediction.dtype)
    print("prediction shape", prediction.shape)
    #print(prediction)

    lon = []
    lat = []
    for idx in prediction:
        lon.append(lonlat[idx, 0])
        lat.append(lonlat[idx, 1])

    if len(lon) == 0 or len(lat) == 0:
        return [lon, lat]

    print(min(lon), max(lon))
    print(min(lat), max(lat))

    return [lon, lat]


if len(sys.argv) != 5:
    print("Usage: %s fileNamePrefix lonlatDirPath predictionDirPath outputFilePath" % sys.argv[0] , file=sys.stderr)
    sys.exit(-1)


fileNamePrefix = sys.argv[1]
lonlatDirPath = sys.argv[2]
predictionDirPath = sys.argv[3]
outputFilePath = sys.argv[4]

# figure out parts from prediction part files
parts = []
for x in os.listdir(predictionDirPath):
    m = re.match(PREDICTION_FILE_PATTERN, x)
    if m == None:
        continue
    parts.append(m.group(1))

if len(parts) == 0:
    print("No part found")
    sys.exit(0)

print("Parts found:", parts)

# collect lon, lat of all prediction points
lon = []
lat = []
for partName in parts:
     print("get lon lat for part", partName)
     x, y = get_lonlat_for_one_part(fileNamePrefix, lonlatDirPath, predictionDirPath, partName)
     lon.extend(x)
     lat.extend(y)

if len(lon) == 0 or len(lat) == 0:
    print("No location predicted")
    sys.exit(0)

print(len(lon))
print(len(lat))
print(min(lon), max(lon))
print(min(lat), max(lat))


# plot them
#plt.axis([13.7, 14.2, 42.1, 42.5])
plt.xlabel("lon")
plt.ylabel("lat")
plt.title("predicted location")

plt.scatter(lon, lat, marker=".", linewidth=1)

print("save prediction plot to", outputFilePath)
plt.savefig(outputFilePath, dpi=100)
