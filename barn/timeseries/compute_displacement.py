import os, sys

import numpy

import scipy.io

import h5py

if len(sys.argv) != 3:
    print("Usage: %s inputDirPath outputDirPath" % sys.argv[0] , file=sys.stderr)
    sys.exit(-1)

inputDirPath = sys.argv[1]
outputDirPath = sys.argv[2]

numberOfFirstPoints = 13579
#numberOfFirstPoints = 1000000

#-------
# vars from ps2.mat
fileName = "ps2.mat"
path = os.path.join(inputDirPath, fileName)
z = scipy.io.loadmat(path)

# lis of days
day = z["day"]
print("Number of days is", len(day))

# master day
master_day = z["master_day"]
print("Master day is", master_day)

# number of persistent points
n_ps = z["n_ps"]
print("Number of persistent points is", n_ps)

# this enables original full array
#numberOfFirstPoints = n_ps[0][0]

# extract lon and lat
print("Extract lonlat for", numberOfFirstPoints, "first points")
lonlat = z["lonlat"][0:numberOfFirstPoints,:]
print("lonlat dtype:", lonlat.dtype)
print("lonlat shape:", lonlat.shape)
# dump out as npy
fileName = "lonlat.npy"
path = os.path.join(outputDirPath, fileName)
numpy.save(path, lonlat)
print("lonlat dumped into", path)

"""
n_ifg = z["n_ifg"]
print("n_ifg is", n_ifg)

master_ix = z["master_ix"]
print("master_ix is", master_ix)

master_ix = numpy.sum(day<master_day) + 1
print(master_ix)
"""

#------
# vars from phuw2.mat
fileName = "phuw2.mat"
path = os.path.join(inputDirPath, fileName)

# this file is > 2GB, so the following won't work
#z = scipy.io.loadmat(path)
#ph_uw = z["ph_uw"]
#
f = h5py.File(path)
print("Extract ph_uw for", numberOfFirstPoints, "first points")
ph_uw = f.get("ph_uw")[:,0:numberOfFirstPoints]
f.close()
print("ph_uw dtype", ph_uw.dtype)
print("ph_uw shape", ph_uw.shape)

#------
# vars from scla2.mat
fileName = "scla2.mat"
path = os.path.join(inputDirPath, fileName)

# this file is > 2GB, so the following won't work
#z = scipy.io.loadmat(path)
#ph_scla = z["ph_scla"]
#
f = h5py.File(path)
print("Extract ph_scla for", numberOfFirstPoints, "first points")
ph_scla = f.get("ph_scla")[:,0:numberOfFirstPoints]
f.close()
print("ph_scla dtype", ph_scla.dtype)
print("ph_scla shape", ph_scla.shape)

#------
# compute displacement
print("compute displacement in meters ...")
ph_all = ph_uw
#
#ph_all[:, master_ix] = 0
#print(ph_all)
#print(ph_all[0, :])
#print(ph_all[1, :])
#print(ph_all[2, :])
#
# displacement_all
lam6da = 0.056 # in meters
displacement_all = (-lam6da/(4 * numpy.pi)) * ph_all
print("displacement_all dtype", displacement_all.dtype)
print("displacement_all shape", displacement_all.shape)
print(displacement_all)
# do transpose it!!!
print("transpose displacement")
displacement_all = displacement_all.transpose()
print("displacement_all dtype", displacement_all.dtype)
print("displacement_all shape", displacement_all.shape)
print(displacement_all)

# dump out
fileName = "displacement.npy"
path = os.path.join(outputDirPath, fileName)
numpy.save(path, displacement_all)
print("displacement dumped into", path)
