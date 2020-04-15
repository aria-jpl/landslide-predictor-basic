import os, sys

import numpy

import scipy.io

import h5py


#------
# read varName as parts from path
# using read_as_parts_by_scipy() or read_as_parts_by_h5py()
# depending on file size
def read_as_parts(path, varName, parts):
    z_parts = None
    try:
        z_parts = read_as_parts_by_scipy(path, varName, parts)
        print("using read_as_parts_by_scipy()")
    except NotImplementedError as e:
        print("However read using read_as_parts_by_scipy() failed with NotImplementedError:", e)
        print("so using read_as_parts_by_h5py() instead.")
        z_parts = read_as_parts_by_h5py(path, varName, parts)
    except Exception as e:
        raise Exception("Unable to read due to exception:", e)
    return z_parts


#------
# if file is < 2GB
# read varName as parts from path using scipy.io
def read_as_parts_by_scipy(path, varName, parts):
    z = scipy.io.loadmat(path)
    var = z[varName]
    print("var.dtype", var.dtype)
    print("var.shape", var.shape)
    var_parts = []
    for i in range(len(parts)):
        part = parts[i]
        start, stop = part
        print("read", varName, "part", i)
        var_parts.append(var[start:stop,:])
    return var_parts


#------
# if file is > 2GB
# read varName as parts from path using h5py
def read_as_parts_by_h5py(path, varName, parts):
    var_parts = []
    f = h5py.File(path)
    for i in range(len(parts)):
        part = parts[i]
        start, stop = part
        print("read", varName, "part", i)
        var = f.get(varName)[:,start:stop]
        print("var.dtype", var.dtype)
        print("var.shape", var.shape)
        # do transpose it!!!
        print("need to transpose")
        var = var.transpose()
        print("var.dtype", var.dtype)
        print("var.shape", var.shape)
        var_parts.append(var)
    f.close()
    return var_parts


#------
# application entry

if len(sys.argv) != 4:
    print("Usage: %s inputDirPath outputDirPath outputFileNamePrefix" % sys.argv[0] , file=sys.stderr)
    sys.exit(-1)

inputDirPath = sys.argv[1]
outputDirPath = sys.argv[2]
outputFileNamePrefix = sys.argv[3]

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

# number of ps points
n_ps = z["n_ps"]
print("Number of ps points is", n_ps)

#partSize = 1048576 # 2^20
#partSize = 1000000
#partSize = 524288 # 2^19
partSize = 500000
#partSize = 262144 # 2^18
#partSize = 250000
#partSize = 123456
# will split ps points into parts, each of which is of partSize
l = [*range(0, n_ps[0][0], partSize)]
l.append(n_ps[0][0])
starts = l[:-1]
stops = l[1:]
parts = []
for i in range(len(starts)):
    parts.append([starts[i], stops[i]])
print("ps points will be split into", len(parts), "parts:", parts)

# extract lonlat
for i in range(len(parts)):
    part = parts[i]
    start, stop = part

    # extract lon and lat
    lonlat = z["lonlat"][start:stop,:]
    print("Extract lonlat for", part)
    print("lonlat dtype:", lonlat.dtype)
    print("lonlat shape:", lonlat.shape)
    # dump out as npy
    fileName = "%s.%04d.lonlat.npy" % (outputFileNamePrefix, i)
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
print("Extract ph_uw as", len(parts), "parts from", path)
ph_uw_parts = read_as_parts(path, "ph_uw", parts)

#------
# vars from scla2.mat
fileName = "scla2.mat"
path = os.path.join(inputDirPath, fileName)
print("Extract ph_scla as", len(parts), "parts from", path)
ph_scla_parts = read_as_parts(path, "ph_scla", parts)

# compute displacement and save
for i in range(len(parts)):
    ph_uw = ph_uw_parts[i]
    ph_scla = ph_scla_parts[i]

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
    # displacement
    lam6da = 0.056 # in meters
    displacement = (-lam6da/(4 * numpy.pi)) * ph_all
    print("displacement dtype", displacement.dtype)
    print("displacement shape", displacement.shape)
    print(displacement)
    
    # dump out as parts
    fileName = "%s.%04d.displacement.npy" % (outputFileNamePrefix, i)
    path = os.path.join(outputDirPath, fileName)
    numpy.save(path, displacement)
    print("displacement dumped into", path)
