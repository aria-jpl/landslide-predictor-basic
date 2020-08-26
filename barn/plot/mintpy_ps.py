
import matplotlib
matplotlib.use("AGG")

from matplotlib import pyplot as plt
import numpy as np
import pickle as pc
import os
import h5py

import sys
if len(sys.argv) != 5:
    print("Usage: %s mintpyOutputDirPath predictorInputDirPath predictorOutputDirPath outputFilePath" % sys.argv[0], file=sys.stderr)
    sys.exit(1)

mintpyOutputDirPath = sys.argv[1]
predictorInputDirPath = sys.argv[2]
predictorOutputDirPath = sys.argv[3]
outputFilePath = sys.argv[4]

#ddir = '/home/giangi/Workspace/Data/ts_slc/mintpy/'
ddir = mintpyOutputDirPath
ts = h5py.File(os.path.join(ddir,'timeseries.h5'))
tc = h5py.File(os.path.join(ddir,'temporalCoherence.h5'))
i = 0
#pred = np.fromfile('/home/giangi/Downloads/test_mintpy_pred_out/pred_input_' + str(i) + '_data.pred',np.int32)
pred = np.fromfile(predictorOutputDirPath+'/pred_input_' + str(i) + '_data.pred',np.int32)
#dt = pc.load(open('/home/giangi/Downloads/test_mintpy_pred/pred_input_' + str(i) + '_data.pck','rb'))
dt = pc.load(open(predictorInputDirPath+'/pred_input_' + str(i) + '_data.pickle','rb'))
#locs = pc.load(open('/home/giangi/Downloads/test_mintpy_pred/pred_input_' + str(i) + '_locs.pck','rb'))
locs = pc.load(open(predictorInputDirPath+'/pred_input_' + str(i) + '_locs.pickle','rb'))
tcv = np.array(tc['temporalCoherence'])
tsv = np.array(ts['timeseries'])
tc.close()
ts.close()

dt_pred = dt['data'][pred]
yx = [locs[0][pred],locs[1][pred]]

mask = np.zeros_like(tcv)
mask[yx[0],yx[1]] = 1
plt.figure(figsize=[10,10])
plt.subplot(2,1,1)
plt.imshow(mask)
plt.colorbar()
plt.subplot(2,1,2)
plt.imshow(10**3*np.abs(tsv).max(0))
plt.colorbar()

plt.savefig(outputFilePath, dpi=100)
