import sys
import numpy as np
import pickle as pc
import os
import h5py
from scipy.interpolate import interp1d
import argparse
import json
def parse(inps):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-c', '--co_mask_fname', type = str, default = 'temporalCoherence.h5', help = 'coherence mask')
    parser.add_argument('-d', '--ddir', type = str, required = True, help = 'Data directory')
    parser.add_argument('-e', '--co_keyw', type = str, default = 'temporalCoherence', help = 'temporalCoherence keyword used in h5 file')
    parser.add_argument('-i', '--input', type = str, default = '', help = 'Input json file. If provided it contains the same key:values of the parser namespace. The json has precedence over the parser.')
    parser.add_argument('-k', '--is_mask', action= 'store_true', help = 'The coherence mask is indeed a mask and not float.')
    parser.add_argument('-l', '--ts_length', type = int, default = 101, help = 'Time steps in the interpolated timeseries')
    parser.add_argument('-m', '--ts_keyw', type = str, default = 'timeseries', help = 'timeseries keyword used in h5 file')
    parser.add_argument('-n', '--name', type = str, default = 'pred_input', help = 'prefix of the output filenames. A suffix _x.pck is appended where x is  a counter')
    parser.add_argument('-o', '--odir', type = str, default = './', help = 'Output directory where the pickle files are saved')
    parser.add_argument('-p', '--pck_size', type = int, default = 500000, help = 'Max number of timeseries per pickle file')
    parser.add_argument('-r', '--th_co', type = float, default = 0.5, help = 'coherence threshold')
    parser.add_argument('-s', '--disp', type = float, default = 0.1, help = 'minimum displacement from t0 to t_last in m')
    parser.add_argument('-t', '--ts_fname', type = str, default = 'timeseries.h5', help = 'timeseries file')

    return parser.parse_args(inps)

def create_ts(args):
    th = args['th_co']
    disp = args['disp']
    ddir = args['ddir']
    odir = args['odir']
    fname_prefix = args['name']
    ts_name = os.path.join(ddir,args['ts_fname'])
    co_mask = os.path.join(ddir,args['co_mask_fname'])
    tsv = np.array(h5py.File(ts_name)[args['ts_keyw']])
    cov = np.array(h5py.File(co_mask)[args['co_keyw']])
    if args['is_mask']:
        sel_good = np.nonzero(cov*(np.abs(tsv[-1] - tsv[0]) >= disp))
    else:
        sel_good = np.nonzero((cov > th)*(np.abs(tsv[-1] - tsv[0]) >= disp))
    sel_ts = tsv[:,sel_good[0],sel_good[1]]
    x = np.linspace(0,1,tsv.shape[0])
    xp = np.linspace(0,1,args['ts_length'])
    fp = interp1d(x,sel_ts,3,0)
    data = fp(xp)
    for i in range(int(np.ceil(data.shape[1]/args['pck_size']))):
        data_now = data[:,i*args['pck_size']:(i+1)*args['pck_size']]*10**3#convert to mm
        sel_now = (sel_good[0][i*args['pck_size']:(i+1)*args['pck_size']],sel_good[1][i*args['pck_size']:(i+1)*args['pck_size']])
        fname = os.path.join(odir,'{0}_{1:d}_data.pickle'.format(fname_prefix,i))
        pc.dump({'data':data_now.astype(np.float32).T},open(fname,'wb'))
        fname = os.path.join(odir,'{0}_{1:d}_locs.pickle'.format(fname_prefix,i))
        pc.dump(sel_now,open(fname,'wb'))
    
def update_inps(args):
    if len(args.input) > 0:
        inps = json.load(open(args.input))
    else:
        inps = {}
    for k,v in args.__dict__.items():
        if not k in inps: 
            inps[k] = v
    return inps

def main(inps):
    args = parse(inps)
    nargs = update_inps(args)
    create_ts(nargs)
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
    
