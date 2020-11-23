import sys
import numpy as np
import pickle as pc
import os
import h5py
from scipy.interpolate import interp1d
import argparse
import json
from datetime import datetime
from glob import glob
from matplotlib import pyplot as plt
from scipy.stats import linregress
def parse(inps):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-a', '--delta', type = float, default = 0.01, help = 'Step used when filtering ts.')
    parser.add_argument('-b', '--no_filter', action= 'store_true', help = 'Do not filter time series.')
    parser.add_argument('-c', '--co_mask_fname', type = str, default = 'temporalCoherence.h5', help = 'coherence mask')
    parser.add_argument('-d', '--ddir', type = str, required = True, help = 'Data directory')
    parser.add_argument('-e', '--co_keyw', type = str, default = 'mask', help = 'temporalCoherence keyword used in h5 file')
    parser.add_argument('-f', '--factor', type = float, default = 2, help = 'Fraction of delta used for filtering.')
    parser.add_argument('-g', '--pdir', type = str, default = './', help = 'Directory with the prediction files.')
    parser.add_argument('-i', '--input', type = str, default = '', help = 'Input json file. If provided it contains the same key:values of the parser namespace. The json has precedence over the parser.')
    parser.add_argument('-k', '--is_mask', action= 'store_true', help = 'The coherence mask is indeed a mask and not float.')
    parser.add_argument('-l', '--ts_length', type = int, default = 101, help = 'Time steps in the interpolated timeseries')
    parser.add_argument('-m', '--ts_keyw', type = str, default = 'timeseries', help = 'timeseries keyword used in h5 file')
    parser.add_argument('-n', '--name', type = str, default = 'pred_input', help = 'prefix of the output filenames. A suffix _x.pck is appended where x is  a counter')
    parser.add_argument('-o', '--odir', type = str, default = './', help = 'Output directory where the pickle files are saved')
    parser.add_argument('-p', '--pck_size', type = int, default = 500000, help = 'Max number of timeseries per pickle file')
    parser.add_argument('-r', '--th_co', type = float, default = 0.5, help = 'coherence threshold')
    parser.add_argument('-s', '--disp', type = float, default = 0.1, help = 'minimum displacement per year in m')
    parser.add_argument('-t', '--ts_fname', type = str, default = 'timeseries.h5', help = 'timeseries file')
    parser.add_argument('-u', '--suffix', type = str, default = 'filt', help = 'Suffix to add to the filter ts name')
    parser.add_argument('--action', type = str, default = 'create_ts', help = 'Action to perform.')
    parser.add_argument('--fig_name', type = str, default = None, help = 'Full name of the figure to save. By defaults it uses the ts name and changes extension with png and save in cwd')

    return parser.parse_args(inps)

def filt_ts(tsv,ts_name,x0,dx,y0,dy,delta=0.01,factor=2,suffix='filt'):
    lon = x0 + dx*np.arange(tsv.shape[2])
    lat = y0 + dy*np.arange(tsv.shape[1])
    lat = np.tile(lat.reshape([-1,1]),[1,len(lon)])
    lon = np.tile(lon.reshape([1,-1]),[lat.shape[0],1])
    #move each step by delta. subtract the mean computed in a region of size 2*factor*delta from current location
    #+- delta/2  
    fdelta = factor*delta
    lons = np.r_[np.min(lon) - 2*delta:np.max(lon) + 2*delta:delta]
    lats = np.r_[np.min(lat) - 2*delta:np.max(lat) + 2*delta:delta]
    ntsv = np.zeros_like(tsv)
    maxts = np.abs(tsv).max(0)
    for i in range(1,len(lats)):
        latnow = lats[i]
        for j in range(1,len(lons)):
            lonnow = lons[j]
            sel = np.where((maxts > 0)*(lat >= latnow - fdelta)*(lat < latnow + fdelta)*(lon >= lonnow - fdelta)*(lon < lonnow + fdelta))
            if len(sel[0]) == 0:
                continue
            sel1 = np.where((maxts > 0)*(lat >= latnow - delta/2)*(lat < latnow + delta/2)*(lon >= lonnow - delta/2)*(lon < lonnow + delta/2))
            if len(sel1[0]) == 0:
                continue
            ntsv[:,sel1[0],sel1[1]] = tsv[:,sel1[0],sel1[1]] - np.tile(tsv[:,sel[0],sel[1]].mean(1).reshape([-1,1]),[1,sel1[0].size])

    nts_name = ts_name.replace('.h5','_' + suffix + '.h5')
    cmd = 'cp ' + ts_name + ' ' + nts_name
    os.system(cmd)

    ts = h5py.File(nts_name,'a')
    ts['timeseries'][:,:,:] = ntsv
    ts.close()
    return ntsv


def create_ts(args):
    th = args['th_co']
    disp = args['disp']
    ddir = args['ddir']
    odir = args['odir']
    fname_prefix = args['name']
    ts_name = os.path.join(ddir,args['ts_fname'])
    co_mask = os.path.join(ddir,args['co_mask_fname'])
    ts = h5py.File(ts_name)
    x0 = float(ts.attrs['X_FIRST'])
    dx = float(ts.attrs['X_STEP'])
    y0 = float(ts.attrs['Y_FIRST'])
    dy = float(ts.attrs['Y_STEP'])
    tsv = np.array(ts[args['ts_keyw']])
    if not args['no_filter']:
        tsv = filt_ts(tsv,ts_name,x0,dx,y0,dy,args['delta'],args['factor'],args['suffix'])
    tsv -= tsv[0][None,...]
    cov = np.array(h5py.File(co_mask)[args['co_keyw']])
    sdates = np.array(ts['date']).astype(np.str)
    dates = []
    for i,d in enumerate(sdates):
        dnow = datetime.strptime(d,'%Y%m%d')
        if i == 0:
            d0 = dnow
        dates.append((dnow - d0).days)
    x = np.array(dates)
    if args['is_mask']:
        sel_good = np.nonzero(cov*(np.abs(tsv[-1] - tsv[0]) >= disp*dates[-1]/365))
    else:
        sel_good = np.nonzero((cov > th)*(np.abs(tsv[-1] - tsv[0]) >= disp*dates[-1]/365))
    sel_ts = tsv[:,sel_good[0],sel_good[1]]

    xp = np.linspace(0,dates[-1],args['ts_length'])
    fp = interp1d(x,sel_ts,3,0)
    data = fp(xp)
    for i in range(int(np.ceil(data.shape[1]/args['pck_size']))):
        data_now = data[:,i*args['pck_size']:(i+1)*args['pck_size']]*10**3#convert to mm
        sel_now = (sel_good[0][i*args['pck_size']:(i+1)*args['pck_size']],sel_good[1][i*args['pck_size']:(i+1)*args['pck_size']])
        fname = os.path.join(odir,'{0}_{1:d}_data.pck'.format(fname_prefix,i))
        pc.dump({'data':data_now.astype(np.float32).T},open(fname,'wb'))
        fname = os.path.join(odir,'{0}_{1:d}_locs.pck'.format(fname_prefix,i))
        pc.dump(sel_now,open(fname,'wb'))

def create_png(args):
    ddir = args['ddir']
    pdir = args['pdir']
    pkdir = args['odir']
    pred_input = args['name']
    ts_name = os.path.join(ddir,args['ts_fname'])
    ts = h5py.File(ts_name)
    tsv = np.array(ts['timeseries'])
    tsv -= tsv[0][None,...]
    from datetime import datetime
    sdates = np.array(ts['date']).astype(np.str)
    dates = []
    for i,d in enumerate(sdates):
        dnow = datetime.strptime(d,'%Y%m%d')
        if i == 0:
            d0 = dnow
        dates.append((dnow - d0).days)
    dates = np.array(dates)    
    
    pred_names = sorted(glob(os.path.join(pdir,pred_input + '*_data.pred')))
    loc_names = sorted(glob(os.path.join(pkdir,pred_input + '*_locs.pck')))  
    x0 = float(ts.attrs['X_FIRST'])
    dx = float(ts.attrs['X_STEP'])
    y0 = float(ts.attrs['Y_FIRST'])
    dy = float(ts.attrs['Y_STEP'])
    ylocs = []
    xlocs = []
    for prn,locn in zip(pred_names,loc_names):
        pred = np.fromfile(prn,np.int32)
        locs = pc.load(open(locn,'rb'))
        ylocs.extend(locs[0][pred])
        xlocs.extend(locs[1][pred])
    xlocs = np.array(xlocs)
    ylocs = np.array(ylocs)
    lg_vals = []
    #linregress returns slope, intercept, r_value, p_value, std_err
    for i,v in enumerate(tsv[:,ylocs,xlocs].T):
        lg_vals.append(linregress(dates,v))
    lg_vals = np.array(lg_vals)
    sel0 = lg_vals[:,3] < 0.05
    xlocs = xlocs[sel0]
    ylocs = ylocs[sel0]
    plt.figure()
    x = x0 + dx*xlocs
    y = y0 + dy*ylocs
    plt.plot(x,y,'ro',ms=3)
    plt.axis('equal')
    if args['fig_name'] is None:
        fig_name = os.path.basename(ts_name).replace('.h5','.png')
    else:
        fig_name = args['fig_name'] + '.png'
    plt.axis('off')
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    plt.savefig(fig_name.replace('.png','_{0:.3f}_{1:.3f}_{2:.3f}_{3:.3f}.png'.format(xmin,xmax,ymin,ymax)),transparent=True)
    ts.close()
    
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
    if nargs['action'] == 'create_ts':
        create_ts(nargs)
    elif nargs['action'] == 'create_png':
        create_png(nargs)
    else:
        raise Exception('Unrecognized action {0}.'.format(nargs['action']))
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
    
