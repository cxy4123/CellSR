"""Blind signal construction for the Denayer et al. v3 weighted gait panel."""
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import label
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/cellsr_weighted_gait_v3'
OUT = ROOT / 'benchmark_outputs/cellsr1_revision/weighted_gait_real_v1'
TRIAL = re.compile(r'OpenSim/(SUBJ\d+)/trcFiles/(SUBJ\d+_WeightedWalking_([125])kg_(\d+))_markers.trc$')


def marker_tracks(raw):
    lines = raw.replace(b'\r', b'').decode().splitlines()
    assert lines[2].split()[4] == 'mm'
    names = lines[3].strip().split('\t')[2::3]
    v = pd.read_csv(io.StringIO('\n'.join(lines[5:])), sep='\t', header=None,
                    usecols=list(range(2+3*len(names)))).to_numpy()
    t = v[:,1]
    assert np.allclose(np.diff(t), .01, atol=1e-6)
    return t, dict(zip(names, (v[:,2:]/1000).reshape(len(t),len(names),3).transpose(1,0,2)))


def kinematics(markers, cuff_offset):
    """Unloaded segment barycentres; fixed proportions, no force or mass labels."""
    def avg(*names):
        return np.mean([markers[n] for n in names], axis=0)
    pelvis = avg('LASI','RASI','LPSI','RPSI')
    shoulder = avg('LSHO','RSHO')
    head = avg('LFHD','RFHD','LBHD','RBHD')
    body = .081*head + .497*(pelvis+shoulder)/2
    cuffs = []
    for side in 'LR':
        def loc(*suffix):
            return avg(*(side+k for k in suffix))
        hip = loc('ASI','PSI')
        knee, ankle = loc('KNE','KNEmed'), loc('ANK','ANKmed')
        elbow, wrist = loc('ELB','ELBmed'), loc('WRA','WRB')
        def segment(a,b,f):
            return a + f*(b-a)
        body += (.100*segment(hip,knee,.433) + .0465*segment(knee,ankle,.433)
                 + .0145*loc('HEE','TOE') + .028*segment(loc('SHO'),elbow,.436)
                 + .016*segment(elbow,wrist,.430) + .006*segment(wrist,loc('FIN'),.506))
        shaft = ankle-knee
        cuffs.append(knee+cuff_offset*shaft/np.linalg.norm(shaft,axis=1)[:,None])
    return np.stack([body, cuffs[0]+cuffs[1]], axis=1), pelvis


def contacts(raw):
    text = raw.replace(b'\r',b'').split(b'endheader\n',1)[1]
    ids = [4,2,3]
    names = [['ground_force_%d_v%s'%(plate,axis) for axis in 'xyz'] for plate in ids]
    frame = pd.read_csv(io.BytesIO(text), sep='\t', usecols=['time',*sum(names,[])])
    t = frame.time.to_numpy()
    dt = np.median(np.diff(t))
    assert np.allclose(np.diff(t),dt,atol=1e-7) and abs(dt-.001)<1e-7
    plate = np.stack([frame[cols].to_numpy() for cols in names],axis=1)
    if not np.isfinite(plate).all():
        raise ValueError('missing GRF')
    # Median baseline per channel from low-force samples, unaffected by loading.
    for k in range(3):
        unloaded = np.abs(plate[:,k,1])<35
        if unloaded.sum()<50:
            raise ValueError('no unloaded baseline')
        plate[:,k] -= np.median(plate[unloaded,k],axis=0)
    bouts = []
    for vy in plate[:,:,1].T:
        ids_,count = label(vy>35)
        intervals = [np.flatnonzero(ids_==i) for i in range(1,count+1)]
        intervals = [idx for idx in intervals if len(idx)>=100]
        if len(intervals)!=1:
            raise ValueError('ambiguous force contact')
        idx=intervals[0]
        bouts.append((t[idx[0]],t[idx[-1]]))
    (a,aend),(b,bend),(c,_) = bouts
    if not (a<b<c and aend>b and bend>c):
        raise ValueError('incomplete ordered foot contacts')
    grf = np.where((plate[:,:2,1]>20)[:,:,None], plate[:,:2], 0).sum(axis=1)
    return t, cumulative_trapezoid(grf,t,axis=0,initial=0), b+.025, c-.025


def trial_windows(marker_raw, force_raw):
    t, markers = marker_tracks(marker_raw)
    ft, integral, start, end = contacts(force_raw)
    primary, pelvis = kinematics(markers,.10)
    delta = np.nanmedian(pelvis[-15:],axis=0)-np.nanmedian(pelvis[:15],axis=0)
    delta[1]=0
    if not np.isfinite(delta).all() or np.linalg.norm(delta)<.3:
        raise ValueError('walking direction unclear')
    axes = np.stack([delta/np.linalg.norm(delta), [0,1,0]]).T
    limits = np.flatnonzero((t>=max(start,t[0]+.07,ft[0])) & (t<=min(end,t[-1]-.07,ft[-1])))
    if not len(limits):
        raise ValueError('no complete support window')
    streams = {'primary':primary,'cuff07':kinematics(markers,.07)[0],'cuff13':kinematics(markers,.13)[0]}
    # No retained derivative uses padded samples (the 70 ms margin exceeds
    # the 50 ms half-window). Avoid fitting polynomials to missing end markers.
    velocities = {k:savgol_filter(x@axes,11,3,deriv=1,delta=.01,axis=0,mode='nearest')
                  for k,x in streams.items()}
    imp = np.stack([np.interp(t,ft,col) for col in (integral@axes).T],axis=1)
    rows = []
    for i in range(limits[0], limits[-1]-9, 10):
        j = i+10
        changes = {k:v[j]-v[i] for k,v in velocities.items()}
        if not all(np.isfinite(v).all() for v in changes.values()):
            continue
        for dim,axis in enumerate(['ap','up']):
            gd = 9.80665*(t[j]-t[i]) if dim else 0.
            rows.append(dict(axis=axis,start=t[i],end=t[j],J=imp[j,dim]-imp[i,dim],
                xM=changes['primary'][0,dim]+gd,
                **{k:v[1,dim]+2*gd for k,v in changes.items()}))
    if len(rows)<6:
        raise ValueError('fewer than three valid 100 ms windows')
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    meta=json.loads((DATA/'figshare_v3.json').read_text(encoding='utf8'))
    for name in ['OpenSim.zip','MAT_normalizedData-v21Aug2026.mat']:
        expected=next(f for f in meta['files'] if f['name']==name)
        with (DATA/name).open('rb') as file:
            assert hashlib.file_digest(file,'md5').hexdigest()==expected['computed_md5']
    rows, ledger = [], []
    with zipfile.ZipFile(DATA/'OpenSim.zip') as z:
        for member in z.namelist():
            hit = TRIAL.fullmatch(member)
            if not hit:
                continue
            subject,stub,load,trial = hit.groups()
            key=dict(subject=subject,load=int(load),trial=int(trial))
            force=f'OpenSim/{subject}/forceFilesOriginal/{stub}_forces_COP_original.mot'
            try:
                data=trial_windows(z.read(member),z.read(force))
            except (ValueError,KeyError) as ex:
                ledger.append(dict(**key,windows=0,reason=str(ex)))
                continue
            rows.extend(dict(**key,**r) for r in data)
            ledger.append(dict(**key,windows=len(data)//2,reason='included'))
    pd.DataFrame(ledger).to_csv(OUT/'trial_qc.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'windows.csv',index=False)
    print(pd.DataFrame(ledger).reason.value_counts().to_string(),flush=True)


if __name__=='__main__':
    main()
