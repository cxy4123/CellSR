"""Reconstruct unnormalized Horst work and impulse observations without opening subject masses."""
from pathlib import Path
import re
import sys
import zipfile
import tempfile
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import label
from external_pair_compare import HORST,OUT

sys.path.insert(0,str(HORST/"reader"))
import ezc3d

def markers(c):
    assert c["parameters"]["POINT"]["UNITS"]["value"][0].strip().lower()=="mm"
    a=c["data"]["points"][:3].transpose(1,2,0).copy()/1000
    a[(c["data"]["meta_points"]["residuals"][0]<0)|np.all(a==0,axis=2)]=np.nan
    names=[" ".join(s.strip().replace("WRITS","WRIST").replace("WIRST","WRIST").split()) for s in c["parameters"]["POINT"]["LABELS"]["value"]]
    return dict(zip(names,a))

def reconstruct(c,static):
    m=markers(c)
    def avg(*names):return np.mean([m[n] for n in names],axis=0)
    def joint(side,segment,which):
        names=[f"{side} {segment} {i}" for i in range(1,5)]
        q=np.stack([static[n] for n in names]);p=np.stack([m[n] for n in names],axis=1)
        target=(static[f"{side} {which} MED"]+static[f"{side} {which} LAT"])/2
        valid=np.isfinite(p).all(axis=(1,2));res=np.full((len(p),3),np.nan)
        if not np.isfinite(q).all() or not np.isfinite(target).all():return res
        if valid.any():
            v=p[valid];mean=v.mean(axis=1);base=q-q.mean(axis=0)
            u,_,vt=np.linalg.svd(np.einsum("mi,tmj->tij",base,v-mean[:,None,:]))
            u[:,:,-1]*=np.linalg.det(u@vt)[:,None]
            res[valid]=(target-q.mean(axis=0))@(u@vt)+mean
        return res
    pelvis=avg("R ILIAC","L ILIAC","SACRUM")
    shoulder=avg("R ACROMION","L ACROMION")
    com=.081*avg("HEAD ANT","R HEAD","L HEAD")+.497*(pelvis+shoulder)/2
    for side in "RL":
        hip=m[side+" GTROCH"];knee=joint(side,"THIGH","KNEE");ankle=joint(side,"SHANK","ANKLE")
        elbow=avg(side+" ELBOW MED",side+" ELBOW LAT");wrist=avg(side+" WRIST LAT",side+" WRIST MED")
        def seg(x,y,f):return x+f*(y-x)
        foot=(m[side+" HEEL"]+avg(side+" FOOT MED",side+" FOOT LAT"))/2
        com+=(.100*seg(hip,knee,.433)+.0465*seg(knee,ankle,.433)+.0145*foot+
              .028*seg(m[side+" ACROMION"],elbow,.436)+.016*seg(elbow,wrist,.430)+.006*seg(wrist,m[side+" HAND"],.506))
    return m,com

def extract(c,static):
    m,com=reconstruct(c,static)
    assert c["header"]["points"]["frame_rate"]==250 and c["header"]["analogs"]["frame_rate"]==1000
    plates=c["data"]["platform"]
    if len(plates)!=2:raise ValueError("missing force plates")
    forces=[]
    for plate in plates:
        assert plate["unit_force"]=="N"
        v=plate["force"].T.copy();low=np.abs(v[:,2])<35
        if low.sum()<25:raise ValueError("no unloaded baseline")
        v-=np.median(v[low],axis=0)
        forces.append(v)
    f1,f2=forces
    def stance(f):
        lab,n=label(f[:,2]>35);intervals=[np.flatnonzero(lab==i) for i in range(1,n+1)]
        long=[a for a in intervals if len(a)>=100]
        if len(long)!=1:raise ValueError("ambiguous stance")
        return long[0][0],long[0][-1]
    a,b=stance(f1);u,w=stance(f2)
    if not a<u<b<w:raise ValueError("unverified sequential contact")
    force=np.where(f1[:,2:3]>20,f1,0)+np.where(f2[:,2:3]>20,f2,0)
    idx=4*np.arange(len(com))+3
    if idx[-1]>=len(force): raise ValueError("unaligned streams")
    airborne=np.ones(len(com),bool)
    for n in ["R HEEL","R FOOT MED","R FOOT LAT"]:
        airborne &= m[n][:,2]>static[n][2]+.020
    supported=(idx>=u+25)&(idx<=w-25)&((idx<=b)|airborne)
    supported[:14]=False;supported[-14:]=False
    v=savgol_filter(com,27,3,deriv=1,delta=.004,axis=0,mode="nearest")
    impulse=cumulative_trapezoid(force,dx=.001,initial=0,axis=0)[idx]
    increments=np.sum(np.diff(impulse,axis=0)*(v[1:]+v[:-1])/2,axis=1)
    valid=supported & np.isfinite(com).all(axis=1)&np.isfinite(v).all(axis=1)
    lab,n=label(valid);rows=[]
    for i in range(1,n+1):
        span=np.flatnonzero(lab==i)
        for start in range(span[0],span[-1]-24,25):
            end=start+25;dz=com[end,2]-com[start,2]
            x=(v[end]@v[end]-v[start]@v[start])/2+9.80665*dz
            rows.extend([dict(axis="ap",xM=x,J=increments[start:end].sum()),
                         dict(axis="up",xM=v[end,2]-v[start,2]+.1*9.80665,J=impulse[end,2]-impulse[start,2])])
    if not rows:raise ValueError("no complete observed 100 ms window")
    return rows

def main():
    ledger=[];allrows=[]
    with zipfile.ZipFile(HORST/"Gait_rawdata_c3d.zip") as z,tempfile.TemporaryDirectory() as tmp:
        files={}
        for n in z.namelist():
            hit=re.fullmatch(r"(S\d+)_(\d+)_(static|gait)\.c3d",Path(n).name,re.I)
            if hit:files.setdefault(hit[1].upper(),[]).append((hit[3].lower(),int(hit[2]),n))
        def read(name,plates):
            p=Path(tmp)/"input.c3d";p.write_bytes(z.read(name))
            return ezc3d.c3d(str(p),extract_forceplat_data=plates)
        for s,items in sorted(files.items()):
            statics={}
            for kind,index,name in sorted(items):
                if kind=="static":
                    ms=markers(read(name,False))
                    statics[index]={key:np.nanmedian(x,axis=0) for key,x in ms.items()}
            for k,tr,n in items:
                if k!="gait":continue
                if not statics:
                    ledger.append(dict(subject=s,trial=tr,status="no static calibration",windows=0));continue
                index=max((i for i in statics if i<=tr),default=min(statics))
                static=statics[index]
                try:rows=extract(read(n,True),static)
                except (ValueError,KeyError) as e:
                    ledger.append(dict(subject=s,trial=tr,status=str(e),windows=0));continue
                ledger.append(dict(subject=s,trial=tr,status="included",windows=len(rows)//2))
                allrows.extend(dict(subject=s,trial=tr,**r) for r in rows)
    pd.DataFrame(ledger).to_csv(OUT/"gait_qc.csv",index=False)
    d=pd.DataFrame(allrows);d.to_csv(OUT/"gait_windows.csv",index=False)
    print(pd.DataFrame(ledger).status.value_counts(),d.subject.nunique(),flush=True)
if __name__=="__main__":main()

