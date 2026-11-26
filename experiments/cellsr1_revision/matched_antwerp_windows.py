"""Construct mass-free work/impulse windows from Antwerp raw C3Ds."""
from pathlib import Path
import re
import sys
import tempfile
from zipfile import ZipFile
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import label
from scipy.signal import savgol_filter
from matched_false_share import ROOT, OUT

sys.path.insert(0,str(ROOT/"data/cellsr_horst_external/reader"))
import ezc3d

ARCHIVE = ROOT/"data/cellsr_gait_antwerp/healthy_verified.zip"

def com(mark):
    def avg(*names):
        return np.mean([mark[n] for n in names],axis=0)
    rear = mark["SACR"] if "SACR" in mark else avg("LPSI","RPSI")
    pelvis = (mark["LASI"] + mark["RASI"] + rear)/3
    body = .081*avg("LFHD","RFHD","LBHD","RBHD")+.497*(pelvis+avg("LSHO","RSHO"))/2
    for s in "LR":
        hip = (mark[s+"ASI"]+rear)/2
        knee,ankle,shoulder,elbow,hand = [mark[s+k] for k in ["KNE","ANK","SHO","ELB","FIN"]]
        wrist = avg(s+"WRA",s+"WRB")
        for a,b,f,m in [(hip,knee,.433,.100),(knee,ankle,.433,.0465),(shoulder,elbow,.436,.028),
                        (elbow,wrist,.430,.016),(wrist,hand,.506,.006)]:
            body += m*(a+f*(b-a))
        body += .0145*avg(s+"HEE",s+"TOE")
    return body


def extract(c):
    header=c["header"]["points"]
    rate=header["frame_rate"]
    analog=c["header"]["analogs"]["frame_rate"]
    ratio=round(analog/rate)
    assert abs(analog/rate-ratio)<1e-6
    steps=round(rate*.10)
    assert abs(steps/rate-.1)<1e-6
    params=c["parameters"]
    assert params["POINT"]["UNITS"]["value"][0].strip().lower()=="mm"
    names=[n.split(":")[-1].strip().upper() for n in params["POINT"]["LABELS"]["value"]]
    xyz=c["data"]["points"][:3].transpose(1,2,0).copy()/1000
    xyz[(c["data"]["meta_points"]["residuals"][0]<0)|np.all(xyz==0,axis=2)] = np.nan
    markers=dict(zip(names,xyz))
    pos=com(markers)
    t=(header["first_frame"]+np.arange(len(pos)))/rate
    idx=ratio*np.arange(len(t))
    velocity=savgol_filter(pos,2*round(.05*rate)+1,3,deriv=1,delta=1/rate,axis=0,mode="nearest")
    events=params["EVENT"]
    event_time=np.array([60,1])@events["TIMES"]["value"]
    def state(side, begin, end):
        mask=np.zeros(len(t),dtype=bool)
        ordered=sorted((time,lbl) for side_name,lbl,time in zip(events["CONTEXTS"]["value"],events["LABELS"]["value"],event_time)
                       if side_name.strip().lower()==side and lbl in ("Foot Strike","Foot Off"))
        for (a,la),(b,lb) in zip(ordered,ordered[1:]):
            if la==begin and lb==end:
                mask |= (t>=a+.025)&(t<=b-.025)
        return mask
    plates=c["data"]["platform"]
    if not plates: raise ValueError("no raw force plate")
    forces=[]; cops=[]
    for plate in plates:
        assert plate["unit_force"]=="N" and plate["unit_position"]=="mm"
        force=plate["force"].T.copy()
        baseline=force[:,2] <= np.median(force[:,2])
        if not baseline.any():raise ValueError("invalid force")
        force-=np.median(force[baseline],axis=0)
        forces.append(np.where(force[:,2:3]>35,force,0))
        cops.append(plate["center_of_pressure"].T[idx,:2]/1000)
    forces=np.asarray(forces)
    loaded=forces[:,idx,2]>0
    ok=np.zeros(len(t),dtype=bool)
    for s,side,opp in [("L","left","right"),("R","right","left")]:
        single=state(side,"Foot Strike","Foot Off")&state(opp,"Foot Off","Foot Strike")
        foot=(markers[s+"HEE"]+markers[s+"TOE"])/2
        near=np.linalg.norm(np.asarray(cops)-foot[None,:,:2],axis=2)<.18
        ok|=single & (loaded.sum(axis=0)==1) & np.any(near&loaded,axis=0)
    ok &= np.isfinite(velocity).all(axis=1)
    ok[:steps]=False;ok[-steps:]=False
    impulse=cumulative_trapezoid(forces.sum(axis=0),dx=1/analog,initial=0,axis=0)[idx]
    work=np.sum(np.diff(impulse,axis=0)*(velocity[:-1]+velocity[1:])/2,axis=1)
    labels,count=label(ok)
    rows=[]
    for j in range(1,count+1):
        span=np.flatnonzero(labels==j)
        for a in range(span[0],span[-1]-steps+1,steps):
            b=a+steps
            x=(velocity[b]@velocity[b]-velocity[a]@velocity[a])/2+9.80665*(pos[b,2]-pos[a,2])
            rows.extend([("ap",x,work[a:b].sum()),("up",velocity[b,2]-velocity[a,2]+.1*9.80665,impulse[b,2]-impulse[a,2])])
    return rows


def main(mode):
    rows=[];qc=[]
    with ZipFile(ARCHIVE) as archive,tempfile.TemporaryDirectory() as tmp:
        members={}
        for name in archive.namelist():
            match=re.search(r"/(SUBJ\d+)/[^/]+\((\d+)\)\.c3d$",name,re.I)
            if match:members.setdefault(match[1],{})[int(match[2])]=name
        assert len(members)==138
        for subject,trials in sorted(members.items()):
            if len([j for j in trials if j>0])<4:continue
            for j,name in sorted(trials.items()):
                if mode!="ref" and j==0:continue
                path=Path(tmp)/"trial.c3d"
                path.write_bytes(archive.read(name))
                c=ezc3d.c3d(str(path),extract_forceplat_data=mode!="ref")
                if mode=="ref":
                    proc=c["parameters"].get("PROCESSING",{})
                    for key in proc:
                        if key.lower().split(":")[-1]=="bodymass":
                            rows.append((subject,float(proc[key]["value"][0])))
                else:
                    try:
                        found=extract(c)
                        qc.append((subject,j,len(found)//2,"ok" if found else "no verified window"))
                        rows.extend((subject,j,*r) for r in found)
                    except (KeyError,ValueError) as e:
                        qc.append((subject,j,0,str(e)))
    if mode=="ref":
        refs=pd.DataFrame(rows,columns=["subject","mass"]).drop_duplicates()
        assert refs.subject.is_unique and refs.mass.between(25,200).all()
        refs.to_csv(OUT/"antwerp_reference.csv",index=False)
    else:
        pd.DataFrame(rows,columns=["subject","trial","axis","xM","J"]).to_csv(OUT/"antwerp_windows.csv",index=False)
        pd.DataFrame(qc,columns=["subject","trial","windows","status"]).to_csv(OUT/"antwerp_qc.csv",index=False)
        print(pd.Series([x[-1] for x in qc]).value_counts())


if __name__=="__main__":
    main(sys.argv[1])
