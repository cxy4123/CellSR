"""Screen a real PV irradiance case using IEC indoor calibration and a separate outdoor pyranometer."""
from itertools import product
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from paderborn_blind_fit import choose,lawset,edgewise

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"data/cellsr_pv_case_screen"
OUT=ROOT/"benchmark_outputs/cellsr1_revision/real_case_screen_pv"
def calibrate():
    with (DATA/"characterization").open("rb") as stream:
        rows=list(load_workbook(stream,read_only=True,data_only=True).active.values)
    def points(label):
        start=next(i for i,r in enumerate(rows) if r[1] and str(r[1]).lower().startswith(label))
        grid=rows[start+1][1:5]
        result=[]
        for r in rows[start+2:start+9]:
            for j,temp in enumerate(grid):
                if isinstance(r[j+1],(int,float)):
                    result.append([r[0]/1000,temp-25,r[j+1]])
        return np.array(result).T
    e,t,y=points("isc"); a=np.linalg.lstsq(np.column_stack([e,e*t]),y,rcond=None)[0]
    e,t,y=points("voc"); b=np.linalg.lstsq(np.column_stack([np.ones(len(e)),t,np.log(e),t*np.log(e)]),y,rcond=None)[0]
    return a,b
def read(cols):
    d=pd.read_csv(DATA/"2018_07",sep=";",skipinitialspace=True,usecols=["Date","Time",*cols])
    idx=pd.to_datetime(d.Date+" "+d.Time,format="%d.%m.%Y %H:%M:%S")
    assert idx.is_unique
    return d.set_index(idx).sort_index()[cols]
def arrays(values):
    return {a:np.column_stack([np.ones(2),values[ix]]) for a,ix in [
        ("fit",[0,3]),("val",[1,4]),("test",[2,5])]}
def main():
    a,b=calibrate();d=read(["Isc","Voc","Tbom"])
    x=d.Tbom-25
    d["i"]=d.Isc/(a[0]+a[1]*x)
    d["v"]=np.exp((d.Voc-b[0]-b[1]*x)/(b[2]+b[3]*x))
    rng=np.random.default_rng(81590)
    rows=[];keys=[];expected=0
    for date,hour,minute in product(d.index.normalize().unique(),[9,11,13],[0,30]):
        starts=[date+pd.Timedelta(hours=hour+j,minutes=minute) for j in [0,1]]
        blocks=[d.reindex(pd.date_range(t,periods=6,freq="5min")) for t in starts]
        expected+=2
        if any(not np.isfinite(z).all().all() or (z[["Isc","Voc","i","v"]]<=0).any().any() for z in blocks):continue
        for k in [0,1]:
            case=f"{len(keys):04d}"
            perm=rng.permutation(2);same="BC"[list(perm).index(k)]
            pack={arm+"_"+role:value for arm,series in zip("ABC",[blocks[k].v,blocks[perm[0]].i,blocks[perm[1]].i])
                  for role,value in arrays(series.to_numpy()).items()}
            laws=lawset(pack);full=lawset(pack,True)
            methods=choose(laws)
            estimates=edgewise(laws,(0,0)).coefficients
            closest=min("BC",key=lambda arm:abs(estimates[arm].item()-estimates["A"].item()))
            methods["nearest"]=(np.inf,0) if closest=="B" else (0,np.inf)
            for method,pair in methods.items():
                pred=edgewise(full,pair).coefficients["A"].item()*1000
                rows.append(dict(case=case,method=method,pred=pred,B=pair[0],C=pair[1]))
            keys.append(dict(case=case,same=same,day=str(date.date()),start=starts[k],other=starts[1-k]))
    r=pd.DataFrame(rows);r.to_csv(OUT/"predictions_before_scoring.csv",index=False)
    seal=hashlib.sha256((OUT/"predictions_before_scoring.csv").read_bytes()).hexdigest()
    (OUT/"prediction_hash.txt").write_text(seal)
    ref=read(["Gpoa"])
    def reference(t):return ref.reindex(pd.date_range(t,periods=6,freq="5min")).Gpoa.mean(skipna=False)
    key=pd.DataFrame(keys);key["ref"]=key.start.map(reference);key["contrast"]=(key.ref-key.other.map(reference)).abs()
    assert np.isfinite(key[["ref","contrast"]]).all().all()
    r=r.merge(key,on="case",validate="many_to_one")
    r["mae"]=(r.pred-r.ref).abs()
    r["correct"]=np.where(r.same=="B",(r.B==np.inf)&(r.C==0),(r.C==np.inf)&(r.B==0))
    r["fallback"]=(r.B==0)&(r.C==0)
    r["improved"]=r.mae<r.case.map(r[r.method=="independent"].set_index("case").mae)-1e-8
    r["both"]=r.correct&r.improved
    oracle=r[r.method=="fixed_"+r.same].copy();oracle["method"]="known_single"
    r=pd.concat([r,oracle]);r.to_csv(OUT/"scored.csv",index=False)
    stats=[];paired=[]
    for name,g in [("all",r),("near",r[r.contrast<50]),("clear",r[r.contrast>=100])]:
        by=g.groupby(["day","method"])[["mae","correct","both","fallback"]].mean()
        t=by.groupby("method").mean().reset_index();t["stratum"]=name;t["n_cases"]=g.case.nunique()
        t["days"]=g.day.nunique();stats.append(t)
        wide=by.mae.unstack()
        ix=np.random.default_rng(1051).integers(len(wide),size=(10000,len(wide)))
        for m in wide:
            diff=(wide.CellSR_B-wide[m]).to_numpy()
            lo,hi=np.quantile(diff[ix].mean(1),[.025,.975])
            paired.append(dict(stratum=name,versus=m,delta=diff.mean(),lo=lo,hi=hi))
    pd.concat(stats).to_csv(OUT/"summary.csv",index=False);pd.DataFrame(paired).to_csv(OUT/"paired.csv",index=False)
    (OUT/"coverage.json").write_text(json.dumps(dict(raw_rows=len(d),scheduled=expected,included=len(key),days=key.day.nunique(),
        calibration_I=a.tolist(),calibration_V=b.tolist(),sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in DATA.iterdir() if f.is_file()}),indent=2))
    print(pd.concat(stats).query("stratum=='all'")[["method","mae","correct","both","n_cases","days"]].round(3).to_string(index=False))
if __name__=="__main__":main()
