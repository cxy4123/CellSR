"""External checks of the two frozen fixed-structure sharing tasks."""
from pathlib import Path
import hashlib
import json
from itertools import product
import numpy as np
import pandas as pd
from screen_real_case_pv import calibrate,arrays
from screen_real_case_mass import bank
from paderborn_blind_fit import choose,lawset,edgewise

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"benchmark_outputs/cellsr1_revision/external_case_comparison_v1"
HORST=ROOT/"data/cellsr_horst_external"
SOLAR=ROOT/"data/cellsr_pv_external"

def raw_pv(path,cols):
    frame=pd.read_csv(path,sep=";",skipinitialspace=True,encoding="latin1")
    frame.columns=frame.columns.str.strip()
    # Only requested channels leave the source adapter.
    date=pd.to_datetime(frame["Date"].astype(str).str.strip()+" "+frame["Time"].astype(str).str.strip(),dayfirst=True,format="mixed",errors="raise")
    result=frame[cols].apply(pd.to_numeric,errors="coerce").set_axis(date).sort_index()
    return result[~result.index.duplicated(keep=False)]

def predict(arms):
    data={a+"_"+role:v for a,p in zip("ABC",arms) for role,v in p.items()}
    train=lawset(data);full=lawset(data,True)
    methods=choose(train)
    def slopes(role):
        return np.array([np.linalg.lstsq(g[role][:,:1],g[role][:,1],rcond=None)[0][0] for g in arms])
    fit,val=slopes("fit"),slopes("val")
    j=1+np.argmin(abs(fit[1:]-fit[0]))
    pair=(np.inf,0) if j==1 else (0,np.inf)
    methods["nearest"]=pair
    methods["nearest_reject"]=pair if abs(fit[0]-fit[j])<=abs(fit[0]-val[0])+abs(fit[j]-val[j]) else (0,0)
    coefs={p:edgewise(full,p).coefficients["A"].item() for p in set(methods.values())}
    return [(m,coefs[p],"soft" if any(0<v<np.inf for v in p) else "".join(k for k,v in zip("BC",p) if v) or "-") for m,p in methods.items()]

def pv_cases():
    a,b=calibrate();rng=np.random.default_rng(81590)
    for path in sorted(SOLAR.glob("*.csv")):
        d=raw_pv(path,["Isc","Voc","Tbom"]);temp=d.Tbom-25
        d["i"]=d.Isc/(a[0]+a[1]*temp)
        d["v"]=np.exp((d.Voc-b[0]-b[1]*temp)/(b[2]+b[3]*temp))
        for date in d.index.normalize().unique():
            blocks={}
            for hour,minute in product(range(9,15),[0,30]):
                t=date+pd.Timedelta(hours=hour,minutes=minute)
                g=d.reindex(pd.date_range(t,periods=6,freq="5min"))
                if np.isfinite(g).all().all() and (g[["i","v","Voc","Isc"]]>0).all().all():
                    blocks[t]=g
            for t,target in blocks.items():
                shifted=t+pd.Timedelta(hours=1 if t.hour%2 else -1)
                options=sorted((u for u in blocks if u not in (t,shifted)),key=lambda u:(abs(u-t),u))
                if shifted not in blocks or not options:continue
                for kind,donors in [("one",[t,shifted]),("zero",[shifted,options[0]])]:
                    if rng.integers(2):donors.reverse()
                    arms=[arrays(target.v.to_numpy()),*[arrays(blocks[u].i.to_numpy()) for u in donors]]
                    yield arms,dict(kind=kind,cluster=str(date.date()),month=path.stem,target=str(t),donor_b=str(donors[0]),donor_c=str(donors[1]),truth="BC"[donors.index(t)] if t in donors else "-")

def gait_cases():
    d=pd.read_csv(OUT/"gait_windows.csv")
    valid=[]
    for s,g in d.groupby("subject"):
        if g.trial.nunique()<6:continue
        trial_ids=sorted(g.trial.unique())
        if any(len(g[(g.axis==("up" if is_donor else "ap")) & g.trial.isin(trial_ids[is_donor::2][f::3])])<2 for is_donor in (0,1) for f in range(3)):continue
        valid.append(s)
    banks={(s,b,f):bank(d,s,b,f) for s,b,f in product(valid,(False,True),range(3))}
    rng=np.random.default_rng(91229)
    for s in valid:
        others=rng.permutation([n for n in valid if n!=s]).tolist()
        for i,other in enumerate(others):
            for fold,kind in product(range(3),("one","zero")):
                donors=[s,other] if kind=="one" else [other,others[(i+1)%len(others)]]
                order=rng.permutation(donors).tolist()
                arms=[banks[s,False,fold],*[banks[k,True,fold] for k in order]]
                yield arms,dict(kind=kind,cluster=s,month="-",target=s,donor_b=order[0],donor_c=order[1],fold=fold,truth="BC"[order.index(s)] if s in order else "-")

def execute(panel):
    dest=OUT/panel;dest.mkdir(exist_ok=True);rows=[];keys=[]
    cases=pv_cases if panel=="pv" else gait_cases
    for case,(arms,key) in enumerate(cases()):
        for m,value,graph in predict(arms):rows.append((case,m,value*(1000 if panel=="pv" else 70),graph))
        keys.append(dict(case=case,**key))
    r=pd.DataFrame(rows,columns=["case","method","pred","graph"])
    assert not r.empty and np.isfinite(r.pred).all()
    file=dest/"predictions.csv";r.to_csv(file,index=False)
    (dest/"prediction_hash.txt").write_text(hashlib.sha256(file.read_bytes()).hexdigest())
    score(panel, dest, r, pd.DataFrame(keys))

def score(panel, dest, r, key):
    if panel=="pv":
        refs=pd.concat([raw_pv(p,["Gpoa"]) for p in sorted(SOLAR.glob("*.csv"))]).Gpoa
        times=set(key.target)|set(key.donor_b)|set(key.donor_c)
        ref={t:refs.reindex(pd.date_range(t,periods=6,freq="5min")).mean(skipna=False) for t in times}
    else:
        frame=pd.read_csv(HORST/"Gait_subject_info_v3.csv",sep=";")
        ref=dict(zip(frame.iloc[:,0].astype(str).str.strip(),pd.to_numeric(frame["body mass [kg]"].astype(str).str.replace(",",".",regex=False))))
    key["ref"]=key.target.map(ref);key["contrast"]=abs(key.donor_b.map(ref)-key.donor_c.map(ref))
    key.to_csv(dest/"key_scored.csv",index=False)
    r=r.merge(key,on="case",validate="many_to_one")
    r=r[np.isfinite(r.ref)].copy()
    r["mae"]=(r.pred-r.ref).abs()
    r["exact"]=r.graph==r.truth
    r["both"]=(r.kind=="one")&r.exact&(r.mae<r.case.map(r[r.method=="independent"].set_index("case").mae)-1e-8)
    r["fallback"]=r.graph=="-"
    oracle=r[((r.kind=="zero")&(r.method=="independent"))|((r.kind=="one")&(r.method=="fixed_"+r.truth))].copy()
    oracle["method"]="informed";r=pd.concat([r,oracle],ignore_index=True)
    r.to_csv(dest/"scored.csv",index=False)
    metrics=["mae","exact","both","fallback"]
    agg=r.groupby(["kind","cluster","method"])[metrics].mean()
    summary=agg.groupby(["kind","method"]).mean()
    summary.to_csv(dest/"by_design.csv");summary.groupby("method").mean().to_csv(dest/"balanced.csv")
    if panel=="pv":
        r.groupby(["month","kind","cluster","method"])[metrics].mean().groupby(["month","kind","method"]).mean().to_csv(dest/"by_month.csv")
    cut=50 if panel=="pv" else 5
    one=r[r.kind=="one"].copy();one["near"]=one.contrast<=cut
    one.groupby(["near","cluster","method"])[metrics].mean().groupby(["near","method"]).mean().to_csv(dest/"by_contrast.csv")
    # Recompute differences deleting every role occupied by a person, or an entire month.
    diff=[]
    for label in (sorted(r.month.unique()) if panel=="pv" else sorted(r.cluster.unique())):
        if panel=="pv":remaining=r[r.month!=label]
        else:remaining=r[(r.cluster!=label)&(r.donor_b!=label)&(r.donor_c!=label)]
        w=remaining.groupby(["cluster","kind","method"]).mae.mean().groupby(["kind","method"]).mean().unstack().mean()
        for m in w.index:diff.append(dict(removed=label,baseline=m,delta=w.CellSR_B-w[m]))
    pd.DataFrame(diff).to_csv(dest/"deletion_sensitivity.csv",index=False)
    print(panel,key.shape,len(r),summary.loc[(slice(None),["CellSR_B","independent","nearest","nearest_reject"]),:].round(4).to_string(),flush=True)
if __name__=="__main__":
    import sys
    if sys.argv[1]=="rescore-gait":
        dest=OUT/"gait"
        file=dest/"predictions.csv"
        assert hashlib.sha256(file.read_bytes()).hexdigest()==(dest/"prediction_hash.txt").read_text().strip()
        key=pd.read_csv(dest/"key_scored.csv").drop(columns=["ref","contrast"])
        score("gait",dest,pd.read_csv(file),key)
    else:
        execute(sys.argv[1])

