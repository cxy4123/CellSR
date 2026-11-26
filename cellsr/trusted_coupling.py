"""Fixed-basis shared-coefficient prototype with explicit rejection and validation.

This module is opt-in and is not a topology-discovery or adaptive-search theorem.
Sharing maps refer to linear coefficients, not nonlinear inverse parameters.
"""
from dataclasses import dataclass,field
import numpy as np
from scipy.linalg import block_diag,null_space
from scipy.stats import chi2

@dataclass
class LinearLaw:
    name: str
    fit_x: np.ndarray
    fit_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    shared_columns: dict[str,int]

@dataclass
class CouplingDecision:
    status: str
    coefficients: dict[str,np.ndarray]
    diagnostics: dict=field(default_factory=dict)

def trusted_joint_fit(laws:list[LinearLaw],alpha=.05,max_law_degradation=.05,
                      minimum_gain=0.,max_parameter_se:dict[str,float]|None=None):
    """Return independent fits unless identifiability, conflict and validation pass.

    HC3 covariance and chi-square conflict diagnostics are approximate conditional
    on a fixed basis. They are not calibrated after adaptive expression selection.
    Every law contributes equal normalized training loss. Validation labels never
    fit coefficients; independent and joint predictions share the same split.
    """
    if not 0<alpha<1 or max_law_degradation<0 or minimum_gain<0:
        raise ValueError('Invalid gate configuration')
    if len({law.name for law in laws})!=len(laws):raise ValueError('Duplicate law names')
    independent={};covariances={};scales={};offsets={};blocks=[];targets=[];links={};cursor=0
    # Compute a complete independent fallback before any identifiability exit.
    for law in laws:
        x=np.asarray(law.fit_x,float);y=np.asarray(law.fit_y,float).reshape(-1)
        v=np.asarray(law.validation_x,float);vy=np.asarray(law.validation_y,float).reshape(-1)
        if x.ndim!=2 or v.ndim!=2 or len(x)!=len(y) or len(v)!=len(vy) or v.shape[1]!=x.shape[1] or not len(v):raise ValueError('Invalid split shape')
        if not all(np.isfinite(a).all() for a in [x,y,v,vy]):raise ValueError('Nonfinite input')
        independent[law.name]=np.linalg.lstsq(x,y,rcond=None)[0]
    for law in laws:
        x=np.asarray(law.fit_x,float);y=np.asarray(law.fit_y,float).reshape(-1)
        v=np.asarray(law.validation_x,float);vy=np.asarray(law.validation_y,float).reshape(-1)
        if x.ndim!=2 or v.ndim!=2 or len(x)!=len(y) or len(v)!=len(vy) or v.shape[1]!=x.shape[1] or not len(v):raise ValueError('Invalid split shape')
        if not all(np.isfinite(a).all() for a in [x,y,v,vy]):raise ValueError('Nonfinite input')
        c=independent[law.name]
        if len(y)<=x.shape[1] or np.linalg.matrix_rank(x)<x.shape[1]:
            return CouplingDecision('unidentifiable',independent,{'law':law.name})
        gram_inv=np.linalg.inv(x.T@x);leverage=np.einsum('ij,jk,ik->i',x,gram_inv,x)
        residual=(y-x@c)/np.maximum(1-leverage,1e-8)
        covariances[law.name]=gram_inv@(x.T@(residual[:,None]**2*x))@gram_inv
        scale=max(float(np.std(y)),1e-12);scales[law.name]=scale
        weight=1/(scale*np.sqrt(len(y)));blocks.append(x*weight);targets.append(y*weight)
        offsets[law.name]=cursor;cursor+=x.shape[1]
        for parameter,index in law.shared_columns.items():
            if not isinstance(index,int) or not 0<=index<x.shape[1]:raise ValueError('Invalid sharing column')
            links.setdefault(parameter,[]).append((law.name,index))
    links={p:members for p,members in links.items() if len(members)>=2}
    if not links:return CouplingDecision('sharing_unavailable',independent)
    constraints=[];diagnostics={'conflicts':{},'conditional_fixed_basis_only':True}
    conflict=False
    for parameter,members in links.items():
        values=np.array([independent[name][i] for name,i in members])
        variances=np.array([max(float(covariances[name][i,i]),1e-12) for name,i in members])
        if max_parameter_se and parameter in max_parameter_se:
            bound=max_parameter_se[parameter]
            if not np.isfinite(bound) or bound<=0:raise ValueError('Invalid uncertainty bound')
            if np.sqrt(variances).max()>bound:
                diagnostics['uncertainty_rejection']={'parameter':parameter,'max_se':float(np.sqrt(variances).max()),'bound':bound}
                return CouplingDecision('precision_rejected',independent,diagnostics)
        precision=1/variances;consensus=np.sum(values*precision)/precision.sum()
        q=float(np.sum((values-consensus)**2/variances));p=float(chi2.sf(q,len(members)-1))
        diagnostics['conflicts'][parameter]={'q':q,'p':p,'participants':len(members)}
        conflict|=p<alpha/len(links)
        first,index0=members[0]
        for name,index in members[1:]:
            row=np.zeros(cursor);row[offsets[first]+index0]=1;row[offsets[name]+index]=-1;constraints.append(row)
    if conflict:return CouplingDecision('conflict_rejected',independent,diagnostics)
    z=null_space(np.asarray(constraints));matrix=block_diag(*blocks);target=np.concatenate(targets)
    reduced=matrix@z
    if np.linalg.matrix_rank(reduced)<reduced.shape[1]:return CouplingDecision('unidentifiable_joint',independent,diagnostics)
    solution=z@np.linalg.lstsq(reduced,target,rcond=None)[0];joint={};gains=[];eligible=True;metrics={}
    for law in laws:
        start=offsets[law.name];joint[law.name]=solution[start:start+law.fit_x.shape[1]]
        base=float(np.mean((law.validation_y-law.validation_x@independent[law.name])**2)/scales[law.name]**2)
        shared=float(np.mean((law.validation_y-law.validation_x@joint[law.name])**2)/scales[law.name]**2)
        eligible&=shared<=base*(1+max_law_degradation)+1e-12
        gains.append(base-shared);metrics[law.name]={'independent':base,'joint':shared}
    diagnostics.update(validation_losses=metrics,mean_validation_gain=float(np.mean(gains)),independent_fits=len(laws),joint_fits=1,fit_count=len(laws)+1)
    accepted=eligible and float(np.mean(gains))>minimum_gain
    return CouplingDecision('shared_accepted' if accepted else 'validation_rejected',joint if accepted else independent,diagnostics)
