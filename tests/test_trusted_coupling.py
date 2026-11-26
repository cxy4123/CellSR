import numpy as np
from cellsr.trusted_coupling import LinearLaw,trusted_joint_fit

def laws(theta=(2.,2.),seed=7):
    rng=np.random.default_rng(seed);out=[]
    for i,t in enumerate(theta):
        x=rng.uniform(-2,2,80);v=rng.uniform(-2,2,240)
        out.append(LinearLaw(str(i),np.c_[np.ones(len(x)),x],t*x+rng.normal(0,.4,len(x)),np.c_[np.ones(len(v)),v],t*v,{'theta':1}))
    return out

def test_joint_constraint_and_no_validation_coefficient_fitting():
    data=laws();result=trusted_joint_fit(data,max_law_degradation=10.)
    assert result.status=='shared_accepted'
    assert abs(result.coefficients['0'][1]-result.coefficients['1'][1])<1e-12
    changed=laws()
    for law in changed:law.validation_y=law.validation_x@result.coefficients[law.name]
    other=trusted_joint_fit(changed,max_law_degradation=1e8)
    assert other.status=='shared_accepted'
    for name in result.coefficients:np.testing.assert_allclose(result.coefficients[name],other.coefficients[name])

def test_conflicting_parameters_are_rejected():
    assert trusted_joint_fit(laws((1.,4.))).status=='conflict_rejected'

def test_one_law_does_not_establish_sharing():
    assert trusted_joint_fit(laws()[:1]).status=='sharing_unavailable'

def test_rank_failure_is_reported():
    data=laws();data[0].fit_x[:,1]=1
    assert trusted_joint_fit(data).status=='unidentifiable'
    assert len(trusted_joint_fit(data).coefficients)==2

def test_validation_can_reject_compatible_fit():
    data=laws();result=trusted_joint_fit(data,minimum_gain=100.)
    assert result.status=='validation_rejected'

def test_declared_uncertainty_budget_rejects_low_information():
    data=laws()
    for law in data:law.fit_x[:,1]*=1e-5;law.validation_x[:,1]*=1e-5
    assert trusted_joint_fit(data,max_parameter_se={'theta':.25}).status=='precision_rejected'
