import numpy as np
import pytest
from cellsr.trusted_coupling import LinearLaw
from cellsr.joint_information import (CoefficientLink, declared_star_links, fit_linear_links,
                                     select_declared_links)


def complementary():
    x = np.linspace(-2, 2, 30); v = np.linspace(-1.9, 1.9, 40)
    return [LinearLaw(str(i), np.c_[x, sign*x], (1.4 + sign*.6)*x,
            np.c_[v, sign*v], (1.4 + sign*.6)*v, {'theta': 0, 'beta': 1})
            for i, sign in enumerate([1, -1])]


def test_joint_information_without_independent_identifiability():
    laws = complementary(); result = fit_linear_links(laws, declared_star_links(laws))
    assert result.diagnostics['independent_ranks'] == {'0': 1, '1': 1}
    assert result.diagnostics['identifiable']
    for coeff in result.coefficients.values(): np.testing.assert_allclose(coeff, [1.4, .6], atol=1e-12)


def test_wrong_or_incomplete_links_do_not_create_joint_information():
    laws = complementary()
    result = fit_linear_links(laws, [CoefficientLink('theta', '0', '1')])
    assert not result.diagnostics['identifiable']


def test_penalty_does_not_turn_prior_into_observational_rank():
    result = fit_linear_links(complementary(), declared_star_links(complementary()), penalty=100.)
    assert not result.diagnostics['identifiable']


def test_fit_coefficients_do_not_use_validation_labels():
    laws = complementary(); links = declared_star_links(laws)
    before = fit_linear_links(laws, links)
    for law in laws: law.validation_y[:] = 1e6
    after = fit_linear_links(laws, links)
    for name in before.coefficients: np.testing.assert_array_equal(before.coefficients[name], after.coefficients[name])


def test_subset_selector_retains_unidentifiable_independent_fallback():
    # Validation on the same degenerate designs cannot identify the null directions.
    result = select_declared_links(complementary())
    assert result.status == 'independent_fallback'
    assert not result.estimate.diagnostics['identifiable']
    assert result.diagnostics['fit_count'] == 4


def test_validation_with_new_sensitivity_selects_complementary_links():
    laws = complementary(); rng = np.random.default_rng(4)
    for law in laws:
        law.validation_x = rng.normal(size=(100, 2));law.validation_y = law.validation_x @ np.array([1.4, .6])
    result = select_declared_links(laws)
    assert len(result.links) == 2 and result.estimate.diagnostics['identifiable']


def test_invalid_graph_and_inputs_fail_before_silent_fits():
    laws = complementary()
    with pytest.raises(ValueError): fit_linear_links(laws, [CoefficientLink('unknown', '0', '1')])
    with pytest.raises(ValueError): select_declared_links(laws, max_links=1)
    laws[0].fit_x[0, 0] = np.nan
    with pytest.raises(ValueError): fit_linear_links(laws)


def test_local_bad_link_does_not_discard_useful_shared_pair():
    x=np.linspace(-2,2,40);v=np.linspace(-1.9,1.9,80);laws=[]
    for i,(truth,training_shift) in enumerate([(2.,.2),(2.,-.2),(5.,0.)]):
        laws.append(LinearLaw(str(i),np.c_[np.ones(len(x)),x],(truth+training_shift)*x,
            np.c_[np.ones(len(v)),v],truth*v,{'theta':1}))
    selected=select_declared_links(laws)
    assert selected.links==(CoefficientLink('theta','0','1'),)
    assert abs(selected.estimate.coefficients['0'][1]-2.)<.05
    np.testing.assert_allclose(selected.estimate.coefficients['2'],[0.,5.],atol=1e-12)
