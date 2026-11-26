"""Opt-in fixed-basis coupling using joint, rather than individual, rank.

Declared links are hypotheses. Validation selection provides no topology test,
post-selection confidence interval, or out-of-sample noninferiority guarantee.
The legacy trusted_joint_fit API and symbolic-search pipeline are unchanged.
"""
from dataclasses import dataclass, field
from itertools import combinations
import numpy as np
from scipy.linalg import block_diag, null_space
from .trusted_coupling import LinearLaw


@dataclass(frozen=True)
class CoefficientLink:
    parameter: str
    left: str
    right: str


@dataclass
class JointEstimate:
    coefficients: dict[str, np.ndarray]
    diagnostics: dict = field(default_factory=dict)


@dataclass
class LinkSelection:
    estimate: JointEstimate
    links: tuple[CoefficientLink, ...]
    status: str
    diagnostics: dict = field(default_factory=dict)


def declared_star_links(laws):
    """Return a deterministic spanning-star declaration for each named parameter."""
    participants = {}
    for law in laws:
        for name in law.shared_columns:
            participants.setdefault(name, []).append(law.name)
    return tuple(CoefficientLink(name, names[0], other)
                 for name, names in sorted(participants.items()) for other in names[1:])


def fit_linear_links(laws: list[LinearLaw], links=(), penalty=None):
    """Fit using fit labels only, retaining rank diagnostics for every estimate.

penalty=None imposes exact equality; finite nonnegative penalty adds squared
coefficient differences to the normalized law loss. Parameter units must match.
Rank-deficient outputs are minimum-norm predictors, not identified parameters.
"""
    if not laws or len({law.name for law in laws}) != len(laws):
        raise ValueError('Provide nonempty laws with unique names')
    if penalty is not None and (not np.isfinite(penalty) or penalty < 0):
        raise ValueError('Penalty must be finite and nonnegative')
    offsets = {}; blocks = []; targets = []; ranks = {}; cursor = 0
    by_name = {law.name: law for law in laws}
    for law in laws:
        x = np.asarray(law.fit_x, float); y = np.asarray(law.fit_y, float)
        if x.ndim != 2 or y.ndim != 1 or len(y) != len(x) or not len(y):
            raise ValueError('Invalid fit shape')
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Nonfinite fit data')
        offsets[law.name] = cursor; cursor += x.shape[1]
        ranks[law.name] = int(np.linalg.matrix_rank(x))
        weight = 1 / (max(float(np.std(y)), 1e-12) * np.sqrt(len(y)))
        blocks.append(x * weight); targets.append(y * weight)
    matrix = block_diag(*blocks); target = np.concatenate(targets)
    rows = []
    for link in links:
        if link.left == link.right:
            raise ValueError('A link must join different laws')
        row = np.zeros(cursor)
        for sign, name in [(1, link.left), (-1, link.right)]:
            if name not in by_name or link.parameter not in by_name[name].shared_columns:
                raise ValueError('Link references an undeclared parameter')
            column = by_name[name].shared_columns[link.parameter]
            if not isinstance(column, (int, np.integer)) or not 0 <= column < by_name[name].fit_x.shape[1]:
                raise ValueError('Invalid linked column')
            row[offsets[name] + column] = sign
        rows.append(row)
    constraints = np.asarray(rows).reshape(-1, cursor)
    z = null_space(constraints) if len(rows) and penalty is None else np.eye(cursor)
    reduced = matrix @ z
    # Identifiability comes from observations plus exact declared equalities.
    # A finite penalty must never be counted as new observational information.
    reduced_solution, _, information_rank, singular = np.linalg.lstsq(reduced, target, rcond=None)
    solve_x = reduced; solve_y = target
    if len(rows) and penalty is not None:
        solve_x = np.vstack([matrix, np.sqrt(penalty) * constraints])
        solve_y = np.r_[target, np.zeros(len(rows))]
    solution = z @ (np.linalg.lstsq(solve_x, solve_y, rcond=None)[0]
                    if len(rows) and penalty is not None else reduced_solution)
    coefficients = {law.name: solution[offsets[law.name]:offsets[law.name] + law.fit_x.shape[1]].copy()
                    for law in laws}
    return JointEstimate(coefficients, dict(independent_ranks=ranks,
        information_rank=int(information_rank), free_columns=z.shape[1],
        identifiable=bool(information_rank == z.shape[1]),
        rank_conditional_on_declared_links=penalty is None and bool(rows),
        smallest_singular_value=float(singular[-1]) if len(singular) else 0.,
        fit_count=1, least_squares_calls=2 if len(rows) and penalty is not None else 1,
        design_rank_checks=len(laws), constraint_svd_count=int(bool(rows) and penalty is None), penalty=penalty))


def validation_losses(laws, estimate):
    losses = []
    for law in laws:
        x = np.asarray(law.validation_x, float); y = np.asarray(law.validation_y, float)
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(y) or x.shape[1] != law.fit_x.shape[1]:
            raise ValueError('Invalid validation shape')
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Nonfinite validation data')
        prediction = x @ estimate.coefficients[law.name]
        losses.append(float(np.mean((y - prediction)**2) / max(float(np.std(law.fit_y))**2, 1e-24)))
    return np.asarray(losses)


def validation_eligible(base, candidate, max_law_degradation=.05):
    return bool(np.isfinite(candidate).all() and
                np.all(candidate <= base * (1 + max_law_degradation) + 1e-12) and
                np.mean(candidate) < np.mean(base) - 1e-12)


def select_declared_links(laws, links=None, mode='subsets', max_links=8, max_law_degradation=.05):
    """Small-graph development selector, with every refit charged explicitly.

Full/subsets admit identifiable constrained models plus the independent fallback.
Greedy explores predictive rank-deficient intermediates to avoid artificially
blocking complementary links; its output rank is reported without certification.
"""
    if mode not in ['full', 'subsets', 'greedy'] or max_law_degradation < 0:
        raise ValueError('Invalid selection configuration')
    links = tuple(declared_star_links(laws) if links is None else links)
    if len(set(links)) != len(links) or len(links) > max_links:
        raise ValueError('Duplicate links or graph exceeds declared enumeration budget')
    base = fit_linear_links(laws); base_loss = validation_losses(laws, base)
    best = base; best_links = (); best_score = float(np.mean(base_loss))
    fit_count = 1; validation_count = 1; unidentifiable = 0
    least_squares_calls = base.diagnostics['least_squares_calls']
    cache = {}
    def evaluate(indices):
        nonlocal fit_count, validation_count, unidentifiable, least_squares_calls
        indices = tuple(sorted(indices))
        if indices not in cache:
            chosen = tuple(links[i] for i in indices)
            est = fit_linear_links(laws, chosen); loss = validation_losses(laws, est)
            fit_count += 1; validation_count += 1
            least_squares_calls += est.diagnostics['least_squares_calls']
            unidentifiable += int(not est.diagnostics['identifiable'])
            cache[indices] = (est, loss, chosen)
        return cache[indices]
    if mode == 'greedy':
        current = ()
        while len(current) < len(links):
            options = []
            for i in range(len(links)):
                if i in current: continue
                est, loss, chosen = evaluate(current + (i,))
                if validation_eligible(base_loss, loss, max_law_degradation):
                    options.append((float(np.mean(loss)), i, est, chosen))
            if not options: break
            score, i, est, chosen = min(options, key=lambda v: (v[0], v[1]))
            if score >= best_score - 1e-12: break
            current = tuple(sorted(current + (i,)))
            best, best_links, best_score = est, chosen, score
    else:
        subsets = [tuple(range(len(links)))] if mode == 'full' else (
            indices for size in range(1, len(links)+1) for indices in combinations(range(len(links)), size))
        for indices in subsets:
            if not indices: continue
            est, loss, chosen = evaluate(indices)
            if not est.diagnostics['identifiable']: continue
            score = float(np.mean(loss))
            if validation_eligible(base_loss, loss, max_law_degradation) and score < best_score - 1e-12:
                best, best_links, best_score = est, chosen, score
    status = 'independent_fallback' if not best_links else (
        'links_selected' if best.diagnostics['identifiable'] else 'predictive_unidentifiable')
    return LinkSelection(best, best_links, status, dict(fit_count=fit_count, least_squares_calls=least_squares_calls,
        validation_passes=validation_count, per_law_validation_evaluations=validation_count*len(laws),
        unidentifiable_candidates=unidentifiable, mean_validation_loss=best_score,
        independent_parameter_identifiable=base.diagnostics['identifiable'],
        no_test_safety_or_topology_guarantee=True))
