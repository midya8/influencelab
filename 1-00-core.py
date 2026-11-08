"""Small, auditable ridge-logistic fits and sample-weight derivatives."""
import math


def number(value, name, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + ' must be a finite number')
    try:
        v = float(value)
    except (OverflowError, ValueError):
        raise ValueError(name + ' must be finite') from None
    if not math.isfinite(v) or (low is not None and v < low) or (high is not None and v > high):
        raise ValueError(name + ' outside supported range')
    return v


def rows(x, y):
    if not isinstance(x, (list, tuple)) or not 1 <= len(x) <= 128:
        raise ValueError('X needs 1..128 rows')
    if not isinstance(y, (list, tuple)) or len(y) != len(x):
        raise ValueError('labels must align')
    if not isinstance(x[0], (list, tuple)) or not 1 <= len(x[0]) <= 12:
        raise ValueError('X needs 1..12 columns; add a constant column for a regularized intercept')
    d = len(x[0])
    out = []
    for row in x:
        if not isinstance(row, (list, tuple)) or len(row) != d:
            raise ValueError('ragged X')
        out.append([number(v, 'feature', -100, 100) for v in row])
    if any(type(v) is not int or v not in (0, 1) for v in y):
        raise ValueError('labels must be integer 0 or 1')
    return out, list(y)


def dot(a, b):
    return math.fsum(v * w for v, w in zip(a, b))


def point(theta, x, y):
    """Stable logistic loss, gradient, and scalar curvature."""
    z = dot(theta, x)
    # Evaluate the error directly to avoid sigmoid(z)-1 cancellation.
    e = math.exp(-abs(z))
    p = 1 / (1 + e) if z >= 0 else e / (1 + e)
    error = (e / (1 + e) if z < 0 else 1 / (1 + e)) if y == 0 else (-1 / (1 + e) if z < 0 else -e / (1 + e))
    loss = max((1 - 2 * y) * z, 0) + math.log1p(e)
    return loss, [error * v for v in x], e / (1 + e) ** 2, p


def _objective(theta, x, y, weights, ridge, normalizer):
    d = len(theta)
    pieces = [point(theta, row, label) for row, label in zip(x, y)]
    loss = math.fsum(w * q[0] / normalizer for w, q in zip(weights, pieces)) + ridge * dot(theta, theta) / 2
    gradient = [math.fsum(w * q[1][j] / normalizer for w, q in zip(weights, pieces)) + ridge * theta[j] for j in range(d)]
    hessian = [[math.fsum(w * q[2] * row[j] * row[k] / normalizer for w, q, row in zip(weights, pieces, x)) + (ridge if j == k else 0) for k in range(d)] for j in range(d)]
    return loss, gradient, hessian


def solve(matrix, rhs):
    """Unjittered Cholesky plus triangular solves; fail on nonpositive pivots."""
    d = len(rhs)
    lower = [[0.0] * d for _ in range(d)]
    for j in range(d):
        for k in range(j + 1):
            value = matrix[j][k] - math.fsum(lower[j][q] * lower[k][q] for q in range(k))
            if j == k:
                if value <= 0 or not math.isfinite(value):
                    raise ValueError('matrix is not numerically positive definite; no hidden jitter')
                lower[j][k] = math.sqrt(value)
            else:
                lower[j][k] = value / lower[k][k]
    intermediate = []
    for j in range(d):
        intermediate.append((rhs[j] - dot(lower[j][:j], intermediate)) / lower[j][j])
    result = [0.0] * d
    for j in range(d - 1, -1, -1):
        result[j] = (intermediate[j] - math.fsum(lower[k][j] * result[k] for k in range(j + 1, d))) / lower[j][j]
    return result, lower


def fit(x, y, ridge=0.1, weights=None, normalizer=None, tolerance=1e-10, max_steps=100):
    """Minimize sum_i w_i loss_i / normalizer + ridge ||theta||²/2.

    Every coordinate, including an explicit constant column, is regularized.
    Raises rather than handing an unconverged fit to the influence calculation.
    """
    x, y = rows(x, y)
    ridge = number(ridge, 'ridge', 1e-6, 100)
    tolerance = number(tolerance, 'tolerance', 1e-12, 1e-6)
    if type(max_steps) is not int or not 1 <= max_steps <= 200:
        raise ValueError('max_steps needs integer 1..200')
    normalizer = len(x) if normalizer is None else number(normalizer, 'normalizer', 1, 128)
    weights = [1.0] * len(x) if weights is None else weights
    if not isinstance(weights, (list, tuple)) or len(weights) != len(x):
        raise ValueError('weights must align')
    weights = [number(w, 'weight', 0, 10) for w in weights]
    theta = [0.0] * len(x[0])
    history = []
    for iteration in range(max_steps + 1):
        loss, gradient, hessian = _objective(theta, x, y, weights, ridge, normalizer)
        norm = max(abs(v) for v in gradient)
        history.append({'iteration': iteration, 'objective': loss, 'max_abs_gradient': norm})
        if norm <= tolerance:
            _, lower = solve(hessian, [0.0] * len(theta))
            return {'theta': theta, 'objective': loss, 'gradient': gradient, 'hessian': hessian,
                    'cholesky': lower, 'max_abs_gradient': norm, 'tolerance': tolerance,
                    'parameter_error_l2_bound': math.sqrt(dot(gradient, gradient)) / ridge,
                    'ridge': ridge, 'normalizer': normalizer, 'weights': weights,
                    'converged': True, 'history': history}
        if iteration == max_steps:
            break
        step, _ = solve(hessian, gradient)
        decrement = dot(gradient, step)
        scale = 1.0
        for _ in range(60):
            candidate = [v - scale * s for v, s in zip(theta, step)]
            new_loss, new_gradient, _ = _objective(candidate, x, y, weights, ridge, normalizer)
            # Near machine precision, allow objective roundoff ONLY with a strict gradient decrease.
            armijo = new_loss <= loss - 1e-4 * scale * decrement
            roundoff = new_loss <= loss + 8 * math.ulp(max(1.0, abs(loss))) and max(map(abs, new_gradient)) < norm
            if armijo or roundoff:
                history[-1]['step_scale'] = scale
                history[-1]['roundoff_acceptance'] = not armijo
                theta = candidate
                break
            scale /= 2
        else:
            raise ValueError('line search failed; no fit returned')
    raise ValueError('Newton iteration limit reached without convergence')


def audit(x, y, query, label, ridge=0.1, damping=0.0, refit=True):
    """Differentiate sample weights, then compare finite deletion to exact refits."""
    x, y = rows(x, y)
    q, qy = rows([query], [label])
    if len(q[0]) != len(x[0]):
        raise ValueError('query dimensions must match')
    if type(refit) is not bool:
        raise ValueError('refit must be bool')
    damping = number(damping, 'damping', 0, 100)
    base = fit(x, y, ridge=ridge)
    hessian = base['hessian']
    d, n = len(query), len(x)
    effective = [[hessian[j][k] + (damping if j == k else 0) for k in range(d)] for j in range(d)]
    theta = base['theta']
    query_loss, query_gradient, _, probability = point(theta, q[0], label)
    inverse_query, lower = solve(effective, query_gradient)
    diagnostics = []
    for i, (row, yi) in enumerate(zip(x, y)):
        training_loss, gradient, _, _ = point(theta, row, yi)
        rhs = [-v / n for v in gradient]
        direction, _ = solve(effective, rhs)
        residual = [dot(effective[j], direction) - rhs[j] for j in range(d)]
        derivative = dot(query_gradient, direction)
        record = {'row': i, 'training_loss': training_loss, 'sample_gradient': gradient,
                  'dtheta_dweight': direction, 'dquery_loss_dweight': derivative,
                  'adjoint_derivative': -dot(inverse_query, gradient) / n,
                  'solve_residual': residual, 'max_abs_solve_residual': max(map(abs, residual)),
                  'predicted_delete_theta_delta': [-v for v in direction],
                  'predicted_delete_query_loss_delta': -derivative}
        if refit:
            weights = [1.0] * n
            weights[i] = 0.0
            deleted = fit(x, y, ridge=base['ridge'], weights=weights, normalizer=n)
            actual = point(deleted['theta'], q[0], label)[0] - query_loss
            record['fixed_normalizer_refit'] = deleted
            record['actual_delete_query_loss_delta'] = actual
            record['actual_delete_theta_delta'] = [v - b for v, b in zip(deleted['theta'], theta)]
            record['delete_query_loss_error'] = -derivative - actual
            # Distinct conventional retained-row mean with same ridge coefficient.
            if n > 1:
                conventional = fit(x, y, ridge=base['ridge'], weights=weights, normalizer=n - 1)
                record['retained_mean_refit'] = conventional
                record['retained_mean_query_loss_delta'] = point(conventional['theta'], q[0], label)[0] - query_loss
        diagnostics.append(record)
    return {'version': 1, 'model': 'fully regularized binary logistic; explicit feature columns',
            'objective': 'sum(w_i * logistic_loss_i) / original_N + ridge * ||theta||^2 / 2',
            'derivative_parameter': 'one unit of sample weight; base weights all 1',
            'damping': damping, 'is_exact_undamped_derivative': damping == 0,
            'damping_scope': 'H+damping*I is an inverse approximation, not a refit with larger ridge',
            'fit': base, 'effective_hessian': effective, 'effective_cholesky': lower,
            'query': q[0], 'query_label': qy[0], 'query_probability': probability,
            'query_loss': query_loss, 'query_gradient': query_gradient, 'rows': diagnostics,
            'scope': 'local supplied-model sensitivities, not causal attribution or guaranteed finite deletion'}
