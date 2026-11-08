"""Finite deletion through a convex retraining path, not an unlearning method."""
import math
from .core import dot, fit, point, rows, solve
from .demo import demo


def _simpson(values):
    """Uniform-grid composite Simpson on [0,1]; caller supplies even intervals."""
    n = len(values) - 1
    return math.fsum((1 if i in (0, n) else 4 if i % 2 else 2) * value for i, value in enumerate(values)) / (3 * n)


def trace_deletion(x, y, query, label, indices, ridge=0.1, segments=16):
    """Refit at t=k/segments under weights w_i(t)=1-t for selected i.

    H(t) theta'(t) = sum_{i in group} grad loss_i(theta(t)) / original_N.
    Query-loss derivatives integrate to the endpoint loss change in exact
    calculus. Report finite Simpson estimates and their measured endpoint gaps;
    there is no rigorous quadrature or floating-point error certification.
    """
    x, y = rows(x, y)
    q, labels = rows([query], [label])
    if len(q[0]) != len(x[0]):
        raise ValueError('query dimensions must match')
    if not isinstance(indices, (list, tuple)) or not indices:
        raise ValueError('indices must be a nonempty unique list')
    if any(type(i) is not int or not 0 <= i < len(x) for i in indices) or len(set(indices)) != len(indices):
        raise ValueError('indices must be unique in-range integers')
    if type(segments) is not int or not 2 <= segments <= 64 or segments & (segments - 1):
        raise ValueError('segments must be a power of two from 2 through 64')
    # Canonical order avoids input-order changes to arithmetic and reports.
    indices = sorted(indices)
    n, d = len(x), len(x[0])
    nodes = []
    for k in range(segments + 1):
        t = k / segments
        weights = [1 - t if i in indices else 1.0 for i in range(n)]
        model = fit(x, y, ridge=ridge, weights=weights)
        theta, hessian = model['theta'], model['hessian']
        gradients = [point(theta, x[i], y[i])[1] for i in indices]
        rhs = [math.fsum(g[j] for g in gradients) / n for j in range(d)]
        direction, _ = solve(hessian, rhs)
        residual = [dot(row, direction) - v for row, v in zip(hessian, rhs)]
        loss, gradient, _, probability = point(theta, q[0], labels[0])
        nodes.append({'t': t, 'fit': model, 'group_sample_gradients': gradients,
                      'rhs': rhs, 'dtheta_dt': direction, 'solve_residual': residual,
                      'max_abs_solve_residual': max(map(abs, residual)),
                      'query_loss': loss, 'query_probability': probability,
                      'query_gradient': gradient, 'dquery_loss_dt': dot(gradient, direction)})
    delta = nodes[-1]['query_loss'] - nodes[0]['query_loss']
    first_order = nodes[0]['dquery_loss_dt']
    levels = []
    count = 2
    while count <= segments:
        values = [nodes[i]['dquery_loss_dt'] for i in range(0, segments + 1, segments // count)]
        value = _simpson(values)
        trapezoid = (math.fsum(values) - (values[0] + values[-1]) / 2) / count
        levels.append({'segments': count, 'simpson': value, 'trapezoid': trapezoid,
                       'simpson_minus_endpoint_delta': value - delta,
                       'trapezoid_minus_endpoint_delta': trapezoid - delta,
                       'simpson_minus_previous': None if not levels else value - levels[-1]['simpson']})
        count *= 2
    return {'version': 1, 'indices': indices, 'segments': segments, 'ridge': nodes[0]['fit']['ridge'],
            'objective': 'sum(w_i(t)*loss_i)/original_N + ridge*||theta||^2/2',
            'weight_path': 'selected weights = 1-t; other weights = 1; original_N fixed',
            'query': q[0], 'query_label': labels[0], 'nodes': nodes, 'levels': levels,
            'first_order_query_loss_delta': first_order, 'endpoint_query_loss_delta': delta,
            'first_order_minus_endpoint_delta': first_order - delta,
            'integrated_query_loss_delta': levels[-1]['simpson'],
            'integration_minus_endpoint_delta': levels[-1]['simpson_minus_endpoint_delta'],
            'max_abs_fit_gradient': max(v['fit']['max_abs_gradient'] for v in nodes),
            'max_abs_solve_residual': max(v['max_abs_solve_residual'] for v in nodes),
            'scope': 'converged convex refit path with numerical quadrature, not a certified error bound or unlearning'}


def path_demo():
    original = demo()
    source = original['fixture']
    x, y, query, label = source['X'], source['y'], source['query'], source['label']
    paths = {'row5': [5], 'mixed_group': [5, 6], 'all_rows': list(range(len(x)))}
    result = {name: trace_deletion(x, y, query, label, group) for name, group in paths.items()}
    for name, group in paths.items():
        individual_sum = math.fsum(original['undamped']['rows'][i]['actual_delete_query_loss_delta'] for i in group)
        result[name]['sum_of_individual_endpoint_deltas'] = individual_sum
        result[name]['group_minus_sum_of_individual_deltas'] = result[name]['endpoint_query_loss_delta'] - individual_sum
    return {'version': 1, 'fixture': source, 'paths': result}
