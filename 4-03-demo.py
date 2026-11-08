"""Deterministic developed fixture, not an untouched benchmark."""
from .core import audit


def demo():
    x = [[1, -2], [1, -1], [1, -0.5], [1, 0.5], [1, 1], [1, 2], [1, 3]]
    y = [0, 0, 1, 0, 1, 1, 0]
    return {'version': 1, 'fixture': {'X': x, 'y': y, 'query': [1, 2], 'label': 1},
            'undamped': audit(x, y, [1, 2], 1, ridge=0.1),
            'damped': audit(x, y, [1, 2], 1, ridge=0.1, damping=0.5, refit=False)}
