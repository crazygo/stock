"""Production declaration of the frozen v9 forward input schema.

This module describes an input; it does not build features or authorize a model.
"""
from __future__ import annotations

from ...focus_v8.core import TARGETS
from .. import data
from .features import (CAUSAL_GROUP_PEERS, GROUP_ROUTES, LEGACY_GROUP_PEERS,
                       ROUTES, SHAPES)


def build_schema(route: str, recipe: dict, peer_semantics: str = 'excluded') -> dict:
    if route not in ROUTES:
        raise ValueError('unknown frozen route')
    if not isinstance(recipe, dict) or not recipe or not all(v is True for v in recipe.values()):
        raise ValueError('invalid frozen recipe')
    grouped = route in GROUP_ROUTES
    if grouped and peer_semantics not in (LEGACY_GROUP_PEERS, CAUSAL_GROUP_PEERS):
        raise ValueError('group route needs explicit peer semantics')
    if not grouped and peer_semantics != 'excluded':
        raise ValueError('no-group route must exclude group peers')
    width = 765 if grouped else 585
    spec = {
        'route_id': route,
        'targets': list(TARGETS),
        'feature_columns': list(data.FEATURE_COLUMNS),
        'input_shapes': {key: list(shape) for key, shape in SHAPES.items()},
        'price_basis': 'NONE',
        'tensor_order': ['x5', 'x60', 'xday', 'group_seq', 'prior'],
        'tabular_width': width,
        'path_width': 84,
        'curve_width': 8,
        'relative_width': 6 if route == 'B_group' else 0,
        'prior_rule': 'own_complete_nine_beta_1_1_last_63',
        'group_mode': 'causal_weekly' if grouped else 'excluded',
        'normalizer': 'fixed_no_fitted_scaler',
        'support_rule': 'checkpoint_buffers' if recipe.get('support') else 'none',
        'tabular_columns': [f'f_{i:03d}' for i in range(width)],
        'path_columns': [f'path_{i:03d}' for i in range(84)],
        'curve_columns': ['open40_ret', 'mid40_ret', 'late40_ret',
                          'prefix_drawdown', 'prefix_runup', 'prefix_efficiency',
                          'late_early_vol_delta', 'late_early_logdollar_delta'],
        'relative_columns': ([f'own_minus_{name}_cutoff_return' for name in
                              ('trend15', 'trend63', 'trend126', 'volatility',
                               'liquidity', 'qqq')] if route == 'B_group' else []),
    }
    if grouped:
        spec['group_peer_semantics'] = peer_semantics
    return spec
