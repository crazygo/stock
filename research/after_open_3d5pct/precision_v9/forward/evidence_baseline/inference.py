"""Static, hash-verified replay of frozen focus_v8 B/C checkpoints."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

# On the pinned macOS runtime, deserializing a LightGBM Booster after torch
# initialized its native runtime segfaults. Fail closed if a host imports this
# module too late; the forward package itself imports LightGBM before torch.
_TORCH_PRELOADED = 'torch' in sys.modules

import lightgbm as lgb
import numpy as np
import scipy.special
import torch

from ...focus_v8 import run as old
from ...focus_v8.core import GROUPS, TARGETS
from ...focus_v8.support import build_c
from ...train_multiscale_v6 import project_monotone
from .. import data as d
from .features import GROUP_ROUTES, ROUTES, SHAPES


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def _files(records, role):
    if not isinstance(records,list) or not records:
        raise ValueError(f'{role}: immutable file manifest missing')
    out=[]
    for record in records:
        if not {'path','sha256'} <= set(record):
            raise ValueError(f'{role}: file path/hash missing')
        path=Path(record['path']).expanduser().resolve(strict=True)
        if _sha(path)!=record['sha256']:
            raise ValueError(f'{role}: source hash mismatch: {path}')
        out.append({'role':role,'path':str(path),'sha256':record['sha256']})
    return out


def load_route(frozen_artifact):
    """Verify every supplied byte and schema; no implicit v8 directory or recipe."""
    manifest=(json.loads(Path(frozen_artifact).read_text()) if isinstance(frozen_artifact,(str,Path))
              else frozen_artifact)
    required={'route_id','model_version','recipe','schema','schema_file','schema_sha256','model_files',
              'model_sha256','calibration_file','calibration_sha256','source_files',
              'source_code_sha256','score_column','thresholds','action_frozen'}
    missing=required-set(manifest)
    if missing:
        raise ValueError(f'route artifact missing: {sorted(missing)}')
    route=manifest['route_id']
    if route not in ROUTES or manifest['schema'].get('route_id')!=route:
        raise ValueError('unknown route or mismatched schema')
    if hashlib.sha256(_canonical(manifest['schema'])).hexdigest()!=manifest['schema_sha256']:
        raise ValueError('schema hash mismatch')
    if tuple(manifest['schema'].get('targets',()))!=tuple(TARGETS):
        raise ValueError('target order differs from frozen nine tasks')
    spec=manifest['schema']
    grouped=route in GROUP_ROUTES
    fixed={'feature_columns':list(d.FEATURE_COLUMNS),
           'input_shapes':{k:list(v) for k,v in SHAPES.items()},
           'price_basis':'NONE','tensor_order':['x5','x60','xday','group_seq','prior'],
           'tabular_width':765 if grouped else 585,'path_width':84,'curve_width':8,
           'relative_width':6 if route=='B_group' else 0,
           'prior_rule':'own_complete_nine_beta_1_1_last_63',
           'group_mode':'causal_weekly' if grouped else 'excluded',
           'normalizer':'fixed_no_fitted_scaler',
           'tabular_columns':[f'f_{i:03d}' for i in range(765 if grouped else 585)],
           'path_columns':[f'path_{i:03d}' for i in range(84)],
           'curve_columns':['open40_ret','mid40_ret','late40_ret','prefix_drawdown',
                            'prefix_runup','prefix_efficiency','late_early_vol_delta',
                            'late_early_logdollar_delta'],
           'relative_columns':([f'own_minus_{name}_cutoff_return' for name in
                               ('trend15','trend63','trend126','volatility','liquidity','qqq')]
                               if route=='B_group' else [])}
    if any(spec.get(key)!=value for key,value in fixed.items()):
        raise ValueError('frozen input shape, transform, or ordered columns changed')
    if (manifest['score_column']!='raw_3d_5pct' or
        set(manifest['thresholds'])!=set(GROUPS) or
        any(v is not None and not 0<=v<=1 for v in manifest['thresholds'].values())):
        raise ValueError('action score/threshold contract missing or changed')
    recipe=manifest['recipe']
    if not isinstance(recipe,dict) or not all(v is True for v in recipe.values()):
        raise ValueError('frozen recipe is invalid')
    if {'within_stock_rank','no_group_categories','group_dropout'} & recipe.keys():
        raise ValueError('failed R01 mechanism in artifact')
    if spec.get('support_rule')!=('checkpoint_buffers' if recipe.get('support') else 'none'):
        raise ValueError('neural support buffer contract changed')
    models=_files(manifest['model_files'],'model')
    calibration=_files([manifest['calibration_file']],'calibration')
    sources=_files(manifest['source_files'],'source_code')
    schema_file=_files([manifest['schema_file']],'schema')
    if (schema_file[0]['sha256']!=manifest['schema_sha256'] or
        json.loads(Path(schema_file[0]['path']).read_text())!=spec):
        raise ValueError('schema file differs from frozen schema')
    if (hashlib.sha256(_canonical([x['sha256'] for x in models])).hexdigest()!=manifest['model_sha256'] or
        calibration[0]['sha256']!=manifest['calibration_sha256'] or
        hashlib.sha256(_canonical([x['sha256'] for x in sources])).hexdigest()!=manifest['source_code_sha256']):
        raise ValueError('artifact aggregate hash mismatch')
    params=json.loads(Path(calibration[0]['path']).read_text())
    if (params.get('method')!='shrunken_nonnegative_platt' or
        np.asarray(params.get('parameters')).shape!=(9,2)):
        raise ValueError('frozen calibration parameters missing or incompatible')
    if route.startswith('B'):
        if _TORCH_PRELOADED:
            raise RuntimeError('B checkpoint loader requires forward import before torch')
        if len(models)!=9:
            raise ValueError('B route requires nine ordered target trees')
        loaded=[lgb.Booster(model_file=item['path']) for item in models]
        width=manifest['schema']['tabular_width']
        if recipe.get('representation'):width+=manifest['schema']['path_width']
        if recipe.get('curves'):
            width+=manifest['schema']['curve_width']
            if route=='B_group':width+=manifest['schema']['relative_width']
        if any(model.num_feature()!=width for model in loaded):
            raise ValueError('tree feature width differs from frozen schema')
    else:
        if len(models)!=1:
            raise ValueError('C route requires exactly one neural checkpoint')
        saved=torch.load(models[0]['path'],map_location='cpu',weights_only=True)
        if saved.get('route')!=route or saved.get('recipe')!=recipe:
            raise ValueError('neural route/recipe differs from checkpoint')
        loaded=build_c(route,recipe)
        loaded.load_state_dict(saved['state_dict'],strict=True)
        loaded.eval()
    return {'manifest':manifest,'model':loaded,'params':params['parameters'],
            'artifacts':models+calibration+sources+schema_file,
            'artifact_files':{'model':models,'source':sources,'schema':schema_file[0],
                              'calibration':calibration[0]}}


def _validate_x(route, X, schema):
    required={'x5','x60','prior','xbase','paths','curve','relative'}
    if route!='C_no_daily':required.add('xday')
    if route in GROUP_ROUTES:required.add('group_seq')
    if set(X)!=required:
        raise ValueError(f'{route}: forbidden or missing route input: {sorted(set(X)^required)}')
    for name in ('x5','x60','xday','group_seq','prior'):
        if name in X and np.asarray(X[name]).shape!=SHAPES[name]:
            raise ValueError(f'{route}: {name} shape differs from frozen schema')
    if np.asarray(X['xbase']).ndim!=1 or np.asarray(X['paths']).ndim!=1:
        raise ValueError('B feature vector dimensions changed')
    if (len(X['xbase'])!=schema['tabular_width'] or len(X['paths'])!=schema['path_width'] or
        len(X['curve'])!=schema['curve_width'] or len(X['relative'])!=schema['relative_width']):
        raise ValueError('feature vector widths differ from frozen schema')


def predict_route(artifact, X):
    """Return nine raw/processed predictions plus forward quality/lineage contract."""
    loaded=artifact if isinstance(artifact,dict) and 'manifest' in artifact else load_route(artifact)
    manifest=loaded['manifest']
    route=manifest['route_id']
    bundle=X if isinstance(X,dict) and 'X' in X else {'X':X,'lineage':{},'rejections':[]}
    if bundle['rejections'] or bundle['X'] is None:
        raise ValueError(f'feature build rejected: {bundle["rejections"]}')
    features=bundle['X']
    _validate_x(route,features,manifest['schema'])
    for item in loaded['artifacts']:
        if _sha(item['path'])!=item['sha256']:
            raise ValueError('artifact changed since route load')
    recipe=manifest['recipe']
    if route.startswith('B'):
        x=np.asarray(features['xbase'],np.float32)
        if recipe.get('representation'):x=np.r_[x,features['paths']]
        if recipe.get('curves'):
            x=np.r_[x,features['curve']]
            if route=='B_group':x=np.r_[x,features['relative']]
        x=x.reshape(1,-1)
        raw=[]
        for j,model in enumerate(loaded['model']):
            z=model.predict(x,raw_score=True,num_threads=1)[0]
            if recipe.get('prior'):
                z+=scipy.special.logit(np.clip(features['prior'][j],.01,.99))
            raw.append(float(scipy.special.expit(z)))
        raw=np.asarray(raw).reshape(1,9)
    else:
        torch.set_num_threads(1)
        model=loaded['model']
        arrays=[]
        for name in ('x5','x60','xday','group_seq','prior'):
            if name not in features:arrays.append(torch.empty(0))
            else:arrays.append(torch.from_numpy(np.asarray(features[name],np.float32)[None]))
        with torch.no_grad():
            raw=torch.sigmoid(model(*arrays)).numpy().reshape(1,9)
    processed=project_monotone(old.apply_cal(raw,np.asarray(loaded['params']))).reshape(9)
    if any(_sha(item['path'])!=item['sha256'] for item in loaded['artifacts']):
        raise ValueError('artifact changed during prediction')
    lineage=dict(bundle.get('lineage',{}))
    receipt=bool(lineage.get('g2_eligible') and lineage.get('receipt_verified'))
    snapshot_file=lineage.get('snapshot_file')
    snapshot_record=None
    if snapshot_file is not None:
        snapshot_path=Path(snapshot_file).expanduser().resolve(strict=True)
        snapshot_digest=_sha(snapshot_path)
        if snapshot_digest!=lineage.get('snapshot_sha256'):
            raise ValueError('snapshot file differs from feature lineage')
        snapshot_record={'role':'snapshot','path':str(snapshot_path),'sha256':snapshot_digest}
    artifact_files={**loaded['artifact_files']}
    if snapshot_record is not None:artifact_files['snapshot']=snapshot_record
    quality={'g1_ready':True,'g2_eligible':receipt,
             'publication_eligible':bool(receipt and manifest['action_frozen'] and snapshot_record),
             'rejections':list(bundle.get('rejections',[])) + ([] if receipt else ['receipt_unverified']),
             'artifacts':{'model_sha256':manifest['model_sha256'],
                          'schema_sha256':manifest['schema_sha256'],
                          'calibration_sha256':manifest['calibration_sha256'],
                          'source_code_sha256':manifest['source_code_sha256'],
                          'files':artifact_files},'prediction_sha256':None}
    prediction={'route_id':route,'model_version':manifest['model_version'],
                'sample_id':lineage.get('sample_id'),'symbol':lineage.get('symbol'),
                'session_date':lineage.get('session_date'),'cutoff_at':lineage.get('cutoff_at'),
                'decision_at':lineage.get('decision_at'),
                'information_deadline_at':lineage.get('information_deadline_at'),
                'raw':{target:float(raw[0,j]) for j,target in enumerate(TARGETS)},
                'processed':{target:float(processed[j]) for j,target in enumerate(TARGETS)},
                'score_column':manifest['score_column'],'lineage':lineage,'quality':quality}
    quality.pop('prediction_sha256')
    quality['prediction_sha256']=hashlib.sha256(_canonical(prediction)).hexdigest()
    return prediction
