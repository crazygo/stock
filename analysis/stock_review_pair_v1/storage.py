"""Immutable shared price versions; runs retain verified references only."""
from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
from pathlib import Path

from common import digest, fingerprint, read


def relative_path(value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('invalid shared data path: ' + str(value))
    return path


def data_root(run):
    run = Path(run).resolve()
    location = run / 'data_location.json'
    if location.exists():
        root = run / relative_path(read(location)['root'])
        if not root.resolve().is_relative_to(run):
            raise ValueError('bundle data root escapes run')
        return root
    return run.parent.parent / 'data'


def resolve_ref(run, ref):
    root = data_root(run).resolve()
    path = root / relative_path(ref['path'])
    if not path.resolve().is_relative_to(root):
        raise ValueError('shared object escapes data root')
    if not path.is_file() or digest(path) != ref['sha256']:
        raise ValueError('shared object missing or checksum mismatch: ' + ref['path'])
    return path


def price_bars(run, entry):
    if 'object_ref' in entry:
        obj = read(resolve_ref(run, entry['object_ref']))
        if obj['security_id'] != entry['security_id']:
            raise ValueError('price object identity mismatch')
        bars = obj['bars']
        if fingerprint([entry['security_id'], bars]) != entry['dataset_id']:
            raise ValueError('normalized price version mismatch: ' + entry['security_id'])
    else:
        bars = entry.get('bars', [])
    if entry.get('raw_ref'):
        resolve_ref(run, entry['raw_ref'])
    elif entry.get('path'):
        path = Path(run) / relative_path(entry['path'])
        if digest(path) != entry['sha256']:
            raise ValueError('dataset checksum mismatch: ' + entry['security_id'])
    return bars


def verify_market(run, manifest=None):
    manifest = manifest or read(Path(run) / 'inputs/data_manifest.json')
    for entry in manifest['datasets']:
        price_bars(run, entry)
    for name, ref in manifest.get('input_refs', {}).items():
        obj = resolve_ref(run, ref)
        if digest(Path(run) / relative_path(name)) != digest(obj):
            raise ValueError('fact snapshot reference mismatch: ' + name)


def hydrate(run, manifest):
    result = copy.deepcopy(manifest)
    for entry in result['datasets']:
        entry['bars'] = price_bars(run, entry)
        if entry.get('raw_ref'):
            entry['path'] = 'shared:' + entry['raw_ref']['path']
    return result


class SharedData:
    def __init__(self, store):
        self.root = Path(store).resolve() / 'data'
        self.metrics = {'objects_created': 0, 'objects_reused': 0, 'bytes_created': 0,
                        'raw_created': 0, 'daily_created': 0, 'facts_created': 0}

    def put(self, source, name, sha, kind):
        destination = self.root / relative_path(name)
        if destination.exists():
            if digest(destination) != sha:
                raise ValueError('immutable shared object is corrupt: ' + name)
            self.metrics['objects_reused'] += 1
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temp = tempfile.mkstemp(dir=destination.parent, prefix='.capture-')
            os.close(fd)
            try:
                shutil.copy2(source, temp)
                if digest(temp) != sha:
                    raise ValueError('source changed while saving shared data')
                os.chmod(temp, 0o444)
                try:
                    os.link(temp, destination)  # Create once; never overwrite an old version.
                    self.metrics['objects_created'] += 1
                    self.metrics[kind + '_created'] += 1
                    self.metrics['bytes_created'] += destination.stat().st_size
                except FileExistsError:
                    if digest(destination) != sha:
                        raise ValueError('concurrent shared object mismatch')
                    self.metrics['objects_reused'] += 1
            finally:
                os.unlink(temp)
        return {'path': name, 'sha256': sha, 'bytes': destination.stat().st_size}

    def prices(self, entry, bars, raw=None):
        result = {k: v for k, v in entry.items() if k not in ['bars', 'path', 'object_ref', 'raw_ref']}
        version = fingerprint([entry['security_id'], bars])
        if version != entry['dataset_id']:
            raise ValueError('legacy price version mismatch')
        code = str(relative_path(entry['security_id']))
        obj = {'schema_version': 'daily_ohlcv_object_v1', 'security_id': code,
               'price_basis': entry['price_basis'], 'bars': bars}
        content = json.dumps(obj, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
        import hashlib
        sha = hashlib.sha256(content).hexdigest()
        fd, temp = tempfile.mkstemp()
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(content)
            result['object_ref'] = self.put(temp, f'daily/{code}/{version}.json', sha, 'daily')
        finally:
            os.unlink(temp)
        if raw is not None:
            sha = digest(raw)
            if entry.get('sha256') and sha != entry['sha256']:
                raise ValueError('raw capture checksum mismatch')
            result['sha256'] = sha
            result['raw_ref'] = self.put(raw, f'raw/{code}/{sha}.parquet', sha, 'raw')
        return result

    def facts(self, run, manifest, source=None):
        refs = {}
        for name, sha in manifest['input_files'].items():
            original = Path(source or run) / relative_path(name)
            ref = self.put(original, f'facts/{sha}.json', sha, 'facts')
            path = Path(run).resolve() / relative_path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() or path.is_symlink():
                path.unlink()
            os.symlink(os.path.relpath(self.root / ref['path'], path.parent), path)
            refs[name] = ref
        manifest['input_refs'] = refs
        manifest['storage_schema'] = 'shared_data_references_v1'

    def replay(self, source, run, manifest):
        result = copy.deepcopy(manifest)
        entries = []
        for entry in manifest['datasets']:
            bars = price_bars(source, entry)
            raw = resolve_ref(source, entry['raw_ref']) if entry.get('raw_ref') else (
                  Path(source) / entry['path'] if entry.get('path') else None)
            entries.append(self.prices(entry, bars, raw))
        result['datasets'] = entries
        self.facts(run, result, source)
        return result
