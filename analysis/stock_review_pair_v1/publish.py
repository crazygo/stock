"""Publish both approved URLs through a single atomic directory pointer."""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from common import ROOT, digest, now, read, write

TARGETS = {
    'analysis/stock_behavior_description_v3/index.html': 'stock_behavior_description_v3/index.html',
    'analysis/stock_behavior_description_v3/detail.html': 'stock_behavior_description_v3/detail.html',
    'analysis/ai_trend_quadrant_v2/index.html': 'ai_trend_quadrant_v2/index.html',
}


def switch(link, destination):
    link, destination = Path(link), Path(destination)
    temp = link.with_name('.' + link.name + '-' + uuid.uuid4().hex)
    os.symlink(os.path.relpath(destination, link.parent), temp, target_is_directory=True)
    os.replace(temp, link)


def initialize(store, root=ROOT):
    current = store / 'current'
    state = store / 'publication.json'
    if state.exists():
        saved = read(state)
        if saved['root'] != str(root.resolve()):
            raise RuntimeError('publication belongs to another workspace')
        for name, target in TARGETS.items():
            link = root / name
            expected = store / 'current/pages' / target
            if not link.is_symlink() or os.readlink(link) != os.path.relpath(expected, link.parent):
                raise RuntimeError(f'Published alias changed by another task: {name}; refusing overwrite')
        return
    if current.exists() or current.is_symlink():
        raise RuntimeError('unregistered publication pointer; refusing overwrite')
    bootstrap = store / 'legacy-' / uuid.uuid4().hex
    originals = {}
    for name, target in TARGETS.items():
        original = root / name
        if not original.is_file() or original.is_symlink():
            raise RuntimeError(f'Expected untouched legacy HTML: {name}')
        dest = bootstrap / 'pages' / target
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, dest)
        originals[name] = digest(original)
    switch(current, bootstrap)
    installed = []
    try:
        for name, target in TARGETS.items():
            original = root / name
            if digest(original) != originals[name]:
                raise RuntimeError(f'Legacy HTML changed concurrently: {name}')
            switch(original, store / 'current/pages' / target)
            installed.append(name)
        write(state, {'root': str(root.resolve()), 'installed_at': now(), 'legacy_backup': str(bootstrap),
                      'legacy_sha256': originals})
    except BaseException:
        for name in installed:
            shutil.copyfile(bootstrap / 'pages' / TARGETS[name], root / (name + '.rollback'))
            os.replace(root / (name + '.rollback'), root / name)
        current.unlink()
        raise


def publish(store, run, root=ROOT):
    acceptance = read(run / 'validation/acceptance.json')
    if acceptance['status'] != 'passed':
        raise ValueError('both pages must pass validation before publication')
    # Reverify every dependency, including both HTML pages, before one pointer switch.
    files = read(run / 'checksums.json')
    for name, sha in files.items():
        if digest(run / name) != sha:
            raise ValueError('artifact changed after validation: ' + name)
    if (run / 'inputs/data_manifest.json').exists():
        from storage import verify_market
        verify_market(run)
    previous = store / 'current/inputs/data_manifest.json'
    if previous.exists():
        old = read(previous)['market_cutoffs']
        new = read(run / 'inputs/data_manifest.json')['market_cutoffs']
        if any(new.get(market, '') < day for market, day in old.items()):
            raise ValueError('candidate would move a published market cutoff backwards; use --no-publish for historical review')
    initialize(store, root)
    switch(store / 'current', run)
    write(store / 'last_success.json', {'run_id': run.name, 'published_at': now(), 'path': str(run)})
