#!/usr/bin/env python3
"""Explicitly materialize a run's shared inputs into a portable offline folder."""
import argparse
import os
import shutil
from pathlib import Path

from common import digest, read, write
from storage import resolve_ref, verify_market


def export_run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(source):
        raise ValueError('export requires a new directory outside the source run')
    manifest = read(source / 'inputs/data_manifest.json')
    verify_market(source, manifest)
    for name, sha in read(source / 'checksums.json').items():
        if digest(source / name) != sha:
            raise ValueError('source artifact checksum mismatch: ' + name)
    shutil.copytree(source, output, symlinks=False)  # Only export copies the input facts.
    refs = [ref for d in manifest['datasets'] for ref in [d.get('object_ref'), d.get('raw_ref')] if ref]
    refs.extend(manifest.get('input_refs', {}).values())
    for ref in {r['path']: r for r in refs}.values():
        target = output / 'shared_data' / ref['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolve_ref(source, ref), target)
    if refs:
        write(output / 'data_location.json', {'schema_version': 'pair_data_location_v1', 'root': 'shared_data'})
    for name, ref in manifest.get('input_refs', {}).items():
        path = output / name
        path.unlink()
        os.link(output / 'shared_data' / ref['path'], path)
    verify_market(output, manifest)
    write(output / 'export_checksums.json', {str(p.relative_to(output)): digest(p)
          for p in output.rglob('*') if p.is_file()})
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(export_run(args.run, args.output))
