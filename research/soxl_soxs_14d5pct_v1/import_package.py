"""Import the supplied ZIP without executing it or contacting data providers."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

from paths import IMPORT_ROOT, ROOT

PREFIX = 'SOXL_SOXS_research/'


def import_package(archive):
    archive = Path(archive).resolve()
    with zipfile.ZipFile(archive) as package:
        members = {}
        for member in package.infolist():
            path = PurePosixPath(member.filename)
            if (path.is_absolute() or '..' in path.parts or '\\' in member.filename
                    or not member.filename.startswith(PREFIX)
                    or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError(f'Unsafe ZIP member: {member.filename}')
            if member.is_dir():
                continue
            name = member.filename[len(PREFIX):]
            if not name or name in members:
                raise ValueError(f'Duplicate or empty ZIP member: {member.filename}')
            members[name] = package.read(member)
        manifest = json.loads(members['MANIFEST.json'])
        if set(members) != set(manifest) | {'MANIFEST.json'}:
            raise ValueError('ZIP members differ from MANIFEST.json')
        for name, expected in manifest.items():
            if hashlib.sha256(members[name]).hexdigest() != expected:
                raise ValueError(f'Checksum mismatch: {name}')
        # The code and protocol reviewed for this migration must stay unchanged.
        captured = json.loads((ROOT / 'original' / 'MANIFEST.json').read_text())
        if manifest != captured:
            raise ValueError('This archive is a different research batch; register it separately')
        for name, content in members.items():
            target = IMPORT_ROOT / name
            if target.exists() and target.read_bytes() != content:
                raise ValueError(f'Existing immutable snapshot differs: {target}')
        for name, content in members.items():
            target = IMPORT_ROOT / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(content)
    report = {
        'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'manifest_checked_files': len(manifest),
        'imported_files': len(members),
        'destination': str(IMPORT_ROOT),
        'network_requests': 0,
        'pickle_loaded': False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    import_package(parser.parse_args().archive)
