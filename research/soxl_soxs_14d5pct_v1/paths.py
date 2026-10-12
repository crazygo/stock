"""Project paths for a frozen, offline research import."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
CACHE_ROOT = PROJECT_ROOT / '.cache' / 'soxl_soxs_14d5pct_v1'
IMPORT_ROOT = CACHE_ROOT / 'imports' / '2026-10-12' / 'SOXL_SOXS_research'
RAW_DIR = IMPORT_ROOT / 'raw'


def writable_results(path):
    """Never overwrite any part of the captured source package."""
    path = Path(path).resolve()
    if path == IMPORT_ROOT or IMPORT_ROOT in path.parents:
        raise ValueError('The imported snapshot is immutable; choose a separate run directory')
    return path
