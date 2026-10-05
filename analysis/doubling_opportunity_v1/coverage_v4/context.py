"""Separate cache namespace for the registered coverage repair; old evidence is read-only."""
from pathlib import Path
import json,os
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
ROOT=PARENT.parents[1]
OLD_CACHE=ROOT/'.cache/doubling_opportunity_v1'
COVERAGE_ROOT=OLD_CACHE/'coverage_v4'
active=COVERAGE_ROOT/'active_source_cache.json'
default=json.loads(active.read_text())['path'] if active.exists() else str(COVERAGE_ROOT)
CACHE=Path(os.environ.get('DOUBLING_COVERAGE_CACHE',default)).resolve()
if not CACHE.is_relative_to((OLD_CACHE/'coverage_v4').resolve()):
    raise ValueError('Coverage source cache must remain inside the isolated v4 namespace')
