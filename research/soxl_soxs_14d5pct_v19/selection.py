"""Select the pre-May hard-switch diagnostic before new v19 case results."""
from pathlib import Path
import hashlib
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'soxl_soxs_14d5pct_v18'))
from adaptive_blend import core, development_gate
from verify_continuation import read

SOURCE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_selection():
    p = read(SOURCE / 'protocol.json'); reg = read(SOURCE / 'evidence/preregistration.json')
    assert reg['protocol_sha256'] == sha(SOURCE / 'protocol.json')
    for name, h in reg['code_sha256'].items():
        assert sha(core.ROOT / name) == h
    selected = reg['development_selection']; source = core.ROOT / p['selection_run']
    result = read(source / 'results.json'); verification = read(source / 'verification.json')
    assert result['phase'] == 'development' and verification['passed']
    assert sha(source / 'results.json') == selected['results_sha256'] == verification['results_sha256']
    assert sha(source / 'verification.json') == selected['verification_sha256']
    assert p['primary'] == selected['route'] == 'state_hard'
    assert result['primary_name'] == 'state_blend' and not result['development_gate']['passed']
    gate = development_gate(result['routes']['state_hard'], result['coverage']['complete'], p)
    if not gate['passed']:
        raise ValueError('Pre-May selected candidate failed the development screen')
    return {'selected_route': p['primary'], 'source_run': str(source), 'gate': gate,
            'development_results_sha256': selected['results_sha256'],
            'v18_primary_remains_failed': True, 'new_v19_case_results_used_for_selection': False}
