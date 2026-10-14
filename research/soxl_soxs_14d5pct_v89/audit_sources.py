"""Integrity, structural and arithmetic audit. Not a full financial-source review."""
import math
from collections import Counter
from catalogue import *
from parse_revenue import actual, guidance
from pair_candidates import require_training_ready


def read(path):
    return json.loads(path.read_text())


def audit():
    cat=read(SOURCE/'catalogue.json')
    inv=read(SOURCE/'evidence/source_inventory.json')
    candidates=read(SOURCE/'evidence/revenue_candidates.json')
    review=read(SOURCE/'review_cases.json')
    pairs=read(SOURCE/'evidence/pair_candidates.json')
    http=read(CACHE/'http_manifest.json')
    rows={r['id']:r for r in candidates['records']}
    origins={r['id']:r for r in inv['records']}
    anchors={r['id']:r for r in review['numeric_anchors']}
    assert len(rows)==len(cat['records'])==128 and len(origins)==128
    assert not inv['failures'] and not candidates['training_use_allowed']
    for path,h in cat['source_sha256'].items():
        assert sha(ROOT/path)==h,path
    counters=Counter();queue=[];input_hashes=dict(cat['source_sha256'])
    for record in cat['records']:
        row=rows[record['id']];origin=origins[record['id']]
        for key in ['url','symbol','published_date','candidate_kind']:
            assert row[key]==origin[key]==record[key],(record['id'],key)
        assert row['training_use_allowed'] is False
        for file_key,hash_key in [('source_file','source_sha256'),('document_file','document_sha256')]:
            assert sha(ROOT/row[file_key])==row[hash_key],row[file_key]
            input_hashes[row[file_key]]=row[hash_key]
        doc=read(ROOT/row['document_file'])
        a=actual(doc);g=guidance(doc)
        if g and g['range_million']:
            value=g['range_million'];value['midpoint']=(value['lower']+value['upper'])/2
        assert a==row['actual_candidate'] and g==row['guidance_candidate'],row['id']
        unit=False
        if a:
            assert g is not None and g['range_million'] is not None
            before=doc['text'][max(0,a['evidence_start']-1500):a['evidence_start']]
            unit=bool(re.search(a['unit_assumption'],before,re.I))
            assert unit,(row['id'],'Unit marker missing')
            assert a['revenue_million']>0 and re.fullmatch(r'20\d{2}Q[1-4]',a['fiscal_period'])
            lo=g['range_million']['lower'];hi=g['range_million']['upper']
            assert 0<lo<=hi and all(map(math.isfinite,[lo,hi]))
            if g['scope']=='quarter':
                year,q=map(int,a['fiscal_period'].split('Q'))
                next_period=f'{year+(q==4)}Q{q%4+1}'
                assert g['fiscal_period']==next_period,(row['id'],g['fiscal_period'],next_period)
                counters['next_quarter_guidance_candidates']+=1
            else:
                assert g['scope']=='annual'
                counters['annual_guidance_candidates']+=1
            counters['actual_candidates_with_nearby_unit_marker']+=1
        else:
            assert row['candidate_kind']=='update_or_preliminary'
            counters['non_regular_release_events']+=1
        # These words are triage aids, not an exhaustive business-scope audit.
        keywords=[word for word in ['deconsolidat','discontinued operations','divest','VMware']
                  if word.lower() in doc['text'].lower()]
        queue.append(dict(id=row['id'],symbol=row['symbol'],published_date=row['published_date'],
            unit_marker_found=unit,numeric_anchor_reviewed=row['id'] in anchors,
            business_scope_keywords=keywords,full_financial_and_publication_review='pending',
            training_use_allowed=False))
    for event in review['updates']+review['annual_guidance_cases']+review['numeric_anchors']:
        assert event['source_sha256']==rows[event['id']]['source_sha256']
    assert len(review['updates'])==6
    for expected in review['annual_guidance_cases']:
        g=rows[expected['id']]['guidance_candidate']
        assert g['scope']=='annual' and g['fiscal_period']==expected['fiscal_year']
        assert g['range_million']['midpoint']==expected['revenue_million']
    events={r['id']:r for r in review['updates']}
    for p in pairs['records']:
        a=rows[p['actual_id']]['actual_candidate']
        regular=p['regular_release_reference_id']
        if regular:
            r=rows[regular];g=r['guidance_candidate']
            assert r['published_date']<p['actual_published_date']
            assert g['scope']=='quarter' and g['fiscal_period']==a['fiscal_period']
            assert p['ratio_to_regular_release_midpoint']==a['revenue_million']/g['range_million']['midpoint']-1
        else:
            assert p['ratio_to_regular_release_midpoint'] is None
        reference=p['last_known_reference_id']
        if reference in events:
            v=events[reference]['new_revenue_million']
            expected=a['revenue_million']/v['midpoint']-1 if isinstance(v,dict) else None
        else:
            expected=p['ratio_to_regular_release_midpoint']
        assert p['ratio_to_last_known_numeric_reference']==expected
        assert p['training_use_allowed'] is False and p['unresolved']
    for dataset in [candidates,pairs]:
        try:require_training_ready(dataset)
        except ValueError:pass
        else:raise AssertionError('Unreviewed data accepted for training')
    for path in [CACHE/'http_manifest.json',CACHE/'old_body_cache_probe.json']:
        input_hashes[str(path.relative_to(ROOT))]=sha(path)
    for r in http['records']:
        if r.get('cache_file') and r.get('sha256'):
            assert sha(ROOT/r['cache_file'])==r['sha256']
            input_hashes[r['cache_file']]=r['sha256']
    for path in (CACHE/'http').glob('*.json'):
        input_hashes[str(path.relative_to(ROOT))]=sha(path)
    result=dict(passed=True,created_at=datetime.now(timezone.utc).isoformat(),
        audit_scope='Source integrity, parser reproducibility, nearby unit markers, eight independently read numeric anchors, six event semantics, three annual cases, and pairing arithmetic. Not full numeric, scope, time or revision-history approval.',
        counts=dict(counters),requested_documents=len(rows),limited_numeric_anchors=len(anchors),
        acquisition_status=dict(Counter(r['status'] for r in http['records'])),
        pairing_summary=pairs['summary'],training_use_allowed=False,goal_complete=False,
        input_sha256=input_hashes)
    dump(SOURCE/'evidence/review_queue.json',dict(records=queue,training_use_allowed=False))
    dump(SOURCE/'evidence/source_verification.json',result)
    print({k:v for k,v in result.items() if k!='input_sha256'})


if __name__=='__main__':
    audit()
