"""Describe source support without changing financial/model features or eligibility."""
from __future__ import annotations
import hashlib,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'joint_v2'))
from financials import TAGS

def profile(payload,asof):
    facts=payload.get('facts',{});gaap=facts.get('us-gaap',{});visible={};currencies=set();forms=set()
    all_amount_units=set()
    for taxonomy,tags in facts.items():
        if taxonomy not in ['us-gaap','ifrs-full']:continue
        for tag in tags.values():
            for unit,rows in tag.get('units',{}).items():
                if re.fullmatch(r'[A-Z]{3}',unit) and any(str(r.get('filed') or '9999')<asof and str(r.get('end') or '9999')<asof for r in rows):all_amount_units.add(unit)
    for field,tags in TAGS.items():
        usd=False
        for tag in tags:
            for unit,rows in gaap.get(tag,{}).get('units',{}).items():
                q=[r for r in rows if str(r.get('filed') or '9999')<asof and str(r.get('end') or '9999')<asof]
                if q:
                    currencies.add(unit);forms.update(r.get('form') for r in q)
                    usd|=unit=='USD'
        visible[field]=usd
    if visible['revenue']:
        support='USD_US_GAAP_revenue_source_present_not_quality_proof'
    elif facts.get('ifrs-full'):
        support='IFRS_taxonomy_not_supported_by_frozen_parser'
    elif gaap and currencies-{'USD'}:
        support='non_USD_financial_currency_not_supported_by_frozen_parser'
    elif gaap:
        support='US_GAAP_source_present_no_supported_USD_revenue'
    else:support='no_supported_standard_financial_taxonomy'
    return {'financial_source_support':support,'financial_reporting_units':sorted(currencies),'financial_amount_unit_candidates':sorted(all_amount_units),
        'financial_source_taxonomies':sorted(facts),'financial_source_forms':sorted(x for x in forms if x),
        'USD_US_GAAP_field_source_presence':visible,'source_support_is_company_health_proof':False}

def build_profiles(cache,asof):
    profiles={}
    for p in sorted((cache/'companyfacts').glob('*.json')):
        raw=p.read_bytes();payload=json.loads(raw);cik=int(p.stem[3:])
        if int(payload.get('cik',0))!=cik:continue
        profiles[str(cik)]={**profile(payload,asof),'raw_sha256':hashlib.sha256(raw).hexdigest()}
    return profiles
