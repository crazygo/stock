#!/usr/bin/env python3
"""Explicit manual publication after acquisition and quality review.

Acquisition never calls this module. Existing shared annual files with
different content are preserved and reported instead of overwritten.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.backfill_model_history import DEST, CACHE, sha, write
from scripts.r2_client import R2Client


def plan():
    manifest=json.loads((DEST/"manifest.json").read_text())
    if not manifest["acquisition_complete"]:
        raise RuntimeError("Acquisition is still incomplete; publication prohibited")
    if any(x["invalid_rows"] or x["duplicates"] for x in manifest["coverage"]):
        raise RuntimeError("Unresolved source-integrity failure")
    files={}
    for item in manifest["files"]:
        p=DEST/item["path"]
        if sha(p)!=item["sha256"]:raise ValueError(f"File changed since validation: {p}")
        files[f"{DEST.name}/{item['path']}"]=p
    for folder in ["corporate_actions","quarantine"]:
        for p in (DEST/folder).glob("**/*.parquet"):
            files[f"{DEST.name}/{p.relative_to(DEST)}"]=p
    for name in ["README.md","REPORT.md","registration.json","calendar.json","security_metadata.json","security_history.json","corporate_actions_audit.json","coverage.json","manifest.json"]:
        p=DEST/name
        if p.exists():files[f"{DEST.name}/{name}"]=p
    evidence=[]
    for p in sorted((DEST/"parts").glob("*/*/*.json")):
        evidence.append({"path":str(p.relative_to(DEST)),"sha256":sha(p),"content":json.loads(p.read_text())})
    write(DEST/"lineage.json",evidence)
    files[f"{DEST.name}/lineage.json"]=DEST/"lineage.json"
    write(DEST/"request_audit.json",[json.loads(p.read_text()) for p in sorted((CACHE/"requests").glob("**/audit.json"))])
    files[f"{DEST.name}/request_audit.json"]=DEST/"request_audit.json"
    conflicts=[]
    for p in sorted((DEST/"us_5m").glob("*/202[345].parquet")):
        shared=ROOT/"market_data/us_5m"/p.parent.name/p.name
        if shared.exists() and sha(shared)!=sha(p):
            conflicts.append({"symbol":p.parent.name,"year":p.stem,"shared_sha256":sha(shared),"version_sha256":sha(p)})
        else:
            files[f"us_5m/{p.parent.name}/{p.name}"]=p
    return files,conflicts


def publish(execute=False):
    files,conflicts=plan()
    result=dict(files=len(files),bytes=sum(p.stat().st_size for p in files.values()),preserved_shared_conflicts=conflicts)
    print(json.dumps(result),flush=True)
    if not execute:return
    client=R2Client()
    remote={x["key"]:x for prefix in [DEST.name+"/","us_5m/"] for x in client.list_objects(prefix)}
    def transfer(item):
        key,p=item
        md5=hashlib.md5(p.read_bytes()).hexdigest()
        old=remote.get(key)
        if old and old["etag"]!=md5 and key.startswith("us_5m/"):
            raise RuntimeError(f"Shared cloud object already differs; needs reconciliation: {key}")
        if not old or old["etag"]!=md5:client.put_object(p,key)
        head=client.head_object(key)
        # The project client uses single PUT objects, whose ETag is MD5.
        etag=str(head.get("etag",head.get("ETag",""))).strip('"') if head else ""
        if etag!=md5:raise RuntimeError(f"Cloud checksum mismatch for {key}: {head}")
        if key.startswith("us_5m/"):
            shared=ROOT/"market_data"/key
            if not shared.exists():shared.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,shared)
        return dict(key=key,bytes=p.stat().st_size,md5=md5,sha256=sha(p),verified=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        verified=list(pool.map(transfer,files.items()))
    write(DEST/"cloud_verification.json",dict(**result,all_verified=True,objects=verified))
    client.put_object(DEST/"cloud_verification.json",f"{DEST.name}/cloud_verification.json")
    print(json.dumps({"all_verified":True,"objects":len(verified),"preserved_shared_conflicts":len(conflicts)}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--execute",action="store_true")
    publish(p.parse_args().execute)
