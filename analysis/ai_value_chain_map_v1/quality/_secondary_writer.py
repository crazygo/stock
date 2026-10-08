import json, os
from pathlib import Path
D=Path(__file__).parent
OUT=D/'reviews_secondary.json'
TMP=D/'reviews_secondary.json.tmp'
def save(records):
    payload={'records':records}
    with open(TMP,'w',encoding='utf-8') as f:
        json.dump(payload,f,ensure_ascii=False,indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(TMP,OUT)
