"""Post-evaluation lossless packaging only; does not train or change any result."""
import argparse
import gzip
from package_delivery import package
from state_calibration import SOURCE, read, dump, sha


def compact(run):
    package(run)
    original=SOURCE/'evidence/event_features.csv'
    target=original.with_suffix('.csv.gz')
    raw=original.read_bytes();digest=sha(original)
    target.write_bytes(gzip.compress(raw,compresslevel=9,mtime=0))
    assert gzip.decompress(target.read_bytes())==raw
    original.unlink()
    path=SOURCE/'evidence/delivery_manifest.json';manifest=read(path)
    assert manifest['delivery_files_sha256'].pop('evidence/event_features.csv')==digest
    manifest['delivery_files_sha256']['evidence/event_features.csv.gz']=sha(target)
    manifest['lossless_delivery_encodings']={'evidence/event_features.csv.gz':{
        'decompressed_sha256':digest,'decompressed_bytes':len(raw),'compressed_bytes':target.stat().st_size,
        'encoding':'gzip','roundtrip_verified':True}}
    manifest['post_evaluation_packaging_only']=True
    dump(path,manifest)
    print({'lossless_roundtrip':True,'raw_bytes':len(raw),'compressed_bytes':target.stat().st_size})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    compact(parser.parse_args().run)
