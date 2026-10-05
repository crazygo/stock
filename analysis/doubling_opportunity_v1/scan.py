#!/usr/bin/env python3
"""Single repeatable entry point; retain explicitly versioned original reports."""
import argparse,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version',choices=['coverage-v4','raw-v3','joint-v2','independent-v1'],default='coverage-v4')
    p.add_argument('--refresh',action='store_true')
    p.add_argument('--top',type=int,default=30)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--debug',action='store_true',help='Generate the selected experiment debug page; joint-v2 also retains its price control.')
    p.add_argument('--status',action='store_true',help='Inspect raw-v3 or coverage-v4 source and training jobs without starting work.')
    args=p.parse_args()
    if args.top<1:p.error('--top must be positive')
    if args.version in ['raw-v3','coverage-v4']:
        if args.output_dir:p.error('raw-v3 and coverage-v4 write immutable versioned runs; --output-dir is not supported')
        directory='coverage_v4' if args.version=='coverage-v4' else 'raw_daily_v3'
        cmd=[sys.executable,str(HERE/directory/'scan.py'),'--top',str(args.top)]
        for enabled,flag in [(args.refresh,'--refresh'),(args.debug,'--debug'),(args.status,'--status')]:
            if enabled:cmd.append(flag)
        subprocess.run(cmd,check=True)
        return
    if args.status:p.error('--status requires --version raw-v3 or coverage-v4')
    if args.debug and args.version!='joint-v2':p.error('--debug currently uses the explicitly named joint-v2 experiment')
    script=HERE/'joint_v2/run.py' if args.version=='joint-v2' else HERE/'run.py'
    if args.refresh:
        subprocess.run([sys.executable,str(HERE/'acquire_quotes.py'),'--refresh','--refresh-universe'],check=True)
    cmd=[sys.executable,str(script),'--top',str(args.top)]
    if args.refresh:cmd.append('--refresh')
    if args.output_dir:cmd+=['--output-dir',str(args.output_dir.resolve())]
    subprocess.run(cmd,check=True)
    if args.debug:
        subprocess.run([sys.executable,str(HERE/'joint_v2/build_debug.py')],check=True)
        pointer=json.loads((HERE/'joint_v2/latest_run.json').read_text())
        report=json.loads((Path(pointer['path'])/'report.json').read_text())
        if report['metadata']['joint_effect_backtested']:
            subprocess.run([sys.executable,str(HERE/'joint_v2/build_debug.py'),'--experiment','financial'],check=True)

if __name__=='__main__':main()
