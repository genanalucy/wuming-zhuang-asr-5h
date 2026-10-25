#!/usr/bin/env python3
"""Score user-supplied reference/prediction JSONL using the original metric."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from common import score_pairs


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('predictions',type=Path);a=p.parse_args()
    data=[json.loads(s) for s in a.predictions.read_text().splitlines() if s.strip()]
    if len({r.get('id',i) for i,r in enumerate(data)})!=len(data):raise ValueError('Duplicate ID')
    print(json.dumps(score_pairs([r['prediction'] for r in data],[r['reference'] for r in data]),indent=2))
