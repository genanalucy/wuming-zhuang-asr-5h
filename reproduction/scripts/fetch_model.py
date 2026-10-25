#!/usr/bin/env python3
"""Fetch the exact public initializer separately; no model weights are bundled."""
import argparse
import json
from pathlib import Path
from verify_package import ROOT,digest


def check_model():
    spec=json.loads((ROOT/'config/public_model.json').read_text())
    directory=ROOT/'model_cache/whisper-large-v3'
    for name,expected in spec['files_sha256'].items():
        if not (directory/name).is_file() or digest(directory/name)!=expected:
            raise ValueError('Public initializer missing or changed: '+name)
    print(json.dumps({'state':'verified','model':spec['repo_id'],'revision':spec['revision']}))


def main():
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');a=p.parse_args()
    spec=json.loads((ROOT/'config/public_model.json').read_text())
    if not a.check_only:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=spec['repo_id'],revision=spec['revision'],
          local_dir=ROOT/'model_cache/whisper-large-v3',allow_patterns=list(spec['files_sha256']))
    check_model()


if __name__=='__main__':main()
