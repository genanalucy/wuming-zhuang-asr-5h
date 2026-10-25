#!/usr/bin/env python3
"""Relocate anonymous train manifests/cache and print the original training command.

No training runs unless --execute is explicitly passed. Development audio and
transcripts are external inputs and are not shipped in this training-only pack.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from verify_package import ROOT,digest,verify,norm


def prepare(workdir,dev_manifest=None):
    verify(full_audio=False)
    if workdir.resolve().is_relative_to(ROOT):
        raise ValueError('Keep workdir outside the immutable extracted dataset directory')
    workdir.mkdir(parents=True,exist_ok=False);cache=workdir/'audio_cache';cache.mkdir()
    data=[json.loads(s) for s in (ROOT/'train/manifest.jsonl').read_text().splitlines()]
    prepared=[]
    for row in data:
        audio=(ROOT/'train'/row['file_name']).resolve()
        (cache/(row['audio_sha256']+'.wav')).symlink_to(audio)
        prepared.append({'id':row['id'],'audio':str(audio),'sha256':row['audio_sha256'],
          'speaker_candidate':row['speaker_id'],'duration':row['source_duration_seconds'],
          'text':row['text']})
    train=workdir/'train.jsonl'
    train.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in prepared))
    if dev_manifest:
        original=[json.loads(s) for s in dev_manifest.read_text().splitlines() if s.strip()]
        if len(original)!=4120:raise ValueError('Original full-dev requires4120rows; no silent replacement subset')
        signature=hashlib.sha256();den_c=den_t=0
        for row in original:
            if not all(k in row for k in ('audio','sha256','speaker_candidate','duration','text')):
                raise ValueError('Development JSONL must have audio/sha256/speaker_candidate/duration/text')
            text=norm(row['text']);den_c+=len(''.join(text.split()));den_t+=len(text.split())
            signature.update(json.dumps([row['sha256'],text],ensure_ascii=False,separators=(',',':')).encode()+b'\n')
            source=Path(row['audio']);source=source if source.is_absolute() else dev_manifest.parent/source
            if not source.is_file() or digest(source)!=row['sha256']:raise ValueError('External dev file missing or SHA differs')
            if row['sha256'] in {r['sha256'] for r in prepared}:raise ValueError('Train/dev identity overlap')
            if row['speaker_candidate'] in {r['speaker_candidate'] for r in prepared}:raise ValueError('Speaker identifiers overlap')
            row['audio']=str(source.resolve())
        identity=json.loads((ROOT/'config/evaluation_identity.json').read_text())
        if (den_c,den_t)!=(178266,33668) or signature.hexdigest()!=identity['original_dev_order_audio_reference_sha256']:
            raise ValueError('External dev is not the original fixed reference/audio order; not comparable reproduction')
        target=workdir/'dev.jsonl';target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in original))
    else:target=Path('EXTERNAL_ORIGINAL_DEV_4120.jsonl')
    config=json.loads((ROOT/'config/training.json').read_text())
    model=ROOT/'model_cache/whisper-large-v3'
    command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=3',
      str(ROOT/'code/train_whisper_aug_ddp.py'),'--mode','train','--augmentation','A0',
      '--model-dir',str(model),'--seed','42','--batch-size','2','--grad-accum','3','--workers','1',
      '--eval-batch-size','1','--gradient-checkpointing','--lr','0.00001',
      '--train-manifest',str(train),'--dev-manifest',str(target),'--full-dev-manifest',str(target),
      '--output',str(workdir/'training'),'--max-steps','1149','--eval-limit','512',
      '--eval-every','46','--full-eval-every','144','--min-steps','431','--early-stopping-patience','3',
      '--warmup-steps','57','--weight-decay','0.01','--max-grad-norm','1.0']
    receipt={'state':'prepared','training_rows':2585,'train_manifest_sha256':digest(train),
      'external_original_dev_verified':bool(dev_manifest),'command':command,
      'env':{'ASR_DATA_EFFICIENCY_TIER':'5h','WZ5H_PUBLIC_MANIFEST_SHA256':digest(train),'ASR_AUDIO_CACHE':str(cache)},
      'no_gpu_or_training_used':True}
    (workdir/'prepared.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--workdir',type=Path,required=True)
    p.add_argument('--dev-manifest',type=Path);p.add_argument('--execute',action='store_true')
    a=p.parse_args();receipt=prepare(a.workdir.resolve(),a.dev_manifest.resolve() if a.dev_manifest else None)
    print(json.dumps(receipt,indent=2))
    if a.execute:
        if not receipt['external_original_dev_verified']:raise ValueError('Original dev is not included; supply it separately for comparable selection')
        from fetch_model import check_model
        check_model()
        env=os.environ.copy();env.update(receipt['env']);env.setdefault('OMP_NUM_THREADS','1')
        subprocess.run(receipt['command'],env=env,check=True)


if __name__=='__main__':main()
