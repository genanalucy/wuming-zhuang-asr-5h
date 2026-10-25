#!/usr/bin/env python3
"""Verify the public package without a GPU or any external dataset."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import unicodedata

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def norm(text):return ' '.join(unicodedata.normalize('NFC',text).split())


def verify(full_audio=False):
    expected=json.loads((ROOT/'config/dataset_identity.json').read_text())
    checked=0
    for line in (ROOT/'CHECKSUMS.sha256').read_text().splitlines():
        wanted,name=line.split('  ',1);path=ROOT/name
        if not path.resolve().is_relative_to(ROOT):raise ValueError('Unsafe checksum path')
        if digest(path)!=wanted:raise ValueError('Checksum mismatch: '+name)
        checked+=1
    records=[json.loads(s) for s in (ROOT/'train/manifest.jsonl').read_text().splitlines()]
    with (ROOT/'train/metadata.csv').open(newline='') as f:metadata=list(csv.DictReader(f))
    assert len(records)==len(metadata)==2585
    assert len({r['speaker_id'] for r in records})==22
    assert len({r['id'] for r in records})==len(records)
    found=set(p.name for p in (ROOT/'train/audio').glob('*.wav'))
    assert found=={Path(r['file_name']).name for r in records}
    texts=hashlib.sha256();pcm_order=hashlib.sha256()
    for i,(r,m) in enumerate(zip(records,metadata)):
        assert r['id']==f'wz5h_{i+1:06d}' and m['id']==r['id'] and m['text']==r['text']
        assert m['file_name']==r['file_name'] and m['speaker_id']==r['speaker_id']
        assert norm(r['text']) and r['num_frames']>0 and r['sample_rate']==16000
        assert r['audio_sha256']==digest(ROOT/'train'/r['file_name'])
        text=norm(r['text']).encode();texts.update(len(text).to_bytes(8,'big'));texts.update(text)
        pcm_order.update(bytes.fromhex(r['pcm_float32_sha256']))
        if full_audio:
            import numpy as np
            import soundfile as sf
            x,rate=sf.read(ROOT/'train'/r['file_name'],dtype='float32',always_2d=False)
            assert rate==16000 and x.ndim==1 and len(x)==r['num_frames'] and np.isfinite(x).all()
            assert hashlib.sha256(x.astype('<f4',copy=False).tobytes()).hexdigest()==r['pcm_float32_sha256']
    assert texts.hexdigest()==expected['normalized_text_order_sha256']
    assert pcm_order.hexdigest()==expected['ordered_pcm_hashes_sha256']
    return {'state':'passed','files_checked':checked,'training_rows':len(records),
            'source_speaker_identifiers':22,'audio_identity':'FLOAT32 PCM full checks' if full_audio else 'file SHA256',
            'dev_test_included':False,'training_launched':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--full-audio',action='store_true');a=p.parse_args()
    print(json.dumps(verify(a.full_audio),indent=2))
