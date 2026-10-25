"""Shared offline audio loading, scoring and durable run artifacts."""
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import unicodedata


def normalize_text(text):
    return " ".join(unicodedata.normalize("NFC", str(text)).split())


def _distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (x != y)))
        previous = current
    return previous[-1]


def score_pairs(predictions, references):
    if len(predictions) != len(references):
        raise ValueError("Prediction/reference cardinalities differ")
    char_errors = word_errors = char_count = word_count = 0
    for prediction, reference in zip(predictions, references):
        pred, ref = normalize_text(prediction), normalize_text(reference)
        pchars, rchars = "".join(pred.split()), "".join(ref.split())
        char_errors += _distance(pchars, rchars)
        word_errors += _distance(pred.split(), ref.split())
        char_count += len(rchars)
        word_count += len(ref.split())
    return {
        "cer": char_errors/char_count if char_count else (0.0 if not char_errors else None),
        "wer": word_errors/word_count if word_count else (0.0 if not word_errors else None),
        "cer_edits": char_errors, "cer_reference_chars": char_count,
        "wer_edits": word_errors, "wer_reference_tokens": word_count,
        "utterances": len(references), "unit": "fraction",
        "normalization": "Unicode NFC and whitespace collapse; case, punctuation, digits preserved; CER excludes whitespace",
        "wer_unit": "whitespace-delimited orthographic token; not certified lexical word or syllable"
    }


def load_manifest(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = stream.name
    os.replace(temporary, path)


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(obj, ensure_ascii=False, allow_nan=False)+"\n")


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_audio_16k(row):
    import numpy as np
    import soundfile as sf
    source = Path(row["audio"])
    cache_root = os.environ.get("ASR_AUDIO_CACHE")
    cached = Path(cache_root)/f"{row['sha256']}.wav" if cache_root and row.get("sha256") else None
    if cached and cached.exists():
        waveform, rate = sf.read(cached, dtype="float32", always_2d=False)
        if rate != 16000 or waveform.ndim != 1:
            raise ValueError(f"Invalid cache format: {cached}")
        return waveform
    # ffmpeg covers all four original containers, resamples and uses a fixed mono downmix.
    proc = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(source),
                           "-vn", "-ar", "16000", "-ac", "1", "-f", "f32le", "pipe:1"],
                          capture_output=True, timeout=90)
    if proc.returncode:
        raise RuntimeError(f"Audio decode failed: {source}: {proc.stderr.decode(errors='replace')[:600]}")
    waveform = np.frombuffer(proc.stdout, dtype="<f4").copy()
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError(f"Empty/non-finite waveform: {source}")
    if float(np.max(np.abs(waveform))) == 0:
        raise ValueError(f"Zero waveform after downmix: {source}")
    if cached:
        cached.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(suffix=".wav", dir=cached.parent)
        os.close(fd)
        try:
            sf.write(temporary, waveform, 16000, subtype="FLOAT")
            os.replace(temporary, cached)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
    return waveform


def test_metrics():
    assert score_pairs(["a b"],["a b"])["cer"] == 0
    assert score_pairs(["a c"],["a b"])["wer"] == .5
    assert score_pairs([""],["ab"])["cer"] == 1
    assert score_pairs(["abc"],["ab"])["cer"] == .5
    assert score_pairs(["a"],["ab"])["cer"] == .5
    assert score_pairs(["é"],["e\u0301"])["cer"] == 0
    assert score_pairs(["A"],["a"])["cer"] == 1
    assert score_pairs(["1"],["2"])["cer"] == 1
    assert score_pairs([""],[""])["cer"] == 0
    assert score_pairs(["a"],[""])["cer"] is None
    assert score_pairs(["a", "abc"],["ab", "abc"])["cer"] == .2
    try: score_pairs(["a"],[])
    except ValueError: pass
    else: raise AssertionError("Mismatched cardinality was accepted")
    print("Scoring self-tests passed", flush=True)


if __name__ == "__main__": test_metrics()
