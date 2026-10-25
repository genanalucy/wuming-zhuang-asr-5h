"""Stateless training-only ASR augmentation; no data loading or GPU operations.

A0=none, A1=light SpecAugment, A2=additive noise, A3=both. Default noise is
explicitly SYNTHETIC white/pink noise, never claimed as natural sound. RNG is
keyed by seed/epoch/sample identity/branch and independent of sampler, rank,
worker, and fetch order. References, duration, sample rate and pitch unchanged.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import multiprocessing
from typing import Mapping

import numpy as np

VERSION = 'asr-augmentation-v1'


def stable_rng(seed, epoch, sample_key, branch):
    if not isinstance(epoch, int) or epoch < 0 or not str(sample_key) or not str(branch):
        raise ValueError('Nonnegative integer epoch and nonempty sample/branch keys required')
    payload = json.dumps([VERSION, int(seed), epoch, str(sample_key), str(branch)],
                         ensure_ascii=False, separators=(',', ':')).encode('utf8')
    entropy = int.from_bytes(hashlib.sha256(payload).digest()[:16], 'big')
    return np.random.Generator(np.random.PCG64(entropy))


class SharedEpoch:
    """Epoch state shared by spawned persistent DataLoader workers.

    Set before iter(loader), including after checkpoint resume; update only once
    the previous epoch iterator is exhausted. Workers read for each __getitem__.
    This solves visibility; caller must separately restore the sampler cursor.
    """
    def __init__(self, epoch=0):
        self._value = multiprocessing.get_context('spawn').Value('q', 0, lock=True)
        self.set(epoch)

    def get(self):
        with self._value.get_lock():
            return int(self._value.value)

    def set(self, epoch):
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError('Epoch must be nonnegative integer')
        with self._value.get_lock():
            self._value.value = epoch


def _waveform(audio):
    audio = np.asarray(audio)
    if audio.ndim != 1 or not len(audio) or not np.issubdtype(audio.dtype, np.floating):
        raise ValueError('Audio must be nonempty mono floating point array')
    if not np.isfinite(audio).all():
        raise ValueError('Audio contains nonfinite samples')
    return audio.astype(np.float64, copy=False)


def rms(audio):
    values = _waveform(audio)
    return float(np.sqrt(np.mean(values * values)))


def waveform_sha256(audio, sample_rate=16000):
    values = _waveform(audio)
    if sample_rate != 16000:
        raise ValueError('Noise assets must be 16kHz mono without implicit resampling')
    digest = hashlib.sha256()
    digest.update(b'16kHz-mono-float32-little-endian-v1\0')
    digest.update(values.astype('<f4').tobytes())
    return digest.hexdigest()


def validate_noise_partitions(train_assets, eval_assets):
    """Reject asset hash reuse and synthetic recipe-key reuse across partitions.

    Each receipt has sha256 and origin, plus role=train/eval. Optional file_sha256
    guards reuse of differently decoded file copies; recipe_sha256 guards a
    synthetic generator's source-key reuse. This function never reads audio.
    """
    keys = {'train': set(), 'eval': set()}
    for role, assets in [('train', train_assets), ('eval', eval_assets)]:
        for asset in assets:
            if asset.get('role') != role or not asset.get('origin'):
                raise ValueError('Noise asset origin/partition missing or incorrect')
            digest = asset.get('sha256')
            if not isinstance(digest, str) or len(digest) != 64 or any(ch not in '0123456789abcdef' for ch in digest):
                raise ValueError('Noise asset requires valid canonical SHA256')
            for field in ('sha256', 'file_sha256', 'recipe_sha256'):
                if asset.get(field):
                    value = asset[field]
                    if not isinstance(value, str) or len(value) != 64 or any(ch not in '0123456789abcdef' for ch in value):
                        raise ValueError('All supplied noise source fingerprints must be valid SHA256')
                    keys[role].add(value)
    if keys['train'] & keys['eval']:
        raise ValueError('Train/evaluation noise source hash overlap')
    return {'train_assets': len(train_assets), 'eval_assets': len(eval_assets),
            'noise_partitions_verified': True}


def synthetic_noise(length, rng, kind='white'):
    if not isinstance(length, int) or length < 1 or kind not in ('white', 'pink'):
        raise ValueError('Positive sample length and white/pink kind required')
    white = rng.standard_normal(length)
    if kind == 'pink' and length > 1:
        spectrum = np.fft.rfft(white)
        frequency = np.fft.rfftfreq(length)
        weight = np.zeros_like(frequency)
        weight[1:] = 1 / np.sqrt(frequency[1:])
        white = np.fft.irfft(spectrum * weight, n=length)
    if length > 1:
        white -= np.mean(white)
    return white.astype(np.float64)


def mix_at_snr(audio, noise, snr_db, peak_limit=.99):
    """Mix matching arrays at RMS SNR; scale clean AND noise if peak clips.

    SNR is 20log10(RMS(clean component)/RMS(noise component)), full utterance
    including silence. The common gain preserves SNR. No clipping operation is
    applied. Silence/zero-energy noise is left unchanged with an explicit reason.
    """
    clean, noise = _waveform(audio), _waveform(noise)
    if clean.shape != noise.shape:
        raise ValueError('Noise and clean waveform length mismatch')
    if not math.isfinite(snr_db) or not 0 < peak_limit <= 1:
        raise ValueError('Finite SNR and peak_limit in (0,1] required')
    clean_rms, noise_rms = rms(clean), rms(noise)
    if clean_rms <= 1e-12 or noise_rms <= 1e-12:
        return np.asarray(audio).copy(), {'applied': False, 'reason': 'silent_clean' if clean_rms <= 1e-12 else 'zero_energy_noise',
                                         'target_snr_db': float(snr_db)}
    noise_gain = clean_rms / (noise_rms * 10 ** (float(snr_db) / 20))
    noise_component = noise * noise_gain
    mixture = clean + noise_component
    peak = float(np.max(np.abs(mixture)))
    common_gain = min(1., peak_limit / peak) if peak else 1.
    scaled_clean, scaled_noise = clean * common_gain, noise_component * common_gain
    mixed = (scaled_clean + scaled_noise).astype(np.asarray(audio).dtype)
    if not np.isfinite(mixed).all() or float(np.max(np.abs(mixed))) > peak_limit + 2e-7:
        raise ValueError('Finite/peak invariants failed after additive mixing')
    actual_snr = 20 * math.log10(rms(scaled_clean) / rms(scaled_noise))
    return mixed, {'applied': True, 'target_snr_db': float(snr_db),
                   'measured_component_snr_db': actual_snr, 'noise_gain': noise_gain,
                   'common_gain': common_gain, 'common_scaled_for_peak': common_gain < 1,
                   'peak_limit': peak_limit, 'output_peak': float(np.max(np.abs(mixed))),
                   'snr_definition': 'full-waveform RMS of separate scaled clean/noise components'}


@dataclass(frozen=True)
class AugmentationConfig:
    condition: str = 'A0'
    seed: int = 42
    spec_probability: float = .5
    noise_probability: float = .3
    time_mask_max_frames: int = 30
    time_mask_max_fraction: float = .05
    frequency_mask_max_bins: int = 8
    snr_min_db: float = 15.
    snr_max_db: float = 30.
    noise_kind: str = 'white'
    peak_limit: float = .99

    def __post_init__(self):
        if not isinstance(self.seed, int) or not isinstance(self.time_mask_max_frames, int) or not isinstance(self.frequency_mask_max_bins, int):
            raise ValueError('Seed and mask widths must be integers')
        if self.condition not in ('A0', 'A1', 'A2', 'A3'):
            raise ValueError('Condition must be A0/A1/A2/A3')
        if not 0 <= self.spec_probability <= 1 or not 0 <= self.noise_probability <= 1:
            raise ValueError('Probabilities must be in [0,1]')
        if not 0 <= self.time_mask_max_frames <= 30 or not 0 <= self.frequency_mask_max_bins <= 8:
            raise ValueError('Predeclared mask limits exceeded')
        if not 0 <= self.time_mask_max_fraction <= .05:
            raise ValueError('Time masking exceeds predeclared 5%')
        if not 15 <= self.snr_min_db <= self.snr_max_db <= 30:
            raise ValueError('Predeclared SNR range is 15 to 30dB')
        if self.noise_kind not in ('white', 'pink') or not 0 < self.peak_limit <= 1:
            raise ValueError('Invalid noise kind/peak limit')

    def receipt(self):
        return {**asdict(self), 'version': VERSION,
                'source': f'synthetic_{self.noise_kind}_v1',
                'natural_noise_claim': False,
                'spec_fill': 'scalar minimum over valid mel region; padding untouched',
                'rng': 'PCG64 keyed SHA256(seed,epoch,sample_key,branch); no global RNG',
                'pitch_or_speed_transform': False, 'training_only': True}


class SpeechAugmentor:
    def __init__(self, config):
        self.config = config

    def waveform(self, audio, sample_key, epoch, training=True, noise=None, noise_receipt=None):
        values = _waveform(audio)
        original = np.asarray(audio)
        config = self.config
        meta = {'applied': False, 'version': VERSION, 'role': 'train' if training else 'eval',
                'epoch': epoch, 'condition': config.condition}
        if not training or config.condition not in ('A2', 'A3'):
            return original.copy(), {**meta, 'reason': 'evaluation_no_augmentation' if not training else 'noise_disabled'}
        if stable_rng(config.seed, epoch, sample_key, 'noise_trigger').random() >= config.noise_probability:
            return original.copy(), {**meta, 'reason': 'probability_not_triggered'}
        if noise is None:
            source = synthetic_noise(len(values), stable_rng(config.seed, epoch, sample_key, 'noise_samples'), config.noise_kind)
            source_receipt = {'origin': f'synthetic_{config.noise_kind}_v1', 'role': 'train',
                              'sha256': waveform_sha256(source), 'natural_noise_claim': False}
        else:
            source = _waveform(noise)
            if not noise_receipt or noise_receipt.get('role') != 'train' or not noise_receipt.get('origin'):
                raise ValueError('Supplied noise requires explicit train-only origin receipt')
            if waveform_sha256(source) != noise_receipt.get('sha256'):
                raise ValueError('Supplied noise waveform hash mismatch')
            source_receipt = dict(noise_receipt)
            offset = stable_rng(config.seed, epoch, sample_key, 'noise_offset').integers(0, len(source))
            source = source[(np.arange(len(values)) + offset) % len(source)]
        snr = float(stable_rng(config.seed, epoch, sample_key, 'noise_snr').uniform(config.snr_min_db, config.snr_max_db))
        result, mixed_meta = mix_at_snr(original, source, snr, config.peak_limit)
        return result, {**meta, **mixed_meta, 'source': source_receipt,
                        'sample_rate': 16000, 'length_unchanged': len(result) == len(original),
                        'pitch_or_speed_transform': False}

    def features(self, features, valid_frames, sample_key, epoch, training=True):
        values = np.asarray(features)
        if values.ndim != 2 or not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
            raise ValueError('Mel features must be finite floating point [mel,time] array')
        if not isinstance(valid_frames, int) or not 1 <= valid_frames <= values.shape[1] or values.shape[0] < 2:
            raise ValueError('Valid frame count outside features or mel bins < 2')
        result = values.copy()
        config = self.config
        meta = {'applied': False, 'version': VERSION, 'valid_frames': valid_frames,
                'time_mask': None, 'frequency_mask': None, 'padding_untouched': True}
        if not training or config.condition not in ('A1', 'A3'):
            return result, {**meta, 'reason': 'evaluation_no_augmentation' if not training else 'spec_disabled'}
        if stable_rng(config.seed, epoch, sample_key, 'spec_trigger').random() >= config.spec_probability:
            return result, {**meta, 'reason': 'probability_not_triggered'}
        # Short clips may receive no time mask: never mask >=5%, one full clip,
        # or padded frames. Frequency mask is independent, valid region only.
        max_time = min(config.time_mask_max_frames, math.floor(valid_frames * config.time_mask_max_fraction), valid_frames - 1)
        max_frequency = min(config.frequency_mask_max_bins, values.shape[0] - 1)
        fill = float(np.min(values[:, :valid_frames]))
        if max_time:
            rng = stable_rng(config.seed, epoch, sample_key, 'spec_time')
            width = int(rng.integers(1, max_time + 1))
            start = int(rng.integers(0, valid_frames - width + 1))
            result[:, start:start + width] = fill
            meta['time_mask'] = {'start': start, 'width': width}
        if max_frequency:
            rng = stable_rng(config.seed, epoch, sample_key, 'spec_frequency')
            width = int(rng.integers(1, max_frequency + 1))
            start = int(rng.integers(0, values.shape[0] - width + 1))
            result[start:start + width, :valid_frames] = fill
            meta['frequency_mask'] = {'start': start, 'width': width}
        return result, {**meta, 'applied': bool(max_time or max_frequency),
                        'fill_value': fill, 'fill_policy': 'minimum_in_valid_mel_region'}

    def record_waveform(self, record, epoch, training=True):
        """Optional adapter for in-memory records; no reference/text mutation."""
        if not isinstance(record, Mapping) or 'waveform' not in record or not record.get('sha256'):
            raise ValueError('In-memory record needs waveform and stable audio SHA256')
        output = dict(record)
        output['waveform'], meta = self.waveform(record['waveform'], record['sha256'], epoch, training)
        output['augmentation_receipt'] = meta
        return output
