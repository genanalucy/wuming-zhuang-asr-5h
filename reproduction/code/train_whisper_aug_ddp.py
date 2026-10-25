#!/usr/bin/env python3
"""Offline, fully supervised Whisper baseline for languages without a language ID.

All CER/WER values are fractions. No test manifest is consumed. Sanity mode is a
deliberate training-set overfit check, never a held-out accuracy claim.
"""
import argparse
from contextlib import nullcontext
import copy
import gc
from datetime import timedelta
import hashlib
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from speech_augmentation import AugmentationConfig, SharedEpoch, SpeechAugmentor
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from common import (
    append_jsonl as common_append_jsonl, load_audio_16k, load_manifest, normalize_text, score_pairs,
    seed_everything, write_json as common_write_json,
)


class SpeechDataset(Dataset):
    def __init__(self, rows, processor, label_prefix, max_tokens):
        self.rows = rows
        self.processor = processor
        self.label_prefix = label_prefix
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        audio = np.asarray(load_audio_16k(row), dtype=np.float32)
        if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
            raise ValueError(f"Invalid decoded waveform: {row['audio']}")
        if len(audio) > 30 * 16000:
            raise ValueError(f"Actual duration exceeds 30s, refusing truncation: {row['audio']}")
        text = normalize_text(row['text'])
        ids = self.label_prefix + self.processor.tokenizer.encode(text, add_special_tokens=False)
        ids.append(self.processor.tokenizer.eos_token_id)
        if len(ids) > self.max_tokens:
            raise ValueError(f"Target exceeds {self.max_tokens} tokens: {row['audio']}")
        features = self.processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors='pt',
            padding='max_length', truncation=False,
        ).input_features[0]
        if features.shape != (self.processor.feature_extractor.feature_size, 3000):
            raise ValueError(f"Unexpected features {features.shape}: {row['audio']}")
        return {'input_features': features, 'labels': ids, 'row': row, 'text': text}


def valid_frames_from_attention_mask(mask, feature_frames, hop_length, actual_samples):
    """Validate the complete same-call mask before any sample-to-frame slicing."""
    values = np.asarray(mask)
    if values.ndim != 1 or hop_length < 1 or actual_samples < 1:
        raise ValueError('Expected one-dimensional mask, positive hop/sample lengths')
    try:
        finite = bool(np.isfinite(values).all())
    except TypeError as exc:
        raise ValueError('Attention mask must contain finite numeric values') from exc
    if not finite or not np.isin(values, [0, 1]).all():
        raise ValueError('Attention mask must be finite and binary')
    original_count = int(values.sum())
    if original_count < 1 or not np.array_equal(values.astype(bool), np.arange(len(values)) < original_count):
        raise ValueError('Full attention mask must be a nonempty contiguous valid prefix')
    if len(values) == feature_frames * hop_length:
        if original_count != actual_samples:
            raise ValueError('Sample-level mask does not match actual waveform length')
        values = values[::hop_length]
    elif len(values) != feature_frames:
        raise ValueError('Attention mask length is neither exact mel nor sample length')
    count = int(values.sum())
    expected = (actual_samples + hop_length - 1) // hop_length
    if count != expected or expected > feature_frames:
        raise ValueError('Attention mask valid frames disagree with actual waveform length')
    return count


class AugmentedSpeechDataset(SpeechDataset):
    """Training only; references and metadata remain identical to SpeechDataset."""
    def __init__(self, rows, processor, label_prefix, max_tokens, augmentor, epoch_state):
        super().__init__(rows, processor, label_prefix, max_tokens)
        self.augmentor = augmentor
        self.epoch_state = epoch_state

    def __getitem__(self, index):
        row = self.rows[index]
        sample_key = row.get('sha256')
        if not sample_key:
            raise ValueError('Augmentation requires stable source audio SHA256')
        epoch = self.epoch_state.get()
        original = np.asarray(load_audio_16k(row), dtype=np.float32)
        if original.ndim != 1 or not len(original) or len(original) > 480000 or not np.isfinite(original).all():
            raise ValueError('Invalid/over-30-second actual waveform; no truncation allowed')
        waveform, wave_receipt = self.augmentor.waveform(original, sample_key, epoch, training=True)
        if waveform.shape != original.shape or waveform.dtype != original.dtype:
            raise ValueError('Augmentation changed waveform shape/dtype')
        extraction = self.processor.feature_extractor(waveform, sampling_rate=16000,
            return_tensors='pt', padding='max_length', truncation=False, return_attention_mask=True)
        features = extraction.input_features[0]
        mask = extraction.attention_mask[0]
        if features.ndim != 2 or tuple(features.shape) != (self.processor.feature_extractor.feature_size, 3000):
            raise ValueError('Unexpected augmented mel feature shape')
        valid = valid_frames_from_attention_mask(mask.cpu().numpy(), features.shape[1],
            int(self.processor.feature_extractor.hop_length), len(waveform))
        transformed, feature_receipt = self.augmentor.features(features.cpu().numpy(), valid,
            sample_key, epoch, training=True)
        text = normalize_text(row['text'])
        labels = self.label_prefix + self.processor.tokenizer.encode(text, add_special_tokens=False)
        labels.append(self.processor.tokenizer.eos_token_id)
        if len(labels) > self.max_tokens:
            raise ValueError('Augmented item target too long; no truncation allowed')
        return {'input_features': torch.from_numpy(transformed), 'labels': labels,
            'row': row, 'text': text, 'augmentation': {'epoch': epoch, 'valid_frames': valid,
                'waveform': wave_receipt, 'features': feature_receipt}}


def rebuild_training_iterator(loader, sampler, epoch_state, epoch, consumed_batches):
    """Recreate a deterministic sampler epoch and skip only consumed local batches.

    Creating/priming DataLoader workers can consume global RNG. A resume must
    preserve the checkpoint's model RNG across this replay operation. Augmentations
    are stateless and do not rely on worker RNG, IDs or prefetch order.
    """
    if epoch < 0 or consumed_batches < 0 or consumed_batches > len(loader):
        raise ValueError('Invalid saved sampler epoch/local batch cursor')
    epoch_state.set(epoch)
    sampler.set_epoch(epoch)
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        iterator = iter(loader)
        for _ in range(consumed_batches):
            next(iterator)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    return iterator


def pause_at_step(step, stop_after_step, max_steps):
    return stop_after_step is not None and step == stop_after_step and step < max_steps


def training_settings_identity(args):
    return {
        'workers': args.workers,
        'eval_batch_size': args.eval_batch_size or args.batch_size,
        'eval_every': args.eval_every,
        'gradient_checkpointing': args.gradient_checkpointing,
        'warmup_steps': args.warmup_steps,
        'effective_warmup_steps': args.warmup_steps if args.warmup_steps is not None else max(1, int(args.max_steps * .05)),
    }


def augmentation_run_spec(args, world_size):
    pretrained = Path(args.model_dir).resolve()
    if pretrained.name != 'whisper-large-v3' or pretrained.parent.name != 'model_cache':
        raise ValueError('Augmentation runs must initialize from audited public model_cache/whisper-large-v3, never a trained A0 checkpoint')
    effective = 'A0' if args.mode == 'sanity' else args.augmentation
    config = AugmentationConfig(condition=effective, seed=args.seed, noise_kind='white')
    receipt = config.receipt()
    receipt.update(method_settings=training_settings_identity(args),
        initialization_policy='audited public model_cache/whisper-large-v3; M0 weights never reused; resume only own exact-method checkpoint',
        attention_mask_policy='same extraction call; binary prefix; verify exact ceil(actual_samples/hop_length)',
        sampler_resume_policy='same world/batch/accum and manifest; saved epoch plus consumed local microbatches; skip replay under RNG guard',
        model_rng_policy='legacy v7 normal DataLoader RNG consumption retained; resume rebuild restores checkpoint model RNG',
        resume_limitations='stateless augmentation/cursor replay verified; no claim of bitwise CUDA/kernel/dropout equivalence',
        utility_sha256=hashlib.sha256(Path(__file__).with_name('speech_augmentation.py').read_bytes()).hexdigest(),
        trainer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    digest = hashlib.sha256(json.dumps(receipt, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if world_size != 3 or args.batch_size != 2 or args.grad_accum != 3:
        raise ValueError('Controlled augmentation recipe requires three ranks, batch2, accumulation3 (global18)')
    if args.mode in ('sanity', 'train'):
        tier = os.environ.get('ASR_DATA_EFFICIENCY_TIER')
        tier_budget = {
            '5h': {'max_steps': 1149, 'full_eval_every': 144, 'min_steps': 431,
                   'warmup_steps': 57, 'eval_every': 46,
                   'manifest_sha256': '92e89a678a019ce19df0f04c3a6ce6d6abff512ee3f13752df403786bb124cb3'},
            '20h': {'max_steps': 4623, 'full_eval_every': 578, 'min_steps': 1734,
                    'warmup_steps': 231, 'eval_every': 186,
                    'manifest_sha256': 'e4175c444f13663af1b0cf8e2b7a0ae746caa1baa7aace22b0867bdd36287f2c'},
        }
        if tier not in tier_budget:
            raise ValueError('Formal data-efficiency trainer requires an approved tier')
        tier_values = tier_budget[tier]
        actual_manifest_sha = hashlib.sha256(Path(args.train_manifest).read_bytes()).hexdigest()
        if actual_manifest_sha != os.environ.get('WZ5H_PUBLIC_MANIFEST_SHA256'):
            raise ValueError('Training manifest does not match the selected frozen tier')
    if args.mode == 'train':
        expected = {'max_steps': tier_values['max_steps'], 'lr': 1e-5, 'seed': 42,
                    'full_eval_every': tier_values['full_eval_every'],
                    'min_steps': tier_values['min_steps'], 'early_stopping_patience': 3,
                    'warmup_steps': tier_values['warmup_steps'], 'eval_every': tier_values['eval_every']}
        for key, value in expected.items():
            if getattr(args, key) != value:
                raise ValueError(f'Data-efficiency recipe mismatch: {key}')
        if args.augmentation != 'A0':
            raise ValueError('Data-efficiency primary curve only permits A0')
        if not args.full_dev_manifest:
            raise ValueError('Controlled augmentation requires complete development checkpoint selection')
    if args.stop_after_step is not None:
        if args.mode != 'train' or not 0 < args.stop_after_step <= args.max_steps:
            raise ValueError('Pilot stop must be in (0,max_steps] and train mode only')
        if not args.full_eval_every or args.stop_after_step % args.full_eval_every:
            raise ValueError('Pilot stop must be a complete full-development evaluation boundary')
    return config, receipt, digest


def collate(items):
    length = max(len(item['labels']) for item in items)
    labels = torch.full((len(items), length), -100, dtype=torch.long)
    for index, item in enumerate(items):
        labels[index, :len(item['labels'])] = torch.tensor(item['labels'])
    return {
        'input_features': torch.stack([item['input_features'] for item in items]),
        'labels': labels, 'rows': [item['row'] for item in items],
        'refs': [item['text'] for item in items],
    }


def filtered_rows(path, processor, label_prefix, max_tokens, log_path):
    accepted, rejected = [], 0
    for row in load_manifest(path):
        reason = None
        duration = float(row.get('duration', 0))
        text = normalize_text(row.get('text', ''))
        if not math.isfinite(duration) or duration <= 0:
            reason = 'invalid_duration'
        elif duration > 30:
            reason = 'over_30s_no_truncation'
        elif not text:
            reason = 'empty_normalized_text'
        elif len(label_prefix) + len(processor.tokenizer.encode(text, add_special_tokens=False)) + 1 > max_tokens:
            reason = 'target_too_long_no_truncation'
        if reason:
            rejected += 1
            append_jsonl(log_path, {'audio': row.get('audio'), 'reason': reason, 'manifest': str(path)})
        else:
            accepted.append(row)
    if not accepted:
        raise RuntimeError(f"No eligible rows in {path}")
    return accepted, rejected


def manifest_digest(rows):
    payload = '\n'.join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def make_generation_config(model):
    config = copy.deepcopy(model.generation_config)
    # Transformers 4.46.3 otherwise detects a language even when language=None.
    # This affects generation routing only, not model weights/config or tokenizer.
    config.is_multilingual = False
    config.language = None
    config.task = None
    config.forced_decoder_ids = None
    # In Transformers 4.46.3 the presence of lang_to_id alone triggers detection.
    if hasattr(config, 'lang_to_id'):
        delattr(config, 'lang_to_id')
    config.return_timestamps = False
    config.do_sample = False
    config.num_beams = 1
    config.use_cache = True
    return config


def forbid_language_detection(*args, **kwargs):
    raise RuntimeError('Language detection called: explicit no-language-token contract violated')


def distributed_ready():
    return dist.is_available() and dist.is_initialized()


def rank_zero():
    return dist.get_rank() == 0 if distributed_ready() else int(os.environ.get('RANK', '0')) == 0


def write_json(path, obj):
    if rank_zero():
        common_write_json(path, obj)


def append_jsonl(path, obj):
    if rank_zero():
        common_append_jsonl(path, obj)


def barrier():
    if distributed_ready():
        dist.barrier()


def unwrap(model):
    return model.module if isinstance(model, DDP) else model


def broadcast_object(value):
    if distributed_ready():
        objects = [value if rank_zero() else None]
        dist.broadcast_object_list(objects, src=0)
        return objects[0]
    return value


def require_all_finite(value, device, message):
    valid = torch.tensor(int(bool(torch.isfinite(value).all())), device=device)
    if distributed_ready():
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    if not valid.item():
        raise RuntimeError(message)


def shard_indices(size, rank, world):
    # Unlike DistributedSampler, never pads or repeats development examples.
    return list(range(rank, size, world))


def merge_eval_payloads(payloads, expected_audio):
    records = [record for payload in payloads for record in payload['records']]
    order = {audio: index for index, audio in enumerate(expected_audio)}
    audio = [record['audio'] for record in records]
    if len(order) != len(expected_audio) or len(audio) != len(set(audio)) or set(audio) != set(order):
        raise RuntimeError('Development shard coverage/uniqueness mismatch')
    records.sort(key=lambda record: order[record['audio']])
    denominator = sum(payload['tokens'] for payload in payloads)
    if denominator <= 0:
        raise RuntimeError('Empty evaluation token denominator')
    metrics = score_pairs([r['prediction'] for r in records], [r['reference'] for r in records])
    metrics.update(teacher_forced_loss=sum(p['loss_sum'] for p in payloads) / denominator,
                   utterances=len(records), coverage_verified=True)
    return records, metrics


def evaluate_synced(model, loader, processor, prefix, generation_config, device, output_path):
    # Update gradients are no longer needed after optimizer.step; keep only DDP
    # bucket storage, then release allocator fragments before inference.
    for parameter in unwrap(model).parameters():
        parameter.grad = None
    gc.collect()
    torch.cuda.empty_cache()
    # Every rank performs its independent, unwrapped forward; unequal shard
    # lengths cannot trigger DDP forward collectives. Only the gather is collective.
    model = unwrap(model)
    model.eval()
    records, loss_sum, token_count, error = [], 0., 0, None
    try:
        for batch in loader:
            features = batch['input_features'].to(device)
            labels = batch['labels'].to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
                loss = model(input_features=features, labels=labels).loss
                decoder_ids = torch.tensor(prefix, device=device).unsqueeze(0).expand(len(features), -1)
                tokens = model.generate(input_features=features, decoder_input_ids=decoder_ids,
                    generation_config=generation_config,
                    max_new_tokens=model.config.max_target_positions - len(prefix))
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite evaluation loss')
            count = int((labels != -100).sum())
            loss_sum += float(loss) * count
            token_count += count
            hypotheses = processor.tokenizer.batch_decode(tokens.cpu(), skip_special_tokens=True)
            for row, ref, hyp in zip(batch['rows'], batch['refs'], hypotheses):
                hyp = normalize_text(hyp)
                record = {'audio': row['audio'], 'sha256': row.get('sha256'),
                    'speaker_candidate': row.get('speaker_candidate'), 'duration': row.get('duration'),
                    'reference': ref, 'prediction': hyp}
                record.update(score_pairs([hyp], [ref]))
                records.append(record)
    except Exception as exc:
        error = str(exc)
    payload = {'records': records, 'loss_sum': loss_sum, 'tokens': token_count, 'error': error}
    payloads = [None] * dist.get_world_size() if distributed_ready() else [payload]
    if distributed_ready():
        dist.all_gather_object(payloads, payload)
    errors = [p['error'] for p in payloads if p['error']]
    if errors:
        raise RuntimeError('Evaluation shard failed: ' + '; '.join(errors))
    result = None
    merge_error = None
    if rank_zero():
        try:
            dataset = loader.dataset.dataset if isinstance(loader.dataset, Subset) else loader.dataset
            merged, result = merge_eval_payloads(payloads, [r['audio'] for r in dataset.rows])
            Path(output_path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in merged), encoding='utf-8')
            write_json(Path(output_path).with_suffix('.metrics.json'), result)
        except Exception as exc:
            merge_error = str(exc)
    merge_error, result = broadcast_object((merge_error, result))
    if merge_error:
        raise RuntimeError(merge_error)
    return result


def rng_state():
    return {'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
            'python_rng': random.getstate(), 'numpy_rng': np.random.get_state()}


def validate_resume_contract(source, target):
    # Single-GPU to DDP migration preserves effective examples/update and all
    # scientific/scheduler settings, but does not claim bitwise shuffle equivalence.
    # Cursor replay is only valid for the exact sampler partition/microbatch.
    ignored = set()
    for key in set(source) | set(target):
        if key not in ignored and source.get(key) != target.get(key):
            raise RuntimeError(f'Checkpoint contract mismatch: {key}')
    source_batch = source['batch_size'] * source['grad_accum'] * source.get('world_size', 1)
    target_batch = target['batch_size'] * target['grad_accum'] * target.get('world_size', 1)
    if source_batch != target_batch:
        raise RuntimeError(f'Global batch conversion mismatch: {source_batch} != {target_batch}')


def save_checkpoint(path, model, optimizer, scheduler, step, best_cer, contract, training_state=None):
    local_rng = rng_state()
    ranks = [None] * dist.get_world_size() if distributed_ready() else [local_rng]
    if distributed_ready():
        dist.all_gather_object(ranks, local_rng)
    error = None
    if rank_zero():
        path = Path(path)
        temporary = None
        try:
            state = {'model': unwrap(model).state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'step': step, 'best_cer': best_cer,
                'contract': contract, 'rank_rng_states': ranks,
                'training_state': training_state or {}}
            fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
            os.close(fd)
            torch.save(state, temporary)
            os.replace(temporary, path)
        except Exception as exc:
            error = str(exc)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
    error = broadcast_object(error)
    if error:
        raise RuntimeError('Checkpoint save failed: ' + error)
    barrier()


def restore_checkpoint(path, model, optimizer, scheduler, contract, return_tracking=False):
    barrier()
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    validate_resume_contract(state['contract'], contract)
    weights = state['model']
    if weights and all(key.startswith('module.') for key in weights):
        weights = {key[7:]: value for key, value in weights.items()}
    unwrap(model).load_state_dict(weights, strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    current_world = dist.get_world_size() if distributed_ready() else 1
    old_world = state['contract'].get('world_size', 1)
    if old_world == current_world and 'rank_rng_states' in state:
        rank = dist.get_rank() if distributed_ready() else 0
        rng = state['rank_rng_states'][rank]
        torch.set_rng_state(rng['torch_rng'])
        torch.cuda.set_rng_state(rng['cuda_rng'])
        random.setstate(rng['python_rng'])
        np.random.set_state(rng['numpy_rng'])
    else:
        rank = dist.get_rank() if distributed_ready() else 0
        seed_everything(contract['seed'] + int(state['step']) * 101 + rank)
    tracking = state.get('training_state', {}).copy()
    tracking['resume_shuffle_bitwise'] = False
    tracking['source_world_size'] = old_world
    result = (int(state['step']), float(state['best_cer']))
    # Optimizer.load_state_dict constructs new CUDA state before replacing old
    # state. Release checkpoint mapping aliases and allocator fragments after all
    # result/tracking/RNG information has been copied, before inference starts.
    del weights, state
    gc.collect()
    torch.cuda.empty_cache()
    barrier()
    return (*result, tracking) if return_tracking else result


@torch.no_grad()
def restore_witness(model, batch, device):
    model = unwrap(model)
    model.eval()
    # A small actual forward witness, evaluated in float32 to avoid BF16 roundoff.
    return model(
        input_features=batch['input_features'][:1].to(device),
        labels=batch['labels'][:1].to(device),
    ).logits.detach().cpu()


def run(args):
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world_size > 1:
        if not dist.is_available():
            raise RuntimeError('torch.distributed is unavailable')
        torch.cuda.set_device(local_rank)
        # NCCL must see the process-local CUDA device before communicator init.
        # Initializing first can select device 0 for every rank and segfault on
        # this host during ncclCommInitRank.
        dist.init_process_group(backend='nccl', timeout=timedelta(hours=2))
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This controlled recipe requires CUDA with BF16 support')
    if args.max_steps < 2 or args.batch_size < 1 or args.grad_accum < 1:
        raise ValueError('max-steps >= 2, batch-size >= 1, grad-accum >= 1 required')
    augmentation_config, augmentation_receipt, augmentation_hash = augmentation_run_spec(args, world_size)
    seed_everything(args.seed)
    device = torch.device('cuda', local_rank)
    output = Path(args.output)
    if rank_zero():
        output.mkdir(parents=True, exist_ok=True)
    barrier()
    if (output / 'run_config.json').exists() and not args.resume:
        raise RuntimeError('Output already contains a run; choose a fresh directory or --resume')
    write_json(output / 'status.json', {'state': 'preparing', 'mode': args.mode})
    processor = WhisperProcessor.from_pretrained(args.model_dir, local_files_only=True)
    processor.tokenizer.language = None
    processor.tokenizer.set_prefix_tokens(language=None, task='transcribe', predict_timestamps=False)
    prefix = list(processor.tokenizer.prefix_tokens)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir, local_files_only=True)
    if len(prefix) != 3 or prefix[0] != model.config.decoder_start_token_id:
        raise RuntimeError(f'Unexpected language-free prefix: {prefix}')
    expected = [
        model.config.decoder_start_token_id,
        processor.tokenizer.convert_tokens_to_ids('<|transcribe|>'),
        processor.tokenizer.convert_tokens_to_ids('<|notimestamps|>'),
    ]
    if prefix != expected:
        raise RuntimeError(f'Prefix includes unexpected token: {prefix} != {expected}')
    model.config.use_cache = False
    # Explicit no-augmentation control for the next accuracy experiment.
    model.config.apply_spec_augment = False
    model.config.forced_decoder_ids = None
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    generation_config = make_generation_config(model)
    # Fail loudly if a future Transformers implementation reintroduces detection.
    model.detect_language = forbid_language_detection
    if world_size > 1:
        model.to(device)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False, gradient_as_bucket_view=True)
    max_tokens = int(unwrap(model).config.max_target_positions)
    train_rows, train_rejected = filtered_rows(
        args.train_manifest, processor, prefix[1:], max_tokens, output / 'excluded.jsonl',
    )
    full_rows = None
    if args.mode == 'sanity':
        if args.skip_initial_eval or args.full_dev_manifest or args.early_stopping_patience:
            raise ValueError('Sanity mode requires its original training-overfit gate')
        selected, seen = [], set()
        for row in train_rows:
            key = row.get('sha256') or row['audio']
            if float(row['duration']) <= 10 and key not in seen:
                selected.append(row)
                seen.add(key)
                if len(selected) == 16:
                    break
        if len(selected) < 16:
            raise RuntimeError('Sanity mode requires 16 distinct audio samples <= 10s')
        train_rows, eval_rows = selected, selected
        dev_rejected = 0
    else:
        if not args.dev_manifest:
            raise ValueError('--dev-manifest required in train mode')
        eval_rows, dev_rejected = filtered_rows(
            args.dev_manifest, processor, prefix[1:], max_tokens, output / 'excluded.jsonl',
        )
        # Deterministic subset spanning the manifest instead of just the first speaker.
        indices = np.linspace(0, len(eval_rows) - 1, min(args.eval_limit, len(eval_rows)), dtype=int)
        eval_rows = [eval_rows[index] for index in indices]
        train_hashes = {row.get('sha256') or row['audio'] for row in train_rows}
        if any((row.get('sha256') or row['audio']) in train_hashes for row in eval_rows):
            raise RuntimeError('Train/evaluation audio overlap')
        train_speakers = {row.get('speaker_candidate') for row in train_rows}
        if None in train_speakers or any(row.get('speaker_candidate') in train_speakers for row in eval_rows):
            raise RuntimeError('Missing speaker identity or train/development speaker overlap')
        if args.full_dev_manifest:
            full_rows, full_rejected = filtered_rows(
                args.full_dev_manifest, processor, prefix[1:], max_tokens, output / 'excluded.jsonl',
            )
            if any((row.get('sha256') or row['audio']) in train_hashes for row in full_rows):
                raise RuntimeError('Train/full-development audio overlap')
            if any(row.get('speaker_candidate') is None or row.get('speaker_candidate') in train_speakers for row in full_rows):
                raise RuntimeError('Missing speaker identity or train/full-development speaker overlap')
            full_keys = {row.get('sha256') or row['audio'] for row in full_rows}
            if not {(row.get('sha256') or row['audio']) for row in eval_rows} <= full_keys:
                raise RuntimeError('Monitoring examples must be contained in full development')
            if args.full_eval_every is None:
                args.full_eval_every = math.ceil(len(train_rows) / (world_size * args.batch_size * args.grad_accum))
    for name, rows in [('train_used', train_rows), ('eval_used', eval_rows)] if rank_zero() else []:
        (output / f'{name}.jsonl').write_text(
            ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8',
        )
    if full_rows is not None and rank_zero():
        (output / 'full_eval_used.jsonl').write_text(
            ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in full_rows), encoding='utf-8',
        )
    contract = {
        'model_dir': str(Path(args.model_dir).resolve()), 'mode': args.mode,
        'train_sha256': manifest_digest(train_rows), 'eval_sha256': manifest_digest(eval_rows),
        'seed': args.seed, 'lr': args.lr, 'max_steps': args.max_steps,
        'batch_size': args.batch_size, 'grad_accum': args.grad_accum,
        'world_size': world_size, 'global_batch_size': world_size * args.batch_size * args.grad_accum,
        'gradient_as_bucket_view': True, 'evaluation_clear_gradients': True,
        'augmentation': augmentation_receipt, 'augmentation_sha256': augmentation_hash,
        **augmentation_receipt['method_settings'],
        'warmup_steps': args.warmup_steps, 'weight_decay': args.weight_decay,
        'max_grad_norm': args.max_grad_norm,
        'full_eval_sha256': manifest_digest(full_rows) if full_rows is not None else None,
        'full_eval_every': args.full_eval_every,
        'early_stopping_patience': args.early_stopping_patience,
        'min_steps': args.min_steps,
    }
    config = dict(vars(args))
    config.update({
        'contract': contract, 'language_free_prefix_ids': prefix,
        'augmentation_requested': args.augmentation, 'augmentation_effective': augmentation_config.condition,
        'augmentation_sha256': augmentation_hash,
        'augmentation_scope': 'M0 forces A0; training dataset only; all evaluation datasets clean',
        'resume_initial_eval_policy': 'skip extra initial evaluation on resume to preserve checkpoint model RNG',
        'language_free_prefix_tokens': processor.tokenizer.convert_ids_to_tokens(prefix),
        'generation_is_multilingual_override': False,
        'evaluation_use_cache': True, 'gradient_as_bucket_view': True,
        'evaluation_clear_gradients': True, 'evaluation_optimizer_offload': False,
        'restore_allocator_cleanup': 'delete checkpoint aliases, gc, CUDA empty_cache before barrier',
        'model_is_still_multilingual': True, 'detect_language_forbidden': True, 'apply_spec_augment': False,
        'train_utterances': len(train_rows), 'eval_utterances': len(eval_rows),
        'train_hours': sum(float(row['duration']) for row in train_rows) / 3600,
        'excluded_train': train_rejected, 'excluded_dev': dev_rejected,
        'full_eval_utterances': len(full_rows) if full_rows is not None else None,
        'excluded_full_dev': full_rejected if full_rows is not None else None,
        'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
        'evaluation_scope': 'training_overfit' if args.mode == 'sanity' else 'fixed_development_subset',
        'checkpoint_selection_scope': 'complete_filtered_development' if full_rows is not None else 'fixed_development_subset',
        'metric_units': 'fraction',
    })
    write_json(output / 'run_config.json', config)
    epoch_state = SharedEpoch(0)
    train_data = AugmentedSpeechDataset(train_rows, processor, prefix[1:], max_tokens,
        SpeechAugmentor(augmentation_config), epoch_state)
    eval_data = SpeechDataset(eval_rows, processor, prefix[1:], max_tokens)
    loader_options = {'num_workers': args.workers}
    if args.workers:
        loader_options.update(multiprocessing_context='spawn', persistent_workers=True)
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=dist.get_rank(), shuffle=True, seed=args.seed, drop_last=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=train_sampler is None, sampler=train_sampler, drop_last=world_size > 1, collate_fn=collate, **loader_options)
    if train_sampler is not None:
        train_sampler.set_epoch(0)
    epoch = 0
    local_batches = 0
    if not len(train_loader):
        raise RuntimeError('Training set too small for DDP batch')
    write_json(output / 'sampler_audit.json', {
        'policy': 'shuffle_each_epoch_drop_tail_no_padding',
        'examples_per_epoch': len(train_loader) * args.batch_size * world_size,
        'dropped_per_epoch': len(train_data) - len(train_loader) * args.batch_size * world_size,
        'global_batch_size': world_size * args.batch_size * args.grad_accum,
        'resume_shuffle_bitwise': False,
        'resume_cursor_exact': True, 'augmentation_replay_stateless': True,
        'loader_model_rng_policy': augmentation_receipt['model_rng_policy'],
    })
    eval_batch_size = args.eval_batch_size or args.batch_size
    rank = dist.get_rank() if distributed_ready() else 0
    eval_data = Subset(eval_data, shard_indices(len(eval_data), rank, world_size))
    eval_loader = DataLoader(eval_data, batch_size=eval_batch_size, shuffle=False, collate_fn=collate, **loader_options)
    full_loader = None
    if full_rows is not None:
        full_loader = DataLoader(
            Subset(SpeechDataset(full_rows, processor, prefix[1:], max_tokens),
                   shard_indices(len(full_rows), rank, world_size)),
            batch_size=eval_batch_size, shuffle=False, collate_fn=collate, **loader_options,
        )
    if world_size == 1:
        model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = args.warmup_steps if args.warmup_steps is not None else max(1, int(args.max_steps * .05))

    def multiplier(step):
        if step < warmup:
            return max(step, 1) / warmup
        return max(0., (args.max_steps - step) / max(1, args.max_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    step, best_cer = 0, float('inf')
    resume_tracking = {}
    if args.resume:
        step, best_cer, resume_tracking = restore_checkpoint(args.resume, model, optimizer, scheduler, contract, return_tracking=True)
        if math.isfinite(best_cer) and not (output / 'best' / 'config.json').exists():
            raise RuntimeError('Resume requires copied source best checkpoint history')
        if 'sampler_epoch' not in resume_tracking or 'local_batches' not in resume_tracking:
            raise RuntimeError('Augmentation resume requires saved sampler epoch and local batch cursor')
        epoch = int(resume_tracking['sampler_epoch'])
        local_batches = int(resume_tracking['local_batches'])
        if args.stop_after_step is not None and args.stop_after_step <= step:
            raise ValueError('Remove/increase stop-after-step to continue a paused pilot')
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
    initial = None
    if not args.skip_initial_eval and not args.resume:
        initial = evaluate_synced(model, eval_loader, processor, prefix, generation_config, device, output / 'initial_predictions.jsonl')
        append_jsonl(output / 'metrics.jsonl', {'step': step, 'phase': 'initial', **initial})
    iterator = (rebuild_training_iterator(train_loader, train_sampler, epoch_state, epoch, local_batches)
                if args.resume else iter(train_loader))
    start_time = time.monotonic()
    checkpoint_witness = None
    interval = min(50, args.eval_every) if args.mode == 'sanity' else args.eval_every
    rolling_loss, rolling_batches = 0., 0
    optimizer_seconds = float(resume_tracking.get('optimizer_seconds', 0.))
    stale_evaluations = int(resume_tracking.get('stale_evaluations', 0))
    stopped_early = False
    paused_for_review = False
    if args.resume and not resume_tracking and (output / 'early_stopping.json').exists():
        stop_state = json.loads((output / 'early_stopping.json').read_text())
        if stop_state['step'] == step:
            stale_evaluations = stop_state['stale_evaluations']
    for step in range(step + 1, args.max_steps + 1):
        update_started = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for microstep in range(args.grad_accum):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                local_batches = 0
                epoch_state.set(epoch)
                train_sampler.set_epoch(epoch)
                iterator = iter(train_loader)
                batch = next(iterator)
            local_batches += 1
            sync = model.no_sync() if isinstance(model, DDP) and microstep < args.grad_accum - 1 else nullcontext()
            with sync:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(input_features=batch['input_features'].to(device),
                                 labels=batch['labels'].to(device)).loss
                require_all_finite(loss, device, f'Non-finite train loss at step {step}')
                # Preserve the recipe's equally weighted microbatch loss objective.
                (loss / args.grad_accum).backward()
            rolling_loss += float(loss.detach())
            rolling_batches += 1
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        require_all_finite(grad_norm, device, f'Non-finite gradient norm at step {step}')
        optimizer.step()
        scheduler.step()
        optimizer_seconds += time.monotonic() - update_started
        if step % 100 == 0 or step == 1 or step == args.max_steps:
            loss_totals = torch.tensor([rolling_loss, rolling_batches], device=device, dtype=torch.float64)
            if distributed_ready():
                dist.all_reduce(loss_totals)
            record = {
                'step': step, 'phase': 'train', 'loss': float(loss_totals[0] / loss_totals[1]),
                'lr': scheduler.get_last_lr()[0], 'grad_norm': float(grad_norm),
                'elapsed_seconds': time.monotonic() - start_time,
                'max_cuda_memory_gb': torch.cuda.max_memory_allocated() / 2**30,
                'optimizer_seconds': optimizer_seconds,
                'approx_epochs': step * world_size * args.batch_size * args.grad_accum / len(train_rows),
            }
            append_jsonl(output / 'metrics.jsonl', record)
            if rank_zero(): print(json.dumps(record), flush=True)
            rolling_loss, rolling_batches = 0., 0
        if args.mode == 'sanity' and step == max(1, args.max_steps // 2):
            witness_batch = next(iter(eval_loader))
            before = restore_witness(model, witness_batch, device)
            checkpoint_path = output / 'resume_checkpoint.pt'
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, step, best_cer, contract)
            # Perturb a parameter to ensure the restore really reloads model weights.
            with torch.no_grad():
                next(model.parameters()).add_(0.01)
            restored_step, restored_best = restore_checkpoint(checkpoint_path, model, optimizer, scheduler, contract)
            after = restore_witness(model, witness_batch, device)
            difference_tensor = (before - after).abs().max().to(device)
            if distributed_ready():
                dist.all_reduce(difference_tensor, op=dist.ReduceOp.MAX)
            difference = float(difference_tensor)
            checkpoint_witness = {
                'restored_step': restored_step, 'max_logit_difference': difference,
                'passed': restored_step == step and difference <= 1e-5,
                'optimizer_state_entries': len(optimizer.state),
                'scheduler_epoch': scheduler.last_epoch,
            }
            if not checkpoint_witness['passed']:
                raise RuntimeError(f'Checkpoint restore failed: {checkpoint_witness}')
            best_cer = restored_best
            write_json(output / 'checkpoint_restore_witness.json', checkpoint_witness)
        paused_for_review = pause_at_step(step, args.stop_after_step, args.max_steps)
        if step % interval == 0 or step == args.max_steps or paused_for_review:
            metrics = evaluate_synced(
                model, eval_loader, processor, prefix, generation_config, device,
                output / f'predictions_step_{step}.jsonl',
            )
            append_jsonl(output / 'metrics.jsonl', {'step': step, 'phase': 'eval', **metrics})
            if rank_zero(): print(json.dumps({'step': step, 'phase': 'eval', **metrics}), flush=True)
            write_json(output / 'status.json', {'state': 'running', 'step': step, 'max_steps': args.max_steps})
        selected_metrics = None
        if full_loader is not None and (step % args.full_eval_every == 0 or step == args.max_steps or paused_for_review):
            write_json(output / 'status.json', {'state': 'evaluating_full_dev', 'step': step, 'max_steps': args.max_steps})
            selected_metrics = evaluate_synced(
                model, full_loader, processor, prefix, generation_config, device,
                output / f'full_dev_predictions_step_{step}.jsonl',
            )
            append_jsonl(output / 'metrics.jsonl', {'step': step, 'phase': 'full_eval', **selected_metrics})
            if rank_zero(): print(json.dumps({'step': step, 'phase': 'full_eval', **selected_metrics}), flush=True)
        elif full_loader is None and (step % interval == 0 or step == args.max_steps or paused_for_review):
            selected_metrics = metrics
        if selected_metrics is not None:
            improved = selected_metrics['cer'] < best_cer
            if improved:
                best_cer = selected_metrics['cer']
                stale_evaluations = 0
                if rank_zero():
                    unwrap(model).save_pretrained(output / 'best')
                    processor.save_pretrained(output / 'best')
                    write_json(output / 'best_metrics.json', {'step': step, 'scope': 'complete_filtered_development', **selected_metrics})
            else:
                stale_evaluations += 1
            write_json(output / 'early_stopping.json', {'step': step, 'stale_evaluations': stale_evaluations, 'best_cer': best_cer})
            if args.early_stopping_patience and step >= args.min_steps and stale_evaluations >= args.early_stopping_patience:
                stopped_early = True
        if args.mode == 'train' and (selected_metrics is not None or step % interval == 0 or paused_for_review):
            save_checkpoint(output / 'resume_checkpoint.pt', unwrap(model), optimizer, scheduler, step, best_cer, contract, {
                'stale_evaluations': stale_evaluations, 'optimizer_seconds': optimizer_seconds,
                'sampler_epoch': epoch, 'local_batches': local_batches,
            })
        write_json(output / 'status.json', {
            'state': 'running', 'step': step, 'max_steps': args.max_steps,
            'approx_epochs': step * world_size * args.batch_size * args.grad_accum / len(train_rows),
            'elapsed_seconds': time.monotonic() - start_time,
            'max_cuda_memory_gb': torch.cuda.max_memory_allocated() / 2**30,
        })
        if distributed_ready():
            stop_tensor = torch.tensor([int(stopped_early) if rank_zero() else 0], device=device)
            dist.broadcast(stop_tensor, src=0)
            stopped_early = bool(stop_tensor.item())
        if stopped_early or paused_for_review:
            break
    if paused_for_review and not stopped_early:
        # This is a review stop within the original schedule, never completion.
        write_json(output / 'pilot_metrics.json', {'step': step, 'max_steps': args.max_steps,
            'metrics': selected_metrics, 'best_cer': best_cer, 'augmentation': augmentation_config.condition,
            'state': 'paused_for_pilot_review', 'scheduler_epoch': scheduler.last_epoch})
        write_json(output / 'status.json', {'state': 'paused_for_pilot_review', 'step': step,
            'max_steps': args.max_steps, 'metrics': selected_metrics, 'best_cer': best_cer,
            'sampler_epoch': epoch, 'local_batches': local_batches, 'scheduler_epoch': scheduler.last_epoch,
            'augmentation': augmentation_config.condition, 'optimizer_seconds': optimizer_seconds})
        barrier()
        if distributed_ready():
            dist.destroy_process_group()
        return
    if rank_zero():
        unwrap(model).save_pretrained(output / 'final')
        processor.save_pretrained(output / 'final')
    final = evaluate_synced(model, full_loader or eval_loader, processor, prefix, generation_config, device, output / 'final_predictions.jsonl')
    write_json(output / 'final_metrics.json', {'step': step, **final})
    if full_loader is not None:
        barrier()
        del optimizer, scheduler, model
        torch.cuda.empty_cache()
        best_model = WhisperForConditionalGeneration.from_pretrained(output / 'best', local_files_only=True)
        best_model.config.use_cache = False
        best_model.detect_language = forbid_language_detection
        best_model.to(device)
        best_metrics = evaluate_synced(best_model, full_loader, processor, prefix, generation_config,
                                      device, output / 'best_full_dev_predictions.jsonl')
        write_json(output / 'best_full_dev_metrics.json', best_metrics)
    passed = True
    if args.mode == 'sanity' and rank_zero():
        criteria = {
            'checkpoint_restored': bool(checkpoint_witness and checkpoint_witness['passed']),
            'cer_below_threshold': final['cer'] <= args.m0_max_cer,
            'loss_below_one': final['teacher_forced_loss'] < 1.,
            'loss_reduced_by_half': final['teacher_forced_loss'] <= initial['teacher_forced_loss'] * .5,
        }
        passed = all(criteria.values())
        write_json(output / 'm0_gate.json', {
            'passed': passed, 'criteria': criteria, 'initial': initial, 'final': final,
            'cer_threshold': args.m0_max_cer, 'evaluation_scope': 'training_overfit_only',
            'language_detection_disabled': True, 'checkpoint_restore': checkpoint_witness,
        })
    if distributed_ready():
        passed_tensor = torch.tensor([int(passed) if rank_zero() else 0], device=device)
        dist.broadcast(passed_tensor, src=0)
        passed = bool(passed_tensor.item())
        barrier()
    if rank_zero():
        write_json(output / 'status.json', {
        'state': 'completed' if passed else 'gate_failed', 'step': step,
        'elapsed_seconds': time.monotonic() - start_time, 'metrics': final,
        'stopped_early': stopped_early, 'optimizer_seconds': optimizer_seconds,
        'world_size': world_size,
    })
    if not passed:
        raise RuntimeError('M0 gate failed; inspect measured loss/CER and checkpoint witness')
    barrier()
    if distributed_ready():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-manifest', required=True)
    parser.add_argument('--dev-manifest')
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['sanity', 'train'], default='sanity')
    parser.add_argument('--max-steps', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-limit', type=int, default=128)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--full-dev-manifest', help='Full development is the only checkpoint selection source when provided')
    parser.add_argument('--full-eval-every', type=int, help='Defaults to approximately one training epoch')
    parser.add_argument('--eval-batch-size', type=int)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--skip-initial-eval', action='store_true', help='Train mode only; never skips the M0 initial witness')
    parser.add_argument('--early-stopping-patience', type=int, default=0)
    parser.add_argument('--min-steps', type=int, default=0)
    parser.add_argument('--warmup-steps', type=int)
    parser.add_argument('--weight-decay', type=float, default=.01)
    parser.add_argument('--max-grad-norm', type=float, default=1.)
    parser.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--m0-max-cer', type=float, default=.15)
    parser.add_argument('--resume', help='Only exact augmentation/cursor checkpoints created by this script')
    parser.add_argument('--augmentation', choices=['A0','A1','A2','A3'], default='A0')
    parser.add_argument('--stop-after-step', type=int, help='Pilot review stop; max_steps/scheduler remain 24808 (two passes:6202)')
    args = parser.parse_args()
    if args.eval_limit < 1 or args.eval_every < 1 or args.lr <= 0 or args.max_grad_norm <= 0:
        parser.error('eval-limit/eval-every/lr/max-grad-norm must be positive')
    if args.workers < 0 or args.early_stopping_patience < 0 or args.min_steps < 0:
        parser.error('workers, patience and min-steps must be non-negative')
    if args.full_eval_every is not None and args.full_eval_every < 1:
        parser.error('full-eval-every must be positive')
    if args.eval_batch_size is not None and args.eval_batch_size < 1:
        parser.error('eval-batch-size must be positive')
    if args.early_stopping_patience and not args.full_dev_manifest:
        parser.error('Early stopping requires full development evaluation')
    try:
        run(args)
    except Exception as error:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / 'status.json', {'state': 'failed', 'error': str(error), 'mode': args.mode})
        if args.mode == 'sanity' and not (output / 'm0_gate.json').exists():
            write_json(output / 'm0_gate.json', {'passed': False, 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
