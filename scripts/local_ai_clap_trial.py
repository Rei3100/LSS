"""CPU-only CLAP same-source caption retrieval proxy; no Product quality decision."""
from __future__ import annotations

try:
    from scripts.dev_state import dev_evidence_root, require_evidence_writable
except ModuleNotFoundError:  # Direct execution outside the checkout root.
    from dev_state import dev_evidence_root, require_evidence_writable

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path('C:/LSS-Work/local-ai-models/clap-htsat-unfused')
EVIDENCE = dev_evidence_root(ROOT) / 'release-evidence/local-ai-challengers-r1/clap'
MANIFEST = dev_evidence_root(ROOT) / 'release-evidence/local-ai-challengers-r1/musiccaps/acquisition.json'
REVISION = '8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a'
WEIGHT_SHA = '1cd3c601bc4afe0fa87be3de4c13dd2cfadd249fac1e29acf74a9b296c3219bb'
WEIGHT_SIZE = 614525833
METADATA = ('config.json', 'merges.txt', 'preprocessor_config.json', 'special_tokens_map.json',
            'tokenizer.json', 'tokenizer_config.json', 'vocab.json')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def verify_model(directory, metadata_path):
    doc = json.loads(metadata_path.read_text(encoding='utf-8'))
    if doc.get('repository') != 'laion/clap-htsat-unfused' or doc.get('revision') != REVISION:
        raise ValueError('model provenance mismatch')
    records = {row['rfilename']: row for row in doc['files']}
    weight = directory / 'pytorch_model.bin'
    if (records['pytorch_model.bin'].get('lfs', {}).get('sha256') != WEIGHT_SHA
            or weight.stat().st_size != WEIGHT_SIZE or sha256(weight) != WEIGHT_SHA):
        raise ValueError('model weight identity mismatch')
    result = {'pytorch_model.bin': {'bytes': WEIGHT_SIZE, 'sha256': WEIGHT_SHA}}
    for name in METADATA:
        path = directory / name
        row = records[name]
        data = path.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if len(data) != row['size'] or blob != row['blobId']:
            raise ValueError('model metadata mismatch: ' + name)
        result[name] = {'bytes': len(data), 'git_blob_sha1': blob, 'sha256': hashlib.sha256(data).hexdigest()}
    return result


def prepared_rows(manifest):
    doc = json.loads(manifest.read_text(encoding='utf-8-sig'))
    rows = [row for row in doc['rows'] if row.get('status') == 'prepared'][:3]
    if len(rows) != 3 or len({row.get('ytid') for row in rows}) != 3:
        raise ValueError('requires three distinct prepared rows')
    if any(not isinstance(r.get('caption'), str) or not r['caption'] or not r.get('fixture_path') or not r.get('fixture_sha256') for r in rows):
        raise ValueError('prepared row incomplete')
    return rows


def validate_fixtures(rows, require_hot):
    paths = []
    for row in rows:
        path = Path(row['fixture_path']).resolve()
        require_hot(path, 'CLAP prepared audio fixture')
        if not path.is_file() or path.stat().st_size > 20_000_000 or sha256(path) != row['fixture_sha256'].lower():
            raise ValueError('fixture missing, too large or SHA mismatch')
        paths.append(path)
    return paths


def convert_audio(audio, rate, np, resample_poly):
    if not isinstance(rate, int) or rate <= 0 or audio.ndim != 2 or audio.shape[1] != 2 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError('fixture must be finite nonempty stereo')
    mono = audio.mean(axis=1, dtype=np.float32)
    factor = math.gcd(rate, 48000)
    up, down = 48000 // factor, rate // factor
    converted = resample_poly(mono, up, down).astype(np.float32) if rate != 48000 else mono
    if not np.isfinite(converted).all(): raise ValueError('resampling produced nonfinite audio')
    return converted, {'source_rate': rate, 'source_channels': 2, 'source_frames': len(audio),
        'downmix': 'arithmetic mean of stereo channels', 'resample': 'scipy.signal.resample_poly',
        'up': up, 'down': down, 'model_rate': 48000, 'model_samples': len(converted)}


def rank_scores(caption_ids, scores, expected_id, epsilon=1e-7):
    if len(caption_ids) != len(scores) or len(set(caption_ids)) != len(caption_ids) or expected_id not in caption_ids:
        raise ValueError('invalid retrieval rows')
    values = [float(x) for x in scores]
    if any(not math.isfinite(x) for x in values): raise ValueError('nonfinite cosine')
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    winner = [i for i in order if abs(values[i] - values[order[0]]) <= epsilon]
    return {'cosines': [{'caption_id': caption_ids[i], 'cosine': values[i],
                         'rank': 1 + sum(v > values[i] + epsilon for v in values)} for i in range(len(values))],
            'expected_source_id': expected_id, 'top_ids': [caption_ids[i] for i in winner],
            'tie_unresolved': len(winner) > 1,
            'top1_same_source': None if len(winner) > 1 else caption_ids[winner[0]] == expected_id}


class MemorySampler:
    """Sample OS RAM only; the parent process owns scheduling and inference."""
    def __init__(self, reader, initial, interval=2.0):
        self.reader, self.minimum, self.interval = reader, initial, interval
        self.tripped = False
        self.error = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.wait(self.interval):
            try:
                ram = self.reader()[1]
                self.minimum = min(self.minimum, ram)
                self.tripped |= ram < 4
            except Exception as exc:
                self.error = str(exc)
                return

    def start(self): self.thread.start()
    def stop(self):
        self.stop_event.set()
        self.thread.join(10)
        if self.thread.is_alive(): raise RuntimeError('RAM sampler did not stop')


def run(output):
    from dev_hot_paths import require_hot
    from gpu_run import memory_headroom, MIN_COMMIT_GB
    from local_llm import peak_working_set_gb
    require_evidence_writable(output)
    output = output.resolve()
    if not output.is_relative_to(EVIDENCE.resolve()) or output.exists():
        raise ValueError('output must be a new file beneath CLAP evidence')
    for label, path in [('repository', ROOT), ('model directory', MODEL), ('model metadata receipt', EVIDENCE / 'hub-metadata.json'),
                        ('musiccaps manifest', MANIFEST), ('output', output), ('Python environment', sys.prefix)]:
        require_hot(path, 'CLAP ' + label)
    for name in (*METADATA, 'pytorch_model.bin'):
        require_hot(MODEL / name, 'CLAP model file ' + name)
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {'status': 'starting', 'stage': 'verify', 'device': 'CPU float32', 'model_revision': REVISION,
              'scope': 'same-source caption top-1 retrieval proxy; not Product, naming, acoustic certification or music quality', 'cases': []}
    start = time.monotonic()
    sampler = None
    def checkpoint():
        temp = output.with_suffix('.partial')
        temp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temp, output)
    try:
        rows = prepared_rows(MANIFEST)
        paths = validate_fixtures(rows, require_hot)
        record['manifest_sha256'] = sha256(MANIFEST)
        record['model_files'] = verify_model(MODEL, EVIDENCE / 'hub-metadata.json')
        record['metadata_receipt_sha256'] = sha256(EVIDENCE / 'hub-metadata.json')
        commit, ram = memory_headroom()
        minimum = WEIGHT_SIZE / 1e9 + 2 + 4
        if ram < minimum or commit < MIN_COMMIT_GB: raise RuntimeError('CPU memory admission refused')
        record['admission'] = {'available_ram_gb': ram, 'commit_headroom_gb': commit, 'required_ram_gb': minimum}
        record['ram_min_gb'] = ram
        sampler = MemorySampler(memory_headroom, ram)
        sampler.start()
        def margin():
            commit_now, ram_now = memory_headroom()
            record['ram_min_gb'] = min(record['ram_min_gb'], ram_now)
            if sampler.error: raise RuntimeError('RAM sampler failed: ' + sampler.error)
            if ram_now < 4 or sampler.tripped: raise RuntimeError('available RAM fell below 4 GB')
        record['stage'] = 'import'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['OMP_NUM_THREADS'] = '2'
        os.environ['MKL_NUM_THREADS'] = '2'
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
        import torch
        import transformers
        from transformers import ClapModel, ClapProcessor
        if torch.cuda.is_initialized(): raise RuntimeError('CUDA already initialized')
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        if transformers.__version__ != '4.52.3': raise RuntimeError('unexpected transformers version')
        np.random.seed(0)
        record['versions'] = {'torch': torch.__version__, 'transformers': transformers.__version__, 'numpy': np.__version__, 'soundfile': sf.__version__}
        record['stage'] = 'model_load'
        margin(); load_start = time.monotonic()
        processor = ClapProcessor.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False)
        model = ClapModel.from_pretrained(str(MODEL), local_files_only=True, trust_remote_code=False, weights_only=True,
                                          use_safetensors=False).to(device='cpu', dtype=torch.float32).eval()
        margin()
        params = list(model.parameters())
        if not params or any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in params) or torch.cuda.is_initialized():
            raise RuntimeError('model not exclusively CPU float32')
        record['parameter_count'] = sum(p.numel() for p in params)
        record['model_load_s'] = round(time.monotonic() - load_start, 3)
        record['stage'] = 'caption_features'
        # Rotate the fixed pool; score using source IDs, never choice positions.
        candidate_rows = rows[1:] + rows[:1]
        caption_ids = [r['ytid'] for r in candidate_rows]
        text_inputs = processor(text=[r['caption'] for r in candidate_rows], padding=True, return_tensors='pt')
        record['processor_class'] = type(processor).__name__
        record['model_class'] = type(model).__name__
        record['text_input_shapes'] = {k: list(v.shape) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)
            text_features = torch.nn.functional.normalize(text_features.float(), dim=-1)
            record['text_feature_shape'] = list(text_features.shape)
            for row, path in zip(rows, paths):
                margin(); record['stage'] = 'case:' + row['ytid']
                if sha256(path) != row['fixture_sha256'].lower(): raise ValueError('fixture changed')
                audio, rate = sf.read(path, dtype='float32', always_2d=True)
                audio, conversion = convert_audio(audio, int(rate), np, resample_poly)
                inputs = processor(audios=audio, sampling_rate=48000, return_tensors='pt')
                case_start = time.monotonic()
                audio_features = model.get_audio_features(**inputs)
                audio_features = torch.nn.functional.normalize(audio_features.float(), dim=-1)
                scores = (audio_features @ text_features.T)[0].tolist()
                result = rank_scores(caption_ids, scores, row['ytid'])
                result.update(fixture_sha256=row['fixture_sha256'], conversion=conversion,
                              input_shapes={k: list(v.shape) for k, v in inputs.items()}, audio_feature_shape=list(audio_features.shape),
                              elapsed_s=round(time.monotonic() - case_start, 3))
                record['cases'].append(result)
                margin()
                if torch.cuda.is_initialized(): raise RuntimeError('CUDA initialized unexpectedly')
        record['status'] = 'complete'; record['stage'] = 'complete'
    except Exception as exc:
        record['status'] = 'error'; record['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if sampler is not None:
            try:
                sampler.stop()
                record['ram_min_gb'] = min(record['ram_min_gb'], sampler.minimum)
                if sampler.error or sampler.tripped:
                    record['status'] = 'error'
                    record['error'] = 'RAM sampler failed or reserve crossed'
            except Exception as exc:
                record['status'] = 'error'
                record['error'] = f'RAM sampler stop failed: {exc}'
        record['wall_s'] = round(time.monotonic() - start, 3)
        record['process_peak_working_set_gb'] = peak_working_set_gb(os.getpid())
        checkpoint()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    record = run(args.output)
    print(json.dumps({'status': record['status'], 'stage': record['stage'], 'output': str(args.output), 'cases': len(record['cases'])}))
    return 0 if record['status'] == 'complete' else 1


if __name__ == '__main__': raise SystemExit(main())
