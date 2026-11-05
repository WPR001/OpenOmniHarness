"""ASR stage: transcribe every benchmark video once with Qwen3-ASR and the forced aligner, before any agent runs.

    python asr/build.py --bench daily_omni worldsense omnivideobench lvomnibench --backend vllm
    python asr/build.py --videos my_list.txt --backend hf            # any list of video paths, one per line

Runs in the ASR env (requirements-asr.txt; GPU, through Slurm: asr/slurm/build_asr.sbatch). qwen-asr cuts long audio at quiet points
into <= 180 s pieces, transcribes them, aligns the words and merges everything back on the video's time axis.
The aligner scores the whole vocabulary at every position of a piece (1-2 GB per 180 s piece), so pieces are aligned
--align-batch at a time; after an out-of-memory error a video is retried with fewer pieces per step, down to one.
When alignment still fails (e.g. a language the aligner does not cover), the video is transcribed again in 30 s chunks
without word times. Output: <cache>/<video key>.json (see omniharness/media/transcript.py); existing files are skipped
(--redo-unaligned also redoes the ones without word times).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from omniharness.config import ALIGNER_MODEL, ASR_CACHE, ASR_MODEL  # noqa: E402
from omniharness.media.transcript import INDEX, make_record, store_path, write_index  # noqa: E402

SR = 16000
CHUNK_S = 30.0


def probe_audio(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type:format=duration', '-of', 'json', path],
                         capture_output=True, text=True).stdout
    data = json.loads(out or '{}')
    has_audio = any(s.get('codec_type') == 'audio' for s in data.get('streams', []))
    return float(data.get('format', {}).get('duration') or 0.0), has_audio


def load_audio(path):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', path, '-vn', '-map', '0:a:0', '-ac', '1', '-ar', str(SR),
                          '-f', 'f32le', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def slurm_uuids_to_indices():
    """Slurm sets CUDA_VISIBLE_DEVICES to GPU UUIDs; vLLM 0.14 (pinned by qwen-asr) only parses integers. Rewrite the
    same assigned GPUs as their nvidia-smi indices (PCI bus order, which CUDA then follows). Nothing else changes."""
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if 'GPU-' not in visible:
        return
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], capture_output=True,
                         text=True).stdout
    index = {uuid.strip(): idx.strip() for idx, uuid in (line.split(',', 1) for line in out.splitlines() if ',' in line)}
    uuids = [u.strip() for u in visible.split(',') if u.strip()]
    if all(u in index for u in uuids):
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(index[u] for u in uuids)
        print(f'CUDA_VISIBLE_DEVICES {visible} -> {os.environ["CUDA_VISIBLE_DEVICES"]} (same GPUs, as indices)', flush=True)


def collect_videos(benches):
    from omniharness.benchmarks import get_benchmark
    videos = []
    for name in benches:
        videos += [s.audio or s.video for s in get_benchmark(name).load()]
    return sorted(set(videos))


def load_model(args):
    import torch
    from qwen_asr import Qwen3ASRModel
    aligner_kwargs = {'dtype': torch.bfloat16, 'device_map': 'cuda:0'}
    if args.backend == 'vllm':
        return Qwen3ASRModel.LLM(model=args.model, forced_aligner=args.aligner, forced_aligner_kwargs=aligner_kwargs,
                                 max_inference_batch_size=args.align_batch, max_new_tokens=4096,
                                 gpu_memory_utilization=args.gpu_memory_utilization, max_num_seqs=args.max_num_seqs)
    return Qwen3ASRModel.from_pretrained(args.model, dtype=torch.bfloat16, device_map='cuda:0',
                                         forced_aligner=args.aligner, forced_aligner_kwargs=aligner_kwargs,
                                         max_inference_batch_size=args.align_batch, max_new_tokens=4096)


def items_of(result):
    stamps = getattr(result, 'time_stamps', None)
    if stamps is None:
        return []
    return [(float(it.start_time), float(it.end_time), str(it.text)) for it in getattr(stamps, 'items', stamps)]


def transcribe_one(model, wav, align_batch):
    """One video with word times; after an out-of-memory error, align fewer 180 s pieces per step (down to one)."""
    import torch
    size = align_batch
    try:
        while True:
            model.max_inference_batch_size = size
            try:
                return model.transcribe(audio=[(wav, SR)], return_time_stamps=True)[0]
            except torch.cuda.OutOfMemoryError:
                if size == 1:
                    raise
                torch.cuda.empty_cache()
                size = max(1, size // 4)
                print(f'    out of memory; aligning {size} piece(s) per step', flush=True)
    finally:
        model.max_inference_batch_size = align_batch


def unaligned(path):
    d = json.load(open(path, encoding='utf-8'))
    return d['has_audio'] and not d['aligned']


def chunked(model, wav):
    """Fallback: 30 s chunks without word times."""
    pieces = [(wav[int(t * SR):int((t + CHUNK_S) * SR)], t) for t in np.arange(0, len(wav) / SR, CHUNK_S)]
    pieces = [(w, t) for w, t in pieces if len(w) > SR // 2]
    if not pieces:
        return '', '', []
    results = model.transcribe(audio=[(w, SR) for w, _ in pieces], return_time_stamps=False)
    chunks = [(t, t + len(w) / SR, r.text) for (w, t), r in zip(pieces, results)]
    langs = [r.language for r in results if r.language]
    return (max(set(langs), key=langs.count) if langs else ''), ' '.join(r.text for r in results), chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bench', nargs='*', default=[])
    parser.add_argument('--videos', default='', help='text file with one video path per line')
    parser.add_argument('--cache', default=ASR_CACHE)
    parser.add_argument('--backend', choices=['vllm', 'hf'], default='vllm')
    parser.add_argument('--model', default=ASR_MODEL)
    parser.add_argument('--aligner', default=ALIGNER_MODEL)
    parser.add_argument('--batch', type=int, default=16, help='videos per transcribe() call')
    parser.add_argument('--align-batch', type=int, default=8, help='180 s pieces per aligner step')
    parser.add_argument('--redo-unaligned', action='store_true', help='also redo stored transcripts without word times')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.6)
    parser.add_argument('--max-num-seqs', type=int, default=64)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-id', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    videos = collect_videos(args.bench) if args.bench else []
    if args.videos:
        videos += [l.strip() for l in open(args.videos) if l.strip()]
    videos = sorted(set(videos))[args.shard_id::args.num_shards]
    os.makedirs(args.cache, exist_ok=True)
    todo = [v for v in videos if os.path.exists(v) and (not os.path.exists(store_path(args.cache, v)) or
                                                        (args.redo_unaligned and unaligned(store_path(args.cache, v))))]
    if args.limit:
        todo = todo[:args.limit]
    print(f'{len(videos)} videos in this shard, {len(todo)} to transcribe -> {args.cache}', flush=True)
    if not todo:
        return
    slurm_uuids_to_indices()
    model = load_model(args)
    name = os.path.basename(args.model.rstrip('/'))
    t_start, audio_s = time.time(), 0.0
    with ThreadPoolExecutor(8) as pool:
        for start in range(0, len(todo), args.batch):
            batch = todo[start:start + args.batch]
            meta = list(pool.map(probe_audio, batch))
            with_audio = [v for v, (_, has) in zip(batch, meta) if has]
            wavs = dict(zip(with_audio, pool.map(load_audio, with_audio)))
            results = {}
            usable = [v for v in with_audio if len(wavs[v]) >= SR // 2]
            if usable:
                try:
                    outs = model.transcribe(audio=[(wavs[v], SR) for v in usable], return_time_stamps=True)
                    results = dict(zip(usable, outs))
                except Exception as e:   # isolate the failing video by going one at a time
                    print(f'batch failed ({type(e).__name__}: {e}); retrying one by one', flush=True)
                    import torch
                    torch.cuda.empty_cache()
                    for v in usable:
                        try:
                            results[v] = transcribe_one(model, wavs[v], args.align_batch)
                        except Exception as e2:
                            print(f'  {v}: aligned run failed ({type(e2).__name__}: {e2}); 30 s chunks instead', flush=True)
            for v, (duration, has_audio) in zip(batch, meta):
                if v in results and (items_of(results[v]) or not results[v].text.strip()):
                    r = results[v]
                    record = make_record(v, duration, True, r.language, r.text, items_of(r), model=name,
                                         aligner=os.path.basename(args.aligner.rstrip('/')))
                elif has_audio and v in wavs and len(wavs[v]) >= SR // 2:
                    language, text, chunks = chunked(model, wavs[v])
                    record = make_record(v, duration, True, language, text, None, chunks, model=name)
                else:
                    record = make_record(v, duration, has_audio, '', '', None, [], model=name)
                with open(store_path(args.cache, v), 'w', encoding='utf-8') as f:
                    json.dump(record, f, ensure_ascii=False)
                audio_s += duration if has_audio else 0.0
            elapsed = time.time() - t_start
            print(f'[{min(start + args.batch, len(todo))}/{len(todo)}] {audio_s / 3600:.1f} h of audio in {elapsed / 60:.1f} min '
                  f'({audio_s / max(elapsed, 1e-6):.0f}x real time)', flush=True)
    print(f'{write_index(args.cache)} transcripts in {os.path.join(args.cache, INDEX)}', flush=True)


if __name__ == '__main__':
    main()
