"""Run the agent on one or more benchmarks (Daily-Omni, WorldSense, OmniVideoBench, LVOmniBench).

    python infer/run_bench.py --bench lvomnibench --backend vllm --limit 8
    python infer/run_bench.py --bench worldsense --backend vllm --num-shards 4 --shard-id 0
    python infer/run_bench.py --bench daily_omni worldsense --uids-file uids.json --tag _subset

The transcripts must exist first (asr/build.py). Results per benchmark: <out>/results_shard{i}of{n}.jsonl, one record
per question, written as soon as it is done; reruns skip finished questions. Several benchmarks share one model and one
pipeline. GPU work goes through Slurm (infer/slurm/run_bench.sbatch).
"""
import argparse
import json
import os
import time

from common import ROOT, add_common_args, build_backend, build_config, check_audio_math_resume, config_dict

from omniharness.agent import Agent
from omniharness.benchmarks import BENCHMARKS, get_benchmark, parse_opts
from omniharness.config import ASR_CACHE
from omniharness.media.transcript import store_path
from score import score_run


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument('--bench', required=True, nargs='+', choices=sorted(BENCHMARKS))
    parser.add_argument('--root', default=None)
    parser.add_argument('--bench-opt', nargs='*', default=[], metavar='KEY=VALUE')
    parser.add_argument('--out', default=None, help='run dir, one benchmark only (default: outputs/<bench>/agent_<backend><tag>)')
    parser.add_argument('--tag', default='', help='suffix of the default run dir, e.g. _subset')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--uids', nargs='*', default=[])
    parser.add_argument('--uids-file', default='', help='json: a list of uids, or {benchmark: [uids]}')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-id', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=16, help='questions worked on at the same time')
    args = parser.parse_args()
    if args.out and len(args.bench) > 1:
        parser.error('--out takes a single benchmark; use --tag with several')

    opts = parse_opts(args.bench_opt)
    wanted = json.load(open(args.uids_file)) if args.uids_file else None
    try:
        cfg = build_config(args)
    except ValueError as e:
        parser.error(str(e))
    runs, todo = {}, []
    for name in args.bench:
        bench = get_benchmark(name, args.root, **opts)
        tag = ''.join(f'_{k}-{v}' for k, v in sorted(opts.items())) + args.tag
        mode_suffix = '_omni' if cfg.agent_mode == 'omni' else ''
        version_suffix = '_audio_math_v1' if cfg.harness_version == 'audio-math-v1' else ''
        out = args.out or os.path.join(ROOT, 'outputs', name, f'agent_{args.backend}{mode_suffix}{version_suffix}{tag}')
        os.makedirs(out, exist_ok=True)
        try:
            check_audio_math_resume(out, args, cfg)
        except ValueError as e:
            parser.error(str(e))
        samples = [s for k, s in enumerate(bench.load()) if k % args.num_shards == args.shard_id]
        if args.uids or wanted is not None:
            keep = set(args.uids) | set(wanted.get(name, []) if isinstance(wanted, dict) else wanted or [])
            samples = [s for s in samples if s.uid in keep]
        if args.limit:
            samples = samples[:args.limit]
        result_file = os.path.join(out, f'results_shard{args.shard_id}of{args.num_shards}.jsonl')
        done = set()
        if os.path.exists(result_file):
            with open(result_file, encoding='utf-8') as f:
                done = {json.loads(l)['uid'] for l in f if l.strip()}
        config_path = os.path.join(out, f'config_shard{args.shard_id}of{args.num_shards}.json')
        if done and args.backend == 'gemini':
            if not os.path.exists(config_path):
                parser.error(f'existing Gemini results have no run configuration: {out}; choose a new --out')
            with open(config_path, encoding='utf-8') as f:
                previous = json.load(f)
            old_args, old_cfg = previous.get('args', {}), previous.get('config', {})
            identity_args = ('backend', 'model', 'gemini_thinking_level')
            identity_cfg = ('agent_mode', 'tools', 'rules', 'max_listen', 'listen_seconds', 'temperature', 'turn_tokens')
            current = config_dict(args, cfg)
            # JSON normalizes tuples before comparing a saved run with this invocation.
            current = json.loads(json.dumps(current))
            if (any(old_args.get(k) != current['args'].get(k) for k in identity_args)
                    or any(old_cfg.get(k) != current['config'].get(k) for k in identity_cfg)):
                parser.error(f'Gemini run configuration differs from existing results in {out}; choose a new --out/--tag')
        no_asr = [s for s in samples if not os.path.exists(store_path(ASR_CACHE, s.audio or s.video))]
        mine = [s for s in samples if s.uid not in done and os.path.exists(s.video)]
        print(f'{name}: {len(samples)} questions in this shard, {len(done)} done, {len(mine)} to run, '
              f'{len(no_asr)} without a transcript yet -> {result_file}', flush=True)
        with open(config_path, 'w') as f:
            json.dump(config_dict(args, cfg), f, indent=1, default=str)
        runs[name] = (bench, out, result_file)
        todo += [(name, s) for s in mine]

    if todo:
        # question ids may repeat across benchmarks (OmniVideoBench and LVOmniBench both have 0199_video_80); a record
        # carries its sample's own meta dict, so that object tells which benchmark's file it belongs to
        bench_of = {id(s.meta): name for name, s in todo}
        assert len(bench_of) == len(todo), 'two questions share one meta dict'
        agent = Agent(build_backend(args, cfg), cfg, workers=args.workers)
        files = {name: open(runs[name][2], 'a', encoding='utf-8') for name in runs}
        correct = total = 0
        t0 = time.time()

        def write(r):   # each question as soon as it is done, into its benchmark's file
            nonlocal correct, total
            f = files[bench_of[id(r['meta'])]]
            f.write(json.dumps(r, ensure_ascii=False, default=str) + '\n')
            f.flush()
            total += 1
            correct += bool(r.get('correct'))
            print(f'[{total}/{len(todo)}] acc {correct / total:.4f} ({correct}/{total}) '
                  f'{(time.time() - t0) / total:.1f} s/question', flush=True)

        try:
            agent.run([s for _, s in todo], concurrency=args.batch_size, on_result=write)
        finally:
            for f in files.values():
                f.close()
            close = getattr(agent.backend, 'close', None)
            if close:
                close()
    for name, (bench, out, _) in runs.items():
        print(name, json.dumps(score_run(out, bench), indent=1, ensure_ascii=False)[:3000])


if __name__ == '__main__':
    main()
