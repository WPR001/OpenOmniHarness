"""Score a run dir: accuracy overall and per benchmark dimension, plus agent behaviour (turns, tools, stops, time).

    python infer/score.py --bench lvomnibench --run-dir outputs/lvomnibench/agent_vllm
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load_records(run_dir):
    records = {}
    for path in sorted(glob.glob(os.path.join(run_dir, 'results_shard*.jsonl'))):
        with open(path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    records[r['uid']] = r
    return list(records.values())


def _acc(records):
    n = len(records)
    c = sum(1 for r in records if r.get('pred') and r.get('pred') == r.get('answer'))
    return {'accuracy': round(c / n, 4) if n else None, 'correct': c, 'total': n}


def _mean(values):
    values = list(values)
    return round(sum(values) / len(values), 2) if values else None


def score_records(records, dims=()):
    scores = {'overall': _acc(records), 'no_answer': sum(1 for r in records if not r.get('pred')),
              'stops': dict(Counter(r.get('stop') for r in records)),
              'mean_turns': _mean(r.get('turns', 0) for r in records),
              'mean_images': _mean(r.get('images_viewed', 0) for r in records),
              'mean_seconds': _mean(r.get('seconds', 0) for r in records),
              'tools': dict(sum((Counter(r.get('tool_counts') or {}) for r in records), Counter())),
              'denied_commands': sum(len(r.get('denied') or []) for r in records)}
    for dim in dims:
        groups = defaultdict(list)
        for r in records:
            value = (r.get('meta') or {}).get(dim)
            for v in (value if isinstance(value, list) else [value]):
                groups[str(v)].append(r)
        scores[dim] = {k: _acc(v) for k, v in sorted(groups.items())}
    return scores


def score_run(run_dir, bench=None):
    scores = score_records(load_records(run_dir), bench.dims if bench else ())
    with open(os.path.join(run_dir, 'scores.json'), 'w', encoding='utf-8') as f:
        json.dump(scores, f, indent=1, ensure_ascii=False)
    return scores


def main():
    from omniharness.benchmarks import BENCHMARKS, get_benchmark
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bench', choices=sorted(BENCHMARKS), default=None)
    p.add_argument('--run-dir', required=True)
    a = p.parse_args()
    print(json.dumps(score_run(a.run_dir, get_benchmark(a.bench) if a.bench else None), indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
