"""Run the agent on one video and one multiple-choice question.

    python asr/build.py --videos <(echo /path/clip.mp4)        # transcript first, in the ASR env
    python infer/run_single.py --video /path/clip.mp4 --question "What does he say after the applause?" \
        --options "A. Thanks" "B. Wow, what a crowd" "C. Let's go" "D. Nothing" --backend vllm

Prints the record (answer, stop reason, every tool call) and writes it to --out when given.
"""
import argparse
import json
import os

from common import add_common_args, build_backend, build_config

from omniharness.agent import Agent
from omniharness.benchmarks.base import letter_options
from omniharness.sample import Sample


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument('--video', required=True)
    parser.add_argument('--question', required=True)
    parser.add_argument('--options', nargs='+', required=True)
    parser.add_argument('--out', default='')
    args = parser.parse_args()
    cfg = build_config(args)
    options = letter_options(args.options)
    sample = Sample(uid=os.path.splitext(os.path.basename(args.video))[0], video=os.path.abspath(args.video),
                    question=args.question, options=options, prompt=args.question)
    record = Agent(build_backend(args, cfg), cfg, workers=args.workers).run([sample])[0]
    text = json.dumps(record, ensure_ascii=False, indent=1, default=str)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(text)
    print(f"\nAnswer: {record.get('pred')} (stop: {record.get('stop')}, turns: {record.get('turns')})")


if __name__ == '__main__':
    main()
