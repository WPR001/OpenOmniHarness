"""Command-line options shared by run_single.py and run_bench.py, and building the backend and agent from them."""
import argparse
import os
import sys
from dataclasses import asdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from omniharness import rules, tools  # noqa: E402
from omniharness.config import AGENT_MODEL, AgentConfig  # noqa: E402


def add_common_args(parser: argparse.ArgumentParser):
    g = parser.add_argument_group('model and backend (both load Qwen3.8-27B locally)')
    g.add_argument('--model', default=AGENT_MODEL)
    g.add_argument('--backend', choices=['vllm', 'hf'], default='vllm')
    g.add_argument('--tp', type=int, default=1)
    g.add_argument('--gpu-memory-utilization', type=float, default=0.92)
    g.add_argument('--max-model-len', type=int, default=131072)
    g.add_argument('--max-num-seqs', type=int, default=16)
    g.add_argument('--no-prefix-caching', action='store_true')
    g.add_argument('--enforce-eager', action='store_true')
    g.add_argument('--attn', default='sdpa', help='transformers attn_implementation')
    g.add_argument('--workers', type=int, default=16, help='threads that run tools')

    g = parser.add_argument_group('agent')
    d = AgentConfig()   # the per-question budget (turns, images, watch clips, minutes) is fixed there
    g.add_argument('--reasoning-effort', choices=['low', 'medium', 'xhigh'], default=d.reasoning_effort)
    g.add_argument('--no-thinking', action='store_true')
    g.add_argument('--no-preserve-thinking', action='store_true')
    g.add_argument('--tools', nargs='+', choices=sorted(tools.REGISTRY), default=list(d.tools), metavar='NAME',
                   help='tools offered to the model, in prompt order (omniharness/tools)')
    g.add_argument('--rules', nargs='*', choices=sorted(rules.REGISTRY), default=list(d.rules), metavar='NAME',
                   help='optional rules to switch on (omniharness/rules/extra.py)')
    g.add_argument('--temperature', type=float, default=d.temperature)
    g.add_argument('--turn-tokens', type=int, default=d.turn_tokens)
    g.add_argument('--keep-sandbox', action='store_true')
    g.add_argument('--scratch', default=os.environ.get('OH_SCRATCH', '/tmp/omniharness_sandbox'),
                   help='where per-question sandboxes are made (use /dev/shm/<user>_<job> in Slurm jobs)')
    return parser


def build_config(args):
    cfg = AgentConfig()
    cfg.tools, cfg.rules = tuple(args.tools), tuple(args.rules)
    cfg.reasoning_effort, cfg.enable_thinking = args.reasoning_effort, not args.no_thinking
    cfg.preserve_thinking = not args.no_preserve_thinking
    cfg.temperature, cfg.turn_tokens, cfg.keep_sandbox = args.temperature, args.turn_tokens, args.keep_sandbox
    cfg.context_limit = min(cfg.context_limit, args.max_model_len - cfg.turn_tokens - 8192)
    cfg.sandbox_root = args.scratch
    os.makedirs(cfg.sandbox_root, exist_ok=True)
    return cfg


def build_backend(args, cfg):
    from omniharness.backends import make_backend
    specs = tools.specs(cfg)
    if args.backend == 'hf':
        return make_backend('hf', args.model, cfg, specs, attn_implementation=args.attn, max_model_len=args.max_model_len)
    return make_backend('vllm', args.model, cfg, specs, tensor_parallel_size=args.tp,
                        gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
                        max_num_seqs=args.max_num_seqs, enable_prefix_caching=not args.no_prefix_caching,
                        enforce_eager=args.enforce_eager)


def config_dict(args, cfg):
    return {'args': vars(args), 'config': asdict(cfg)}
