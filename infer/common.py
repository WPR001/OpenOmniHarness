"""Command-line options shared by run_single.py and run_bench.py, and building the backend and agent from them."""
import argparse
import math
import os
import sys
from dataclasses import asdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from omniharness import rules, tools  # noqa: E402
from omniharness.config import AGENT_MODEL, AgentConfig  # noqa: E402


def add_common_args(parser: argparse.ArgumentParser):
    g = parser.add_argument_group('model and backend')
    g.add_argument('--agent-mode', choices=['visual', 'omni'], default='visual',
                   help='visual: V+ASR (default); omni: enable on-demand original audio')
    g.add_argument('--model', default=None, help='local model path/ID, or official Gemini model ID')
    g.add_argument('--backend', choices=['vllm', 'hf', 'gemini'], default=None,
                   help='default: vllm for visual, official Gemini API for omni')
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
    g.add_argument('--reasoning-effort', choices=['low', 'medium', 'xhigh'], default=None,
                   help='local Qwen reasoning effort (default: medium)')
    g.add_argument('--no-thinking', action='store_true')
    g.add_argument('--no-preserve-thinking', action='store_true')
    g.add_argument('--tools', nargs='+', choices=sorted(tools.REGISTRY), default=None, metavar='NAME',
                   help='tools offered in prompt order; omni mode adds listen_audio to the default tools')
    g.add_argument('--max-listen', type=int, default=d.max_listen, help='audio calls per question; 0 disables listening')
    g.add_argument('--listen-seconds', type=float, default=d.listen_seconds, help='maximum seconds per audio window')
    g.add_argument('--rules', nargs='*', choices=sorted(rules.REGISTRY), default=list(d.rules), metavar='NAME',
                   help='optional rules to switch on (omniharness/rules/extra.py)')
    g.add_argument('--temperature', type=float, default=d.temperature)
    g.add_argument('--turn-tokens', type=int, default=d.turn_tokens)
    g.add_argument('--keep-sandbox', action='store_true')
    g.add_argument('--scratch', default=os.environ.get('OH_SCRATCH', '/tmp/omniharness_sandbox'),
                   help='where per-question sandboxes are made (use /dev/shm/<user>_<job> in Slurm jobs)')
    g = parser.add_argument_group('official Gemini API (GEMINI_API_KEY environment variable only)')
    g.add_argument('--api-workers', type=int, default=4, help='maximum concurrent official API requests')
    g.add_argument('--api-timeout', type=float, default=180.0, help='timeout per API request in seconds')
    g.add_argument('--api-retries', type=int, default=2, help='additional attempts for transient API errors')
    g.add_argument('--gemini-thinking-level', choices=['low', 'medium', 'high'], default=None,
                   help='optional Gemini thinking level; omit to use the model default')
    return parser


def build_config(args):
    cfg = AgentConfig()
    args.backend = args.backend or ('gemini' if args.agent_mode == 'omni' else 'vllm')
    args.model = args.model or ('gemini-3.8-flash' if args.backend == 'gemini' else AGENT_MODEL)
    if args.agent_mode == 'omni' and args.backend != 'gemini':
        raise ValueError('omni mode currently supports --backend gemini; the local backends are visual-only')
    if args.max_listen < 0 or not math.isfinite(args.listen_seconds) or args.listen_seconds <= 0:
        raise ValueError('--max-listen must be nonnegative and --listen-seconds must be positive and finite')
    if args.api_workers < 1 or not 0 <= args.api_retries <= 10 or not math.isfinite(args.api_timeout) or args.api_timeout <= 0:
        raise ValueError('API workers/timeout must be positive and API retries must be between 0 and 10')
    if args.backend == 'gemini':
        if args.reasoning_effort is not None or args.no_thinking or args.no_preserve_thinking:
            raise ValueError('Qwen thinking flags are local-only; use --gemini-thinking-level for Gemini')
    elif args.gemini_thinking_level is not None:
        raise ValueError('--gemini-thinking-level requires --backend gemini')
    if args.tools is None:
        args.tools = list(cfg.tools)
        if args.agent_mode == 'omni' and args.max_listen > 0:
            args.tools.insert(-1, 'listen_audio')
    if 'listen_audio' in args.tools and (args.agent_mode != 'omni' or args.max_listen == 0):
        raise ValueError('listen_audio requires --agent-mode omni and --max-listen greater than 0')
    cfg.agent_mode = args.agent_mode
    cfg.max_listen, cfg.listen_seconds = args.max_listen, args.listen_seconds
    cfg.tools, cfg.rules = tuple(args.tools), tuple(args.rules)
    cfg.reasoning_effort, cfg.enable_thinking = args.reasoning_effort or cfg.reasoning_effort, not args.no_thinking
    cfg.preserve_thinking = not args.no_preserve_thinking
    cfg.temperature, cfg.turn_tokens, cfg.keep_sandbox = args.temperature, args.turn_tokens, args.keep_sandbox
    cfg.context_limit = min(cfg.context_limit, args.max_model_len - cfg.turn_tokens - 8192)
    cfg.sandbox_root = args.scratch
    os.makedirs(cfg.sandbox_root, exist_ok=True)
    return cfg


def build_backend(args, cfg):
    from omniharness.backends import make_backend
    specs = tools.specs(cfg)
    if args.backend == 'gemini':
        return make_backend('gemini', args.model, cfg, specs, workers=args.api_workers,
                            timeout=args.api_timeout, max_retries=args.api_retries,
                            thinking_level=args.gemini_thinking_level)
    if args.backend == 'hf':
        return make_backend('hf', args.model, cfg, specs, attn_implementation=args.attn, max_model_len=args.max_model_len)
    return make_backend('vllm', args.model, cfg, specs, tensor_parallel_size=args.tp,
                        gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
                        max_num_seqs=args.max_num_seqs, enable_prefix_caching=not args.no_prefix_caching,
                        enforce_eager=args.enforce_eager)


def config_dict(args, cfg):
    return {'args': vars(args), 'config': asdict(cfg)}
