"""Paths and settings. Environment variables override the model, cache and scratch paths."""
import os
from dataclasses import dataclass, field

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.environ.get('OH_MODEL_DIR', os.path.join(REPO, 'models'))   # local copies of the models below


def _model(env, name):
    """$env, else MODEL_DIR/<name> when it is there, else the Hugging Face id (downloaded on first use)."""
    local = os.path.join(MODEL_DIR, name)
    return os.environ.get(env) or (local if os.path.isdir(local) else f'Qwen/{name}')


AGENT_MODEL = _model('QWEN38_PATH', 'Qwen3.8-27B')
ASR_MODEL = _model('QWEN3_ASR_PATH', 'Qwen3-ASR-1.7B')
ALIGNER_MODEL = _model('QWEN3_ALIGNER_PATH', 'Qwen3-ForcedAligner-0.6B')
ASR_CACHE = os.environ.get('OH_ASR_CACHE', f'{REPO}/cache/asr')          # one transcript json per video
MUTED_CACHE = os.environ.get('OH_MUTED_CACHE', f'{REPO}/cache/muted')    # videos with the audio track removed


@dataclass
class AgentConfig:
    max_turns: int = 50              # model calls per question
    max_images: int = 30             # images shown through `view` (a contact sheet counts as one)
    max_watch: int = 10              # clips shown through `watch`
    max_seconds: float = 1800.0      # wall-clock budget per question
    bash_timeout: int = 120
    output_chars: int = 8000         # tool output kept in the conversation (head + tail)
    turn_tokens: int = 8192          # generation budget per turn (thinking included)
    answer_tokens: int = 4096
    enable_thinking: bool = True
    reasoning_effort: str = 'medium' # xhigh | medium | low (Qwen3.8 chat template)
    preserve_thinking: bool = True   # keep earlier thinking in the history (append-only prompts reuse the cache)
    temperature: float = 1.0         # Qwen3.8 recommended sampling for thinking mode
    top_p: float = 0.95
    top_k: int = 20
    context_limit: int = 110000      # above this many prompt tokens, older images are dropped from the history
    tools: tuple = ('bash', 'view', 'watch', 'answer')   # tools offered, in prompt order (omniharness/tools)
    rules: tuple = ()                # optional rules switched on by name (omniharness/rules/extra.py)
    keep_sandbox: bool = False
    sandbox_root: str = field(default_factory=lambda: os.environ.get('OH_SCRATCH', '/tmp/omniharness_sandbox'))
