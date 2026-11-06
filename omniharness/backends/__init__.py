"""Local Qwen backends and an opt-in official Gemini API backend.

Local backends return text for the model-specific parser; API backends may return native tool calls and opaque
provider content, which the agent preserves across turns.
"""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Request:
    messages: list
    max_new_tokens: int = 4096
    tools: list = None     # tool schemas for this request; None = the backend's default set


@dataclass
class Output:
    text: str
    prompt_tokens: int = 0
    error: str = ''
    tool_calls: list = None         # None: local text parsing; []: native response without function calls
    usage: dict = field(default_factory=dict)
    provider_content: Any = None    # complete provider model message, including opaque thought signatures


class Batched:
    """submit/poll over a backend that only has generate(): each poll serves every queued request in one call."""

    def __init__(self, backend):
        self.backend, self.queue = backend, []

    def submit(self, key, request):
        self.queue.append((key, request))

    def poll(self):
        queue, self.queue = self.queue, []
        return list(zip([k for k, _ in queue], self.backend.generate([r for _, r in queue]))) if queue else []


def make_backend(name, model_path, cfg, tools, **kwargs):
    if name == 'gemini':
        from .gemini import GeminiBackend
        return GeminiBackend(model_path, cfg, tools, **kwargs)
    if name == 'hf':
        from .hf import HFBackend
        return HFBackend(model_path, cfg, tools, **kwargs)
    if name == 'vllm':
        from .vllm_backend import VLLMBackend
        return VLLMBackend(model_path, cfg, tools, **kwargs)
    raise ValueError(f'unknown backend {name}')
