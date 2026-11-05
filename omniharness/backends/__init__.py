"""Local backends for Qwen3.8-27B. Both render the conversation with the model's own chat template (tools included)
and return raw text; the agent parses thinking and tool calls itself, so both behave the same."""
from dataclasses import dataclass


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
    if name == 'hf':
        from .hf import HFBackend
        return HFBackend(model_path, cfg, tools, **kwargs)
    if name == 'vllm':
        from .vllm_backend import VLLMBackend
        return VLLMBackend(model_path, cfg, tools, **kwargs)
    raise ValueError(f'unknown backend {name}')
