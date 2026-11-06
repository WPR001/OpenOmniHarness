"""Tool interface. A tool is a function the model calls by name with JSON arguments. It registers itself with
@register (name, description and JSON-schema parameters, as rendered by the Qwen3.8 chat template) and is offered to
the model when its name is in cfg.tools.

The function gets the question's ToolBox (box.sb the sandbox, box.cfg the config, box.images / box.watches the budgets
used so far, box.state free per-question storage) and the arguments, and returns the content of the tool message:
a list of {'type': 'text', 'text': str}, {'type': 'image', 'image': PIL.Image},
{'type': 'video', 'video': frames, 'metadata': dict}, or (audio-capable backends only)
{'type': 'audio', 'audio': WAV bytes, 'mime_type': 'audio/wav', 't0': float, 't1': float} items.
Raise ToolInputError for a bad call; the model sees
"[error] <message>" and the question goes on."""
from dataclasses import dataclass
from typing import Callable

REGISTRY = {}


class ToolInputError(Exception):
    pass


@dataclass
class Tool:
    name: str
    spec: dict           # the 'function' part of the tool schema: name, description, parameters
    run: Callable        # run(box, args) -> content items
    prompt: str = ''     # text appended to the system prompt when the tool is offered


def register(name, description, parameters, prompt=''):
    def wrap(fn):
        REGISTRY[name] = Tool(name, {'name': name, 'description': description, 'parameters': parameters}, fn, prompt)
        return fn
    return wrap
