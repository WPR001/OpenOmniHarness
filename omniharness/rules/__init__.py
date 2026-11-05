"""Rules: what the agent may not do, stated to the model and enforced in code.

The guard (guard.py) is always on: it keeps bash commands inside the sandbox. Further rules register under a name
(extra.py) and are switched on with cfg.rules (--rules NAME ...). A rule can
- add a line to the Rules section of the system prompt (`prompt`),
- refuse bash commands that match a pattern (`command`), showing `reason` to the model,
- refuse a tool call after looking at its name and arguments (`check(name, args, box) -> reason or None`).
Refused commands are kept in the record's `denied`, refused tool calls in its `refused`."""
from dataclasses import dataclass
from typing import Callable, Optional

from .guard import GUARD

REGISTRY = {}


@dataclass
class Rule:
    name: str
    prompt: str = ''                    # line for the Rules section of the system prompt
    command: Optional[object] = None    # compiled regex: bash commands that match are refused
    reason: str = ''                    # shown to the model when `command` refuses
    check: Optional[Callable] = None    # check(tool_name, args, box) -> reason to refuse the call, or None


def register(rule):
    REGISTRY[rule.name] = rule
    return rule


from . import extra  # noqa: E402,F401  (registers the optional rules)


def active(cfg):
    return [REGISTRY[name] for name in cfg.rules]


def guard(command):
    for pattern, reason in GUARD:
        if pattern.search(command):
            return reason
    return None


def refuse_command(cfg, command):
    """Why a bash command is not run, or None."""
    reason = guard(command)
    if reason:
        return reason
    for rule in active(cfg):
        if rule.command is not None and rule.command.search(command):
            return rule.reason
    return None


def refuse_call(cfg, name, args, box):
    """Why a tool call is not run, or None."""
    for rule in active(cfg):
        reason = rule.check(name, args, box) if rule.check else None
        if reason:
            return reason
    return None


def prompt_lines(cfg):
    """Lines the active rules add to the end of the system prompt's Rules section."""
    return ''.join(f'\n- {rule.prompt}' for rule in active(cfg) if rule.prompt)
