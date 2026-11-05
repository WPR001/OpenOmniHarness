"""Tools the model calls by name (function calling). Each tool is one module here that registers itself (base.py);
cfg.tools names the tools offered and their order, which is the order of their schemas in the prompt.

To add a tool: write tools/<name>.py with @register, import it below, and offer it with --tools ... <name>
(or add it to AgentConfig.tools). ToolBox runs the calls of one question."""
import json
import shlex
import time

from .. import commands, rules
from .base import REGISTRY, Tool, ToolInputError, register
from . import bash, view, watch, answer  # noqa: F401,E402  (registers the tools)


def specs(cfg):
    """Schemas of the offered tools, as passed to the chat template."""
    return [{'type': 'function', 'function': REGISTRY[name].spec} for name in cfg.tools]


class ToolBox:
    def __init__(self, sandbox, cfg):
        self.sb = sandbox
        self.cfg = cfg
        self.images = 0
        self.watches = 0
        self.state = {}               # per-question storage for tools that keep more than the two budgets above
        self.refused = []             # tool calls refused by a rule
        self.offered = [REGISTRY[name] for name in cfg.tools]

    def run(self, name, args):
        """Returns (content items for the tool message, info dict for the trace)."""
        t = time.time()
        args = args or {}
        try:
            reason = rules.refuse_call(self.cfg, name, args, self)
            if reason:
                self.refused.append({'tool': name, 'args': args, 'reason': reason})
                content = [{'type': 'text', 'text': f'[not run] {reason}'}]
            elif name in self.cfg.tools:
                content = REGISTRY[name].run(self, args)
            elif name in commands.COMMANDS or name.split('_')[0] in commands.COMMANDS:
                content = self._as_command(name, args)
            else:
                raise ToolInputError(f'there is no tool "{name}". Tools: {", ".join(self.cfg.tools)}. '
                                     'Commands such as vsheet, vframes, vcrop, vprobe and asr run inside bash.')
        except ToolInputError as e:
            content = [{'type': 'text', 'text': f'[error] {e}'}]
        except Exception as e:  # the question goes on after a failing command or file
            content = [{'type': 'text', 'text': f'[error] {type(e).__name__}: {e}'}]
        text = ' '.join(c['text'] for c in content if c['type'] == 'text')
        n_img = sum(c['type'] == 'image' for c in content)
        n_vid = sum(c['type'] == 'video' for c in content)
        return content, {'tool': name, 'args': args, 'output': text[:2000], 'images': n_img, 'videos': n_vid,
                         'seconds': round(time.time() - t, 2)}

    def _as_command(self, name, args):
        """A sandbox command called as if it were a tool (e.g. vsheet {"t0": 0, "t1": 30, "every": 2}):
        run it through bash and say so, so the model learns the intended form."""
        command = command_line(name, args)
        out, code = self.sb.run(command)
        tail = '' if code in (0, None) else f'\n[exit code {code}]'
        note = (f'[{name} is a command, not a tool; it was run as: bash {{"command": {json.dumps(command)}}}. '
                'Call commands through the bash tool.]\n')
        return [{'type': 'text', 'text': note + (out or '(no output)') + tail}]


def command_line(name, args):
    """Shell form of a command called as a tool: a 'command' string is used as is (prefixed with the program when
    missing); otherwise positional arguments come first and the rest become --flags."""
    program = name.split('_')[0]
    sub = name.split('_', 1)[1] if '_' in name else ''
    if isinstance(args.get('command'), str):
        cmd = args['command'].strip()
        head = cmd.split()[0] if cmd else ''
        return cmd if head == program else ' '.join(x for x in (program, sub, cmd) if x)
    parts = [program] + ([sub] if sub else [])
    rest = dict(args)
    if program == 'asr':
        for key in ('words', 'window', 'search', 'gaps', 'info', 'full'):
            if key in rest and not sub:
                value = rest.pop(key)
                parts.append(key)
                if value not in (True, None, ''):
                    parts += [str(value)] if key == 'search' else str(value).replace(',', ' ').split()
                break
    for key in commands.POSITIONAL.get(program, ()):
        if key in rest:
            parts.append(str(rest.pop(key)))
    for key, value in rest.items():
        flag = key.lstrip('-')
        flag = ('--' if len(flag) > 1 else '-') + flag
        if value is True:
            parts.append(flag)
        elif value not in (False, None):
            parts += [flag, str(value)]
    return ' '.join(shlex.quote(p) if i and (' ' in p or '"' in p) else p for i, p in enumerate(parts))


__all__ = ['REGISTRY', 'Tool', 'ToolBox', 'ToolInputError', 'command_line', 'register', 'specs']
