"""bash: run a command in the sandbox (through the rules in omniharness/rules)."""
from .base import ToolInputError, register


@register('bash',
          'Run a bash command in the sandbox (its directory holds video.mp4 and question.md). Output longer '
          'than 8000 characters is cut in the middle.',
          {'type': 'object', 'properties': {'command': {'type': 'string', 'description': 'the command'}},
           'required': ['command']})
def bash(box, args):
    command = str(args.get('command', '')).strip()
    if not command:
        raise ToolInputError('give a command')
    out, code = box.sb.run(command)
    tail = '' if code in (0, None) else f'\n[exit code {code}]'
    return [{'type': 'text', 'text': (out or '(no output)') + tail}]
