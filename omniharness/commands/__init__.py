"""Shell commands the model runs inside its sandbox through the bash tool (vprobe, vsheet, vframes, vcrop, asr).

Each command is a function taking argv, registered with @command; bin/<name> is a two-line wrapper that the sandbox
puts first on PATH. To add a command: write it in a module here with @command('<name>', positional=(...)), import
that module below, copy any wrapper in bin/ to bin/<name>, and mention the command in the system prompt
(omniharness/agent/prompts.py). `positional` names the arguments that come before the flags, so that a call of the
command as if it were a tool can be turned into its shell form (omniharness/tools/__init__.py: command_line)."""
import os
import sys

COMMANDS = {}      # name -> function(argv)
POSITIONAL = {}    # name -> positional argument names
BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bin')


def command(name, positional=()):
    def wrap(fn):
        COMMANDS[name] = fn
        POSITIONAL[name] = tuple(positional)
        return fn
    return wrap


from . import video, asr  # noqa: E402,F401  (registers the commands)


def main():
    name = os.path.basename(sys.argv[0])
    COMMANDS[name](sys.argv[1:])
