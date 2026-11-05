"""The per-question sandbox (sandbox.py). What runs inside it: the commands in omniharness/commands; what may not:
omniharness/rules."""
from .sandbox import Sandbox

__all__ = ['Sandbox']
