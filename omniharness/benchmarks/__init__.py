"""Loaders for the four omni benchmarks; each one reads the official annotation files and emits Samples."""
from .daily_omni import DailyOmni
from .lvomnibench import LVOmniBench
from .omnivideobench import OmniVideoBench
from .worldsense import WorldSense

BENCHMARKS = {cls.name: cls for cls in (DailyOmni, WorldSense, OmniVideoBench, LVOmniBench)}


def get_benchmark(name, root=None, **opts):
    if name not in BENCHMARKS:
        raise ValueError(f'unknown benchmark {name}; choose from {sorted(BENCHMARKS)}')
    return BENCHMARKS[name](root, **opts)


def parse_opts(pairs):
    """['setting=sub'] -> {'setting': 'sub'}"""
    out = {}
    for pair in pairs or []:
        key, sep, value = pair.partition('=')
        if not sep:
            raise ValueError(f'--bench-opt takes key=value, got {pair}')
        out[key.strip()] = value.strip()
    return out


__all__ = ['BENCHMARKS', 'get_benchmark', 'parse_opts']
