"""CPU-only entry point for deterministic audio measurements."""
import json
from pathlib import Path
import sys
from .dsp import analyze


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    out = Path(request['output_dir'])
    try:
        result = analyze(request['audio_path'], request['t0'], request['t1'], request['mode'], request['params'], out)
        result.setdefault('status','ok')
    except Exception as exc:
        result = {'status':'error','error':f'{type(exc).__name__}: {exc}'}
    (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
