"""CPU worker for model-authored audio measurements (audio-math-v1).

Run with ``python -m omniharness.audio_math.python_worker request.json``.  The
parent controls wall time and supplies a fresh output directory per call.
Model code receives the decoded audio window and writes analysis artifacts
inside that call's directory.
"""
import ast
import builtins
import contextlib
import io
import json
import math
import os
from pathlib import Path
import sys


MAX_CODE_CHARS = 24000
MAX_STDOUT_CHARS = 12000
MAX_RESULT_BYTES = 24000
ALLOWED_IMPORTS = {"numpy", "scipy", "matplotlib", "math", "statistics"}
FORBIDDEN_NAMES = {"eval", "exec", "compile", "open", "input", "breakpoint",
                   "getattr", "setattr", "delattr", "globals", "locals", "vars",
                   "exit", "quit", "help"}
FORBIDDEN_ATTRIBUTES = {"os", "sys", "subprocess", "ctypes", "f2py", "mro",
                        "builtins", "system", "popen", "spawn", "fork"}


def validate_code(code):
    if not isinstance(code, str) or not code.strip():
        raise ValueError("code must be a nonempty Python string")
    if len(code) > MAX_CODE_CHARS:
        raise ValueError(f"code must contain at most {MAX_CODE_CHARS} characters")
    tree = ast.parse(code, filename="analysis_code.py", mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if isinstance(node, ast.ImportFrom) and node.level:
                raise ValueError("relative imports are not available")
            if any(name.split(".")[0] not in ALLOWED_IMPORTS for name in modules):
                raise ValueError("imports available: numpy, scipy, matplotlib, math, statistics")
            if any(a.name.startswith("_") or (a.asname or "").startswith("_") for a in node.names):
                raise ValueError("private imports are not available")
        if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in FORBIDDEN_NAMES):
            raise ValueError(f"{node.id} is not available in audio_python")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRIBUTES):
            raise ValueError(f"attribute {node.attr} is not available in audio_python")
    return compile(tree, "analysis_code.py", "exec")


class LimitedOutput(io.TextIOBase):
    def __init__(self):
        self.parts = []
        self.size = 0
        self.truncated = False

    def write(self, text):
        text = str(text)
        available = max(0, MAX_STDOUT_CHARS - self.size)
        if available:
            self.parts.append(text[:available])
        self.size += min(available, len(text))
        self.truncated |= len(text) > available
        return len(text)

    def getvalue(self):
        return "".join(self.parts) + ("\n[stdout truncated; save detailed data as CSV]" if self.truncated else "")


def install_io_guard(output_dir):
    """Limit model code's file I/O; allow installed packages and font reads."""
    root = output_dir.resolve()
    read_roots = [root, Path(sys.base_prefix).resolve(), Path(sys.prefix).resolve()]
    for path in sys.path:
        if path and ("site-packages" in path or "lib/python" in path):
            read_roots.append(Path(path).resolve())
    read_roots += [Path(p) for p in ("/usr/share/fonts", "/usr/local/share/fonts", "/etc/fonts",
                                     "/System/Library/Fonts", "/Library/Fonts")]

    def inside(path, roots):
        return any(path == r or r in path.parents for r in roots)

    def audit(event, args):
        if event == "open":
            name, mode, flags = args
            if isinstance(name, int):
                return
            p = Path(os.fsdecode(name)).resolve()
            writing = bool((flags or 0) & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writing and not inside(p, [root]):
                raise PermissionError("write artifacts inside this analysis directory")
            if not writing and not inside(p, read_roots):
                raise PermissionError("read x, sr and t, or files in this analysis directory")
        elif event.startswith("socket.") or event in {"subprocess.Popen", "os.system", "os.posix_spawn", "os.fork", "os.exec"}:
            raise PermissionError("audio_python executes numerical analysis without network or subprocess calls")
        elif event in {"os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.rename", "os.link", "os.symlink"}:
            count = 2 if event in {"os.rename", "os.link", "os.symlink"} else 1
            for name in args[:count]:
                if isinstance(name, (str, bytes)) and not inside(Path(os.fsdecode(name)).resolve(), [root]):
                    raise PermissionError("file operations must stay inside this analysis directory")

    sys.addaudithook(audit)


def run(request):
    out = Path(request["output_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)
    code = request["code"]
    if isinstance(code, str):
        (out / "analysis_code.py").write_text(code, encoding="utf-8")
    compiled = validate_code(code)
    t0, t1 = float(request["t0"]), float(request["t1"])
    if not (math.isfinite(t0) and math.isfinite(t1) and 0 <= t0 < t1):
        raise ValueError("require finite 0 <= t0 < t1")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(out / ".mplconfig")
    import numpy as np
    import scipy
    import scipy.signal
    import scipy.fft
    import scipy.stats
    import scipy.ndimage
    import scipy.optimize
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .dsp import decode_window
    x, sr = decode_window(request["audio_path"], t0, t1)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if not len(x):
        raise ValueError("audio window contains no samples")
    t = t0 + np.arange(len(x), dtype=np.float64) / sr
    np.set_printoptions(threshold=32, edgeitems=3)

    def target(name):
        name = Path(name)
        if name.is_absolute():
            raise ValueError("artifact names must be relative paths")
        path = (out / name).resolve()
        if path == out or out not in path.parents:
            raise ValueError("artifact path must stay inside the analysis directory")
        if path.name in {"result.json", "request.json", "analysis_code.py"}:
            raise ValueError("this filename is reserved for the trace")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def json_value(value):
        if isinstance(value, np.ndarray):
            return json_value(value.tolist())
        if isinstance(value, np.generic):
            return json_value(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {str(k): json_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_value(v) for v in value]
        return value

    def save_json(name, value):
        path = target(name)
        path.write_text(json.dumps(json_value(value), ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        return str(path.relative_to(out))

    def save_csv(name, rows, header=None):
        import csv
        path = target(name)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if header is not None:
                writer.writerow(header)
            writer.writerows(rows)
        return str(path.relative_to(out))

    def save_plot(name):
        path = target(name)
        plt.savefig(path, dpi=140, bbox_inches="tight")
        return str(path.relative_to(out))

    real_import = builtins.__import__

    def scientific_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name.split(".")[0] not in ALLOWED_IMPORTS:
            raise ImportError("use the provided NumPy, SciPy and matplotlib modules")
        return real_import(name, globals, locals, fromlist, level)

    builtin_names = ("abs all any bool dict enumerate filter float format frozenset int isinstance issubclass "
                     "iter len list map max min next pow print range reversed round set slice sorted str sum "
                     "tuple zip Exception ValueError TypeError RuntimeError ArithmeticError AssertionError "
                     "StopIteration ZeroDivisionError IndexError KeyError")
    safe_builtins = {name: getattr(builtins, name) for name in builtin_names.split()}
    safe_builtins["__import__"] = scientific_import
    env = {"__builtins__": safe_builtins, "x": x, "sr": sr, "t": t, "t0": t0, "t1": t1,
           "np": np, "scipy": scipy, "plt": plt, "math": math,
           "save_json": save_json, "save_csv": save_csv, "save_plot": save_plot, "result": {}}
    # Preload helper imports before the I/O guard and keep limits local to this worker.
    import csv  # noqa: F401
    try:
        import resource
        seconds = max(1, min(600, int(request.get("max_seconds", 120))))
        resource.setrlimit(resource.RLIMIT_CPU, (seconds + 2, seconds + 3))
        resource.setrlimit(resource.RLIMIT_FSIZE, (128 * 1024 * 1024, 128 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    except (ImportError, ValueError, OSError):
        pass
    stdout = LimitedOutput()
    install_io_guard(out)
    status, error = "ok", None
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
            exec(compiled, env, env)
        result = json_value(env["result"])
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
        if len(encoded.encode("utf-8")) > MAX_RESULT_BYTES:
            save_json("full_result.json", result)
            result = {"full_result": "full_result.json", "note": "Detailed result saved; return small summaries in result."}
    except Exception as exc:
        status, error, result = "error", f"{type(exc).__name__}: {exc}", {}
    finally:
        plt.close("all")
    artifacts = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name not in {"request.json", "result.json"} and not any(p.startswith(".") for p in path.relative_to(out).parts):
            artifacts.append({"path": str(path.relative_to(out)), "bytes": path.stat().st_size})
    payload = {"status": status, "summary": "Custom audio measurements completed." if status == "ok" else error,
               "result": result, "stdout": stdout.getvalue(), "artifacts": artifacts,
               "window": [t0, t1], "time_reference": "original_video_seconds",
               "sample_rate_hz": int(sr), "channels": int(x.shape[1]), "samples": int(len(x)),
               "decoded_duration_s": len(x) / sr}
    if error:
        payload["error"] = error
    (out / "result.json").write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    return payload


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    try:
        payload = run(request)
    except Exception as exc:
        payload = {"status": "error", "summary": f"{type(exc).__name__}: {exc}", "result": {}, "artifacts": []}
        out = Path(request["output_dir"]).resolve()
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "result_file": "result.json"}))


if __name__ == "__main__":
    main()
