# OpenOmniHarness

The agentic harness of **SleuthVLM** ([paper](https://arxiv.org/abs/2611.06083), [project page](https://wpr001.github.io/sleuthVLM/)).

- [x] [Offline Omni Harness](#offline-omni-harness): Qwen3.8-27B on Daily-Omni, WorldSense, OmniVideoBench,
  LVOmniBench and single videos
- [ ] [Online Omni Harness](#online-omni-harness): coming soon

## Offline Omni Harness

Qwen3.8-27B answers questions about a video by working in a sandbox with tools and shell commands. The speech
reaches it as a timestamped transcript computed beforehand with Qwen3-ASR and a forced aligner.

| tool | what it does |
|---|---|
| `bash` | runs a command in the sandbox: the commands below, `python3`, `ffmpeg`, `grep`, ... |
| `view` | shows up to 4 image files (contact sheets, frames, crops) |
| `watch` | shows a segment of up to 30 s as video |
| `answer` | gives the option letter and ends the question |

| command (through `bash`) | what it does |
|---|---|
| `vprobe` | duration, size and frame rate |
| `vsheet T0 T1 --every S` | contact sheet of time-stamped frames |
| `vframes T0 T1 --fps F [--full]` | single frames |
| `vcrop --t T --box x0,y0,x1,y1` | crop of a native-resolution frame |
| `asr window / search / words / gaps` | transcript lines, search, word times, stretches without speech |

### Layout

```text
omniharness/
  config.py        paths; AgentConfig (budgets, sampling, tools, rules)
  agent/           loop.py (the agent loop), prompts.py (system prompt and fixed messages)
  tools/           bash.py, view.py, watch.py, answer.py; registry and ToolBox in __init__.py
  sandbox/         sandbox.py: per-question work dir and bash runner
  commands/        video.py (vprobe, vsheet, vframes, vcrop), asr.py; bin/ wrappers
  rules/           guard.py (built-in command guard), extra.py (optional rules)
  media/           video.py (decoding, contact sheets), transcript.py (transcript store)
  backends/        vllm_backend.py, hf.py, render.py (Qwen3.8 chat template)
  benchmarks/      daily_omni.py, worldsense.py, omnivideobench.py, lvomnibench.py
asr/               build.py (transcripts), slurm/build_asr.sbatch
infer/             run_single.py, run_bench.py, score.py, slurm/run_bench.sbatch
scripts/           setup_envs.sh, download_models.sh
```

### Setup

```bash
conda create -n omniharness python=3.12 -y
conda run -n omniharness pip install -r requirements.txt            # agent
conda create -n omniharness-asr python=3.12 -y
conda run -n omniharness-asr pip install -r requirements-asr.txt    # transcripts
OH_MODEL_DIR=models bash scripts/download_models.sh                  # Qwen3.8-27B, Qwen3-ASR-1.7B, Qwen3-ForcedAligner-0.6B
```

`ffmpeg` and `ffprobe` are used for video decoding.

Paths are set with environment variables:

| variable | default | |
|---|---|---|
| `OH_MODEL_DIR` | `./models` | model folder (models not found there load from Hugging Face) |
| `QWEN38_PATH`, `QWEN3_ASR_PATH`, `QWEN3_ALIGNER_PATH` | | one model each |
| `OMNI_BENCH_ROOT` | `./data` | benchmark data |
| `OH_ASR_CACHE` | `./cache/asr` | transcripts |
| `OH_MUTED_CACHE` | `./cache/muted` | videos without audio, made on first use |
| `OH_SCRATCH` | `/tmp/omniharness_sandbox` | per-question sandboxes |

Benchmark data under `OMNI_BENCH_ROOT`:

```text
Daily-Omni/qa.json             Daily-Omni/Videos/<vid>/<vid>_video.mp4
WorldSense/worldsense_qa.json  WorldSense/videos/<id>.mp4
OmniVideoBench/data.parquet    OmniVideoBench/videos/<name>.mp4
LVOmniBench/data.json          LVOmniBench/videos/<video_id>.mp4
```

### Usage

Run from the repository root. The Slurm scripts take one GPU per array task; `PYTHON` sets the environment's python
and `EXTRA` passes more flags.

#### One video

```bash
# transcript (omniharness-asr env)
python asr/build.py --videos <(echo /path/clip.mp4)
# answer (omniharness env)
python infer/run_single.py --video /path/clip.mp4 \
    --question "What does the host say right after the applause?" \
    --options "A. Thank you" "B. Wow, what a crowd" "C. Let's begin" "D. Nothing" \
    --out outputs/single/clip.json
```

It prints the answer and the full record of every tool call.

#### Benchmarks

```bash
# 1. transcripts for the four benchmarks (4 GPUs)
PYTHON=$(conda run -n omniharness-asr which python) \
BENCH="daily_omni worldsense omnivideobench lvomnibench" sbatch --array=0-3 asr/slurm/build_asr.sbatch

# 2. the agent (4 GPUs, one shard each)
PYTHON=$(conda run -n omniharness which python) BENCH=lvomnibench sbatch --array=0-3 infer/slurm/run_bench.sbatch
PYTHON=... BENCH="daily_omni worldsense" sbatch --array=0-3 infer/slurm/run_bench.sbatch
PYTHON=... BENCH=omnivideobench EXTRA="--limit 8 --tag _smoke" sbatch infer/slurm/run_bench.sbatch

# or directly on a GPU
python infer/run_bench.py --bench lvomnibench --num-shards 4 --shard-id 0

# 3. scores
python infer/score.py --bench lvomnibench --run-dir outputs/lvomnibench/agent_vllm
```

Every question has the same budget: 50 turns, 30 viewed images, 10 watch clips and 30 minutes.
Each finished question is written at once, and the same command resumes an unfinished run. The log's first line
per benchmark gives the number of questions with transcripts.

| flag | default | |
|---|---|---|
| `--tools` | `bash view watch answer` | tools offered, in prompt order |
| `--rules` | none | optional rules |
| `--backend` | `vllm` | or `hf` (transformers) |
| `--tp` | 1 | tensor parallel GPUs |
| `--limit`, `--uids`, `--uids-file` | | a subset of questions |
| `--tag`, `--out` | | run directory name |
| `--keep-sandbox` | off | keep the per-question directories |

Outputs in `outputs/<bench>/agent_<backend><tag>/`:
- `results_shard<i>of<n>.jsonl`: one record per question (prediction, stop reason, every turn with its tool calls
  and outputs, images viewed, seconds);
- `config_shard<i>of<n>.json`: the run settings;
- `scores.json`: accuracy overall and per benchmark dimension, with tool and stop statistics.

### Extending

New tools and rules are switched on by name, and the default setting stays as it is.

#### Add a tool

Add a module to `omniharness/tools/`, import it in `omniharness/tools/__init__.py`, and offer it with `--tools`:

```python
# omniharness/tools/ocr.py
from .base import ToolInputError, register


@register('ocr', 'Read the text in an image file from the sandbox.',
          {'type': 'object', 'properties': {'path': {'type': 'string', 'description': 'image file name'}},
           'required': ['path']},
          prompt='\n\nAlso available: the ocr tool reads the text in an image file.')
def ocr(box, args):
    path = box.sb.path(args.get('path', ''))      # a file inside the sandbox
    ...                                           # box.cfg, box.images, box.watches, box.state
    return [{'type': 'text', 'text': text}]       # or {'type': 'image', 'image': PIL.Image}
```

```bash
python infer/run_bench.py --bench daily_omni --tools bash view watch answer ocr
```

`prompt` is added to the system prompt when the tool is offered; `ToolInputError` shows the model an error message.

#### Add a sandbox command

Register an argparse function in `omniharness/commands/`, add its wrapper, and list it in the system prompt
(`omniharness/agent/prompts.py`):

```python
@command('vdiff', positional=('t0', 't1'))
def vdiff(argv=None):
    p = argparse.ArgumentParser(prog='vdiff', description='...')
    ...
```

```bash
cp omniharness/commands/bin/vsheet omniharness/commands/bin/vdiff
```

#### Add a rule

A rule can add a line to the prompt's Rules section, refuse bash commands by pattern, and refuse tool calls
(including `answer`). Register it in `omniharness/rules/extra.py` and switch it on with `--rules`:

```python
register(Rule(
    name='look_before_answer',
    prompt='Look at the video (view or watch) at least once before you answer.',
    check=lambda name, args, box: ('look at the video first (view or watch)'
                                   if name == 'answer' and not (box.images or box.watches) else None)))

register(Rule(
    name='no_python',
    prompt='Use the commands for the video; python3 is not available.',
    command=re.compile(r'\bpython3?\b'),
    reason='python3 is not available in this setting'))
```

```bash
python infer/run_bench.py --bench daily_omni --rules look_before_answer no_python
```

Refused commands go to each record's `denied`, refused tool calls to `refused`.

#### More

- Prompts: `omniharness/agent/prompts.py`.
- Sampling and generation settings: `AgentConfig` in `omniharness/config.py`, flags in `infer/common.py`.
- A new benchmark: a `Benchmark` subclass in `omniharness/benchmarks/` whose `load()` returns `Sample`s, added to
  `BENCHMARKS`.

## Online Omni Harness

Coming soon.

## Citation

```bibtex
@misc{wu2026sleuthvlmreallyneedaudioinvideo,
      title={SleuthVLM: Do We Really Need Audio-in-Video for Omni Video Understanding?},
      author={Peiran Wu and Fangfei Gou and Zhaoxi Mu and Changhao Zhang and Jian Liu and Weiqiang Wang and Yuntao Wang and Yuanchun Shi and Junxiao Shen},
      year={2026},
      eprint={2611.06083},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2611.06083},
}
```

## License

MIT
