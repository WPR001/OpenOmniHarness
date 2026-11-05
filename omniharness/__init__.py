"""OpenOmniHarness: a vision-language model (Qwen3.8-27B) answers questions about a video by working in a sandbox
with tools and commands, with the audio available only as precomputed Qwen3-ASR transcripts.

    agent/       the loop (think -> tool call -> result) and the prompts
    tools/       tools the model calls by name: bash, view, watch, answer
    sandbox/     the per-question work dir and the bash runner
    commands/    shell commands inside the sandbox: vprobe, vsheet, vframes, vcrop, asr
    rules/       what the agent may not do: the built-in command guard and optional named rules
    media/       video probing and decoding, the transcript store
    backends/    Qwen3.8-27B with vLLM or transformers
    benchmarks/  Daily-Omni, WorldSense, OmniVideoBench, LVOmniBench

The package stays import-light (no torch at import time) because the ASR stage, which runs in a different
environment, uses parts of it."""
