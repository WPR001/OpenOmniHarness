"""OpenOmniHarness: video question answering with a sandbox and tools.
The default local visual agent uses precomputed ASR; opt-in omni mode adds bounded original-audio observations
through an official Gemini API backend.

    agent/       the loop (think -> tool call -> result) and the prompts
    tools/       bash, view, watch, answer, and opt-in listen_audio
    sandbox/     the per-question work dir and the bash runner
    commands/    shell commands inside the sandbox: vprobe, vsheet, vframes, vcrop, asr
    rules/       what the agent may not do: the built-in command guard and optional named rules
    media/       video probing and decoding, the transcript store
    backends/    local vLLM/transformers, and official Gemini
    benchmarks/  Daily-Omni, WorldSense, OmniVideoBench, LVOmniBench

The package stays import-light (no torch at import time) because the ASR stage, which runs in a different
environment, uses parts of it."""
