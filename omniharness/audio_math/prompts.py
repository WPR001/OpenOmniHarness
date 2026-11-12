"""Prompt extension for the optional audio-math-v1 tools."""


def augment_system(text, cfg):
    if getattr(cfg, 'harness_version', 'original') != 'audio-math-v1':
        return text
    offered = set(cfg.tools)
    names = [name for name in ('audio_analyze', 'audio_python') if name in offered]
    if not names:
        return text
    tool_names = ' / '.join(names)
    text = text.replace(
        'the transcript files below are your only access to the audio.',
        f'the transcript files provide recognized speech. {tool_names} measures the original audio signal.')
    text = text.replace(
        '- A sound that is not speech is not in the transcript: find its source in the frames, '
        'or use "[no words]" stretches as hints of where it happens.',
        '- Localize non-speech sounds with the frames and transcript, then use audio signal analysis '
        'when measurements can resolve the remaining acoustic question.')
    text += f'''

Audio to Math: audio-math-v1
Follow the visual/ASR sequence above, choosing the operations the question needs. Gather the initial
observations with the existing tools. During localization, consider {tool_names} when music or other
sounds require evidence beyond the transcript and frames. Select the audio interval from the observations
already returned. Keep these audio calls for a subsequent turn after those initial observations.

Use a focused interval and identify the acoustic detail to measure. After an audio result, continue with
visual, ASR or signal tools as needed, and call answer when the combined evidence decides the question.
'''
    if 'audio_analyze' in offered:
        text += '''
- audio_analyze(t0, t1, mode, params): measure a selected interval on the original media timeline.
  overview: energy, frequency structure and acoustic-change candidates, with an overview plot.
  events: transient candidates, onset/peak times and levels, with plots and data tables.
  spectrum: frequency peaks and time-frequency structure.
  rhythm: periodicity candidates and event timing.
  compare: compare this interval with params.comparison_window=[start,end].
  Optional params: channel (mix, left, right or a zero-based channel index), band_hz=[low,high],
  min_peak_dbfs, envelope_smooth_hz, window_ms, hop_ms, min_gap_ms,
  threshold_above_local_floor_db, comparison_window. Start with defaults and refine from observed structure.
  Example: {"t0":12,"t1":24,"mode":"spectrum"}.
'''
    if 'audio_python' in offered:
        text += '''
- audio_python(t0, t1, code): compute a custom signal measurement for the selected interval.
  Preloaded: x (float32 samples x channels, original sample rate), sr, t (absolute sample times),
  t0, t1, np, scipy and plt. Set result to a compact JSON-compatible finding.
  Save artifacts with save_json(name, value), save_csv(name, rows, header=None), save_plot(name).
  Use relative filenames. Each call runs in its own analysis directory with numpy, scipy, matplotlib,
  math and statistics available for imports.
  Example code: mono = x.mean(axis=1); result = {"rms":float(np.sqrt(np.mean(mono**2)))}.
'''
    text += f'''

Read the evidence
- Audio tools return text measurements and artifact paths. Use view to inspect returned PNG plots;
  use bash to read the JSON/CSV tables. Creating an image does not show it to you: call view on its path.
- Compare the same absolute interval with vsheet/vframes/vcrop/watch and ASR. Times are absolute source
  seconds or mm:ss; keep the timestamps and units with the measurements you use.
- For counts, inspect the detected events and list their times. A periodic grid alone is not an event
  list. For pitch, check fundamental frequencies and harmonics; spectral peaks are not separate sources.
- Refine the interval or detector parameters when needed, and compare with the earlier measurement.
  Use existing tables for filtering and counting. Store large arrays as files and return a short summary.
- The first audio analysis call also prepares ./audio_source.mka and ./audio_source.json. The metadata
  records the supplementary track's timing; both tools analyze the original source timeline. The visual
  copy ./video.mp4 remains muted. Audio signal tools provide measurements and plots to the model.

Audio analysis budget: {cfg.audio_analysis_calls} calls per question, up to {cfg.audio_analysis_seconds:g}
seconds per interval and {cfg.audio_analysis_timeout:g} seconds per CPU worker. These calls share the
question's overall time budget. Audio signal analysis uses tools, not shell command names.
'''
    return text
