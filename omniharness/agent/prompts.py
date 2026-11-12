"""System prompt and the fixed messages of the agent loop. system_prompt() fills in the budgets and appends what
the active rules and the offered tools add (nothing, for the default tools and no rules)."""
from .. import rules

SYSTEM = """You answer a question about a video by working in a sandbox with command-line tools.

What you have
- ./video.mp4: the video. Its audio track was removed. The speech in it was transcribed beforehand by a speech recognizer; the transcript files below are your only access to the audio. The transcript contains recognized speech and lyrics with their times; music, applause, laughter, other sounds and tone of voice are NOT in it, and "[no words a-b]" only means that no speech was recognized there.
- ./question.md: the question and its options.
- ./asr.txt: the speech transcript of the whole video, one utterance per line, in time order:
    [mm:ss.s-mm:ss.s] text            what was said between the two times
    [no words mm:ss.s-mm:ss.s]         a stretch of 3 s or more where no speech was recognized
  The first line (starting with #) summarizes it: language, number of lines, whether word times exist.
- ./asr_words.tsv: every recognized word with its times in seconds (columns start, end, word; tab-separated), for
  exact timing. Absent when the video has no speech or no word-level times.
- Commands, run through the bash tool (times are seconds or mm:ss):
  vprobe                                   duration, size and frame rate
  vsheet T0 T1 --every S [-o FILE]         contact sheet of time-stamped frames (up to 64 cells)
  vframes T0 T1 --fps F [--full] [-o DIR]  single frames (up to 32); --full keeps the native resolution
  vcrop --t T --box x0,y0,x1,y1            crop the native-resolution frame at T (box on a 0-1000 scale); or vcrop IMAGE --box ...
  asr window T0 T1 | asr search "words" | asr words T0 T1 | asr gaps   (shortcuts over asr.txt / asr_words.tsv)
  python3 (numpy, PIL, cv2), ffmpeg/ffprobe for the picture, grep, sort, awk, head
- Tools: bash (run commands), view (look at up to 4 image files), watch (see a segment of up to 30 s as video), answer (give the final option letter).
  The commands above are not tools: run them through bash, e.g. bash with command "vsheet 0 30 --every 1 -o sheet.jpg". view, watch and answer are tools, not shell commands.

A sequence that usually works (decide yourself what each question needs, and skip what it does not):
1. Read the question and the transcript: `cat asr.txt` for a video up to about 20 minutes; for a longer one `head -1 asr.txt`, then `grep -i` (or `asr search "words"`) for the key words of the question and read the lines around the hits.
2. Get an overview: one vsheet over the whole video with an interval of about duration/48 seconds, then view it, to see the scenes, the people and the structure.
3. Localize: combine transcript hits, sheet cells and "[no words]" stretches into a candidate time window.
4. Zoom in on that window: a vsheet with a short interval (1-5 s), then vframes at 1-4 fps; watch a segment when motion, direction or the order of events matters; vcrop --t on a native-resolution frame for small details, colors and text; asr words for the exact timing of speech.
5. Check the specific detail the question asks about before answering. When it asks how many, list each occurrence with its time.
6. Call answer as soon as the evidence decides the question.

Rules
- Work only inside the sandbox: write and read files with relative names (e.g. -o frames, view frames/f_01-02.0.jpg); absolute paths and /tmp are not available. Do not repeat a command that already gave its result.
- A sound that is not speech is not in the transcript: find its source in the frames, or use "[no words]" stretches as hints of where it happens.
- Budget: about {max_turns} steps, {max_images} viewed images and {max_minutes} minutes in total. Answer before it runs out; if still unsure, choose the most likely option."""

QUESTION = """{question}

{options}

Video: {duration}, {size}. Speech transcript: {transcript}.
Work in the sandbox with the tools, then call answer with the option letter."""

REMIND = 'Use a tool: continue with bash, view or watch, or call answer with the option letter.'

FORCE = ('The budget is used up. Call the answer tool now with the most likely option letter, '
         'based on what you have found so far.')

DROPPED = '[an image shown earlier was dropped from the conversation to save context]'
DROPPED_AUDIO = '[audio heard earlier was dropped to save context; its source time interval is retained above]'


def reminder(cfg):
    if getattr(cfg, 'harness_version', 'original') == 'audio-math-v1':
        return ('Continue with an available tool, using audio analysis for a localized acoustic question '
                'when needed, or call answer with the option letter.')
    if 'listen_audio' in cfg.tools:
        return 'Use an available tool to gather evidence, including listen_audio when needed, or call answer.'
    return REMIND


def system_prompt(cfg, tools=()):
    """SYSTEM with the budgets, then the lines of the active rules (end of the Rules section), then the prompt text of
    the offered tools that have one."""
    text = SYSTEM.format(max_turns=cfg.max_turns, max_images=cfg.max_images, max_minutes=int(cfg.max_seconds // 60))
    if 'listen_audio' in cfg.tools:
        text = text.replace('the transcript files below are your only access to the audio.',
                            'the transcript files provide recognized speech. You can also request original audio '
                            'from a selected time interval using listen_audio.')
        text = text.replace('watch and answer are tools, not shell commands.',
                            'watch, listen_audio and answer are tools, not shell commands.')
        text = text.replace('answer (give the final option letter).',
                            'listen_audio (hear original audio in a selected time window), '
                            'answer (give the final option letter).')
        text = text.replace('- A sound that is not speech is not in the transcript: find its source in the frames, '
                            'or use "[no words]" stretches as hints of where it happens.',
                            '- ASR contains recognized speech, not all sounds. Use listen_audio when sounds, music, '
                            'voice tone or uncertain transcription matter. "[no words]" is not evidence of silence. '
                            'You choose whether and when to listen; no fixed tool sequence is required.')
        text += (f'\n- Original-audio budget: {cfg.max_listen} calls, up to {cfg.listen_seconds:g} seconds each. '
                 'Specify absolute source times t0 and t1. Audio playback time zero maps to t0 in the source. '
                 'Audio is supplied by the tool; ./video.mp4 stays muted.')
    text += rules.prompt_lines(cfg) + ''.join(tool.prompt for tool in tools if tool.prompt)
    if getattr(cfg, 'harness_version', 'original') == 'audio-math-v1':
        from ..audio_math.prompts import augment_system
        text = augment_system(text, cfg)
    return text
