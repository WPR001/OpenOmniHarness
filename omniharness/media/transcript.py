"""Transcript store shared by the ASR stage (writes) and the `asr` command in the sandbox (reads). Pure Python.

One json per video, named by a key of its path, size and mtime:
  words: [[start, end, word], ...]     aligned words (empty when the language could not be aligned)
  lines: [[start, end, text], ...]     utterances built from the words (or 30 s chunks when unaligned)
  gaps:  [[start, end], ...]           stretches of >= GAP_S with no recognized words
The key changes when the videos move to another machine (new path, new mtime after a fresh download), so the cache
also keeps index.json: "<file name>|<size>" -> json name. store_path falls back to it when the key misses, which makes a
copied cache usable whatever the new paths are (file name and size are unique across the benchmarks).
"""
import hashlib
import json
import os
import re

from ..parsing import mmss

GAP_S = 3.0
LINE_BREAK_S = 0.8
LINE_MAX_S = 15.0
SENTENCE_END = '.!?。！？…'
PUNCT = '.,!?;:…。，！？；：、"”’)]'
CJK = ('Chinese', 'Japanese', 'Cantonese', 'Korean')


def video_key(path):
    real = os.path.realpath(path)
    st = os.stat(real)
    return hashlib.sha1(f'{real}|{st.st_size}|{int(st.st_mtime)}'.encode()).hexdigest()[:16]


INDEX = 'index.json'
_indexes = {}


def portable_key(path):
    """File name plus size: the same on every machine, whatever the directories are called."""
    return f'{os.path.basename(path)}|{os.stat(path).st_size}'


def store_path(cache_dir, video):
    own = os.path.join(cache_dir, f'{video_key(video)}.json')
    if os.path.exists(own):
        return own
    if cache_dir not in _indexes:
        try:
            _indexes[cache_dir] = json.load(open(os.path.join(cache_dir, INDEX), encoding='utf-8'))
        except (OSError, ValueError):
            _indexes[cache_dir] = {}
    name = _indexes[cache_dir].get(portable_key(video))
    found = os.path.join(cache_dir, name) if name else ''
    return found if found and os.path.exists(found) else own


def write_index(cache_dir):
    """Add to index.json every stored transcript whose 'video' exists here (the newest one when a video was transcribed
    twice). Entries are kept, never dropped, so running it where the recorded paths do not exist leaves a copied index
    intact."""
    try:
        index = json.load(open(os.path.join(cache_dir, INDEX), encoding='utf-8'))
    except (OSError, ValueError):
        index = {}
    names = [n for n in os.listdir(cache_dir) if n.endswith('.json') and n != INDEX]
    for name in sorted(names, key=lambda n: os.path.getmtime(os.path.join(cache_dir, n))):
        try:
            video = json.load(open(os.path.join(cache_dir, name), encoding='utf-8'))['video']
            if os.path.exists(video):
                index[portable_key(video)] = name
        except (OSError, ValueError, KeyError):
            continue
    tmp = os.path.join(cache_dir, f'.{INDEX}.{os.getpid()}')   # array tasks may finish together: replace atomically
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=0, sort_keys=True)
    os.replace(tmp, os.path.join(cache_dir, INDEX))
    _indexes.pop(cache_dir, None)
    return len(index)


def punctuate(items, text):
    """Attach the punctuation of the ASR text to the aligned items (the aligner drops it)."""
    out, pos = [], 0
    for start, end, word in items:
        k = text.find(word, pos)
        if k < 0:
            out.append([start, end, word])
            continue
        j = k + len(word)
        while j < len(text) and text[j] in PUNCT:
            j += 1
        out.append([start, end, text[k:j]])
        pos = j
    return out


def build_lines(words, joiner=' '):
    lines, cur = [], []
    for w in words:
        if cur and (w[0] - cur[-1][1] >= LINE_BREAK_S or cur[-1][2][-1:] in SENTENCE_END or w[1] - cur[0][0] > LINE_MAX_S):
            lines.append([cur[0][0], cur[-1][1], joiner.join(x[2] for x in cur)])
            cur = []
        cur.append(w)
    if cur:
        lines.append([cur[0][0], cur[-1][1], joiner.join(x[2] for x in cur)])
    return lines


def build_gaps(words, duration):
    gaps, prev = [], 0.0
    for start, end, _ in words:
        if start - prev >= GAP_S:
            gaps.append([round(prev, 2), round(start, 2)])
        prev = max(prev, end)
    if duration - prev >= GAP_S:
        gaps.append([round(prev, 2), round(duration, 2)])
    return gaps


def make_record(video, duration, has_audio, language, text, items, chunks=None, model='', aligner=''):
    """items: [(start, end, word)] from the aligner, or None; chunks: [(start, end, text)] used when unaligned."""
    joiner = '' if any(lang in (language or '') for lang in CJK) else ' '
    words = punctuate([[round(s, 2), round(e, 2), w] for s, e, w in items], text or '') if items else []
    if words:
        lines = build_lines(words, joiner)
        gaps = build_gaps(words, duration)
    elif has_audio and not (text or '').strip() and not any((c[2] or '').strip() for c in (chunks or [])):
        lines, gaps = [], ([[0.0, round(duration, 2)]] if duration >= GAP_S else [])   # audio, but no speech at all
        words = []
        return {'video': video, 'duration': round(duration, 2), 'has_audio': has_audio, 'language': language or '',
                'aligned': True, 'model': model, 'aligner': aligner, 'text': '', 'words': words, 'lines': lines,
                'gaps': gaps}
    else:
        lines = [[round(a, 2), round(b, 2), t.strip()] for a, b, t in (chunks or []) if t and t.strip()]
        gaps = []
    return {'video': video, 'duration': round(duration, 2), 'has_audio': has_audio, 'language': language or '',
            'aligned': bool(words), 'model': model, 'aligner': aligner, 'text': text or '',
            'words': words, 'lines': lines, 'gaps': gaps}


class Transcript:
    def __init__(self, path):
        with open(path, encoding='utf-8') as f:
            self.d = json.load(f)

    @property
    def lines(self):
        return self.d['lines']

    def render(self, t0=0.0, t1=None, gaps=True):
        t1 = self.d['duration'] if t1 is None else t1
        rows = [(a, f'[{mmss(a)}-{mmss(b)}] {text}') for a, b, text in self.d['lines'] if b >= t0 and a <= t1]
        if gaps and self.d['aligned']:
            rows += [(a, f'[no words {mmss(a)}-{mmss(b)}]') for a, b in self.d['gaps'] if b >= t0 and a <= t1]
        return '\n'.join(r for _, r in sorted(rows))

    def words(self, t0, t1):
        return '\n'.join(f'{mmss(a)} {mmss(b)} {w}' for a, b, w in self.d['words'] if b >= t0 and a <= t1)

    def search(self, query, k=20):
        terms = [w for w in re.findall(r"[\w']+", query.lower()) if len(w) > 1]
        phrase = query.lower().strip()
        scored = []
        for a, b, text in self.d['lines']:
            low = text.lower()
            score = (10 if phrase and phrase in low else 0) + sum(1 for t in terms if t[:6] in low)
            if score:
                scored.append((score, a, b, text))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return '\n'.join(f'[{mmss(a)}-{mmss(b)}] {text}' for _, a, b, text in scored[:k])

    def info(self):
        d = self.d
        n_words = len(d['words'])
        return (f"duration {mmss(d['duration'])}; audio track: {'yes' if d['has_audio'] else 'no'}; language: "
                f"{d['language'] or 'none detected'}; {len(d['lines'])} lines, {n_words} aligned words; "
                f"word-level times: {'yes' if d['aligned'] else 'no (line times are 30 s chunks)'}; "
                f"{len(d['gaps'])} stretches of >= {GAP_S:.0f} s without words; model {d['model']}")
