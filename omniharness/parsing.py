"""Answer letters, model output (thinking / text / tool calls) and time strings."""
import json
import re

ANSWER_PREFIXES = ('the best answer is', 'the correct answer is', 'the answer is', 'best answer:', 'best option:',
                   'answer:', 'option:', 'final answer:')
TOOL_CALL = re.compile(r'<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>', re.S)
PARAM = re.compile(r'<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>', re.S)


def extract_letter(text, letters='ABCD', options=None):
    """First option letter the reply commits to; falls back to matching an option's text. None if nothing fits."""
    if not text:
        return None
    letters = ''.join(letters) or 'ABCD'
    s = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
    boxed = re.search(r'\\boxed\{([^}]*)\}', s)
    if boxed:
        s = boxed.group(1)
    low = s.lower()
    for prefix in ANSWER_PREFIXES:
        k = low.find(prefix)
        if k >= 0:
            m = re.search(rf'\b([{letters}])\b', s[k + len(prefix):])
            if m:
                return m.group(1)
    head = s.strip().strip('*`"\'( ')
    if head and head[0] in letters and (len(head) == 1 or head[1] in '.):,' or (head[1] == ' ' and len(head.split()) <= 2)):
        return head[0]
    m = re.search(rf'\b([{letters}])\b', s)
    if m:
        return m.group(1)
    if options:
        for letter, option in zip(letters, options):
            body = re.sub(r'^\s*[A-Z][.)]\s*', '', str(option)).strip().lower()
            if body and body in low:
                return letter
    return None


def _value(raw):
    raw = raw.strip()
    if raw[:1] in '[{"' or raw in ('true', 'false', 'null') or re.fullmatch(r'-?\d+(\.\d+)?', raw):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return raw


def split_output(text):
    """(thinking, visible text, tool calls) of one generation. The prompt already opened <think>, so the
    thinking runs up to </think>. Tool calls use the Qwen3 XML form:
    <tool_call><function=name><parameter=key>value</parameter></function></tool_call>."""
    text = text.replace('<|im_end|>', '').replace('<|endoftext|>', '')
    if '</think>' in text:
        thinking, _, rest = text.partition('</think>')
        thinking = thinking.replace('<think>', '').strip()
    else:
        thinking, rest = '', text
    calls = []
    for name, body in TOOL_CALL.findall(rest):
        args = {key: _value(val) for key, val in PARAM.findall(body)}
        calls.append({'name': name.strip(), 'arguments': args})
    if not calls:   # some generations put a JSON object inside <tool_call>
        for blob in re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', rest, re.S):
            try:
                obj = json.loads(blob)
                calls.append({'name': obj.get('name', ''), 'arguments': obj.get('arguments') or {}})
            except json.JSONDecodeError:
                continue
    content = re.split(r'<tool_call>', rest, maxsplit=1)[0].strip()
    return thinking, content, calls


def parse_time(value):
    """Seconds from 12.5, '12.5', '01:02', '1:02:03' or '01:02.5'."""
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).strip().split(':')
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def mmss(t):
    t = round(max(0.0, float(t)), 1)
    if t >= 3600:
        return f'{int(t // 3600)}:{int(t // 60 % 60):02d}:{t % 60:04.1f}'
    return f'{int(t // 60):02d}:{t % 60:04.1f}'
