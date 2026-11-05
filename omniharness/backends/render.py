"""Turn the agent conversation into (prompt text, images, videos) with the Qwen3.8 chat template.

Message content is a string or a list of items {'type': 'text'|'image'|'video', ...}; images are PIL images and
videos are (frames uint8 (T, H, W, 3), metadata) with absolute frame indices (see media.frames_for_video).
"""


class Renderer:
    def __init__(self, processor, cfg, tools):
        self.processor = processor
        self.cfg = cfg
        self.tools = tools

    def render(self, messages, tools=None):
        chat, images, videos = [], [], []
        for m in messages:
            content = m.get('content')
            if isinstance(content, list):
                items = []
                for c in content:
                    if c['type'] == 'image':
                        items.append({'type': 'image'})
                        images.append(c['image'])
                    elif c['type'] == 'video':
                        items.append({'type': 'video'})
                        videos.append((c['video'], c['metadata']))
                    else:
                        items.append({'type': 'text', 'text': c['text']})
                content = items
            entry = {'role': m['role'], 'content': content}
            for key in ('reasoning_content', 'tool_calls'):
                if m.get(key):
                    entry[key] = m[key]
            chat.append(entry)
        text = self.processor.apply_chat_template(
            chat, tools=tools or self.tools, add_generation_prompt=True, tokenize=False,
            enable_thinking=self.cfg.enable_thinking, reasoning_effort=self.cfg.reasoning_effort,
            preserve_thinking=self.cfg.preserve_thinking)
        return text, images, videos
