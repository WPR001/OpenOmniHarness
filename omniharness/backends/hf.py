"""transformers backend: Qwen3.8-27B (Qwen3_5ForConditionalGeneration) loaded locally, one request at a time."""
import torch

from . import Output
from .render import Renderer


class HFBackend:
    name = 'hf'

    def __init__(self, model_path, cfg, tools, attn_implementation='sdpa', max_model_len=131072, device_map='auto'):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16, device_map=device_map,
                                                                 attn_implementation=attn_implementation).eval()
        self.renderer = Renderer(self.processor, cfg, tools)
        self.cfg = cfg
        self.max_model_len = max_model_len

    def encode(self, messages, tools=None):
        text, images, videos = self.renderer.render(messages, tools)
        kwargs = dict(text=[text], return_tensors='pt')
        if images:
            kwargs['images'] = images
        if videos:
            from dataclasses import fields
            from transformers.video_utils import VideoMetadata
            known = {f.name for f in fields(VideoMetadata)}
            kwargs.update(videos=[v for v, _ in videos], do_sample_frames=False, cap_pixels_per_frame=True,
                          video_metadata=[{k: x for k, x in m.items() if k in known} for _, m in videos])
        return self.processor(**kwargs)

    @torch.inference_mode()
    def _one(self, request):
        inputs = self.encode(request.messages, request.tools).to(self.model.device)
        n = int(inputs['input_ids'].shape[1])
        if n + request.max_new_tokens > self.max_model_len:
            return Output('', n, f'context_overflow: {n} prompt tokens')
        out = self.model.generate(**inputs, max_new_tokens=request.max_new_tokens, do_sample=self.cfg.temperature > 0,
                                  temperature=self.cfg.temperature, top_p=self.cfg.top_p, top_k=self.cfg.top_k)
        text = self.processor.batch_decode(out[:, n:], skip_special_tokens=False)[0]
        return Output(text.replace('<|endoftext|>', '').replace('<|im_end|>', ''), n)

    def generate(self, requests):
        outputs = []
        for r in requests:
            try:
                outputs.append(self._one(r))
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                outputs.append(Output('', 0, f'oom: {e}'))
            except Exception as e:
                outputs.append(Output('', 0, f'{type(e).__name__}: {e}'))
        return outputs
