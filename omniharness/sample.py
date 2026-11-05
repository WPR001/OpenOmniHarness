"""One question about one video, as every benchmark loader and the single-input script produce it."""
from dataclasses import dataclass, field


@dataclass
class Sample:
    uid: str
    video: str
    question: str
    options: list                     # option strings as shown to the model, e.g. ["A. Wife", "B. Friend"]
    prompt: str                       # the benchmark's own question text
    system: str = ''                  # benchmark system prompt (kept for reference; the agent uses its own)
    prefix: str = ''                  # benchmark text that precedes the media (kept for reference)
    answer: str = ''                  # gold letter, empty when unknown
    audio: str = ''                   # separate audio file; empty means the video's own audio track
    meta: dict = field(default_factory=dict)

    @property
    def letters(self):
        return [chr(ord('A') + k) for k in range(len(self.options))] if self.options else []
