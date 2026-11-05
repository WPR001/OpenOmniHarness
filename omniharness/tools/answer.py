"""answer: the final option letter. The agent loop reads it and ends the question (omniharness/agent/loop.py)."""
from .base import register


@register('answer',
          'Give the final answer and end the task.',
          {'type': 'object', 'properties': {
              'answer': {'type': 'string', 'description': 'the option letter, e.g. "B"'},
              'reason': {'type': 'string', 'description': 'one or two sentences on the evidence'}},
           'required': ['answer']})
def answer(box, args):
    return [{'type': 'text', 'text': 'answer recorded'}]
