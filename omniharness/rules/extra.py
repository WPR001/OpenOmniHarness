"""Optional rules, off unless named in cfg.rules (--rules NAME ...). Register each one here with register(Rule(...));
see __init__.py for the fields. Two examples:

    register(Rule(
        name='look_before_answer',
        prompt='Look at the video (view or watch) at least once before you answer.',
        check=lambda name, args, box: ('look at the video first (view or watch)'
                                       if name == 'answer' and not (box.images or box.watches) else None)))

    register(Rule(
        name='no_python',
        prompt='Use the commands for the video; python3 is not available.',
        command=re.compile(r'\\bpython3?\\b'),
        reason='python3 is not available in this setting'))

A rule's check also sees the answer call, so a rule can hold the answer back; once the budget is used up, the forced
answer is always accepted."""
import re  # noqa: F401

from . import Rule, register  # noqa: F401
