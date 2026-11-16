"""Decode one complete JSON reply, optionally removing one exact Markdown fence.

No substring/repair extraction is allowed: prose, multiple blocks, rewritten
values, missing fields and malformed JSON remain failures. Raw replies survive.
"""
import json
import re


def decode_json(text):
    try:
        return json.loads(text), 'none'
    except (ValueError, TypeError):
        if not isinstance(text, str):
            raise ValueError('JSON reply must be text') from None
    match = re.fullmatch(r'\s*```(?:json)?[ \t]*\r?\n(.*?)\r?\n```\s*', text, flags=re.S)
    if not match:
        raise ValueError('expected one complete JSON value or one complete JSON fence')
    return json.loads(match.group(1)), 'single_json_fence'
