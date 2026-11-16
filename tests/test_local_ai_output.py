import pytest
from scripts.local_ai_output import decode_json


def test_exact_fence_removal_preserves_unicode_values():
    value, adapter = decode_json('```json\n{"title":" café — 東京。"}\n```')
    assert value == {'title': ' café — 東京。'}
    assert adapter == 'single_json_fence'
    assert decode_json('{"a":1}') == ({'a': 1}, 'none')


@pytest.mark.parametrize('text', [
    'Explanation\n```json\n{"a":1}\n```',
    '```json\n{"a":1}\n```\n```json\n{"a":2}\n```',
    '```python\n{"a":1}\n```',
    '```json\n{"a":1\n```',
])
def test_prose_multiple_blocks_other_languages_and_broken_json_stay_rejected(text):
    with pytest.raises(ValueError):
        decode_json(text)
