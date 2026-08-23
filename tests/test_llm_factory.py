"""Tests for the JSON extraction helper in llm_factory.

Local models are messy: they wrap JSON in markdown fences or chat around it.
_extract_json is the recovery layer, so it gets its own tests.
"""

import json

import pytest

from llm_factory import _extract_json


def test_plain_json():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_json_with_surrounding_whitespace():
    assert _extract_json('   \n {"a": 1} \n ') == {"a": 1}


def test_markdown_fenced_json():
    raw = '```json\n{"trends": []}\n```'
    assert _extract_json(raw) == {"trends": []}


def test_fenced_without_language_tag():
    raw = '```\n{"a": 1}\n```'
    assert _extract_json(raw) == {"a": 1}


def test_json_embedded_in_prose():
    raw = 'Sure, here is the result: {"a": 1} Hope this helps!'
    assert _extract_json(raw) == {"a": 1}


def test_nested_braces_in_prose():
    raw = 'Answer: {"outer": {"inner": [1, 2]}} done.'
    assert _extract_json(raw) == {"outer": {"inner": [1, 2]}}


def test_no_json_raises():
    with pytest.raises(json.JSONDecodeError):
        _extract_json("no json here at all")
