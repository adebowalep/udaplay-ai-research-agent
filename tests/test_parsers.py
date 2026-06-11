"""Tests for udaplay.parsers — output parser classes."""
import json
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from udaplay.parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from udaplay.messages import AIMessage


def _ai_msg(content: str) -> AIMessage:
    return AIMessage(content=content)


class TestStrOutputParser:
    def test_returns_content_string(self):
        parser = StrOutputParser()
        msg = _ai_msg("Hello, world!")
        assert parser.parse(msg) == "Hello, world!"

    def test_empty_content(self):
        parser = StrOutputParser()
        assert parser.parse(_ai_msg("")) == ""

    def test_multiline_content(self):
        parser = StrOutputParser()
        text = "Line 1\nLine 2\nLine 3"
        assert parser.parse(_ai_msg(text)) == text


class TestJsonOutputParser:
    def test_parses_dict(self):
        parser = JsonOutputParser()
        data = {"key": "value", "number": 42}
        msg = _ai_msg(json.dumps(data))
        result = parser.parse(msg)
        assert result == data

    def test_parses_list(self):
        parser = JsonOutputParser()
        data = [1, 2, 3]
        msg = _ai_msg(json.dumps(data))
        assert parser.parse(msg) == data

    def test_invalid_json_raises(self):
        parser = JsonOutputParser()
        with pytest.raises(Exception):
            parser.parse(_ai_msg("not json"))


class TestPydanticOutputParser:
    class _SimpleModel(BaseModel):
        name: str
        value: int

    def test_parses_valid_json(self):
        parser = PydanticOutputParser(model_class=self._SimpleModel)
        data = {"name": "test", "value": 42}
        msg = _ai_msg(json.dumps(data))
        result = parser.parse(msg)
        assert isinstance(result, self._SimpleModel)
        assert result.name == "test"
        assert result.value == 42

    def test_invalid_schema_raises(self):
        parser = PydanticOutputParser(model_class=self._SimpleModel)
        with pytest.raises(Exception):
            parser.parse(_ai_msg('{"wrong_field": "oops"}'))

    def test_invalid_json_raises(self):
        parser = PydanticOutputParser(model_class=self._SimpleModel)
        with pytest.raises(Exception):
            parser.parse(_ai_msg("not valid json"))
