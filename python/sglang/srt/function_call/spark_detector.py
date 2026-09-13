import json
import logging
import re
from typing import List, Tuple

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import infer_type_from_json_schema

logger = logging.getLogger(__name__)

TOOL_CALL_BEGIN = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
ARG_KEY_BEGIN = "<arg_key>"
ARG_KEY_END = "</arg_key>"
ARG_VALUE_BEGIN = "<arg_value>"
ARG_VALUE_END = "</arg_value>"

ARG_PAIR_PATTERN = re.compile(
    rf"{re.escape(ARG_KEY_BEGIN)}(.*?){re.escape(ARG_KEY_END)}"
    rf"{re.escape(ARG_VALUE_BEGIN)}(.*?){re.escape(ARG_VALUE_END)}",
    re.DOTALL,
)


def _convert_value(value: str, param_type) -> object:
    """Convert Spark-X2.5 XML text into a Python value using the declared schema."""
    if value.lower() == "null":
        return None

    normalized_type = (param_type or "string").lower()
    try:
        if normalized_type in {"string", "str", "text"}:
            return value
        if normalized_type in {"integer", "int"}:
            return int(value)
        if normalized_type in {"number", "float"}:
            number = float(value)
            return int(number) if number.is_integer() else number
        if normalized_type in {"boolean", "bool"}:
            normalized_value = value.strip().lower()
            if normalized_value not in {"true", "1", "false", "0"}:
                raise ValueError(f"invalid boolean: {value}")
            return normalized_value in {"true", "1"}
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        try:
            return json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return value


def _partial_tag_overlap(text: str, tag: str) -> int:
    """Length of the trailing substring of text that is a prefix of tag."""
    for i in range(1, min(len(text) + 1, len(tag))):
        if tag.startswith(text[-i:]):
            return i
    return 0


class SparkDetector(BaseFormatDetector):
    """
    Detector for the Spark-X2.5 XML-KV tool call format:

        <tool_call>get_weather<arg_key>city</arg_key><arg_value>上海</arg_value>...</tool_call>

    All tags are plain text labels; tool name precedes the first <arg_key> and
    arguments are <arg_key>/<arg_value> pairs. Values are converted to the JSON
    Schema type declared for each parameter in the tool definition.
    """

    def __init__(self):
        super().__init__()
        self.bot_token = TOOL_CALL_BEGIN
        self.eot_token = TOOL_CALL_END

    def has_tool_call(self, text: str) -> bool:
        return TOOL_CALL_BEGIN in text

    def supports_structural_tag(self) -> bool:
        return False

    def parses_required_natively(self) -> bool:
        return True

    def structure_info(self) -> _GetInfoFunc:
        def get_info(name: str) -> StructureInfo:
            return StructureInfo(
                begin=f"{TOOL_CALL_BEGIN}{name}",
                end=TOOL_CALL_END,
                trigger=TOOL_CALL_BEGIN,
            )

        return get_info

    def _tool_indices(self, tools: List[Tool]) -> dict:
        return self._get_tool_indices(tools)

    def _get_param_type(self, tools: List[Tool], function_name: str, param_name: str):
        for tool in tools:
            if tool.function.name == function_name:
                parameters = tool.function.parameters
                properties = (parameters or {}).get("properties", {})
                definition = properties.get(param_name)
                if isinstance(definition, dict):
                    return infer_type_from_json_schema(definition)
                return None
        return None

    def _parse_tool_call_xml(
        self, tool_xml: str, tools: List[Tool]
    ) -> Tuple[str, dict] | None:
        """Parse one complete <tool_call>...</tool_call> block into (name, arguments)."""
        if not tool_xml.startswith(TOOL_CALL_BEGIN) or not tool_xml.endswith(
            TOOL_CALL_END
        ):
            return None

        body = tool_xml[len(TOOL_CALL_BEGIN) : -len(TOOL_CALL_END)]
        first_arg = body.find(ARG_KEY_BEGIN)
        function_name = (body if first_arg < 0 else body[:first_arg]).strip()
        if not function_name:
            return None
        if tools and function_name not in self._tool_indices(tools):
            return None

        arguments = {}
        for match in ARG_PAIR_PATTERN.finditer(body):
            key, raw_value = match.group(1), match.group(2)
            if not key:
                continue
            arguments[key] = _convert_value(
                raw_value,
                self._get_param_type(tools, function_name, key),
            )
        return function_name, arguments

    def _has_unknown_tool_name(self, tool_xml: str, tools: List[Tool]) -> bool:
        if not tools:
            return False
        if not tool_xml.startswith(TOOL_CALL_BEGIN) or not tool_xml.endswith(
            TOOL_CALL_END
        ):
            return False
        body = tool_xml[len(TOOL_CALL_BEGIN) : -len(TOOL_CALL_END)]
        first_arg = body.find(ARG_KEY_BEGIN)
        function_name = (body if first_arg < 0 else body[:first_arg]).strip()
        return bool(function_name) and function_name not in self._tool_indices(tools)

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        calls = []
        normal_parts = []
        cursor = 0

        while cursor < len(text):
            start = text.find(TOOL_CALL_BEGIN, cursor)
            if start < 0:
                normal_parts.append(text[cursor:])
                break

            normal_parts.append(text[cursor:start])
            end = text.find(TOOL_CALL_END, start + len(TOOL_CALL_BEGIN))
            if end < 0:
                normal_parts.append(text[start:])
                break

            end += len(TOOL_CALL_END)
            raw_tool_call = text[start:end]
            parsed = self._parse_tool_call_xml(raw_tool_call, tools)
            if parsed is None:
                if not self._has_unknown_tool_name(raw_tool_call, tools):
                    normal_parts.append(raw_tool_call)
            else:
                name, arguments = parsed
                arguments_json = json.dumps(
                    arguments, ensure_ascii=False, separators=(",", ":")
                )
                calls.append(
                    ToolCallItem(
                        tool_index=len(calls),
                        name=name,
                        parameters=arguments_json,
                    )
                )
            cursor = end

        return StreamingParseResult(normal_text="".join(normal_parts), calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """Incremental streaming parse: buffer partial tags, emit complete tool calls."""
        self._buffer += new_text
        normal_parts = []
        calls = []

        while self._buffer:
            start = self._buffer.find(TOOL_CALL_BEGIN)
            if start < 0:
                keep = _partial_tag_overlap(self._buffer, TOOL_CALL_BEGIN)
                if keep:
                    normal_parts.append(self._buffer[:-keep])
                    self._buffer = self._buffer[-keep:]
                else:
                    normal_parts.append(self._buffer)
                    self._buffer = ""
                break

            if start > 0:
                normal_parts.append(self._buffer[:start])
                self._buffer = self._buffer[start:]

            end = self._buffer.find(TOOL_CALL_END, len(TOOL_CALL_BEGIN))
            if end < 0:
                break

            end += len(TOOL_CALL_END)
            raw_tool_call = self._buffer[:end]
            self._buffer = self._buffer[end:]
            parsed = self._parse_tool_call_xml(raw_tool_call, tools)
            if parsed is None:
                if not self._has_unknown_tool_name(raw_tool_call, tools):
                    normal_parts.append(raw_tool_call)
                continue

            name, arguments = parsed
            arguments_json = json.dumps(
                arguments, ensure_ascii=False, separators=(",", ":")
            )
            self.current_tool_id += 1
            self.prev_tool_call_arr.append({"name": name, "arguments": arguments})
            self.streamed_args_for_tool.append(arguments_json)
            calls.append(
                ToolCallItem(
                    tool_index=self.current_tool_id,
                    name=name,
                    parameters=arguments_json,
                )
            )

        return StreamingParseResult(normal_text="".join(normal_parts), calls=calls)

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        del tools
        pending = self._buffer
        self._buffer = ""
        return StreamingParseResult(normal_text=pending)
