# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, NamedTuple, Tuple, Union

from typing_extensions import override


class FunctionCall(NamedTuple):
    name: str
    arguments: str


DEFAULT_TOOL_PROMPT = (
    "You have access to the following tools:\n{tool_text}"
    "You may call one or more tools in a single response when needed.\n"
    "Use the following format if using a tool:\n"
    "```\n"
    "Action: tool name (one of [{tool_names}])\n"
    "Action Input: the input to the tool, in a JSON format representing the kwargs "
    """(e.g. ```{{"input": "hello world", "num_beams": 5}}```)\n"""
    "```\n"
)

GLM4_TOOL_PROMPT = (
    "你是一个名为 ChatGLM 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，"
    "你的任务是针对用户的问题和要求提供适当的答复和支持。# 可用工具{tool_text}"
)

LLAMA3_TOOL_PROMPT = (
    "Cutting Knowledge Date: December 2023\nToday Date: {date}\n\n"
    "You have access to the following functions. To call a function, please respond with JSON for a function call. "
    """Respond in the format {{"name": function name, "parameters": dictionary of argument name and its value}}. """
    "Do not use variables.\n\n{tool_text}"
)

QWEN_TOOL_PROMPT = (
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>{tool_text}"
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
    """<tool_call></tool_call> XML tags:\n<tool_call>\n{{"name": <function-name>, """
    """"arguments": <args-json-object>}}\n</tool_call>"""
)


GPT_OSS_TOOL_PROMPT = (
    "You have access to the following tools:\n{tool_text}\n\n"
    "Tool calling rules:\n"
    "- If you need to call tools, respond with JSON only (no markdown/code fences).\n"
    "- You may call multiple tools in a single response.\n"
    "- Use the following OpenAI-compatible JSON format:\n"
    '  {{"tool_calls": [{{"type": "function", "function": {{"name": "tool_name", "arguments": {{ ... }} }} }}, ...]}}\n'
)


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.lstrip()
        if stripped.startswith("json"):
            stripped = stripped[4:].lstrip()
    return stripped.strip()


def _coerce_tool_calls(obj: Any) -> Union[None, List["FunctionCall"]]:
    """Coerces common tool-calling JSON shapes into a FunctionCall list."""

    def normalize_args(args: Any) -> str:
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                return json.dumps(parsed, ensure_ascii=False)
            except json.JSONDecodeError:
                return json.dumps(args, ensure_ascii=False)
        return json.dumps(args, ensure_ascii=False)

    def normalize_call(call_obj: Any) -> Union[None, "FunctionCall"]:
        if not isinstance(call_obj, dict):
            return None

        # OpenAI-style: {"type": "function", "function": {"name": ..., "arguments": ...}}
        if "function" in call_obj and isinstance(call_obj["function"], dict):
            fn = call_obj["function"]
            if "name" not in fn or "arguments" not in fn:
                return None
            return FunctionCall(str(fn["name"]), normalize_args(fn["arguments"]))

        # Simple: {"name": ..., "arguments": ...}
        if "name" in call_obj and "arguments" in call_obj:
            return FunctionCall(str(call_obj["name"]), normalize_args(call_obj["arguments"]))

        return None

    if isinstance(obj, dict) and "tool_calls" in obj:
        tool_calls = obj["tool_calls"]
        if not isinstance(tool_calls, list):
            return None
        calls: List[FunctionCall] = []
        for item in tool_calls:
            call = normalize_call(item)
            if call is None:
                return None
            calls.append(call)
        return calls

    if isinstance(obj, list):
        calls: List[FunctionCall] = []
        for item in obj:
            call = normalize_call(item)
            if call is None:
                return None
            calls.append(call)
        return calls

    if isinstance(obj, dict):
        call = normalize_call(obj)
        if call is None:
            return None
        return [call]

    return None


def _find_tool_calls_in_text(text: str) -> Union[None, List["FunctionCall"]]:
    """Find embedded JSON tool calls inside a larger text blob."""
    decoder = json.JSONDecoder()
    stripped = _strip_json_fence(text)

    # Fast path: entire content is JSON.
    try:
        parsed = json.loads(stripped)
        coerced = _coerce_tool_calls(parsed)
        if coerced is not None and len(coerced):
            return coerced
    except json.JSONDecodeError:
        pass

    # Slow path: scan from the end for a JSON object/array that decodes cleanly.
    # Limit scan window to keep this cheap for long generations.
    scan = stripped[-8000:]
    base = len(stripped) - len(scan)

    best: Union[None, List[FunctionCall]] = None
    for rel_i in range(len(scan) - 1, -1, -1):
        ch = scan[rel_i]
        if ch not in "[{":
            continue
        i = base + rel_i
        try:
            obj, _end = decoder.raw_decode(stripped, idx=i)
        except json.JSONDecodeError:
            continue

        coerced = _coerce_tool_calls(obj)
        if coerced is not None and len(coerced):
            if best is None or len(coerced) > len(best):
                best = coerced

            # Early exit if we already found parallel calls.
            if best is not None and len(best) > 1:
                break

    return best


@dataclass
class ToolUtils(ABC):
    """
    Base class for tool utilities.
    """

    @staticmethod
    @abstractmethod
    def tool_formatter(tools: List[Dict[str, Any]]) -> str:
        r"""
        Generates the system message describing all the available tools.
        """
        ...

    @staticmethod
    @abstractmethod
    def function_formatter(functions: List["FunctionCall"]) -> str:
        r"""
        Generates the assistant message including all the tool calls.
        """
        ...

    @staticmethod
    @abstractmethod
    def tool_extractor(content: str) -> Union[str, List["FunctionCall"]]:
        r"""
        Extracts all the function calls from the assistant message.

        It should be an inverse function of `function_formatter`.
        """
        ...


class DefaultToolUtils(ToolUtils):
    r"""
    Default tool using template.
    """

    @override
    @staticmethod
    def tool_formatter(tools: List[Dict[str, Any]]) -> str:
        tool_text = ""
        tool_names = []
        for tool in tools:
            param_text = ""
            for name, param in tool["parameters"]["properties"].items():
                required, enum, items = "", "", ""
                if name in tool["parameters"].get("required", []):
                    required = ", required"

                if param.get("enum", None):
                    enum = ", should be one of [{}]".format(", ".join(param["enum"]))

                if param.get("items", None):
                    items = ", where each item should be {}".format(param["items"].get("type", ""))

                param_text += "  - {name} ({type}{required}): {desc}{enum}{items}\n".format(
                    name=name,
                    type=param.get("type", ""),
                    required=required,
                    desc=param.get("description", ""),
                    enum=enum,
                    items=items,
                )

            tool_text += "> Tool Name: {name}\nTool Description: {desc}\nTool Args:\n{args}\n".format(
                name=tool["name"], desc=tool.get("description", ""), args=param_text
            )
            tool_names.append(tool["name"])

        return DEFAULT_TOOL_PROMPT.format(tool_text=tool_text, tool_names=", ".join(tool_names))

    @override
    @staticmethod
    def function_formatter(functions: List["FunctionCall"]) -> str:
        function_text = ""
        for name, arguments in functions:
            function_text += f"Action: {name}\nAction Input: {arguments}\n"

        return function_text

    @override
    @staticmethod
    def tool_extractor(content: str) -> Union[str, List["FunctionCall"]]:
        # Accept JSON tool calls (e.g. OpenAI-style `tool_calls`) in addition to the default "Action:" format.
        try:
            parsed = json.loads(_strip_json_fence(content))
            coerced = _coerce_tool_calls(parsed)
            if coerced is not None and len(coerced):
                return coerced
        except json.JSONDecodeError:
            pass

        regex = re.compile(r"Action:\s*([a-zA-Z0-9_]+)\s*Action Input:\s*(.+?)(?=\s*Action:|\s*$)", re.DOTALL)
        action_match: List[Tuple[str, str]] = re.findall(regex, content)
        if not action_match:
            return content

        results = []
        for match in action_match:
            tool_name = match[0].strip()
            tool_input = match[1].strip().strip('"').strip("```")
            try:
                arguments = json.loads(tool_input)
                results.append(FunctionCall(tool_name, json.dumps(arguments, ensure_ascii=False)))
            except json.JSONDecodeError:
                return content

        return results


class GPTOssToolUtils(ToolUtils):
    r"""GPT-OSS tool calling template.

    Supports OpenAI-style `tool_calls` JSON and parallel (multi-tool) calls.
    """

    @override
    @staticmethod
    def tool_formatter(tools: List[Dict[str, Any]]) -> str:
        wrapped_tools = [{"type": "function", "function": tool} for tool in tools]
        tool_text = json.dumps(wrapped_tools, indent=2, ensure_ascii=False)
        return GPT_OSS_TOOL_PROMPT.format(tool_text=tool_text)

    @override
    @staticmethod
    def function_formatter(functions: List["FunctionCall"]) -> str:
        tool_calls = []
        for name, arguments in functions:
            try:
                args_obj = json.loads(arguments)
            except json.JSONDecodeError:
                args_obj = arguments

            if isinstance(args_obj, str):
                try:
                    args_obj = json.loads(args_obj)
                except json.JSONDecodeError:
                    pass
            tool_calls.append({"type": "function", "function": {"name": name, "arguments": args_obj}})

        return json.dumps({"tool_calls": tool_calls}, ensure_ascii=False)

    @override
    @staticmethod
    def tool_extractor(content: str) -> Union[str, List["FunctionCall"]]:
        coerced = _find_tool_calls_in_text(content)
        return coerced if coerced is not None else content


class GLM4ToolUtils(ToolUtils):
    r"""
    GLM-4 tool using template.
    """

    @override
    @staticmethod
    def tool_formatter(tools: List[Dict[str, Any]]) -> str:
        tool_text = ""
        for tool in tools:
            tool_text += "\n\n## {name}\n\n{body}\n在调用上述函数时，请使用 Json 格式表示调用的参数。".format(
                name=tool["name"], body=json.dumps(tool, indent=4, ensure_ascii=False)
            )

        return GLM4_TOOL_PROMPT.format(tool_text=tool_text)

    @override
    @staticmethod
    def function_formatter(functions: List["FunctionCall"]) -> str:
        if len(functions) > 1:
            raise ValueError("GLM-4 does not support parallel functions.")

        return f"{functions[0].name}\n{functions[0].arguments}"

    @override
    @staticmethod
    def tool_extractor(content: str) -> Union[str, List["FunctionCall"]]:
        if "\n" not in content:
            return content

        tool_name, tool_input = content.split("\n", maxsplit=1)
        try:
            arguments = json.loads(tool_input.strip())
        except json.JSONDecodeError:
            return content

        return [FunctionCall(tool_name, json.dumps(arguments, ensure_ascii=False))]


class Llama3ToolUtils(ToolUtils):
    r"""
    Llama 3.x tool using template with `tools_in_user_message=False`.

    Reference: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/#json-based-tool-calling
    """

    @override
    @staticmethod
    def tool_formatter(tools: List[Dict[str, Any]]) -> str:
        date = datetime.now().strftime("%d %b %Y")
        tool_text = ""
        for tool in tools:
            wrapped_tool = {"type": "function", "function": tool}
            tool_text += json.dumps(wrapped_tool, indent=4, ensure_ascii=False) + "\n\n"

        return LLAMA3_TOOL_PROMPT.format(date=date, tool_text=tool_text)

    @override
    @staticmethod
    def function_formatter(functions: List["FunctionCall"]) -> str:
        if len(functions) > 1:
            raise ValueError("Llama-3 does not support parallel functions.")

        return f'{{"name": "{functions[0].name}", "parameters": {functions[0].arguments}}}'

    @override
    @staticmethod
    def tool_extractor(content: str) -> Union[str, List["FunctionCall"]]:
        try:
            tool = json.loads(content.strip())
        except json.JSONDecodeError:
            return content

        if "name" not in tool or "parameters" not in tool:
            return content

        return [FunctionCall(tool["name"], json.dumps(tool["parameters"], ensure_ascii=False))]


class MistralToolUtils(ToolUtils):
    r"""
    Mistral v0.3 tool using template.
    """

    @override
    @staticmethod
    def tool_formatter(tools: List[Dict[str, Any]]) -> str:
        wrapped_tools = []
        for tool in tools:
            wrapped_tools.append({"type": "function", "function": tool})

        return "[AVAILABLE_TOOLS] " + json.dumps(wrapped_tools, ensure_ascii=False) + "[/AVAILABLE_TOOLS]"

    @override
    @staticmethod
    def function_formatter(functions: List["FunctionCall"]) -> str:
        function_texts = []
        for name, arguments in functions:
            function_texts.append(f'{{"name": "{name}", "arguments": {arguments}}}')

        return "[" + ", ".join(function_texts) + "]"

    @override
    @staticmethod
    def tool_extractor(content: str) -> Union[str, List["FunctionCall"]]:
        try:
            tools = json.loads(content.strip())
        except json.JSONDecodeError:
            return content

        if not isinstance(tools, list):
            tools = [tools]

        results = []
        for tool in tools:
            if "name" not in tool or "arguments" not in tool:
                return content

            results.append(FunctionCall(tool["name"], json.dumps(tool["arguments"], ensure_ascii=False)))

        return results


class QwenToolUtils(ToolUtils):
    r"""
    Qwen 2.5 tool using template.
    """

    @override
    @staticmethod
    def tool_formatter(tools: List[Dict[str, Any]]) -> str:
        tool_text = ""
        for tool in tools:
            wrapped_tool = {"type": "function", "function": tool}
            tool_text += "\n" + json.dumps(wrapped_tool, ensure_ascii=False)

        return QWEN_TOOL_PROMPT.format(tool_text=tool_text)

    @override
    @staticmethod
    def function_formatter(functions: List["FunctionCall"]) -> str:
        function_texts = []
        for name, arguments in functions:
            function_texts.append(
                "<tool_call>\n" + f'{{"name": "{name}", "arguments": {arguments}}}' + "\n</tool_call>"
            )

        return "\n".join(function_texts)

    @override
    @staticmethod
    def tool_extractor(content: str) -> Union[str, List["FunctionCall"]]:
        regex = re.compile(r"<tool_call>(.+?)</tool_call>(?=\s*<tool_call>|\s*$)", re.DOTALL)
        tool_match: List[str] = re.findall(regex, content)
        if not tool_match:
            return content

        results = []
        for tool in tool_match:
            try:
                tool = json.loads(tool.strip())
            except json.JSONDecodeError:
                return content

            if "name" not in tool or "arguments" not in tool:
                return content

            results.append(FunctionCall(tool["name"], json.dumps(tool["arguments"], ensure_ascii=False)))

        return results


TOOLS = {
    "default": DefaultToolUtils(),
    "glm4": GLM4ToolUtils(),
    "llama3": Llama3ToolUtils(),
    "mistral": MistralToolUtils(),
    "qwen": QwenToolUtils(),
    "gpt_oss": GPTOssToolUtils(),
}


def get_tool_utils(name: str) -> "ToolUtils":
    tool_utils = TOOLS.get(name, None)
    if tool_utils is None:
        raise ValueError(f"Tool utils `{name}` not found.")

    return tool_utils
