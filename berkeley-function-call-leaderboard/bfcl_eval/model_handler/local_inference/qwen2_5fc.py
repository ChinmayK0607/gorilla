import json
import re
from typing import Any

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import convert_to_function_call
from overrides import override


class Qwen2_5FCHandler(OSSHandler):
    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_name_huggingface = model_name

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return [
            {call["name"]: {k: v for k, v in call["arguments"].items()}}
            for call in tool_calls
        ]

    @override
    def decode_execute(self, result, has_tool_call_tag):
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        decoded_result = []
        for item in tool_calls:
            if type(item) == str:
                item = eval(item)
            decoded_result.append({item["name"]: item["arguments"]})
        return convert_to_function_call(decoded_result)

    @override
    def _format_prompt(self, messages, function):
        """
        Matches the `chat_template` shipped with Qwen2.5 Instruct models.
        """
        default_system_prompt = (
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        )
        formatted_prompt = ""
        has_tools = len(function) > 0
        first_is_system = len(messages) > 0 and messages[0]["role"] == "system"

        if has_tools:
            formatted_prompt += "<|im_start|>system\n"
            if first_is_system:
                formatted_prompt += messages[0]["content"]
            else:
                formatted_prompt += default_system_prompt

            formatted_prompt += (
                "\n\n# Tools\n\nYou may call one or more functions to assist with the user query."
                "\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
            )
            for tool in function:
                formatted_prompt += f"\n{json.dumps(tool)}"
            formatted_prompt += (
                '\n</tools>\n\nFor each function call, return a json object with function name and arguments within '
                '<tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, '
                '"arguments": <args-json-object>}\n</tool_call><|im_end|>\n'
            )
        else:
            formatted_prompt += "<|im_start|>system\n"
            if first_is_system:
                formatted_prompt += messages[0]["content"]
            else:
                formatted_prompt += default_system_prompt
            formatted_prompt += "<|im_end|>\n"

        for idx, message in enumerate(messages):
            role = message["role"]
            content = message.get("content", "")
            tool_calls = message.get("tool_calls") or []

            if (
                role == "user"
                or (role == "system" and idx != 0)
                or (role == "assistant" and not tool_calls)
            ):
                formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

            elif role == "assistant":
                formatted_prompt += f"<|im_start|>{role}"

                if content:
                    formatted_prompt += f"\n{content}"

                for tool_call in tool_calls:
                    if hasattr(tool_call, "function"):
                        tool_call = tool_call.function
                    elif "function" in tool_call:
                        tool_call = tool_call["function"]

                    formatted_prompt += "\n"
                    formatted_prompt += '<tool_call>\n{"name": "'
                    formatted_prompt += tool_call["name"]
                    formatted_prompt += '", "arguments": '

                    if isinstance(tool_call["arguments"], str):
                        formatted_prompt += tool_call["arguments"]
                    else:
                        formatted_prompt += json.dumps(tool_call["arguments"])

                    formatted_prompt += "}\n</tool_call>"

                formatted_prompt += "<|im_end|>\n"

            elif role == "tool":
                prev_role = messages[idx - 1]["role"] if idx > 0 else None
                next_role = messages[idx + 1]["role"] if idx < len(messages) - 1 else None

                if idx == 0 or prev_role != "tool":
                    formatted_prompt += "<|im_start|>user"

                formatted_prompt += f"\n<tool_response>\n{content}\n</tool_response>"

                if idx == len(messages) - 1 or next_role != "tool":
                    formatted_prompt += "<|im_end|>\n"

        formatted_prompt += "<|im_start|>assistant\n"
        return formatted_prompt

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        # FC models use their own system prompt and tool header; skip default injection.
        functions: list = test_entry["function"]
        return {"message": [], "function": functions}

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        model_response = api_response.choices[0].text
        extracted_tool_calls = self._extract_tool_calls(model_response)

        if len(extracted_tool_calls) > 0:
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": "",
                "tool_calls": extracted_tool_calls,
            }
        else:
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": model_response,
            }

        return {
            "model_responses": model_response,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"],
        )
        return inference_data

    @staticmethod
    def _extract_tool_calls(input_string):
        pattern = r"<tool_call>\n(.*?)\n</tool_call>"
        matches = re.findall(pattern, input_string, re.DOTALL)

        result = []
        for match in matches:
            try:
                match = json.loads(match)
                result.append(match)
            except Exception:
                pass
        return result
