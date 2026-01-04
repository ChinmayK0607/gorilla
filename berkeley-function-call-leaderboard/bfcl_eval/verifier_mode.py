import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bfcl_eval.constants.enums import Language, ReturnFormat
from bfcl_eval.constants.eval_config import (
    LOCAL_SERVER_PORT,
    PROJECT_ROOT,
    RESULT_PATH,
    RESULT_VERIFIER_PATH,
    SCORE_VERIFIER_PATH,
)
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.eval_checker.eval_runner import (
    _evaluate_single_agentic_entry,
    _evaluate_single_ast_entry,
    _evaluate_single_multi_turn_entry,
    _evaluate_single_relevance_entry,
    _subset_entries_by_model_ids,
    get_handler,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
)
from bfcl_eval.model_handler.utils import parse_prompt_variation_params
from bfcl_eval.utils import (
    extract_test_category,
    get_directory_structure_by_category,
    get_file_name_by_category,
    is_agentic,
    is_format_sensitivity,
    is_java,
    is_js,
    is_multi_turn,
    is_relevance_or_irrelevance,
    load_dataset_entry,
    load_file,
    load_ground_truth_entry,
    parse_test_category_argument,
)
from openai import OpenAI
from tqdm import tqdm


def _load_system_prompt() -> str:
    prompt_path = PROJECT_ROOT / "prompt.md"
    if not prompt_path.exists():
        return "You are a verifier. Return JSON {\"score\": 0.0, \"critic\": \"...\"}."
    return prompt_path.read_text()


def _safe_json_extract(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try to extract first JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def _format_task_text(prompt_entry: dict) -> str:
    turns = []
    for idx, turn in enumerate(prompt_entry.get("question", []), start=1):
        if not turn:
            continue
        content = turn[0].get("content", "")
        turns.append(f"Turn {idx}: {content}")
    return "\n".join(turns)


def _format_available_tools(prompt_entry: dict) -> str:
    tools = prompt_entry.get("path") or []
    if not tools:
        tools = prompt_entry.get("involved_classes") or []
    return ", ".join(tools)


def _decode_multi_turn_calls(
    handler, model_result_list: list, has_tool_call_tag: bool = False
) -> list[list[list[str]]]:
    decoded: list[list[list[str]]] = []
    for single_turn_model_response_list in model_result_list:
        decoded_turn: list[list[str]] = []
        for model_result_item in single_turn_model_response_list:
            try:
                decoded_result: list[str] = handler.decode_execute(
                    model_result_item, has_tool_call_tag=has_tool_call_tag
                )
            except Exception:
                decoded_result = []
            decoded_turn.append(decoded_result)
        decoded.append(decoded_turn)
    return decoded


def _simulate_multi_turn_execution(
    decoded_turns: list[list[list[str]]],
    prompt_entry: dict,
    model_name_tag: str,
) -> list[list[list[str]]]:
    initial_config: dict = prompt_entry.get("initial_config", {})
    involved_classes: list = prompt_entry.get("involved_classes", [])
    test_entry_id: str = prompt_entry["id"]
    test_category: str = extract_test_category(test_entry_id, raise_error=False) or ""
    long_context = ("long_context" in test_category) or ("composite" in test_category)

    execution_results: list[list[list[str]]] = []
    for single_turn in decoded_turns:
        single_turn_exec: list[list[str]] = []
        for func_call_list in single_turn:
            exec_result, _ = execute_multi_turn_func_call(
                func_call_list=func_call_list,
                initial_config=initial_config,
                involved_classes=involved_classes,
                model_name=model_name_tag,
                test_entry_id=test_entry_id,
                long_context=long_context,
                is_evaL_run=True,
            )
            single_turn_exec.append(exec_result)
        execution_results.append(single_turn_exec)
    return execution_results


def _build_verifier_payload(
    prompt_entry: dict,
    decoded_calls: list[list[list[str]]],
    execution_results: list[list[list[str]]],
) -> str:
    lines = []
    lines.append("Task:")
    lines.append(_format_task_text(prompt_entry))
    lines.append("")
    lines.append(f"Available tools: {_format_available_tools(prompt_entry)}")
    lines.append("")
    lines.append("Trajectory tool calls and responses:")
    for turn_idx, (turn_calls, turn_execs) in enumerate(
        zip(decoded_calls, execution_results), start=1
    ):
        if not turn_calls:
            lines.append(f"Turn {turn_idx}: (no tool calls)")
            continue
        lines.append(f"Turn {turn_idx}:")
        for step_idx, (calls, execs) in enumerate(
            zip(turn_calls, turn_execs), start=1
        ):
            call_text = "; ".join(calls) if isinstance(calls, list) else str(calls)
            exec_text = "; ".join(execs) if isinstance(execs, list) else str(execs)
            lines.append(f"- Step {step_idx}: call = {call_text}; tool_response = {exec_text}")
    return "\n".join(lines)


def _chat_complete(
    client: OpenAI, model_name: str, system_prompt: str, user_prompt: str, temperature: float
) -> str:
    resp = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content


def _meta_validity(
    handler,
    test_category: str,
    model_result_item: dict,
    prompt_entry: dict,
    possible_answer_entry: dict,
    model_name: str,
) -> bool:
    if is_multi_turn(test_category):
        result = _evaluate_single_multi_turn_entry(
            handler,
            model_result_item["id"],
            model_result_item["result"],
            possible_answer_entry["ground_truth"],
            prompt_entry,
            model_name,
            test_category,
        )
    elif is_agentic(test_category):
        result = _evaluate_single_agentic_entry(
            handler,
            model_result_item["id"],
            model_result_item["result"],
            possible_answer_entry["ground_truth"],
            prompt_entry,
            model_name,
            test_category,
        )
    elif is_relevance_or_irrelevance(test_category):
        result = _evaluate_single_relevance_entry(
            handler,
            model_result_item["id"],
            model_result_item["result"],
            prompt_entry,
            model_name,
            test_category,
        )
    elif is_format_sensitivity(test_category):
        index = model_result_item["id"]
        assert (
            ":" in index and len(index.split(":")) == 3
        ), f"Test entry ID {index} should contain exactly two colons for format sensitivity ids."
        format_sensitivity_config = index.split(":")[1]
        (
            return_format,
            has_tool_call_tag,
            _function_doc_format,
            _prompt_format,
            _prompt_style,
        ) = parse_prompt_variation_params(format_sensitivity_config)

        result = _evaluate_single_ast_entry(
            handler,
            model_result_item["id"],
            model_result_item["result"],
            possible_answer_entry["ground_truth"],
            prompt_entry,
            model_name,
            test_category,
            language=Language.PYTHON,
            return_format=ReturnFormat(return_format),
            has_tool_call_tag=has_tool_call_tag,
        )
    else:
        if is_java(test_category):
            language = Language.JAVA
            return_format = ReturnFormat.JAVA
        elif is_js(test_category):
            language = Language.JAVASCRIPT
            return_format = ReturnFormat.JAVASCRIPT
        else:
            language = Language.PYTHON
            return_format = ReturnFormat.PYTHON

        result = _evaluate_single_ast_entry(
            handler,
            model_result_item["id"],
            model_result_item["result"],
            possible_answer_entry["ground_truth"],
            prompt_entry,
            model_name,
            test_category,
            language=language,
            return_format=return_format,
            has_tool_call_tag=False,
        )
    return bool(result.get("valid", False))


def run_verifier_mode(args):
    system_prompt = _load_system_prompt()

    verifier_model_names = args.model if isinstance(args.model, list) else [args.model]
    if len(verifier_model_names) != 1:
        raise ValueError("Verifier mode expects exactly one verifier model in --model.")
    verifier_model = verifier_model_names[0]

    if not args.verifier_target_model:
        raise ValueError("--verifier-target-model is required in verifier mode.")
    target_model = args.verifier_target_model
    if target_model not in MODEL_CONFIG_MAPPING:
        raise ValueError(
            f"Unknown target model '{target_model}'. Please use the registry name used during generation."
        )

    base_url = args.verifier_base_url or os.getenv(
        "VERIFIER_BASE_URL",
        f"http://{os.getenv('LOCAL_SERVER_ENDPOINT', 'localhost')}:{os.getenv('LOCAL_SERVER_PORT', LOCAL_SERVER_PORT)}/v1",
    )
    api_key = args.verifier_api_key or os.getenv("VERIFIER_API_KEY", "EMPTY")
    verifier_client = OpenAI(base_url=base_url, api_key=api_key)

    temperature = getattr(args, "temperature", 0.001) or 0.0

    # Resolve dirs
    result_dir = PROJECT_ROOT / args.result_dir if args.result_dir else RESULT_PATH
    verifier_out_dir = (
        PROJECT_ROOT / args.verifier_output_dir
        if getattr(args, "verifier_output_dir", None)
        else RESULT_VERIFIER_PATH
    )
    score_out_dir = SCORE_VERIFIER_PATH

    test_categories = (
        parse_test_category_argument(args.test_category)
        if isinstance(args.test_category, list)
        else parse_test_category_argument([args.test_category])
    )

    target_handler = get_handler(target_model)
    model_name_dir = target_model.replace("/", "_")

    meta_summary = []

    for test_category in test_categories:
        # Locate target model results
        result_file = (
            result_dir
            / model_name_dir
            / get_directory_structure_by_category(test_category)
            / get_file_name_by_category(test_category, is_result_file=True)
        )
        if not result_file.exists():
            tqdm.write(f"⚠️ Missing result file for {target_model} / {test_category} at {result_file}, skipping.")
            continue

        model_results = load_file(result_file, sort_by_id=True)
        prompt_entries = load_dataset_entry(
            test_category, include_prereq=False, include_language_specific_hint=False
        )
        ground_truth_entries = load_ground_truth_entry(test_category)
        prompt_entries, ground_truth_entries = _subset_entries_by_model_ids(
            model_results, prompt_entries, ground_truth_entries, allow_missing=True
        )

        id_to_result = {entry["id"]: entry for entry in model_results}
        paired_entries = []
        for p_entry, gt_entry in zip(prompt_entries, ground_truth_entries):
            if p_entry["id"] in id_to_result:
                paired_entries.append((id_to_result[p_entry["id"]], p_entry, gt_entry))

        out_entries = []
        meta_correct = 0

        for model_result_item, prompt_entry, gt_entry in tqdm(
            paired_entries, total=len(paired_entries), desc=f"Verifier {test_category}"
        ):
            actual_valid = _meta_validity(
                target_handler,
                test_category,
                model_result_item,
                prompt_entry,
                gt_entry,
                target_model,
            )

            if is_multi_turn(test_category):
                decoded_calls = _decode_multi_turn_calls(
                    target_handler, model_result_item["result"]
                )
            else:
                try:
                    decoded_single = target_handler.decode_execute(
                        model_result_item["result"], has_tool_call_tag=False
                    )
                except Exception:
                    decoded_single = [str(model_result_item.get("result"))]
                decoded_calls = [[decoded_single]]

            execution_results = (
                _simulate_multi_turn_execution(
                    decoded_calls,
                    prompt_entry,
                    model_name_tag=f"verifierSim_{target_model}",
                )
                if is_multi_turn(test_category)
                else [[[]]]
            )

            user_payload = _build_verifier_payload(
                prompt_entry, decoded_calls, execution_results
            )

            try:
                raw_response = _chat_complete(
                    verifier_client, verifier_model, system_prompt, user_payload, temperature
                )
            except Exception as e:
                raw_response = f"Error calling verifier model: {e}"

            parsed = _safe_json_extract(raw_response) or {}
            verifier_score = parsed.get("score")
            critic = parsed.get("critic")

            meta_correct_flag = (
                isinstance(verifier_score, (int, float))
                and (verifier_score >= 0.5) == actual_valid
            )
            if meta_correct_flag:
                meta_correct += 1

            out_entries.append(
                {
                    "id": model_result_item["id"],
                    "test_category": test_category,
                    "target_model": target_model,
                    "verifier_model": verifier_model,
                    "verifier_input": user_payload,
                    "verifier_response": raw_response,
                    "parsed": parsed,
                    "actual_valid": actual_valid,
                    "meta_correct": meta_correct_flag,
                }
            )

        # Write verifier results per category
        verifier_model_dir = verifier_model.replace("/", "_")
        out_dir = (
            verifier_out_dir
            / verifier_model_dir
            / get_directory_structure_by_category(test_category)
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{get_file_name_by_category(test_category).replace('.json', '_verifier_result.json')}"
        with open(out_file, "w") as f:
            for entry in out_entries:
                f.write(json.dumps(entry) + "\n")

        total = len(out_entries)
        accuracy = meta_correct / total if total else 0.0
        meta_summary.append(
            {
                "test_category": test_category,
                "verifier_model": verifier_model,
                "target_model": target_model,
                "correct": meta_correct,
                "total": total,
                "accuracy": accuracy,
            }
        )

        # Score file
        score_dir = (
            score_out_dir
            / verifier_model_dir
            / get_directory_structure_by_category(test_category)
        )
        score_dir.mkdir(parents=True, exist_ok=True)
        score_file = score_dir / f"{get_file_name_by_category(test_category, is_score_file=True).replace('_score.json', '_verifier_score.json')}"
        with open(score_file, "w") as f:
            header = {
                "accuracy": accuracy,
                "correct_count": meta_correct,
                "total_count": total,
                "target_model": target_model,
                "verifier_model": verifier_model,
            }
            f.write(json.dumps(header) + "\n")

    # Save overall summary
    summary_dir = SCORE_VERIFIER_PATH / verifier_model.replace("/", "_")
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_file = summary_dir / "verifier_meta_summary.json"
    with open(summary_file, "w") as f:
        json.dump(meta_summary, f, indent=2)

    print(f"Verifier run complete. Results saved to {RESULT_VERIFIER_PATH}, scores to {SCORE_VERIFIER_PATH}.")
