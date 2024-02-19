from datetime import datetime
import os
import asyncio
import pathlib
import requests
from common_functions import (
    get_datasets_info,
    initialize_system_prompt,
    initialize_files,
    generate_gold_file,
    write_to_file,
    log,
    get_elapsed_time,
    sql_match,
    get_parsed_args,
)
from typing import Tuple, Dict
from common_constants import Defaults, Environments

CURRENT_FILE_PATH = pathlib.Path(__file__).parent.resolve()

# Set environment variables
exec(open(f"{CURRENT_FILE_PATH}/../set_env_vars.py").read())
HOST_ENV = Environments.HUGGING_FACE

MODEL_META_CODELLAMA_70B = "codellama/CodeLlama-70b-Instruct-hf"
MODEL_META_CODELLAMA_34B = "codellama/CodeLlama-34b-Instruct-hf"

supported_models = {
    "cl-70": MODEL_META_CODELLAMA_70B,
    "cl-34": MODEL_META_CODELLAMA_34B,
}


def run_queries_on_hugging_face(
    total_user_query: Tuple[str, str, str],
    output_file_path: str,
    metrics_file_path: str,
    log_file_path: str,
    model_name: str,
    system_prompt: str,
    instruction_size: int,
    dataset_length: int,
) -> None:
    for context, question, hardness in total_user_query:
        try:
            data_to_log = {
                "environment": HOST_ENV,
                "model": model_name,
                "instruction_size": instruction_size,
                "dataset_length": dataset_length,
                "severity": "info",
                "is_sql": 0,
            }
            payload = f"<s>Source: system\n\n {system_prompt.replace("[context]", context).replace(
                        "[question]", ""
                    )}\n  <step> Source: user\n\n {question}  <step> Source: user: assistant\nDestination: user"

            body = {
                "inputs": payload,
                "parameters": {"max_new_tokens": 200, "stop": ["</s>", "<step>"]},
            }

            data_to_log["request"] = body
            headers = {"Authorization": f"Bearer { os.getenv("HUGGING_FACE_TOKEN")}"}
            API_URL = f"https://api-inference.huggingface.co/models/{model_name}"

            response_time_start = datetime.now()
            with requests.Session().post(
                API_URL, headers=headers, json=body
            ) as api_response:
                hf_response = api_response.json()
            response_time_stop = datetime.now()
            data_to_log["response"] = hf_response

            llm_response_content = hf_response[0]["generated_text"]
            llm_response_tokens = 0
            llm_prompt_tokens = 0

            if "select" in llm_response_content.lower():
                is_sql_match, sql_response = sql_match(llm_response_content)
            else:
                is_sql_match = False

            if not is_sql_match:
                data_to_log["severity"] = "warn"
                log("No SQL Output detected", data_to_log, log_file_path)
                write_to_file(
                    output_file_path,
                    metrics_file_path,
                    "I don't know\n\n",
                    f"{0},{llm_prompt_tokens},{llm_response_tokens},{hardness}\n",
                )
                continue

            data_to_log["is_sql"] = 1
            data_to_log["sql_response"] = sql_response
            log("SQL Response successful", data_to_log, log_file_path)

            response_time = response_time_stop - response_time_start
            write_to_file(
                output_file_path,
                metrics_file_path,
                f"{sql_response}\n\n",
                f"{response_time},{llm_prompt_tokens},{llm_response_tokens},{hardness}\n",
            )
        except Exception as ex:
            data_to_log["severity"] = "error"
            log(ex, data_to_log, log_file_path)
            print("exception: ", ex)
            write_to_file(
                output_file_path,
                metrics_file_path,
                f"An error occurred: {ex}\n\n",
                f"{0},{0},{0},{hardness}\n",
            )


def run_inferences(args: Dict, model_instructions: Dict) -> None:
    inference_length_in_args = [int(inst) for inst in args.inf_length.split(",")]
    dataset_length_list = inference_length_in_args or Defaults.INFERENCE_LENGTH_LIST

    datasets_info = get_datasets_info(dataset_length_list)

    for model_name_from_args in args.models.split(","):
        if model_instructions:
            instruction_size_list = model_instructions[model_name_from_args]
        elif args.inst:
            instruction_size_list = [int(inst) for inst in args.inst.split(",")]
        else:
            instruction_size_list = Defaults.INSTRUCTION_SIZE_LIST

        model_name = supported_models[model_name_from_args]

        for instruction_size in instruction_size_list:
            system_prompt = initialize_system_prompt(instruction_size)

            for dataset_length, query_list, gold_file_list in datasets_info:
                model_file_path = f"{CURRENT_FILE_PATH}/{HOST_ENV}/{model_name}/{instruction_size}_Instructions/{dataset_length}_Inferences"

                output_file_path, metrics_file_path, log_file_path = initialize_files(
                    model_file_path
                )
                print(
                    f"Starting loop for {model_name} - {instruction_size} instructions - {dataset_length} records"
                )
                loop_start_time = datetime.now()
                run_queries_on_hugging_face(
                    query_list,
                    output_file_path,
                    metrics_file_path,
                    log_file_path,
                    model_name,
                    system_prompt,
                    instruction_size,
                    dataset_length,
                )
                generate_gold_file(gold_file_list, model_file_path)
                loop_end_time = datetime.now()
                total_secs = (loop_end_time - loop_start_time).total_seconds()
                print(
                    f"Time taken for {dataset_length} records: {get_elapsed_time(total_secs)}"
                )


if __name__ == "__main__":
    args, model_instructions = get_parsed_args(supported_models, HOST_ENV)
    run_inferences(args, model_instructions)
