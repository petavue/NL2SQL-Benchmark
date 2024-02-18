import os
import boto3
import json
from datetime import datetime
import pathlib
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
from typing import Any, Tuple, Dict
from common_constants import Defaults, Environments

CURRENT_FILE_PATH = pathlib.Path(__file__).parent.resolve()

# Set environment variables
exec(open(f"{CURRENT_FILE_PATH}/../set_env_vars.py").read())
HOST_ENV = Environments.AMZ_BEDROCK

# Get environment variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

MODEL_META_LLAMA = "meta.llama2-70b-chat-v1"
MODEL_ANTHROPIC_CLAUDE = "anthropic.claude-v2"

supported_models = {
    "claude": MODEL_ANTHROPIC_CLAUDE,
    "llama": MODEL_META_LLAMA,
}


def initialize_amz_bedrock() -> Any:
    return boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def run_queries_on_bedrock(
    total_user_query: Tuple[str, str, str],
    output_file_path: str,
    metrics_file_path: str,
    log_file_path: str,
    model_name: str,
    system_prompt: str,
    bedrock_runtime_client: Any,
    data_to_log: Dict,
) -> None:
    for context, question, hardness in total_user_query:
        try:
            prompt = system_prompt.replace("[context]", context).replace(
                "[question]", question
            )
            if model_name == MODEL_ANTHROPIC_CLAUDE:
                prompt = f"\n\nHuman: ${prompt}\n\nAssistant:"

            body = {"prompt": prompt}

            if model_name == MODEL_ANTHROPIC_CLAUDE:
                body["max_tokens_to_sample"] = 3000

            data_to_log["request"] = body

            response_time_start = datetime.now()
            bedrock_response = bedrock_runtime_client.invoke_model(
                modelId=model_name, body=json.dumps(body)
            )
            response_time_stop = datetime.now()
            response_body = json.loads(bedrock_response["body"].read())

            # remove 'body' entry from bedrock_response due to 'StreamingBody is not JSON serializable' exception while logging
            data_to_log["response"] = {
                key: bedrock_response[key] for key in bedrock_response if key != "body"
            }
            data_to_log["response_body"] = response_body

            bedrock_response_headers = bedrock_response["ResponseMetadata"][
                "HTTPHeaders"
            ]
            llm_response_tokens = bedrock_response_headers[
                "x-amzn-bedrock-output-token-count"
            ]
            llm_prompt_tokens = bedrock_response_headers[
                "x-amzn-bedrock-input-token-count"
            ]

            if model_name == MODEL_ANTHROPIC_CLAUDE:
                llm_response_content = response_body["completion"]
            else:
                llm_response_content = response_body["generation"]

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
    bedrock_runtime_client = initialize_amz_bedrock()

    for model_name_from_args in args.models.split(","):
        if model_instructions:
            instruction_size_list = model_instructions[model_name_from_args]
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

                data_to_log = {
                    "environment": HOST_ENV,
                    "model": model_name,
                    "instruction_size": instruction_size,
                    "dataset_length": dataset_length,
                    "severity": "info",
                    "is_sql": 0,
                }
                print(
                    f"Starting loop for {model_name} - {instruction_size} instructions - {dataset_length} records"
                )
                loop_start_time = datetime.now()
                run_queries_on_bedrock(
                    query_list,
                    output_file_path,
                    metrics_file_path,
                    log_file_path,
                    model_name,
                    system_prompt,
                    bedrock_runtime_client,
                    data_to_log,
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
