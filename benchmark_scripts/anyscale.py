from datetime import datetime
import os
import pathlib
import requests
from openai import AsyncOpenAI
import asyncio
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
    get_few_shot_sample_string,
)
from typing import Tuple, Dict, Any
from common_constants import Defaults, Environments

CURRENT_FILE_PATH = pathlib.Path(__file__).parent.resolve()

# Set environment variables
exec(open(f"{CURRENT_FILE_PATH}/../set_env_vars.py").read())
HOST_ENV = Environments.ANYSCALE

# Get environment variables
ANY_SCALE_API_KEY = os.getenv("ANY_SCALE_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_KEY")

ANY_SCALE_BASE_URL = "https://api.endpoints.anyscale.com/v1"

SHOT_SAMPLE = "0-shot-cot"

MODEL_META_LLAMA = "meta-llama/Llama-2-70b-chat-hf"
MODEL_META_CODELLAMA_70B = "codellama/CodeLlama-70b-Instruct-hf"
MODEL_META_CODELLAMA_34B = "codellama/CodeLlama-34b-Instruct-hf"
MODEL_MISTRALAI_MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.1"
MODEL_MISTRALAI_MIXTRAL_8X7B = "mistralai/Mixtral-8x7B-Instruct-v0.1"
MODEL_GPT_3 = "gpt-3.5-turbo-16k"
MODEL_GPT_4 = "gpt-4-turbo-preview"

supported_models = {
    "cl-70": MODEL_META_CODELLAMA_70B,
    "cl-34": MODEL_META_CODELLAMA_34B,
    "mistral": MODEL_MISTRALAI_MISTRAL_7B,
    "mixtral": MODEL_MISTRALAI_MIXTRAL_8X7B,
    "llama": MODEL_META_LLAMA,
    "gpt-4": MODEL_GPT_4,
    "gpt-3": MODEL_GPT_3,
}


async def run_queries_on_anyscale(
    total_user_query: Tuple[str, str, str],
    output_file_path: str,
    metrics_file_path: str,
    log_file_path: str,
    model_name: str,
    system_prompt: str,
    instruction_size: int,
    dataset_length: int,
    client: Any,
    few_shot_samples,
) -> None:
    for context, question, hardness, db_id in total_user_query:
        try:
            data_to_log = {
                "environment": HOST_ENV,
                "model": model_name,
                "instruction_size": instruction_size,
                "dataset_length": dataset_length,
                "severity": "info",
                "is_sql": 0,
            }

            system_prompt = get_few_shot_sample_string(few_shot_samples[db_id], system_prompt)
            req = [
                {
                    "role": "system",
                    "content": system_prompt.replace("[context]", context).replace(
                        "[question]", ""
                    ),
                },
                {
                    "role": "user",
                    "content": f"{question}",
                    # "content": f"{question} \n    Let's think step by step.",
                },
            ]
            data_to_log["request"] = req

            response_time_start = datetime.now()
            anyscale_response = await client.chat.completions.create(
                model=model_name,
                messages=req,
            )
            response_time_stop = datetime.now()
            data_to_log["response"] = str(anyscale_response)

            llm_response_content = anyscale_response.choices[0].message.content
            llm_response_tokens = anyscale_response.usage.completion_tokens
            llm_prompt_tokens = anyscale_response.usage.prompt_tokens

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


async def run_inferences(args: Dict, model_instructions: Dict) -> None:
    inference_length_in_args = [int(inst) for inst in args.inf_length.split(",")]
    dataset_length_list = inference_length_in_args or Defaults.INFERENCE_LENGTH_LIST

    datasets_info, few_shot_samples = get_datasets_info(dataset_length_list)

    for model_name_from_args in args.models.split(","):
        if model_instructions:
            instruction_size_list = model_instructions[model_name_from_args]
        elif args.inst:
            instruction_size_list = [int(inst) for inst in args.inst.split(",")]
        else:
            instruction_size_list = Defaults.INSTRUCTION_SIZE_LIST

        model_name = supported_models[model_name_from_args]

        if model_name in [MODEL_GPT_4, MODEL_GPT_3]:
            client = AsyncOpenAI(api_key=OPENAI_KEY)
        else:
            client = AsyncOpenAI(api_key=ANY_SCALE_API_KEY, base_url=ANY_SCALE_BASE_URL)

        for instruction_size in instruction_size_list:
            system_prompt = initialize_system_prompt(instruction_size)

            for dataset_length, query_list, gold_file_list in datasets_info:
                model_file_path = f"{CURRENT_FILE_PATH}/{HOST_ENV}/{SHOT_SAMPLE}/{model_name}/{instruction_size}_Instructions/{dataset_length}_Inferences"

                output_file_path, metrics_file_path, log_file_path = initialize_files(
                    model_file_path
                )

                print(
                    f"Starting loop for {model_name} - {instruction_size} instructions - {dataset_length} inferences"
                )
                loop_start_time = datetime.now()
                await run_queries_on_anyscale(
                    query_list,
                    output_file_path,
                    metrics_file_path,
                    log_file_path,
                    model_name,
                    system_prompt,
                    instruction_size,
                    dataset_length,
                    client,
                    few_shot_samples,
                )
                generate_gold_file(gold_file_list, model_file_path)
                loop_end_time = datetime.now()
                total_secs = (loop_end_time - loop_start_time).total_seconds()
                print(
                    f"Time taken for {dataset_length} records: {get_elapsed_time(total_secs)}"
                )


if __name__ == "__main__":
    args, model_instructions = get_parsed_args(supported_models, HOST_ENV)

    asyncio.run(run_inferences(args, model_instructions))
