from datetime import datetime
import re
import os
import asyncio
import pathlib
import requests
from common_functions import (
    get_dataset_dataframes,
    initialize_system_prompt,
    initialize_files,
    generate_gold_file,
    write_to_file,
    log,
    get_elapsed_time,
    sql_match,
)

CURRENT_FILE_PATH = pathlib.Path(__file__).parent.resolve()

# Set environment variables
exec(open(f"{CURRENT_FILE_PATH}/../set_env_vars.py").read())
HOST_ENV = "anyscale"

# Get environment variables
ANY_SCALE_API_KEY = os.getenv("ANY_SCALE_API_KEY")

ANY_SCALE_BASE_URL = "https://api.endpoints.anyscale.com/v1"

MODEL_META_LLAMA = "meta-llama/Llama-2-70b-chat-hf"
MODEL_META_CODELLAMA_70B = "codellama/CodeLlama-70b-Instruct-hf"
MODEL_META_CODELLAMA_34B = "codellama/CodeLlama-34b-Instruct-hf"
MODEL_MISTRALAI_MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.1"
MODEL_MISTRALAI_MIXTRAL_8X7B = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# supported_models = {
#     "cl-70": MODEL_META_CODELLAMA_70B,
#     "cl-34": MODEL_META_CODELLAMA_34B,
#     "mistral": MODEL_MISTRALAI_MISTRAL_7B,
#     "mixtral": MODEL_MISTRALAI_MIXTRAL_8X7B,
#     "llama-70": MODEL_META_LLAMA,
# }


supported_models = [
    MODEL_META_LLAMA,
    MODEL_META_CODELLAMA_70B,
    MODEL_META_CODELLAMA_34B,
    MODEL_MISTRALAI_MISTRAL_7B,
    MODEL_MISTRALAI_MIXTRAL_8X7B,
]


async def run_queries_on_anyscale(
    total_user_query,
    output_file_path,
    metrics_file_path,
    model_name,
    system_prompt,
    instruction_size,
    dataset_length,
):
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
            req = [
                {
                    "role": "system",
                    "content": system_prompt.replace("[context]", context).replace(
                        "[question]", ""
                    ),
                },
                {"role": "user", "content": question},
            ]
            data_to_log["request"] = req

            token = os.getenv("ANY_SCALE_API_KEY")
            url = f"{ANY_SCALE_BASE_URL}/chat/completions"
            body = {"model": model_name, "messages": req, "temperature": 0.7}

            response_time_start = datetime.now()
            with requests.Session().post(
                url, headers={"Authorization": f"Bearer {token}"}, json=body
            ) as api_response:
                anyscale_response = api_response.json()
            response_time_stop = datetime.now()
            data_to_log["response"] = anyscale_response

            llm_response_content = anyscale_response["choices"][0]["message"]["content"]
            llm_response_tokens = anyscale_response["usage"]["completion_tokens"]
            llm_prompt_tokens = anyscale_response["usage"]["prompt_tokens"]

            if "select" in llm_response_content.lower():
                is_sql_match, sql_response = sql_match(llm_response_content)
            else:
                is_sql_match = False

            if not is_sql_match:
                data_to_log["severity"] = "warn"
                log("No SQL Output detected", data_to_log, "warning")
                write_to_file(
                    output_file_path,
                    metrics_file_path,
                    "I don't know\n\n",
                    f"{0},{llm_prompt_tokens},{llm_response_tokens},{hardness}\n",
                )
                continue

            data_to_log["is_sql"] = 1
            data_to_log["sql_response"] = sql_response
            log("SQL Response successful", data_to_log)

            response_time = response_time_stop - response_time_start
            write_to_file(
                output_file_path,
                metrics_file_path,
                f"{sql_response}\n\n",
                f"{response_time},{llm_prompt_tokens},{llm_response_tokens},{hardness}\n",
            )
        except Exception as ex:
            data_to_log["severity"] = "error"
            log(ex, data_to_log, "error")
            print("exception: ", ex)
            write_to_file(
                output_file_path,
                metrics_file_path,
                f"An error occurred: {ex}\n\n",
                f"{0},{0},{0},{hardness}\n",
            )


async def main():
    dataset_length_list = [50, 100, 200, 400]
    instruction_size_list = [5, 7, 9, 11, 13]
    df_array = get_dataset_dataframes(dataset_length_list)

    for model_name in supported_models:
        for instruction_size in instruction_size_list:
            system_prompt = initialize_system_prompt(instruction_size)
            for dataset_length, df, query_list in df_array:
                model_file_path = f"{CURRENT_FILE_PATH}/{HOST_ENV}/{instruction_size}/{model_name}/{dataset_length}"
                loop_start_time = datetime.now()
                print(f"Starting loop for {dataset_length} records")
                output_file_path, metrics_file_path = initialize_files(model_file_path)
                await run_queries_on_anyscale(
                    query_list,
                    output_file_path,
                    metrics_file_path,
                    model_name,
                    system_prompt,
                    instruction_size,
                    dataset_length,
                )
                generate_gold_file(df, model_file_path)
                loop_end_time = datetime.now()
                total_secs = (loop_end_time - loop_start_time).total_seconds()
                print(
                    f"Time taken for {dataset_length} records: {get_elapsed_time(total_secs)}"
                )


if __name__ == "__main__":
    asyncio.run(main())
