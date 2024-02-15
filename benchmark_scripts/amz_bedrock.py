import os
import boto3
import json
from datetime import datetime
import pathlib
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
HOST_ENV = "amz_bedrock"

# Get environment variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

MODEL_META_LLAMA = "meta.llama2-70b-chat-v1"
MODEL_ANTHROPIC_CLAUDE = "anthropic.claude-v2"

supported_models = [MODEL_META_LLAMA, MODEL_ANTHROPIC_CLAUDE]


def initialize_amz_bedrock():
    return boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def run_queries_on_bedrock(output_file_path, metrics_file_path, model_name):
    for context, question, hardness in query_list:
        try:
            data_to_log = {
                "environment": HOST_ENV,
                "model": model_name,
                "instruction_size": instruction_size,
                "dataset_length": dataset_length,
                "severity": "info",
                "is_sql": 0,
            }
            prompt = system_prompt.replace("[context]", context).replace(
                "[question]", question
            )
            if model_name == MODEL_ANTHROPIC_CLAUDE:
                prompt = f"\n\nHuman: ${prompt}\n\nAssistant:"

            body = {
                "prompt": prompt,
            }

            if model_name == MODEL_ANTHROPIC_CLAUDE:
                body["max_tokens_to_sample"] = 3000

            data_to_log["request"] = body
            response_time_start = datetime.now()
            bedrock_response = bedrock_runtime_client.invoke_model(
                modelId=model_name, body=json.dumps(body)
            )
            response_time_stop = datetime.now()
            data_to_log["response"] = bedrock_response

            response_body = json.loads(bedrock_response["body"].read())
            data_to_log["response_body"] = response_body
            
            llm_response_tokens = bedrock_response["ResponseMetadata"]["HTTPHeaders"][
                "x-amzn-bedrock-output-token-count"
            ]
            llm_prompt_tokens = bedrock_response["ResponseMetadata"]["HTTPHeaders"][
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


dataset_length_list = [50, 100, 200, 400]
instruction_size_list = [5, 7, 9, 11, 13]
df_array = get_dataset_dataframes(dataset_length_list)

bedrock_runtime_client = initialize_amz_bedrock()

for instruction_size in instruction_size_list:
    system_prompt = initialize_system_prompt(instruction_size)
    for model_name in supported_models:
        for dataset_length, df, query_list in df_array:
            model_file_path = f"{CURRENT_FILE_PATH}/{HOST_ENV}/{instruction_size}/{model_name}/{dataset_length}"
            loop_start_time = datetime.now()
            print(f"Starting loop for {dataset_length} records")
            output_file_path, metrics_file_path = initialize_files(model_file_path)
            run_queries_on_bedrock(output_file_path, metrics_file_path, model_name)
            generate_gold_file(df, model_file_path)
            loop_end_time = datetime.now()
            total_secs = (loop_end_time - loop_start_time).total_seconds()
            print(
                f"Time taken for {dataset_length} records: {get_elapsed_time(total_secs)}"
            )
