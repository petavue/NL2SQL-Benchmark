import pandas as pd
import os
import re
import boto3
import json
import math
from datetime import datetime
import pathlib

CURRENT_FILE_PATH = pathlib.Path(__file__).parent.resolve()

# Set environment variables
exec(open(f"{CURRENT_FILE_PATH}/../set_env_vars.py").read())

# Get environment variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

MODEL_META_LLAMA = "meta.llama2-70b-chat-v1"
MODEL_ANTHROPIC_CLAUDE = "anthropic.claude-v2"

supported_models = [MODEL_META_LLAMA, MODEL_ANTHROPIC_CLAUDE]


def get_dataset_dataframes():
    df_list = []
    for dataset_length in dataset_length_list:
        file_path = f"../spider_data/spider_equal_split_{str(dataset_length)}.csv"
        df = pd.read_csv(file_path)
        df = df[df.columns[1:]]
        query_list = list(zip(df.context, df.question, df.hardness))
        df_list.append((dataset_length, df, query_list))

    return df_list


def initialize_system_prompt(instruction_size):
    INSTRUCTIONS_6_TO_7 = """
    6. Spend time to get the right databases,tables,and columns required for the question
    7. give attention to primary and foreign keys"""
    INSTRUCTIONS_8_TO_9 = """
    8. analyze the table constraints for columns such as unique etc.
    9. understand the question and analyze where to correctly use 'GROUP BY', 'HAVING', 'UNION' etc."""
    INSTRUCTIONS_10_TO_11 = """
    10. first, thoroughly go through the question, and figure out which columns of tables need to be chosen. comprehend what data is required and execute.
    11. use proper alias names for tables where grouping or joining is required to ensure clarity"""
    INSTRUCTIONS_12_TO_13 = """
    12. You should not perform any write operations, such as modifying, updating, deleting, or dropping data in the database. If a task requires such operations, you should return a message indicating that you are 'I'm sorry, but I can't assist with that.
    13. the query must be compatible with the Database request"""

    extra_instruction = []
    if instruction_size >= 7:
        extra_instruction.append(INSTRUCTIONS_6_TO_7)
    if instruction_size >= 9:
        extra_instruction.append(INSTRUCTIONS_8_TO_9)
    if instruction_size >= 11:
        extra_instruction.append(INSTRUCTIONS_10_TO_11)
    if instruction_size >= 13:
        extra_instruction.append(INSTRUCTIONS_12_TO_13)

    return """
    You are an SQL query generator. Given a question, you must generate a SQL query. If unsure do not assume the answer and give the default answer as "I don't know". Refer to the below context:
    [context]
    
    Also, Adhere to the following instructions:
    1. The answer generated must only be an SQL query
    2. make sure you use data only from the tables provided
    3. Be aware of case-sensitive data and Make sure all the required data is taken from the required columns and tables.
    4. Analyse the usage of JOINS if required between two or more tables. 
    5. use SQL functions like 'wildcards', 'procedures', 'exists', and 'case' to simplify the query if needed. {extra_instructions}
    
    [question]
    """.format(extra_instructions="".join(extra_instruction))


def initialize_amz_bedrock():
    return boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def initialize_files(model_file_path):
    if not os.path.exists(model_file_path):
        os.makedirs(model_file_path)

    file_path = f"{model_file_path}/predicted.txt"
    metrics_file_path = f"{model_file_path}/metrics.csv"
    metrics_file = open(metrics_file_path, "w", encoding="utf-8")
    metrics_file.write("response_time,llm_response_tokens,hardness \n")

    open(file_path, "w", encoding="utf-8")
    return (file_path, metrics_file_path)


def write_to_file(file_path, metrics_file_path, predicted_file_text, metrics_file_text):
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(predicted_file_text)
    with open(metrics_file_path, "a", encoding="utf-8") as file:
        file.write(metrics_file_text)


def run_queries_on_bedrock(file_path, metrics_file_path, model_name):
    res_query = []
    for context, question, hardness in query_list:
        try:
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

            response_time_start = datetime.now()
            anyscale_response = bedrock_runtime_client.invoke_model(
                modelId=model_name, body=json.dumps(body)
            )
            response_time_stop = datetime.now()

            response_body = json.loads(anyscale_response["body"].read())
            llm_response_tokens = anyscale_response["ResponseMetadata"]["HTTPHeaders"][
                "x-amzn-bedrock-output-token-count"
            ]
            if model_name == MODEL_ANTHROPIC_CLAUDE:
                llm_response_content = response_body["completion"]
            else:
                llm_response_content = response_body["generation"]

            regex_pattern = r"\bSELECT\b[^;]+;"
            match = re.search(regex_pattern, llm_response_content)
            if match:
                llm_response_content = match.group(0)
            else:
                write_to_file(
                    file_path,
                    metrics_file_path,
                    "I don't know\n\n",
                    f"{0},{llm_response_tokens},{hardness} \n",
                )
                continue

            sql_response = re.sub(r"\n", " ", llm_response_content)

            res_query.append(llm_response_content)
            response_time = response_time_stop - response_time_start
            write_to_file(
                file_path,
                metrics_file_path,
                f"{sql_response}\n\n",
                f"{response_time},{llm_response_tokens},{hardness} \n",
            )
        except Exception as e:
            print(f"An error occurred: {e}")
            write_to_file(
                file_path,
                metrics_file_path,
                f"An error occurred: {e}\n\n",
                f"{0},{0},{hardness} \n",
            )


def generate_gold_file(df):
    qry_lis = df["query"]
    db_id_lis = df["db_id"]
    with open(f"{model_file_path}/gold.txt", "w") as f:
        for i in range(len(qry_lis)):
            f.write(f"{qry_lis[i]}\t{db_id_lis[i]}\n\n")


def get_elapsed_time(time_in_sec):
    minutes = time_in_sec // 60.0
    sec = time_in_sec % 60.0
    hr = minutes // 60.0
    minutes = minutes % 60.0
    return f"{math.trunc(hr)}hrs {math.trunc(minutes)}min {math.trunc(sec)}sec"


dataset_length_list = [50, 100, 200, 400]
instruction_size_list = [5, 7, 9, 11, 13]
df_array = get_dataset_dataframes()

bedrock_runtime_client = initialize_amz_bedrock()

for instruction_size in instruction_size_list:
    system_prompt = initialize_system_prompt(instruction_size)
    for llm_model in supported_models:
        for dataset_length, df, query_list in df_array:
            model_file_path = (
                f"{CURRENT_FILE_PATH}/{instruction_size}/{llm_model}/{dataset_length}"
            )
            loop_start_time = datetime.now()
            print(f"Starting loop for {dataset_length} records")
            file_path, metrics_file_path = initialize_files(model_file_path)
            run_queries_on_bedrock(file_path, metrics_file_path, llm_model)
            generate_gold_file(df)
            loop_end_time = datetime.now()
            total_secs = (loop_end_time - loop_start_time).total_seconds()
            print(
                f"Time taken for {dataset_length} records: {get_elapsed_time(total_secs)}"
            )
