# Import torch for datatype attributes

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import pandas as pd
import math
import re
from datetime import datetime
import os
import pathlib

CURRENT_FILE_PATH = pathlib.Path(__file__).parent.resolve()

# Set environment variables
exec(open(f"{CURRENT_FILE_PATH}/../set_env_vars.py").read())

# Get environment variables
auth_token = os.getenv("HUGGING_FACE_TOKEN")

MODEL_META_CODELLAMA_70B = "codellama/CodeLlama-70b-Instruct-hf"
MODEL_META_CODELLAMA_34B = "codellama/CodeLlama-34b-Instruct-hf"
MODEL_MISTRALAI_MISTRL_7B = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_MISTRALAI_MIXTRAL_8X7B = "mistralai/Mixtral-8x7B-Instruct-v0.1"
MODEL_WIZARDLM_WIZARD_CODER_33B = "WizardLM/WizardCoder-33B-V1.1"
MODEL_DEFOG_SQLCODER_70B = "defog/sqlcoder-70b-alpha"
MODEL_DEFOG_SQLCODER_7B_2 = "defog/sqlcoder-7b-2"


supported_models = [
    # MODEL_META_CODELLAMA_70B,
    MODEL_META_CODELLAMA_34B,
    # MODEL_MISTRALAI_MISTRL_7B,
]

model_tensor_types = {
    MODEL_META_CODELLAMA_70B: torch.bfloat16,
    MODEL_META_CODELLAMA_34B: torch.bfloat16,
    MODEL_WIZARDLM_WIZARD_CODER_33B: torch.bfloat16,
    MODEL_DEFOG_SQLCODER_70B: torch.float16,
    MODEL_DEFOG_SQLCODER_7B_2:torch.float16
}


def get_dataset_dataframes(dataset_length_list):
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
    
    """.format(extra_instructions="".join(extra_instruction))


def initialize_files(model_file_path):
    if not os.path.exists(model_file_path):
        os.makedirs(model_file_path)

    file_path = f"{model_file_path}/predicted.txt"
    metrics_file_path = f"{model_file_path}/metrics.csv"
    metrics_file = open(metrics_file_path, "w", encoding="utf-8")
    metrics_file.write("response_time,llm_response_tokens,hardness\n")

    open(file_path, "w", encoding="utf-8")
    return (file_path, metrics_file_path)


def write_to_file(file_path, metrics_file_path, predicted_file_text, metrics_file_text):
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(predicted_file_text)
    with open(metrics_file_path, "a", encoding="utf-8") as file:
        file.write(metrics_file_text)


def initialize_model_and_tokenizer(model_name):
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir="./model/", token=auth_token
    )

    # Create model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        max_memory={0: "40GB", 1: "40GB", 2: "40GB", 3: "80GB"},
        cache_dir="./model/",
        token=auth_token,
        torch_dtype=model_tensor_types[model_name],
        rope_scaling={"type": "dynamic", "factor": 2},
    )

    return (tokenizer, model)


def test_model(tokenizer, model):
    # Setup a prompt
    prompt = "### User:What is the fastest car in  \
            the world and how much does it cost? \
            ### Assistant:"
    # Pass the prompt to the tokenizer
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Setup the text streamer
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Actually run the thing
    output = model.generate(
        **inputs, streamer=streamer, use_cache=True, max_new_tokens=float("inf")
    )

    # Covert the output tokens back to text
    return tokenizer.decode(output[0], skip_special_tokens=True)


def run_queries_on_model(total_user_query, file_path, metrics_file_path, system_prompt):
    torch.cuda.empty_cache()
    for context, question, hardness in total_user_query:
        try:
            if model_name == MODEL_META_CODELLAMA_70B:
                prompt = [
                    {
                        "role": "system",
                        "content": system_prompt.replace("[context]", context),
                    },
                    {"role": "user", "content": question},
                ]
                inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(
                    "cuda"
                )
                response_time_start = datetime.now()
                output = model.generate(
                    input_ids=inputs, use_cache=True, max_new_tokens=500
                )
                response_time_stop = datetime.now()
            else:
                prompt = system_prompt.replace("[context]",context).replace(
                    "[question]", question
                )
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                # Setup the text streamer
                streamer = TextStreamer(
                    tokenizer, skip_prompt=True, skip_special_tokens=True
                )

                response_time_start = datetime.now()
                output = model.generate(
                    **inputs,
                    streamer=streamer,
                    use_cache=True,
                    max_new_tokens=float("inf"),
                )
                response_time_stop = datetime.now()

            llm_response_content = tokenizer.decode(output[0], skip_special_tokens=True)
            llm_response_tokens = len(output[0])

            regex_pattern = r"\bSELECT\b[^;]+;"
            match = re.search(regex_pattern, llm_response_content)
            if match:
                llm_response_content = match.group(0)
            else:
                write_to_file(
                    file_path,
                    metrics_file_path,
                    "I don't know\n\n",
                    f"{0},{llm_response_tokens},{hardness}\n",
                )
                continue

            sql_response = re.sub(r"\n", " ", llm_response_content)
            response_time = response_time_stop - response_time_start
            write_to_file(
                file_path,
                metrics_file_path,
                f"{sql_response}\n\n",
                f"{response_time},{llm_response_tokens},{hardness}\n",
            )
        except Exception as e:
            print(f"An error occurred: {e}")
            write_to_file(
                file_path,
                metrics_file_path,
                f"An error occurred: {e}\n\n",
                f"{0},{0},{hardness}\n",
            )


def generate_gold_file(df, model_file_path):
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
df_array = get_dataset_dataframes(dataset_length_list)

for instruction_size in instruction_size_list:
    system_prompt = initialize_system_prompt(instruction_size)
    for model_name in supported_models:
        tokenizer, model = initialize_model_and_tokenizer(model_name)
        for dataset_length, df, query_list in df_array:
            model_file_path = (
                f"{CURRENT_FILE_PATH}/{instruction_size}/{model_name}/{dataset_length}"
            )
            loop_start_time = datetime.now()
            print(f"Starting loop for {dataset_length} records")
            file_path, metrics_file_path = initialize_files(model_file_path)
            run_queries_on_model(
                query_list, file_path, metrics_file_path, system_prompt
            )
            generate_gold_file(df, model_file_path)
            loop_end_time = datetime.now()
            total_secs = (loop_end_time - loop_start_time).total_seconds()
            print(
                f"Time taken for {dataset_length} records: {get_elapsed_time(total_secs)}"
            )
