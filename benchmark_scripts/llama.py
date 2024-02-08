# Import torch for datatype attributes

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import torch
import pandas as pd
import math
import re
from datetime import datetime

start_limit = 0
csv_file_to_read = "../dev-data/sample_cleaned200.csv"
chunk_size = 30
file_path = "predicted.txt"
metrics_file_path = "metrics.csv"
execution_log_file_path = "execution-log.txt"
metrics_to_consider = ["hardness","gpu-memory-allocation", "encoding-time", "token-generation-time", "decoding-time"]

# Define variable to hold llama2 weights naming
name = "meta-llama/Llama-2-70b-chat-hf"

# Set auth token variable from hugging face
auth_token = "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"


def initialize_model_and_tokenizer():
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        name, cache_dir="./model/", token=auth_token
    )

    # Create model

    model = AutoModelForCausalLM.from_pretrained(
        name,
        device_map="auto",
        max_memory={0: "19GB", 1: "38GB", 2: "57GB", 3: "80GB"},
        cache_dir="./model/",
        token=auth_token,
        torch_dtype=torch.float16,
        rope_scaling={"type": "dynamic", "factor": 2},
        load_in_8bit=True,
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


def initialize_system_prompt():
    return """
    You are an SQL query generator. Given a question, you must generate a SQL query. If unsure do not assume the answer and give the default answer as "I don't know". Refer to the below context:
    {context}
    
    Also, Adhere to the following instructions:
    1. The answer generated must only be an SQL query
    2. make sure you use data only from the tables provided
    3. Be aware of case-sensitive data and Make sure all the required data is taken from the required columns and tables.
    4. Analyse the usage of JOINS if required between two or more tables. 
    5. use SQL functions like 'wildcards', 'procedures', 'exists', and 'case' to simplify the query if needed.
    6. Spend time to get the right databases,tables,and columns required for the question
    7. give attention to primary and foreign keys
    8. analyze the table constraints for columns such as unique etc.
    9. understand the question and analyze where to correctly use 'GROUP BY', 'HAVING', 'UNION' etc.
    10. first, thoroughly go through the question, and figure out which columns of tables need to be chosen. comprehend what data is required and execute.
    11. use proper alias names for tables where grouping or joining is required to ensure clarity
    12. You should not perform any write operations, such as modifying, updating, deleting, or dropping data in the database. If a task requires such operations, you should return a message indicating that you are 'I'm sorry, but I can't assist with that.
    13. the query must be compatible with the Database request
    
    {question}
    """


def initialize_files():
    open(file_path, "w", encoding="utf-8")
    open(execution_log_file_path, "w", encoding="utf-8")
    metrics_file = open(metrics_file_path, "w", encoding="utf-8")
    metrics_file.write("context, question \n")


def fetch_query_list(df):
    total_user_query = []
    for index, row in df.iterrows():
        context_col = row["context"]
        question_col = row["question"]
        total_user_query.append((context_col, question_col))
    return total_user_query


def run_queries_on_model(total_user_query):
    for idx, prompt_info in enumerate(total_user_query):
        try:
            prompt = system_prompt.replace("{context}", prompt_info[0]).replace(
                "{question}", prompt_info[1]
            )
            # Pass the prompt to the tokenizer
            inputs = tokenizer(prompt, return_tensors="pt").to(f"cuda:{idx % 3}")
            # Setup the text streamer
            streamer = TextStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            output = model.generate(
                **inputs, streamer=streamer, use_cache=True, max_new_tokens=float("inf")
            )

            assistant_response = tokenizer.decode(output[0], skip_special_tokens=True)

            with open(execution_log_file_path, "a", encoding="utf-8") as file:
                file.write(
                    f"{assistant_response}\n******************************************\n"
                )

            regex_pattern = r"\bSELECT\b[^;]+;"
            match = re.search(regex_pattern, assistant_response)
            if match:
                assistant_response = match.group(0)
            else:
                with open(file_path, "a", encoding="utf-8") as file:
                    file.write("I don't know\n\n")
                continue

            act_res = re.sub(r"\n", "", assistant_response)
            with open(file_path, "a", encoding="utf-8") as file:
                file.write(f"{act_res}\n\n")

        except Exception as e:
            print(f"An error occurred: {e}")
            with open(file_path, "a", encoding="utf-8") as file:
                file.write(f"No Response:: {e}\n\n")


def get_elapsed_time(time_in_sec):
    minutes = time_in_sec // 60.0
    sec = time_in_sec % 60.0
    hr = minutes // 60.0
    minutes = minutes % 60.0
    return f"{math.trunc(hr)}hrs {math.trunc(minutes)}min {math.trunc(sec)}sec"


df = pd.read_csv(csv_file_to_read)
df = df[df.columns[1:]]

query_list = fetch_query_list(df)

if start_limit == 0:
    initialize_files()

tokenizer, model = initialize_model_and_tokenizer()
test_model(tokenizer, model)

system_prompt = initialize_system_prompt()

# loop_start_time = datetime.now()
# run_queries_on_model(query_list[start_limit : start_limit + chunk_size])
# loop_end_time = datetime.now()

# total_secs = (loop_end_time - loop_start_time).total_seconds()
# print(f"Time taken for 30 records: {get_elapsed_time(total_secs)}")
