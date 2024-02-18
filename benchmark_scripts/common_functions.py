import pandas as pd
import os
import math
import re
import argparse
from typing import Any, List, Tuple, Dict
import json


def get_datasets_info(
    dataset_length_list: List[int],
) -> List[Tuple[int, Any, Tuple[str, str, str]]]:
    df_list = []
    for dataset_length in dataset_length_list:
        output_file_path = (
            f"../spider_data/spider_equal_split_{str(dataset_length)}.csv"
        )
        df = pd.read_csv(output_file_path)
        df = df[df.columns[1:]]
        query_list = list(zip(df.context, df.question, df.hardness))
        gold_file_list = (df["query"], df["db_id"])
        df_list.append((dataset_length, query_list, gold_file_list))

    return df_list


def initialize_system_prompt(instruction_size: int) -> str:
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
    1. The answer generated must only be an SQL query ending with delimiter ";"
    2. make sure you use data only from the tables provided
    3. Be aware of case-sensitive data and Make sure all the required data is taken from the required columns and tables.
    4. Analyse the usage of JOINS if required between two or more tables. 
    5. use SQL functions like 'wildcards', 'procedures', 'exists', and 'case' to simplify the query if needed. {extra_instructions}
    
    [question]
    """.format(extra_instructions="".join(extra_instruction))


def initialize_files(model_file_path: str) -> Tuple[str, str]:
    if not os.path.exists(model_file_path):
        os.makedirs(model_file_path)

    log_file_path = f"{model_file_path}/execution-log.jsonl"

    output_file_path = f"{model_file_path}/predicted.txt"
    metrics_file_path = f"{model_file_path}/metrics.csv"
    metrics_file = open(metrics_file_path, "w", encoding="utf-8")
    metrics_file.write("response_time,llm_prompt_tokens,llm_response_tokens,hardness\n")

    open(output_file_path, "w", encoding="utf-8")
    return (output_file_path, metrics_file_path, log_file_path)


def generate_gold_file(gold_file_list: Tuple[Any, Any], model_file_path: str) -> None:
    query_list, db_id_list = gold_file_list
    with open(f"{model_file_path}/gold.txt", "w") as f:
        for i in range(len(query_list)):
            f.write(f"{query_list[i]}\t{db_id_list[i]}\n\n")


def get_elapsed_time(time_in_sec: int) -> None:
    minutes = time_in_sec // 60.0
    sec = time_in_sec % 60.0
    hr = minutes // 60.0
    minutes = minutes % 60.0
    return f"{math.trunc(hr)}hrs {math.trunc(minutes)}min {math.trunc(sec)}sec"


def write_to_file(
    output_file_path: str,
    metrics_file_path: str,
    output_file_text: str,
    metrics_file_text: str,
) -> None:
    with open(output_file_path, "a", encoding="utf-8") as file:
        file.write(output_file_text)
    with open(metrics_file_path, "a", encoding="utf-8") as file:
        file.write(metrics_file_text)


def log(log_text: str, data: str, log_file_path) -> None:
    data.update({"message": log_text})
    with open(log_file_path, "a") as json_file:
        json.dump(data, json_file)
        json_file.write("\n")


def get_parsed_args(supported_models: Dict, host_env: str) -> Tuple[Any, Dict]:
    parser = argparse.ArgumentParser(
        description=f"Run {host_env} specific NL-to-SQL LLM benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help=f"The model to use for this benchmark test. Supported models: {supported_models}",
    )
    parser.add_argument(
        "--model-instructions",
        "--model-inst",
        type=str,
        dest="model_instructions",
        default="",
        help=(
            "A semi-colon separated list of model specific instruction set to include in the results, e.g. "
            "cl-34=7,9,11;cl-70=5,7,9. The models specifed will run inferences for these instruction sets alone"
        ),
    )
    parser.add_argument(
        "--instructions",
        "--inst",
        dest="inst",
        type=str,
        default="5,7,9,11,13",
        help=(
            "A comma separated list of instructions set to include in the results, e.g. "
            "5,7,9. The models specifed will run inferences for these instruction sets alone"
        ),
    )
    parser.add_argument(
        "--inferences-length",
        "--il",
        dest="inf_length",
        type=str,
        default="50,100,200,400",
        help=(
            "A comma separated list of inferences to include for each results, currently supported inference lengths: "
            "50,100,200,400. The models specifed will run inferences for these infernce-lengths alone"
        ),
    )

    parsed_args = parser.parse_args()

    model_instructions = {}
    if parsed_args.model_instructions:
        for item in parsed_args.model_instructions.split(";"):
            key, value = item.split("=")
            model_instructions[key] = [int(inst) for inst in value.split(",")]

    return (parsed_args, model_instructions)


def initial_sql_match(sql_string: str) -> str:
    intial_regex = r"SELECT\b[^;]+"

    sql_string = (
        sql_string.replace("\n", " ")
        .replace("\\n", " ")
        .replace("\r", " ")
        .replace("\t", " ")
        .replace("```", " ")
        .replace("...", " ")
    )
    match = re.findall(intial_regex, sql_string, re.IGNORECASE)
    if match:
        return match[0]
    else:
        return sql_string


def intermediate_sql_match(sql_string: str) -> Tuple[bool, str]:
    processed_str = initial_sql_match(sql_string)

    intermediate_regex = r"^(.*?)(?:\s*Explanation|\s*Caution|\s*This query|\s*The code|\s*Please|\s*The above query|\s*This SQL query|$)"
    match = re.findall(intermediate_regex, processed_str, re.IGNORECASE)
    if match:
        return (True, match[0])
    else:
        return (False, processed_str)


def sql_match(sql_string: str) -> Tuple[bool, str]:
    is_sql_match, processed_str = intermediate_sql_match(sql_string)
    if is_sql_match:
        return (True, processed_str)

    sql_keywords = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "CREATE",
        "ALTER",
        "DROP",
        "TRUNCATE",
        "HAVING",
        "UNION",
        "INTERSECT",
        "EXCEPT",
        "IN",
        "NOT IN",
        "BETWEEN",
        "LIKE",
        "IS NULL",
        "IS NOT NULL",
        "AND",
        "OR",
        "AS",
        "DISTINCT",
        "COUNT",
        "SUM",
        "AVG",
        "MAX",
        "MIN",
        "CASE",
        "WHEN",
        "THEN",
        "ELSE",
        "END",
        "ALL",
        "ANY",
        "EXISTS",
        "UNIQUE",
        "PRIMARY KEY",
        "FOREIGN KEY",
        "REFERENCES",
        "CHECK",
        "INDEX",
        "CASCADE",
        "CONSTRAINT",
        "DEFAULT",
        "AUTO_INCREMENT",
        "SEQUENCE",
        "VIEW",
        "TABLE",
        "DATABASE",
        "PROCEDURE",
        "FUNCTION",
        "TRIGGER",
        "GRANT",
        "REVOKE",
        "COMMIT",
        "ROLLBACK",
        "SAVEPOINT",
        "START TRANSACTION",
        "LOCK",
        "UNLOCK",
        "SELECT",
        "FROM",
        "WHERE",
        "JOIN",
        "ON",
        "GROUP BY",
        "ORDER BY",
        "LIMIT",
        "insert",
        "update",
        "delete",
        "create",
        "alter",
        "drop",
        "truncate",
        "having",
        "union",
        "intersect",
        "except",
        "in",
        "not in",
        "between",
        "like",
        "is null",
        "is not null",
        "and",
        "or",
        "as",
        "distinct",
        "count",
        "sum",
        "avg",
        "max",
        "min",
        "case",
        "when",
        "then",
        "else",
        "end",
        "all",
        "any",
        "exists",
        "unique",
        "primary key",
        "foreign key",
        "references",
        "check",
        "index",
        "cascade",
        "constraint",
        "default",
        "auto_increment",
        "sequence",
        "view",
        "table",
        "database",
        "procedure",
        "function",
        "trigger",
        "grant",
        "revoke",
        "commit",
        "rollback",
        "savepoint",
        "start transaction",
        "lock",
        "unlock",
        "select",
        "from",
        "where",
        "join",
        "on",
        "group by",
        "order by",
        "limit",
        "=",
        "*",
        "group",
        "GROUP",
        "by",
        "BY",
        ">",
        "<",
        ">=",
        "<=",
    ]
    answer = ""
    words = processed_str.split()
    for idx1 in range(len(words)):
        flag = 0
        if words[idx1].lower() == "from":
            for idx2 in range(idx1, len(words)):
                choice = 0
                for keyword in sql_keywords:
                    if words[idx2] == keyword:
                        choice = 1
                        answer += words[idx2]
                        answer += " "
                        break
                if choice == 1:
                    continue
                else:
                    if (idx2 + 1) >= len(words):
                        answer += words[idx2]
                        answer += " "
                        continue
                    if len(words[idx2 + 1]) == 1:
                        answer += words[idx2]
                        answer += " "
                        continue
                    for key in sql_keywords:
                        if words[idx2 + 1] == key:
                            choice = 1
                            answer += words[idx2]
                            answer += " "
                            break
                    if choice == 1:
                        continue
                    else:
                        if (idx2 + 2) >= len(words):
                            answer += words[idx2]
                            answer += " "
                            continue
                        if len(words[idx2 + 2]) == 1:
                            answer += words[idx2]
                            answer += " "
                            continue
                        for keyword in sql_keywords:
                            if words[idx2 + 2] == keyword:
                                choice = 1
                                answer += words[idx2]
                                answer += " "
                                break
                        if choice == 0:
                            flag = 1
                            if words[idx2 - 1].lower() != "end":
                                answer += words[idx2]
                                answer += " "
                            break
                if flag == 1:
                    break
            break
        else:
            answer += words[idx1]
            answer += " "

    if answer == processed_str:
        return (False, answer)
    else:
        return (True, answer)
