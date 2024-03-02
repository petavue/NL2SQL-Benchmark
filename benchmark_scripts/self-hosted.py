# Import torch for datatype attributes

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
import torch
from datetime import datetime
import os
import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial
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
    multi_process_setup,
)
from typing import Tuple, Dict, Any
from common_constants import Defaults, Environments

CURRENT_FILE_PATH = pathlib.Path(__file__).parent.resolve()

# Set environment variables
exec(open(f"{CURRENT_FILE_PATH}/../set_env_vars.py").read())
HOST_ENV = Environments.SELF_HOSTED

# Get environment variables
auth_token = os.getenv("HUGGING_FACE_TOKEN")

MODEL_META_CODELLAMA_70B = "codellama/CodeLlama-70b-Instruct-hf"
MODEL_META_CODELLAMA_34B = "codellama/CodeLlama-34b-Instruct-hf"
MODEL_MISTRALAI_MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.1"
MODEL_MISTRALAI_MIXTRAL_8X7B = "mistralai/Mixtral-8x7B-Instruct-v0.1"
MODEL_WIZARDLM_WIZARD_CODER_33B = "WizardLM/WizardCoder-33B-V1.1"
MODEL_DEFOG_SQLCODER_70B = "defog/sqlcoder-70b-alpha"
MODEL_DEFOG_SQLCODER_7B_2 = "defog/sqlcoder-7b-2"

supported_models = {
    "cl-70": MODEL_META_CODELLAMA_70B,
    "cl-34": MODEL_META_CODELLAMA_34B,
    "mistral": MODEL_MISTRALAI_MISTRAL_7B,
    "mixtral": MODEL_MISTRALAI_MIXTRAL_8X7B,
    "wc-33": MODEL_WIZARDLM_WIZARD_CODER_33B,
    "sqlc-70-a": MODEL_DEFOG_SQLCODER_70B,
    "sqlc-7-2": MODEL_DEFOG_SQLCODER_7B_2,
}

model_tensor_types = {
    MODEL_DEFOG_SQLCODER_70B: torch.float16,
    MODEL_DEFOG_SQLCODER_7B_2: torch.float16,
    MODEL_META_CODELLAMA_70B: torch.bfloat16,
    MODEL_META_CODELLAMA_34B: torch.bfloat16,
    MODEL_MISTRALAI_MISTRAL_7B: torch.bfloat16,
    MODEL_MISTRALAI_MIXTRAL_8X7B: torch.bfloat16,
    MODEL_WIZARDLM_WIZARD_CODER_33B: torch.bfloat16,
}


def initialize_model_and_tokenizer(model_name):
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir="./model/", token=auth_token
    )

    if model_name in [MODEL_MISTRALAI_MISTRAL_7B, MODEL_MISTRALAI_MIXTRAL_8X7B]:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            max_memory={0: "40GB", 1: "40GB", 2: "40GB", 3: "80GB"},
            cache_dir="./model/",
            token=auth_token,
            torch_dtype=model_tensor_types[model_name],
        )
    else:
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


def test_model(tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, model: Any):
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
        **inputs, streamer=streamer, use_cache=True, max_new_tokens=300
    )

    # Covert the output tokens back to text
    return tokenizer.decode(output[0], skip_special_tokens=True)


def run_queries_on_model(
    total_user_query: Tuple[str, str, str],
    output_file_path: str,
    metrics_file_path: str,
    log_file_path: str,
    model_name: str,
    system_prompt: str,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    model: Any,
    instruction_size: int,
    dataset_length: int,
) -> None:
    for context, question, hardness in total_user_query:
        torch.cuda.empty_cache()
        try:
            data_to_log = {
                "environment": HOST_ENV,
                "model": model_name,
                "instruction_size": instruction_size,
                "dataset_length": dataset_length,
                "severity": "info",
                "is_sql": 0,
            }
            if model_name in [
                MODEL_META_CODELLAMA_70B,
                MODEL_MISTRALAI_MISTRAL_7B,
                MODEL_MISTRALAI_MIXTRAL_8X7B,
            ]:
                if model_name == MODEL_META_CODELLAMA_70B:
                    prompt = [
                        {
                            "role": "system",
                            "content": system_prompt.replace(
                                "[context]", context
                            ).replace("[question]", ""),
                        },
                        {"role": "user", "content": question},
                    ]
                else:
                    prompt = [
                        {
                            "role": "user",
                            "content": system_prompt.replace(
                                "[context]", context
                            ).replace("[question]", ""),
                        },
                        {"role": "assistant", "content": question},
                    ]
                data_to_log["request"] = prompt

                inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(
                    "cuda"
                )
                llm_prompt_tokens = len(inputs)

                response_time_start = datetime.now()
                output = model.generate(
                    input_ids=inputs,
                    use_cache=True,
                    max_new_tokens=300,
                    pad_token_id=tokenizer.eos_token_id,
                )
                response_time_stop = datetime.now()
            else:
                prompt = system_prompt.replace("[context]", context).replace(
                    "[question]", question
                )
                data_to_log["request"] = prompt
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                llm_prompt_tokens = len(inputs)

                # Setup the text streamer
                streamer = TextStreamer(
                    tokenizer, skip_prompt=True, skip_special_tokens=True
                )

                response_time_start = datetime.now()
                output = model.generate(
                    **inputs,
                    streamer=streamer,
                    use_cache=True,
                    max_new_tokens=300,
                    pad_token_id=tokenizer.eos_token_id,
                )
                response_time_stop = datetime.now()

            llm_response_content = tokenizer.decode(output[0], skip_special_tokens=True)
            llm_response_tokens = len(output[0])
            data_to_log["response"] = llm_response_content

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


async def multi_process(
    instruction_size: int, datasets_info: list, model_name: str, args: Any
) -> None:
    system_prompt = initialize_system_prompt(instruction_size)
    tokenizer, model = initialize_model_and_tokenizer(model_name)
    for dataset_length, query_list, gold_file_list in datasets_info:
        model_file_path = f"{args.target_dir}/{HOST_ENV}/{model_name}/{instruction_size}_Instructions/{dataset_length}_Inferences"

        output_file_path, metrics_file_path, log_file_path = initialize_files(
            model_file_path
        )

        print(
            f"Starting loop for {model_name} - {instruction_size} instructions - {dataset_length} inferences"
        )
        loop_start_time = datetime.now()
        run_queries_on_model(
            query_list,
            output_file_path,
            metrics_file_path,
            log_file_path,
            model_name,
            system_prompt,
            tokenizer,
            model,
            instruction_size,
            dataset_length,
        )
        generate_gold_file(gold_file_list, model_file_path)
        loop_end_time = datetime.now()
        total_secs = (loop_end_time - loop_start_time).total_seconds()
        print(
            f"Time taken for {dataset_length} records: {get_elapsed_time(total_secs)}"
        )


async def main():
    args, model_instructions = get_parsed_args(supported_models, HOST_ENV)
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

        await multi_process_setup(
            multi_process, instruction_size_list, datasets_info, model_name, args
        )


if __name__ == "__main__":
    asyncio.run(main())
