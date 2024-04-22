# Import torch for datatype attributes

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
import torch
from datetime import datetime, timezone
import os
import pathlib
from common_functions import (
    get_datasets_info,
    initialize_files,
    generate_gold_file,
    write_to_file,
    log,
    get_elapsed_time,
    sql_match,
    get_parsed_args,
    get_instruction_shot_specific_prompt,
    generate_model_specific_prompt_for_self_hosted_model,
)
from typing import Tuple, Any
from common_constants import Defaults, Environments, SelfHostedModels, SelfHosted

CURRENT_FILE_PATH = pathlib.Path(__file__).parent.resolve()

# Set environment variables
exec(open(f"{CURRENT_FILE_PATH}/../set_env_vars.py").read())
HOST_ENV = Environments.SELF_HOSTED

# Get environment variables
auth_token = os.getenv("HUGGING_FACE_TOKEN")


supported_models = {
    "cl-70": SelfHostedModels.MODEL_META_CODELLAMA_70B,
    "cl-34": SelfHostedModels.MODEL_META_CODELLAMA_34B,
    "mistral": SelfHostedModels.MODEL_MISTRALAI_MISTRAL_7B,
    "mixtral": SelfHostedModels.MODEL_MISTRALAI_MIXTRAL_8X7B,
    "wc-33": SelfHostedModels.MODEL_WIZARDLM_WIZARD_CODER_33B,
    "sqlc-70-a": SelfHostedModels.MODEL_DEFOG_SQLCODER_70B,
    "sqlc-7-2": SelfHostedModels.MODEL_DEFOG_SQLCODER_7B_2,
    "mistral-v2": SelfHostedModels.MODEL_MISTRALAI_MISTRAL_7B_V2,
    "dbrx": SelfHostedModels.MODEL_DATABRICKS_DBRX,
    "cg-7b": SelfHostedModels.MODEL_GOOGLE_CODEGEMMA_7B,
    "mixtral8x22": SelfHostedModels.MODEL_MISTRALAI_MIXTRAL_8X22B,
    "llama-3-70b": SelfHostedModels.MODEL_META_LLAMA_3_70B  
}

model_tensor_types = {
    SelfHostedModels.MODEL_DEFOG_SQLCODER_70B: torch.float16,
    SelfHostedModels.MODEL_DEFOG_SQLCODER_7B_2: torch.float16,
    SelfHostedModels.MODEL_META_CODELLAMA_70B: torch.bfloat16,
    SelfHostedModels.MODEL_META_CODELLAMA_34B: torch.bfloat16,
    SelfHostedModels.MODEL_MISTRALAI_MISTRAL_7B: torch.bfloat16,
    SelfHostedModels.MODEL_MISTRALAI_MIXTRAL_8X7B: torch.bfloat16,
    SelfHostedModels.MODEL_WIZARDLM_WIZARD_CODER_33B: torch.bfloat16,
    SelfHostedModels.MODEL_MISTRALAI_MISTRAL_7B_V2: torch.bfloat16,
    SelfHostedModels.MODEL_DATABRICKS_DBRX: torch.bfloat16,
    SelfHostedModels.MODEL_GOOGLE_CODEGEMMA_7B: torch.bfloat16,
    SelfHostedModels.MODEL_MISTRALAI_MIXTRAL_8X22B: torch.bfloat16,
    SelfHostedModels.MODEL_META_LLAMA_3_70B: torch.bfloat16
}


def initialize_model_and_tokenizer(
    model_name: str,
) -> Tuple[PreTrainedTokenizer | PreTrainedTokenizerFast, Any]:
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=SelfHosted.MODEL_WEIGHTS_DIRECTORY,
        token=auth_token,
        trust_remote_code=model_name == SelfHostedModels.MODEL_DATABRICKS_DBRX,
    )

    if model_name == SelfHostedModels.MODEL_DATABRICKS_DBRX:
        gpu_max_memory = {0: "80GB", 1: "160GB", 2: "240GB", 3: "320GB"}
    else:
        gpu_max_memory = {0: "40GB", 1: "40GB", 2: "40GB", 3: "80GB"}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        max_memory=gpu_max_memory,
        cache_dir=SelfHosted.MODEL_WEIGHTS_DIRECTORY,
        token=auth_token,
        torch_dtype=model_tensor_types[model_name],
        trust_remote_code=model_name == SelfHostedModels.MODEL_DATABRICKS_DBRX,
    )

    return (tokenizer, model)


def test_model(tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, model: Any):
    # Setup a prompt
    prompt = "### User:What is the fastest car in  \
            the world and how much does it cost? \
            ### Assistant:"
    # Pass the prompt to the tokenizer
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Actually run the thing
    output = model.generate(
        **inputs, use_cache=True, max_new_tokens=Defaults.MAX_TOKENS_TO_GENERATE
    )

    # Covert the output tokens back to text
    return tokenizer.decode(output[0], skip_special_tokens=True)


def run_queries_on_model(
    total_user_query: Tuple,
    output_file_path: str,
    metrics_file_path: str,
    log_file_path: str,
    model_name: str,
    instruction_size: int,
    dataset_length: int,
    shot_size: str,
    model: Any,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> None:
    for index, user_query_data in enumerate(total_user_query):
        context, question, hardness, db_id, evidence = user_query_data
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

            system_prompt, examples = get_instruction_shot_specific_prompt(
                instruction_size, shot_size, db_id
            )

            if model_name == SelfHostedModels.MODEL_DATABRICKS_DBRX:
                prompt = [
                    {
                        "role": "system",
                    "content": system_prompt.replace("[context]", "")
                    .replace("[question]", "")
                    .replace("[hint]", "").replace("[examples]", ""),
                    },
                    {"role": "user", "content": f"Question: {question} \n Hint: {str(evidence)} \n Here is the schema of the tables which are needed for the SQL generation: \n {context}\n {examples}"},
                ]
                inputs = tokenizer.apply_chat_template(
                    prompt,
                    return_dict=True,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to("cuda")
                llm_prompt_tokens = len(inputs.input_ids[0])

            else:
                prompt = generate_model_specific_prompt_for_self_hosted_model(
                    model_name, system_prompt, context, question, evidence, examples
                )
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                llm_prompt_tokens =len(inputs[0])

            data_to_log["request"] = prompt
            print(f"Completed tokenization step for record '{index+1}'")

            response_time_start = datetime.now(timezone.utc)
            output = model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=Defaults.MAX_TOKENS_TO_GENERATE,
                pad_token_id=tokenizer.eos_token_id,
            )
            response_time_stop = datetime.now(timezone.utc)
            print(f"Completed output generation step for record '{index+1}'")

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

            data_to_log["response_time_start"] = response_time_start.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            data_to_log["response_time_stop"] = response_time_stop.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
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
            print(
                f"Completed inferencing for record '{index+1}' in {response_time} secs"
            )
        except Exception as ex:
            print(f"Error during inferencing for record: {index+1}")
            exception = str(ex)
            data_to_log["severity"] = "error"
            log(exception, data_to_log, log_file_path)
            print("exception: ", exception)
            write_to_file(
                output_file_path,
                metrics_file_path,
                f"An error occurred: {exception}\n\n",
                f"{0},{0},{0},{hardness}\n",
            )


def run_inferences() -> None:
    args, model_instructions = get_parsed_args(supported_models, HOST_ENV)
    inference_length_in_args = [int(inst) for inst in args.inf_length.split(",")]
    dataset_length_list = inference_length_in_args or Defaults.INFERENCE_LENGTH_LIST
    shot_size_list = args.shot_size.split(",")

    datasets_info = get_datasets_info(dataset_length_list)

    for model_name_from_args in args.models.split(","):
        if model_instructions:
            instruction_size_list = model_instructions[model_name_from_args]
        elif args.inst:
            instruction_size_list = [int(inst) for inst in args.inst.split(",")]
        else:
            instruction_size_list = Defaults.INSTRUCTION_SIZE_LIST

        model_name = supported_models[model_name_from_args]
        tokenizer, model = initialize_model_and_tokenizer(model_name)

        for shot_size in shot_size_list:
            if "cot" in shot_size:
                file_shot_size = shot_size.split("-")[0] + "_shot_cot"
            else:
                file_shot_size = shot_size + "_shot"

            for instruction_size in instruction_size_list:
                for dataset_length, query_list, gold_file_list in datasets_info:
                    model_file_path = f"{args.target_dir}/{HOST_ENV}/{file_shot_size}/{model_name}/{instruction_size}_Instructions/{dataset_length}_Inferences"

                    output_file_path, metrics_file_path, log_file_path = (
                        initialize_files(model_file_path, False)
                    )
                    if os.path.exists(model_file_path) and os.path.isfile(
                        log_file_path
                    ):
                        num_lines = 0
                        with open(log_file_path, "rb") as file:
                            num_lines = sum(1 for _ in file)

                        if num_lines == dataset_length:
                            continue
                        else:
                            print(
                                f"Starting loop for {model_name} - {file_shot_size} prompt - {instruction_size} instructions - {dataset_length} inferences - resuming from {num_lines+1}"
                            )
                            loop_start_time = datetime.now()
                            run_queries_on_model(
                                query_list[num_lines:],
                                output_file_path,
                                metrics_file_path,
                                log_file_path,
                                model_name,
                                instruction_size,
                                dataset_length,
                                file_shot_size,
                                model,
                                tokenizer,
                            )
                            generate_gold_file(
                                gold_file_list, model_file_path, dataset_length
                            )
                            loop_end_time = datetime.now()
                            total_secs = (
                                loop_end_time - loop_start_time
                            ).total_seconds()
                            print(
                                f"Time taken for {model_name} - {file_shot_size} prompt - {instruction_size} instructions - {dataset_length} inferences: {get_elapsed_time(total_secs)}"
                            )

                    else:
                        output_file_path, metrics_file_path, log_file_path = (
                            initialize_files(model_file_path)
                        )

                        print(
                            f"Starting loop for {model_name} - {file_shot_size} prompt - {instruction_size} instructions - {dataset_length} inferences"
                        )
                        loop_start_time = datetime.now()
                        run_queries_on_model(
                            query_list,
                            output_file_path,
                            metrics_file_path,
                            log_file_path,
                            model_name,
                            instruction_size,
                            dataset_length,
                            file_shot_size,
                            model,
                            tokenizer,
                        )
                        generate_gold_file(
                            gold_file_list, model_file_path, dataset_length
                        )
                        loop_end_time = datetime.now()
                        total_secs = (loop_end_time - loop_start_time).total_seconds()
                        print(
                            f"Time taken for {model_name} - {file_shot_size} prompt - {instruction_size} instructions - {dataset_length} inferences: {get_elapsed_time(total_secs)}"
                        )


if __name__ == "__main__":
    run_inferences()
