class Defaults:
    INFERENCE_LENGTH_LIST = [50, 100, 200, 400]
    INSTRUCTION_SIZE_LIST = [5, 7, 9, 11, 13]


class Environments:
    ANYSCALE = "anyscale"
    AMZ_BEDROCK = "amazon-bedrock"
    SELF_HOSTED = "self-hosted"
    HUGGING_FACE = "hugging-face"


class BedrockModels:
    MODEL_META_LLAMA = "meta.llama2-70b-chat-v1"
    MODEL_ANTHROPIC_CLAUDE = "anthropic.claude-v2"


class AnyscaleModels:
    MODEL_META_LLAMA = "meta-llama/Llama-2-70b-chat-hf"
    MODEL_META_CODELLAMA_70B = "codellama/CodeLlama-70b-Instruct-hf"
    MODEL_META_CODELLAMA_34B = "codellama/CodeLlama-34b-Instruct-hf"
    MODEL_MISTRALAI_MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.1"
    MODEL_MISTRALAI_MIXTRAL_8X7B = "mistralai/Mixtral-8x7B-Instruct-v0.1"
