import torch
import random
import numpy as np
from huggingface_hub import hf_hub_download
import os


def get_test_alpaca_no_act_order_model():
    repository = "TheBloke/orca_mini_7B-GPTQ"
    files = [".gitattributes", "README.md", "config.json",
             "generation_config.json", "orca-mini-7b-GPTQ-4bit-128g.no-act.order.safetensors", 
             "quantize_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer.model",
             "tokenizer_config.json"]
    local_paths = {}
    for file in files:
        local_paths[file] = hf_hub_download(
            repo_id=repository,
            filename=file,
        )
    return os.path.dirname(local_paths["config.json"]), local_paths["orca-mini-7b-GPTQ-4bit-128g.no-act.order.safetensors"]


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)