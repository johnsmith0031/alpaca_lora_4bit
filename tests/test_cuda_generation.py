import torch
from alpaca_lora_4bit.autograd_4bit import load_llama_model_4bit_low_ram, model_to_half
from alpaca_lora_4bit.amp_wrapper import AMPWrapper
import alpaca_lora_4bit.matmul_utils_4bit as mm4b
from utils import get_test_alpaca_no_act_order_model, set_seeds
import random
import numpy as np


def test_cuda_generation_no_act_order():
    config_path, weights_path = get_test_alpaca_no_act_order_model()
    model, tokenizer = load_llama_model_4bit_low_ram(
        config_path=config_path,
        model_path=weights_path,
        groupsize=128,
        is_v1_model=False,
    )
    model_to_half(model)
    AMPWrapper(model).apply_generate()

    prompt = '''I think the meaning of life is'''
    batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    batch = {k: v.cuda() for k, v in batch.items()}

    mm4b.act_order = False
    mm4b.faster_mode = "disabled"

    set_seeds(42)
    with torch.no_grad():
        generated = model.generate(inputs=batch["input_ids"],
                                do_sample=True, use_cache=True,
                                repetition_penalty=1.1,
                                max_new_tokens=20,
                                temperature=0.9,
                                top_p=0.95,
                                top_k=40,
                                return_dict_in_generate=True,
                                output_attentions=False,
                                output_hidden_states=False,
                                output_scores=False)
    result_text = tokenizer.decode(generated['sequences'].cpu().tolist()[0])
    raise ValueError(result_text)