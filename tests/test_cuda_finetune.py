import numpy as np
from utils import get_test_alpaca_no_act_order_model, set_seeds
import pytest
import multiprocessing as mp


def inner_cuda_finetune(flash_attn, mm4b_faster_mode):
    """
    Since we do monkey patching - we can not run, for instance:
    1. test with flash-attention patch
    2. than another one without it
    3. than again with it
    So I make this function work inside separated process instead
    """
    import torch
    import torch.nn.functional as F
    from alpaca_lora_4bit import autograd_4bit
    from alpaca_lora_4bit.amp_wrapper import AMPWrapper
    import alpaca_lora_4bit.matmul_utils_4bit as mm4b
    from alpaca_lora_4bit.monkeypatch.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
    from alpaca_lora_4bit.gradient_checkpointing import apply_gradient_checkpointing
    from alpaca_lora_4bit.monkeypatch.peft_tuners_lora_monkey_patch import replace_peft_model_with_int4_lora_model
    from torch.optim import Adam
    
    replace_peft_model_with_int4_lora_model()

    from peft import LoraConfig, get_peft_model

    
    if flash_attn:
        replace_llama_attn_with_flash_attn()

    autograd_4bit.switch_backend_to("cuda")
    mm4b.act_order = False
    mm4b.faster_mode = mm4b_faster_mode

    config_path, weights_path = get_test_alpaca_no_act_order_model()
    model, tokenizer = autograd_4bit.load_llama_model_4bit_low_ram(
        config_path=config_path,
        model_path=weights_path,
        groupsize=128,
        is_v1_model=False,
        bits=4,
    )
    autograd_4bit.model_to_half(model)
    AMPWrapper(model).apply_forward()

    prompt = '''I think the meaning of life is to find happiness, and that's something you have to work for and fight for every day.'''
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    batch_input = {
        "input_ids": input_ids[:, :-1].to(model.device),
        "labels": input_ids[:, 1:].to(model.device),
    }
    
    model.eval()
    with torch.no_grad():
        loss_original = model(**batch_input).loss.item()

    set_seeds(42)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    lora_model = get_peft_model(model, lora_config)
    for _, m in lora_model.named_modules():
        if 'Autograd4bitQuantLinear' in str(type(m)) or 'Linear4bitLt' in str(type(m)):
            if hasattr(m, "is_v1_model") and m.is_v1_model:
                m.zeros = m.zeros.half()
            m.scales = m.scales.half()
    apply_gradient_checkpointing(lora_model)

    optimizer = Adam(lora_model.parameters(), lr=1e-4)
    for _ in range(25):
        optimizer.zero_grad()
        loss = lora_model(**batch_input).loss
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        loss_tuned = model(**batch_input).loss.item()

    assert loss_original >= 10.0 and loss_tuned <= 3.0


@pytest.mark.parametrize("flash_attn, mm4b_faster_mode, fails", [
    (False, "disable", False),
    (False, "old_faster", False),
    (False, "faster", False),
])
def test_cuda_forwardpass_no_act_order(flash_attn, mm4b_faster_mode, fails):
    # I don't do FlashAttention tests here right now because my GPU can't do backward pass with currect flash-attn implementation on LLAMA-like models
    process = mp.Process(target=inner_cuda_finetune, args=(flash_attn, mm4b_faster_mode))
    process.start()
    process.join()
    if fails:
        assert process.exitcode != 0
    else:
        assert process.exitcode == 0
