
import numpy as np
from utils import get_test_alpaca_no_act_order_model, set_seeds
import pytest
import multiprocessing as mp


def inner_cuda_generation_no_act_order(flash_attn, mm4b_faster_mode):
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
    
    if flash_attn:
        replace_llama_attn_with_flash_attn()
    autograd_4bit.backend = "cuda"
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
    AMPWrapper(model).apply_generate()
    prompt = '''I think the meaning of life is to find happiness, and that's something you have to work for and fight for every day.'''
    batch_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    batch_input = {
        k: v.to(model.device)
        for k, v in batch_input.items()
    }
    set_seeds(42)
    with torch.no_grad():
        # Since output_scores gives me -Inf probabilities while text is not gibberish - sounds like some bug in generate method
        # So let's just make 1 more forward pass
        model.eval()
        scores = F.softmax(model(**batch_input).logits, dim=-1)
    scores_np = scores.detach().cpu().numpy()
    
    top_tokens_canonical = np.array([21130,   357,  1153,   287,   266,   322,
                                       289,   333, 12591, 31844,   291,   342,
                                     31876, 31829,   674,   342,   473,   289,
                                       623,   329,   291,   342,   329,   291,
                                       1124, 31843,   571])
    top_tokens_scores_canonical = np.array([3.765e-04, 1.888e-01, 3.259e-02,
                                            8.105e-01, 3.274e-01, 8.257e-01,
                                            6.084e-01, 2.734e-01, 6.250e-01,
                                            4.512e-01, 5.156e-01, 2.930e-01,
                                            2.186e-01, 1.000e+00, 6.196e-01,
                                            8.799e-01, 6.099e-01, 9.922e-01,
                                            8.315e-01, 5.923e-01, 2.910e-01,
                                            2.510e-01, 9.980e-01, 2.646e-01,
                                            7.036e-01, 5.073e-01, 1.936e-01])
    top_tokens = scores_np[0].argmax(axis=-1)
    top_tokens_scores = scores_np[0].max(axis=-1)
    assert all(top_tokens == top_tokens_canonical)
    assert all(np.abs(top_tokens_scores.astype(np.float32) - top_tokens_scores_canonical.astype(np.float32)) < 1e-3)


@pytest.mark.parametrize("flash_attn, mm4b_faster_mode", [
    (True, "disable"),
    (True, "old_faster"),
    (True, "faster"),
    (False, "disable"),
    (False, "old_faster"),
    (False, "faster"),
])
def test_cuda_generation_no_act_order(flash_attn, mm4b_faster_mode):
    process = mp.Process(target=inner_cuda_generation_no_act_order, args=(flash_attn, mm4b_faster_mode))
    process.start()
    process.join()
    assert process.exitcode == 0
