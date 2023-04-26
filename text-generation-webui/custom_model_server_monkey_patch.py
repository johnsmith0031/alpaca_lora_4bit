from alpaca_lora_4bit.server import ModelClient
from transformers import LlamaTokenizer

def load_model_llama(*args, **kwargs):
    config_path = '../llama-13b-4bit/'
    tokenizer = LlamaTokenizer.from_pretrained(config_path)
    tokenizer.truncation_side = 'left'
    model = ModelClient(port=5555, port_sub=5556)
    return model, tokenizer

patch_encode_func = True

# Monkey Patch
from modules import models
from modules import shared
models.load_model = load_model_llama
shared.args.model = 'llama-13b-4bit'
shared.settings['name1'] = 'You'
shared.settings['name2'] = 'Assistant'
shared.settings['chat_prompt_size_max'] = 2048
shared.settings['chat_prompt_size'] = 2048

if patch_encode_func:
    from modules import text_generation
    text_generation.encode_old = text_generation.encode
    def encode_patched(*args, **kwargs):
        input_ids = text_generation.encode_old(*args, **kwargs)
        if input_ids[0,0] == 0:
            input_ids = input_ids[:, 1:]
        return input_ids
    text_generation.encode = encode_patched
    print('Encode Function Patched.')
    
print('Monkey Patch Completed.')

# Apply Generate Monkey Patch
import generate_monkey_patch
