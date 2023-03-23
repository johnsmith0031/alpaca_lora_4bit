import os
import sys
sys.path.insert(0, './repository/transformers/src')
sys.path.insert(0, './repository/GPTQ-for-LLaMa')
sys.path.insert(0, './repository/peft/src')

import peft
import peft.tuners.lora
assert peft.tuners.lora.is_gptq_available()

import time
import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
import accelerate
from modelutils import find_layers
from autograd_4bit import make_quant_for_4bit_autograd
from autograd_4bit import load_llama_model_4bit_low_ram
from datasets import load_dataset, Dataset
import json
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model, get_peft_model_state_dict, PeftModel


# Parameters
DATA_PATH = "./data.txt"
OUTPUT_DIR = "alpaca_lora"
lora_path_old = ''
config_path = './llama-13b-4bit/'
model_path = './llama-13b-4bit.pt'

MICRO_BATCH_SIZE = 1
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 3
LEARNING_RATE = 2e-4
CUTOFF_LEN = 256
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 0
TARGET_MODULES = [
    "q_proj",
    "v_proj",
]
GRADIENT_CHECKPOINTING = False
GRADIENT_CHECKPOINTING_RATIO = 1
warmup_steps = 50
save_steps = 50
save_total_limit = 3
logging_steps = 10

# Load Basic Model
model, tokenizer = load_llama_model_4bit_low_ram(config_path, model_path)

# Config Lora
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
if lora_path_old == '':
    model = get_peft_model(model, config)
else:
    model = PeftModel.from_pretrained(model, lora_path_old)
    print(lora_path_old, 'loaded')

# Scales to half
print('Fitting 4bit scales and zeros to half')
for n, m in model.named_modules():
    if '4bit' in str(type(m)):
        m.zeros = m.zeros.half()
        m.scales = m.scales.half()

# Set tokenizer
tokenizer.pad_token_id = 0

# Load Data
with open(DATA_PATH, 'r', encoding='utf8') as file:
    txt = file.read()
txt = txt.replace('\r\n', '\n')
rows = [r for r in txt.split('\n') if r != '']
data = Dataset.from_dict({"input": rows})
exceed_count = 0
def tokenize(prompt):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    global exceed_count
    prompt = prompt['input']
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    d = {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }
    if sum(d['attention_mask']) >= CUTOFF_LEN:
        exceed_count += 1
    return d
data = data.shuffle().map(lambda x: tokenize(x))
print('Train Data: {:.2f}%'.format(exceed_count / len(data) * 100), 'outliers')
train_data = data

# Use gradient checkpointing
if GRADIENT_CHECKPOINTING:
    print('Applying gradient checkpointing ...')
    from gradient_checkpointing import apply_gradient_checkpointing
    apply_gradient_checkpointing(model, checkpoint_ratio=GRADIENT_CHECKPOINTING_RATIO)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=warmup_steps,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=logging_steps,
        evaluation_strategy="no",
        save_strategy="steps",
        eval_steps=None,
        save_steps=save_steps,
        output_dir=OUTPUT_DIR,
        save_total_limit=save_total_limit,
        load_best_model_at_end=False
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False

# Set Model dict
old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

# Run Trainer
trainer.train()

print('Train completed.')

# Save Model
model.save_pretrained(OUTPUT_DIR)

print('Model Saved.')
