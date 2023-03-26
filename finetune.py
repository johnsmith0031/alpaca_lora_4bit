"""
    llama-4b trainer with support of Stanford Alpaca-like JSON datasets (short for SAD)
    Intended to use with https://github.com/johnsmith0031/alpaca_lora_4bit
    
    SAD structure:
    [
        {
            "instruction": "Give null hypothesis",
            "input": "6 subjects were given a drug (treatment group) and an additional 6 subjects a placebo (control group).",
            "output": "Drug is equivalent of placebo"
        },
        {
            "instruction": "What does RNA stand for?",
            "input": "",
            "output": "RNA stands for ribonucleic acid."
        }
    ]
"""

import sys

import peft
import peft.tuners.lora
assert peft.tuners.lora.is_gptq_available()

import torch
import transformers
from autograd_4bit import load_llama_model_4bit_low_ram
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, PeftModel

# ! Config
from arg_parser import get_config
import train_data

ft_config = get_config()

# * Show loaded parameters
if ft_config.local_rank == 0:
    print(f"{ft_config}\n")

if ft_config.gradient_checkpointing:
    print('Disable Dropout.')

# Load Basic Model
model, tokenizer = load_llama_model_4bit_low_ram(ft_config.llama_q4_config_dir, ft_config.llama_q4_model, device_map=ft_config.device_map)

# Config Lora
lora_config = LoraConfig(
    r=ft_config.lora_r,
    lora_alpha=ft_config.lora_alpha,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=ft_config.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)
if ft_config.lora_apply_dir is None:
    model = get_peft_model(model, lora_config)
else:
    model = PeftModel.from_pretrained(model, ft_config.lora_apply_dir, device_map={'': 0}, torch_dtype=torch.float32)  # ! Direct copy from inference.py
    print(ft_config.lora_apply_dir, 'loaded')

# Scales to half
print('Fitting 4bit scales and zeros to half')
for n, m in model.named_modules():
    if '4bit' in str(type(m)):
        m.zeros = m.zeros.half()
        m.scales = m.scales.half()

# Set tokenizer
tokenizer.pad_token_id = 0

if not ft_config.skip:
    # Load Data
    data = None
    match ft_config.ds_type:
        case "txt" if not ft_config.skip:
            #### LLaMA
            data = train_data.TrainTxt(ft_config.dataset, ft_config.val_set_size, tokenizer, ft_config.cutoff_len)
        case "alpaca" if not ft_config.skip:
            #### Stanford Alpaca-like Data
            data = train_data.TrainSAD(ft_config.dataset, ft_config.val_set_size, tokenizer, ft_config.cutoff_len)
        case _:
            raise NotImplementedError("ERROR: Unknown dataset format")
    data.prepare_data()
    ####

    # Use gradient checkpointing
    if ft_config.gradient_checkpointing:
        print('Applying gradient checkpointing ...')
        from gradient_checkpointing import apply_gradient_checkpointing
        apply_gradient_checkpointing(model, checkpoint_ratio=ft_config.gradient_checkpointing_ratio)

    # Disable Trainer's DataParallel for multigpu
    if not ft_config.ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data.train_data,
        eval_dataset=data.val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=ft_config.mbatch_size,
            gradient_accumulation_steps=ft_config.gradient_accumulation_steps,
            warmup_steps=ft_config.warmup_steps,
            num_train_epochs=ft_config.epochs,
            learning_rate=ft_config.lr,
            fp16=True,
            logging_steps=ft_config.logging_steps,
            evaluation_strategy="no",
            save_strategy="steps",
            eval_steps=None,
            save_steps=ft_config.save_steps,
            output_dir=ft_config.lora_out_dir,
            save_total_limit=ft_config.save_total_limit,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ft_config.ddp else None,
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
model.save_pretrained(ft_config.lora_out_dir)

if ft_config.checkpoint:
    print("Warning: Merge model + LoRA and save the whole checkpoint not implemented yet.")
    
print('Model Saved.')