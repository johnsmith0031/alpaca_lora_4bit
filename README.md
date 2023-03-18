# Alpaca Lora 4bit
Made some adjust for the code in peft and gptq for llama, and make it possible for lora finetuning with a 4 bits base model. The same adjustment can be made for 2, 3 and 8 bits.

# Requirements
gptq-for-llama: https://github.com/qwopqwop200/GPTQ-for-LLaMa
peft: https://github.com/huggingface/peft.git

# Install
copy files from GPTQ-for-LLaMa into GPTQ-for-LLaMa path and re-compile cuda extension
copy files from peft/tuners/lora.py to peft path, replace it

# Finetuning
The same finetune script from https://github.com/tloen/alpaca-lora can be used.

