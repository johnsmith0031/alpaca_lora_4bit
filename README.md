# Alpaca Lora 4bit
Made some adjust for the code in peft and gptq for llama, and make it possible for lora finetuning with a 4 bits base model. The same adjustment can be made for 2, 3 and 8 bits.
<br>
* Install Manual by s4rduk4r: https://github.com/s4rduk4r/alpaca_lora_4bit_readme/blob/main/README.md (**NOTE:** don't use the install script, use the requirements.txt instead.)
<br>
* Also Remember to create a venv if you do not want the packages be overwritten.
<br>

# Update Logs
* Resolved numerically unstable issue
<br>

* Reconstruct fp16 matrix from 4bit data and call torch.matmul largely increased the inference speed.
<br>

* Added install script for windows and linux.
<br>

* Added Gradient Checkpointing. Now It can finetune 30b model 4bit on a single GPU with 24G VRAM with Gradient Checkpointing enabled. (finetune.py updated) (but would reduce training speed, so if having enough VRAM this option is not needed)
<br>

* Added install manual by s4rduk4r
<br>

# Requirements
gptq-for-llama: https://github.com/qwopqwop200/GPTQ-for-LLaMa<br>
peft: https://github.com/huggingface/peft.git<br>
<br>

# Install
~copy files from GPTQ-for-LLaMa into GPTQ-for-LLaMa path and re-compile cuda extension~<br>
~copy files from peft/tuners/lora.py to peft path, replace it~<br>

**NOTE:** Install scripts are no longer needed! requirements.txt now pulls from forks with the necessary patches.

```
pip install -r requirements.txt
```

# Finetune
~The same finetune script from https://github.com/tloen/alpaca-lora can be used.~<br>

After installation, this script can be used:

```
python finetune.py
```

# Inference

After installation, this script can be used:

```
python inference.py
```

# Text Generation Webui Monkey Patch

Clone the latest version of text generation webui and copy all the files into ./text-generation-webui/
```
git clone https://github.com/oobabooga/text-generation-webui.git
```

Open server.py and insert a line at the beginning
```
import custom_monkey_patch # apply monkey patch
import gc
import io
...
```

Use the command to run

```
python server.py
```
