import os
from colorama import init, Fore, Back, Style
init(autoreset=True)

try:
    GPTQ_VERSION = int(os.environ["GPTQ_VERSION"])
except:
    print(Style.BRIGHT + Fore.YELLOW + "GPTQ_VERSION environment not provided. Fallback to GPTQv1")
    GPTQ_VERSION = 1  # Fallback version

loader = None


if GPTQ_VERSION == 1:
    from .autograd_4bit_v1 import Autograd4bitQuantLinear, load_llama_model_4bit_low_ram            
    print(Style.BRIGHT + Fore.GREEN + "GPTQv1 set")
elif GPTQ_VERSION == 2:
    from .autograd_4bit_v2 import Autograd4bitQuantLinear, load_llama_model_4bit_low_ram
    print(Style.BRIGHT + Fore.GREEN + "GPTQv2 set")
else:
    raise ValueError("GPTQ_VERSION not set or invalid")