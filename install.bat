REM This is a install script for Alpaca_LoRA_4bit

REM makedir ./repository/ if not exists
if not exist .\repository mkdir .\repository

REM Clone repos into current repository into ./repository/
git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa.git ./repository/GPTQ-for-LLaMa
git clone https://github.com/huggingface/peft.git ./repository/peft
git clone https://github.com/huggingface/transformers.git ./repository/transformers

REM replace ./repository/peft/src/peft/tuners/lora.py with ./peft/tuners/lora.py
copy .\peft\tuners\lora.py .\repository\peft\src\peft\tuners\lora.py /Y

REM replace ./repository/GPTQ-for-LLaMa/quant_cuda.cpp and quant_cuda_kernel.cu with ./GPTQ-for-LLaMa/quant_cuda.cpp and quant_cuda_kernel.cu
copy .\GPTQ-for-LLaMa\quant_cuda.cpp .\repository\GPTQ-for-LLaMa\quant_cuda.cpp /Y
copy .\GPTQ-for-LLaMa\quant_cuda_kernel.cu .\repository\GPTQ-for-LLaMa\quant_cuda_kernel.cu /Y

REM copy files into ./repository/GPTQ-for-LLaMa/
copy .\GPTQ-for-LLaMa\autograd_4bit.py .\repository\GPTQ-for-LLaMa\autograd_4bit.py /Y
copy .\GPTQ-for-LLaMa\gradient_checkpointing.py .\repository\GPTQ-for-LLaMa\gradient_checkpointing.py /Y

REM install quant_cuda
cd .\repository\GPTQ-for-LLaMa
python setup_cuda.py install

echo "Install finished"
@pause
