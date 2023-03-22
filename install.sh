#!/bin/bash

# This is an install script for Alpaca_LoRA_4bit

# makedir ./repository/ if not exists
if [ ! -d "./repository" ]; then
    mkdir ./repository
fi

# Clone repos into current repository into ./repository/
git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa.git ./repository/GPTQ-for-LLaMa
git clone https://github.com/huggingface/peft.git ./repository/peft
git clone https://github.com/huggingface/transformers.git ./repository/transformers

# Replace ./repository/peft/src/peft/tuners/lora.py with ./peft/tuners/lora.py
cp ./peft/tuners/lora.py ./repository/peft/src/peft/tuners/lora.py

# Replace ./repository/GPTQ-for-LLaMa/quant_cuda.cpp and quant_cuda_kernel.cu with ./GPTQ-for-LLaMa/quant_cuda.cpp and quant_cuda_kernel.cu
cp ./GPTQ-for-LLaMa/quant_cuda.cpp ./repository/GPTQ-for-LLaMa/quant_cuda.cpp
cp ./GPTQ-for-LLaMa/quant_cuda_kernel.cu ./repository/GPTQ-for-LLaMa/quant_cuda_kernel.cu

# Copy autograd_4bit.py into ./repository/GPTQ-for-LLaMa/autograd_4bit.py
cp ./GPTQ-for-LLaMa/autograd_4bit.py ./repository/GPTQ-for-LLaMa/autograd_4bit.py

# Install quant_cuda and cd into ./repository/GPTQ-for-LLaMa
cd ./repository/GPTQ-for-LLaMa
python setup_cuda.py install

echo "Install finished"
read -p "Press [Enter] to continue..."
