#FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
FROM nvidia/cuda:11.7.0-devel-ubuntu22.04

# Get 

RUN apt-get update && apt-get install -y git wget python3 python3-pip
RUN ln -s `which python3` /usr/bin/python

RUN pip3 install --upgrade pip requests tqdm

# Some of the requirements expect some python packages in their setup.py, just install them first.
RUN pip install torch==2.0.0
RUN pip install semantic-version==2.10.0

RUN git clone --depth=1 --branch main https://github.com/andybarry/alpaca_lora_4bit_docker.git && cd alpaca_lora_4bit
# && git checkout 86387a0a3575c82e689a452c20b2c9a5cc94a0f3

WORKDIR alpaca_lora_4bit

COPY requirements2.txt requirements2.txt
RUN pip install -r requirements2.txt

# The docker build environment has trouble detecting CUDA version, build for all reasonable archs
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6"
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN git clone --depth=1 --branch main https://github.com/andybarry/text-generation-webui-4bit.git text-generation-webui-tmp && cd text-generation-webui-tmp 
# && git checkout 378d21e80c3d6f11a4835e57597c69e340008e2c 

RUN mv -f text-generation-webui-tmp/* text-generation-webui/

# Get the model
RUN cd text-generation-webui && python download-model.py --text-only decapoda-research/llama-7b-hf && mv models/decapoda-research_llama-7b-hf ../llama-7b-4bit

RUN wget https://huggingface.co/decapoda-research/llama-7b-hf-int4/resolve/main/llama-7b-4bit.pt -O llama-7b-4bit.pt

# Get LoRA
RUN cd text-generation-webui && python download-model.py samwit/alpaca7B-lora && mv loras/samwit_alpaca7B-lora ../alpaca7b_lora

# Symlink for monkeypatch
RUN cd text-generation-webui && ln -s ../autograd_4bit.py ./autograd_4bit.py && ln -s ../matmul_utils_4bit.py .

# Run the server
WORKDIR /alpaca_lora_4bit/text-generation-webui
CMD ["python", "server.py"]