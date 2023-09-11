# syntax = docker/dockerfile:experimental

# Dockerfile is split into parts because we want to cache building the requirements and downloading the model, both of which can take a long time.

FROM nvidia/cuda:11.7.0-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y python3 python3-pip git

RUN pip3 install --upgrade pip

# Some of the requirements expect some python packages in their setup.py, just install them first.
RUN --mount=type=cache,target=/root/.cache/pip pip install --user torch==2.0.0
RUN --mount=type=cache,target=/root/.cache/pip pip install --user semantic-version==2.10.0 requests tqdm

# The docker build environment has trouble detecting CUDA version, build for all reasonable archs
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6"
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src src
RUN --mount=type=cache,target=/root/.cache pip install --user .

# -------------------------------

# Download the model
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04 AS downloader
RUN apt-get update && apt-get install -y wget

RUN wget --progress=bar:force:noscroll https://huggingface.co/decapoda-research/llama-7b-hf-int4/resolve/main/llama-7b-4bit.pt



# -------------------------------

#FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
FROM nvidia/cuda:11.7.0-devel-ubuntu22.04

RUN --mount=type=cache,target=/var/cache/apt apt-get update && apt-get install -y git python3 python3-pip

RUN ln -s `which python3` /usr/bin/python


# Copy the installed packages from the first stage
COPY --from=builder /root/.local /root/.local

RUN mkdir alpaca_lora_4bit
WORKDIR alpaca_lora_4bit

COPY --from=downloader llama-7b-4bit.pt llama-7b-4bit.pt

#RUN git clone --depth=1 --branch main https://github.com/andybarry/text-generation-webui-4bit.git text-generation-webui-tmp

RUN git clone --depth=1 --branch main https://github.com/oobabooga/text-generation-webui.git text-generation-webui-tmp

RUN --mount=type=cache,target=/root/.cache pip install --user markdown gradio

# Apply monkey patch
RUN cd text-generation-webui-tmp && printf '%s'"import custom_monkey_patch # apply monkey patch\nimport gc\n\n" | cat - server.py > tmpfile && mv tmpfile server.py

# Get the model config
RUN cd text-generation-webui-tmp && python download-model.py --text-only decapoda-research/llama-7b-hf && mv models/decapoda-research_llama-7b-hf ../llama-7b-4bit


# Get LoRA
RUN cd text-generation-webui-tmp && python download-model.py samwit/alpaca7b-lora && mv loras/samwit_alpaca7b-lora ../alpaca7b_lora

COPY src src
COPY text-generation-webui text-generation-webui
COPY src/alpaca_lora_4bit/monkeypatch text-generation-webui/monkeypatch

RUN mv -f text-generation-webui-tmp/* text-generation-webui/

# Symlink for monkeypatch
RUN cd text-generation-webui && ln -s ../src/alpaca_lora_4bit/autograd_4bit.py ./autograd_4bit.py && ln -s ../src/alpaca_lora_4bit/matmul_utils_4bit.py . && ln -s ../src/alpaca_lora_4bit/models.py .

# Swap to the 7bn parameter model
RUN sed -i 's/llama-13b-4bit/llama-7b-4bit/g' text-generation-webui/custom_monkey_patch.py && sed -i 's/alpaca13b_lora/alpaca7b_lora/g' text-generation-webui/custom_monkey_patch.py

# Run the server
WORKDIR /alpaca_lora_4bit/text-generation-webui
CMD ["python", "-u", "server.py", "--listen", "--chat"]
