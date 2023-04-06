# Run LLM chat in realtime on an 8GB NVIDIA GPU

## Dockerfile for alpaca_lora_4bit
Based on https://github.com/johnsmith0031/alpaca_lora_4bit

## Use
Can run real-time LLM chat using alpaca on a 8GB NVIDIA/CUDA GPU (ie 3070 Ti mobile)

## Requirements
- linux with docker
- nvidia GPU

## Installation

```
docker build -t alpaca_lora_4bit .
docker run -p 7086:7086 alpaca_lora_4bit
```
Point your browser to http://localhost:7086

## Results
It's fast on a 3070 Ti.

### Discussion
The model isn't all that good, sometimes it goes crazy.  But hey, "when 4-bits _you reach_ look this good you will not."

But it is fast (on my 3070 Ti mobile at least)



## References

- https://github.com/johnsmith0031/alpaca_lora_4bit
- https://github.com/s4rduk4r/alpaca_lora_4bit_readme/blob/main/README.md
- https://github.com/tloen/alpaca-lora

