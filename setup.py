import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

install_requires = []
with open("./requirements.txt", "r") as requirements_file:
    reqs = [r.strip() for r in requirements_file.readlines()]
    for r in reqs:
        install_requires.append(r)

quant_cuda_module = CUDAExtension(
    'alpaca_lora_4bit.quant_cuda',
    sources=[
        'src/alpaca_lora_4bit/quant_cuda/quant_cuda.cpp',
        'src/alpaca_lora_4bit/quant_cuda/quant_cuda_kernel.cu'
    ])

setup(
    name='alpaca_lora_4bit',
    version='0.1.3',
    description='Alpaca LoRA 4-bit',
    package_dir={'alpaca_lora_4bit': 'src/alpaca_lora_4bit'},
    packages=['alpaca_lora_4bit', 'alpaca_lora_4bit.monkeypatch', 'alpaca_lora_4bit.server', 'alpaca_lora_4bit.quant_cuda'],
    install_requires=install_requires,
    extras_require={
        'triton': 'triton',
        'dev': [
            'pytest',
        ],
    },
    ext_modules=[quant_cuda_module],
    cmdclass={'build_ext': BuildExtension},
)
