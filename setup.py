import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

install_requires = []
with open("./requirements.txt", "r") as requirements_file:
    # don't include gptq_llama by default
    reqs = [r.strip() for r in requirements_file.readlines() if "gptq_llama" not in r]
    for r in reqs:
        install_requires.append(r)

quant_cuda_module = CUDAExtension(
    'alpaca_lora_4bit.quant_cuda',
    sources=[
        'src/alpaca_lora_4bit/quant_cuda/quant_cuda.cpp',
        'src/alpaca_lora_4bit/quant_cuda/quant_cuda_kernel.cu'
    ])

# conditionally only install the cuda extension explicitly
ext_modules = []
cmdclass = {}
if '--cuda' in sys.argv or any(["cuda" in arg for arg in sys.argv]):
    ext_modules.append(quant_cuda_module)
    cmdclass = {'build_ext': BuildExtension}
    sys.argv.remove('--cuda')


setup(
    name='alpaca_lora_4bit',
    version='0.1',
    description='Alpaca LoRA 4-bit',
    package_dir={'alpaca_lora_4bit': 'src/alpaca_lora_4bit'},
    packages=['alpaca_lora_4bit', 'alpaca_lora_4bit.monkeypatch'],
    install_requires=install_requires,
    extras_require={
        'cuda': 'gptq_llama @ git+https://github.com/sterlind/GPTQ-for-LLaMa.git@lora_4bit',
        'triton': 'triton',
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
